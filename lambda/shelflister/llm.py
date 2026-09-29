"""Provider layer: the two LLM call shapes the pipeline needs, for Anthropic and Gemini.

  structured(system, user, schema)          -> (pydantic instance, usage)
  tool_loop(system_blocks, user, tools, run) -> (submitted dict | None, trace, usage)

Model selection: SHELFLISTER_MODEL or --model. Names starting with 'gemini' go to Google, everything
else to Anthropic. Aliases: sonnet -> claude-sonnet-5-5, opus -> claude-opus-5-5, gemini -> gemini-3.8-flash.
Keys: Anthropic from CLAUDE_PAID_API or ANTHROPIC_API_KEY; Gemini from GEMINI_API_KEY / GOOGLE_API_KEY / GOOGLE_AI.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Callable

from pydantic import BaseModel

DEFAULT_MODEL = os.environ.get("SHELFLISTER_MODEL", "gemini-3.8-flash")

MODEL_ALIASES = {
    "gemini": "gemini-3.8-flash",
    "flash": "gemini-3.8-flash",
    "sonnet": "claude-sonnet-5-5",
    "sonnet-5.5": "claude-sonnet-5-5",
    "sonnet-5": "claude-sonnet-5",
    "opus": "claude-opus-5-5",
    "opus-5.5": "claude-opus-5-5",
    "opus-5": "claude-opus-5",
}


@dataclass
class ToolSpec:
    name: str
    description: str
    schema: dict            # JSON schema for the input
    strict: bool = False


@dataclass
class LoopResult:
    submitted: dict | None
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


_NUDGE = ("You have {n} turns left. Stop exploring: decide with what you have, run at most one more check, "
          "and call {submit} now. If something is unresolved, say so in open_questions and set needs_review "
          "rather than continuing to browse.")


class Provider:
    name = "base"
    on_event: Callable[[dict], None] | None = None     # live trace: {"tool", "input", "output"} / {"text"}

    def emit(self, event: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event)
            except Exception:  # noqa: BLE001  (a progress hook must never break the loop)
                pass

    def structured(self, system: str, user: str, schema: type[BaseModel], *, max_tokens: int = 8000) -> tuple[BaseModel, dict]:
        raise NotImplementedError

    def tool_loop(self, system_blocks: list[str], user: str, tools: list[ToolSpec], run: Callable[[str, dict], str],
                  *, submit_tool: str = "submit", max_turns: int = 24, max_tokens: int = 16000,
                  check_submit: Callable[[dict], str | None] | None = None) -> LoopResult:
        """check_submit(input) -> error text or None: a rejected submit goes back to the model as a tool error."""
        raise NotImplementedError


def _take_submit(inp: dict, check_submit) -> tuple[dict | None, str]:
    err = check_submit(inp) if check_submit else None
    return (None, "ERROR: " + err) if err else (inp, "submitted")


# --------------------------------------------------------------------------- Anthropic
def _strict_schema(schema):
    """Anthropic strict tools need additionalProperties: false on every object, nested ones included."""
    if isinstance(schema, dict):
        out = {k: _strict_schema(v) for k, v in schema.items()}
        if out.get("type") == "object":
            out.setdefault("additionalProperties", False)
        return out
    if isinstance(schema, list):
        return [_strict_schema(x) for x in schema]
    return schema


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, model: str, effort: str = "high", client=None):
        import anthropic
        key = os.environ.get("CLAUDE_PAID_API") or os.environ.get("ANTHROPIC_API_KEY")
        if client is None and not key:
            raise RuntimeError("set CLAUDE_PAID_API or ANTHROPIC_API_KEY to use a Claude model")
        self.client = client or anthropic.Anthropic(api_key=key)
        self.model, self.effort = model, effort

    def structured(self, system, user, schema, *, max_tokens=8000):
        resp = self.client.messages.parse(
            model=self.model, max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}], output_format=schema)
        if resp.stop_reason == "refusal":
            raise RuntimeError(f"refused: {resp.stop_details}")
        return resp.parsed_output, {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                                    "cache_read": getattr(resp.usage, "cache_read_input_tokens", 0) or 0}

    def tool_loop(self, system_blocks, user, tools, run, *, submit_tool="submit", max_turns=24, max_tokens=16000, check_submit=None):
        system = [{"type": "text", "text": b, "cache_control": {"type": "ephemeral"}} for b in system_blocks[:2]]
        system += [{"type": "text", "text": b} for b in system_blocks[2:]]
        tdefs = [{"name": t.name, "description": t.description,
                  "input_schema": _strict_schema(t.schema) if t.strict else t.schema,
                  **({"strict": True} if t.strict else {})} for t in tools]
        messages = [{"role": "user", "content": user}]
        trace, usage = [], {"input": 0, "output": 0, "cache_read": 0, "turns": 0}
        submitted = None
        for turn in range(max_turns):
            if turn == max_turns - 4:
                messages.append({"role": "user", "content": _NUDGE.format(n=3, submit=submit_tool)})
            with self.client.messages.stream(model=self.model, max_tokens=max_tokens, system=system, messages=messages,
                                             tools=tdefs, thinking={"type": "adaptive", "display": "summarized"},
                                             output_config={"effort": self.effort}) as stream:
                resp = stream.get_final_message()
            usage["turns"] += 1
            usage["input"] += resp.usage.input_tokens
            usage["output"] += resp.usage.output_tokens
            usage["cache_read"] += getattr(resp.usage, "cache_read_input_tokens", 0) or 0
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"refused: {resp.stop_details}")
            messages.append({"role": "assistant", "content": resp.content})
            for b in resp.content:
                if b.type == "thinking" and b.thinking:
                    trace.append({"thinking": b.thinking})
                elif b.type == "text" and b.text.strip():
                    trace.append({"text": b.text})
                    self.emit({"text": b.text[:300]})
            calls = [b for b in resp.content if b.type == "tool_use"]
            if not calls:
                messages.append({"role": "user", "content": f"Please call {submit_tool} with your final answer."})
                continue
            results = []
            for tu in calls:
                inp = tu.input if isinstance(tu.input, dict) else json.loads(tu.input)
                if tu.name == submit_tool:
                    submitted, out = _take_submit(inp, check_submit)
                else:
                    out = run(tu.name, inp)
                trace.append({"tool": tu.name, "input": inp, "output": out[:4000]})
                self.emit({"tool": tu.name, "input": inp, "output": out[:300]})
                results.append({"type": "tool_result", "tool_use_id": tu.id, "content": out, "is_error": out.startswith("ERROR:")})
            messages.append({"role": "user", "content": results})
            if submitted:
                break
        return LoopResult(submitted, trace, usage)


# --------------------------------------------------------------------------- Gemini
def _gemini_schema(schema: dict) -> dict:
    """Gemini's function-declaration schema subset: drop keys it rejects."""
    if isinstance(schema, dict):
        return {k: _gemini_schema(v) for k, v in schema.items() if k not in ("additionalProperties", "default", "title", "$schema")}
    if isinstance(schema, list):
        return [_gemini_schema(x) for x in schema]
    return schema


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str, client=None, thinking_level: str | None = None):
        import logging
        from google import genai
        logging.getLogger("google_genai.models").setLevel(logging.ERROR)   # AFC advisory is noise for a manual loop
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_AI")
        if client is None and not key:
            raise RuntimeError("set GEMINI_API_KEY, GOOGLE_API_KEY, or GOOGLE_AI to use a Gemini model")
        self.client = client or genai.Client(api_key=key)
        self.model = model
        self.thinking_level = thinking_level or os.environ.get("SHELFLISTER_GEMINI_THINKING")

    def _cfg(self, **kw):
        from google.genai import types
        if self.thinking_level:
            kw["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        return types.GenerateContentConfig(**kw)

    @staticmethod
    def _usage(resp) -> dict:
        u = getattr(resp, "usage_metadata", None)
        return {"input": getattr(u, "prompt_token_count", 0) or 0, "output": getattr(u, "candidates_token_count", 0) or 0,
                "thinking": getattr(u, "thoughts_token_count", 0) or 0, "cache_read": getattr(u, "cached_content_token_count", 0) or 0}

    def structured(self, system, user, schema, *, max_tokens=8000):
        last = None
        for attempt in range(2):
            resp = self.client.models.generate_content(
                model=self.model, contents=user,
                config=self._cfg(system_instruction=system, response_mime_type="application/json", response_schema=schema,
                                 max_output_tokens=max(max_tokens, 16000) * (attempt + 1)))   # Gemini counts thinking in the budget
            try:
                obj = resp.parsed
                if obj is None:
                    obj = schema.model_validate_json(resp.text)
                return obj, self._usage(resp)
            except Exception as e:  # noqa: BLE001  (truncated / invalid JSON: retry once with more room)
                last = e
        raise RuntimeError(f"structured output failed twice: {last}")

    def tool_loop(self, system_blocks, user, tools, run, *, submit_tool="submit", max_turns=24, max_tokens=16000, check_submit=None):
        from google.genai import types
        decls = [types.FunctionDeclaration(name=t.name, description=t.description, parameters=_gemini_schema(t.schema) or None)
                 for t in tools]
        cfg = self._cfg(system_instruction="\n\n".join(system_blocks), tools=[types.Tool(function_declarations=decls)],
                        max_output_tokens=max_tokens,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True))
        contents = [types.Content(role="user", parts=[types.Part.from_text(text=user)])]
        trace, usage = [], {"input": 0, "output": 0, "thinking": 0, "cache_read": 0, "turns": 0}
        submitted = None
        for turn in range(max_turns):
            if turn == max_turns - 4:
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=_NUDGE.format(n=3, submit=submit_tool))]))
            resp = self.client.models.generate_content(model=self.model, contents=contents, config=cfg)
            u = self._usage(resp)
            usage["turns"] += 1
            for k in ("input", "output", "thinking", "cache_read"):
                usage[k] += u[k]
            cand = resp.candidates[0] if resp.candidates else None
            if cand is None or cand.content is None:
                trace.append({"text": f"(empty response: {getattr(resp, 'prompt_feedback', None)})"})
                break
            contents.append(cand.content)
            parts = cand.content.parts or []
            calls = [p.function_call for p in parts if getattr(p, "function_call", None)]
            for p in parts:
                if getattr(p, "text", None) and p.text.strip() and not getattr(p, "thought", False):
                    trace.append({"text": p.text})
                    self.emit({"text": p.text[:300]})
            if not calls:
                contents.append(types.Content(role="user", parts=[types.Part.from_text(text=f"Please call {submit_tool} with your final answer.")]))
                continue
            results = []
            for fc in calls:
                inp = dict(fc.args or {})
                if fc.name == submit_tool:
                    submitted, out = _take_submit(inp, check_submit)
                else:
                    out = run(fc.name, inp)
                trace.append({"tool": fc.name, "input": inp, "output": out[:4000]})
                self.emit({"tool": fc.name, "input": inp, "output": out[:300]})
                results.append(types.Part.from_function_response(name=fc.name, response={"result": out}))
            contents.append(types.Content(role="user", parts=results))
            if submitted:
                break
        return LoopResult(submitted, trace, usage)


def get_provider(model: str | None = None, effort: str | None = None, client=None) -> Provider:
    model = MODEL_ALIASES.get((model or DEFAULT_MODEL).lower(), model or DEFAULT_MODEL)
    if model.lower().startswith("gemini"):
        return GeminiProvider(model, client=client)
    return AnthropicProvider(model, effort=effort or os.environ.get("SHELFLISTER_EFFORT", "high"), client=client)
