"""LLM providers with a single structured-output interface.

    provider = get_provider("claude")      # claude-sonnet-5-5
    provider = get_provider("gemini")      # gemini-3.8-flash
    plan = provider.generate(SYSTEM, user_text, SearchPlan)   # -> SearchPlan instance

Keys: CLAUDE_PAID_API (or ANTHROPIC_API_KEY) and GOOGLE_AI (or GEMINI_API_KEY / GOOGLE_API_KEY).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import TypeVar

from pydantic import BaseModel, ValidationError

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

CLAUDE_DEFAULT_MODEL = "claude-sonnet-5-5"
GEMINI_DEFAULT_MODEL = "gemini-3.8-flash"


class LLMError(RuntimeError):
    pass


class Usage:
    """Running token/cost tally across calls in one pipeline run."""

    def __init__(self):
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cache_read_tokens = 0
        self.seconds = 0.0

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "seconds": round(self.seconds, 2),
        }


class Provider:
    name: str
    model: str

    def __init__(self):
        self.usage = Usage()
        self.trace: list[dict] = []  # optional record of prompts/responses for debugging

    def generate(self, system: str, user: str, schema: type[T]) -> T:
        t0 = time.time()
        self.last_meta = {}
        raw = self._generate(system, user, schema)
        self.usage.seconds += time.time() - t0
        self.usage.calls += 1
        try:
            obj = schema.model_validate_json(raw) if isinstance(raw, str) else schema.model_validate(raw)
        except ValidationError as e:
            raise LLMError(f"{self.name} returned JSON that failed {schema.__name__} validation: {e}\n{raw}") from e
        log.debug("%s %s: %s", self.name, schema.__name__, self.last_meta)
        self.trace.append(
            {"schema": schema.__name__, "system": system, "user": user, "response": obj.model_dump(), "meta": self.last_meta, "raw": raw}
        )
        return obj

    def _generate(self, system: str, user: str, schema: type[BaseModel]) -> str | dict:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Claude
# --------------------------------------------------------------------------- #


class ClaudeProvider(Provider):
    name = "claude"

    def __init__(self, model: str | None = None, api_key: str | None = None, effort: str = "medium"):
        super().__init__()
        import anthropic

        self.anthropic = anthropic
        self.model = model or CLAUDE_DEFAULT_MODEL
        self.effort = effort
        key = api_key or os.environ.get("CLAUDE_PAID_API") or os.environ.get("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key) if key else anthropic.Anthropic()

    def _generate(self, system: str, user: str, schema: type[BaseModel]) -> str:
        a = self.anthropic
        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                messages=[{"role": "user", "content": user}],
                output_config={
                    "effort": self.effort,
                    "format": {"type": "json_schema", "schema": schema.model_json_schema()},
                },
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
            )
        except a.RateLimitError as e:
            raise LLMError(f"Claude rate limited: {e}") from e
        except a.APIStatusError as e:
            raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
        except a.APIConnectionError as e:
            raise LLMError(f"Claude connection error: {e}") from e

        if resp.stop_reason == "refusal":
            detail = getattr(resp, "stop_details", None)
            raise LLMError(f"Claude refused the request: {detail}")
        if resp.stop_reason == "max_tokens":
            raise LLMError("Claude hit max_tokens before finishing the JSON output")
        self.last_meta = {
            "model": resp.model,
            "stop_reason": resp.stop_reason,
            "blocks": [b.type for b in resp.content],
            "fallback": [f"{b.from_.model}->{b.to.model}" for b in resp.content if b.type == "fallback"],
        }
        u = resp.usage
        self.usage.input_tokens += u.input_tokens
        self.usage.output_tokens += u.output_tokens
        self.usage.cache_read_tokens += getattr(u, "cache_read_input_tokens", 0) or 0
        return "".join(b.text for b in resp.content if b.type == "text")


# --------------------------------------------------------------------------- #
# Gemini
# --------------------------------------------------------------------------- #


def gemini_schema(schema: type[BaseModel]) -> dict:
    """Pydantic JSON schema -> the subset Gemini's response_schema accepts.

    Gemini rejects `additionalProperties` (which Claude requires) and `$ref`/`$defs`,
    so strip the former and inline the latter.
    """
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", {})

    def clean(node):
        if isinstance(node, list):
            return [clean(n) for n in node]
        if not isinstance(node, dict):
            return node
        if "$ref" in node:
            name = node["$ref"].split("/")[-1]
            return clean(defs[name])
        out = {}
        for k, v in node.items():
            if k in ("additionalProperties", "title", "default"):
                continue
            out[k] = clean(v)
        return out

    return clean(raw)


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, model: str | None = None, api_key: str | None = None, thinking_level: str | None = None):
        super().__init__()
        from google import genai

        self.model = model or GEMINI_DEFAULT_MODEL
        self.thinking_level = thinking_level
        key = (
            api_key
            or os.environ.get("GOOGLE_AI")
            or os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
        )
        if not key:
            raise LLMError("No Gemini key: set GOOGLE_AI (or GEMINI_API_KEY)")
        self.client = genai.Client(api_key=key)

    def _generate(self, system: str, user: str, schema: type[BaseModel]) -> str:
        from google.genai import types

        cfg = dict(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=gemini_schema(schema),
            temperature=0.2,
        )
        if self.thinking_level:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_level=self.thinking_level)
        last_err = None
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=user,
                    config=types.GenerateContentConfig(**cfg),
                )
                break
            except Exception as e:  # google.genai raises a family of ClientError/ServerError
                last_err = e
                status = getattr(e, "code", None) or getattr(e, "status_code", None)
                if status in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise LLMError(f"Gemini error: {e}") from e
        else:
            raise LLMError(f"Gemini error: {last_err}")

        try:
            self.last_meta = {"model": getattr(resp, "model_version", None), "finish_reason": str(resp.candidates[0].finish_reason)}
        except Exception:
            pass
        um = getattr(resp, "usage_metadata", None)
        if um:
            self.usage.input_tokens += um.prompt_token_count or 0
            self.usage.output_tokens += (um.candidates_token_count or 0) + (getattr(um, "thoughts_token_count", 0) or 0)
            self.usage.cache_read_tokens += getattr(um, "cached_content_token_count", 0) or 0
        text = resp.text
        if not text:
            fr = None
            try:
                fr = resp.candidates[0].finish_reason
            except Exception:
                pass
            raise LLMError(f"Gemini returned no text (finish_reason={fr})")
        return text


# --------------------------------------------------------------------------- #


def get_provider(name: str = "claude", model: str | None = None, **kw) -> Provider:
    name = (name or "claude").lower()
    if name in ("claude", "anthropic", "sonnet"):
        return ClaudeProvider(model=model, **kw)
    if name in ("gemini", "google"):
        return GeminiProvider(model=model, **kw)
    raise ValueError(f"unknown provider {name!r}; use 'claude' or 'gemini'")


def dump_trace(provider: Provider, path: str) -> None:
    with open(path, "w") as f:
        json.dump({"usage": provider.usage.as_dict(), "trace": provider.trace}, f, indent=2, ensure_ascii=False)
