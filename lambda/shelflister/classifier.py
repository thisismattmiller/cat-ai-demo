"""Classification as a schedule-first browsing loop.

The model must find the number in the LC classification schedules (full-text search over
captions and index terms, then browsing the hierarchy) and confirm it against what LC
has actually shelved there (the shelflist). Similar-record retrieval from the catalog
(the embedded-catalog lambda) is available as one more piece of evidence, and can be
switched off. The CSM classification chapters are in context.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import callnum, rules
from .classify import classify as lambda_classify
from .llm import Provider, ToolSpec
from .schedule import get_schedules
from .shelflist_api import ShelflistClient

CLASSIFY_CARDS = ["F010", "F210", "F240", "F275", "F320", "F585", "F615", "F632"]     # always in context
EXTRA_CARDS = ["F175", "F177", "F225", "F250", "F280", "F290", "F300", "F350", "F430", "F565", "F587", "F590",
               "F600", "F634", "F730"]                                                   # via load_rule

SYSTEM = """You are an experienced Library of Congress subject cataloger assigning the LC classification number
for a resource from its metadata (title, creator, summary, contents, subjects ...). Work the way a cataloger
works with the printed schedules:

1. Decide what the work is about and what form it takes (CSM F 010: class by subject, then form; the
   number for the work as a whole, not for one chapter).
2. Search the schedules (schedule_search) with topical words, and with the subject-heading vocabulary
   LC uses. Try several phrasings; add the discipline or country when results are scattered
   (e.g. "women authors English literature"). Index hits point to the number or range for a topic.
3. Browse (schedule_lookup) the candidate numbers: read the caption path, the notes ("Class here",
   "Including", "For ... see", "Cf."), the A-Z lists and tables. Follow "see" references. Prefer the
   most specific caption that describes the whole work; do not stop at a general number when a specific
   one exists, and do not use a number whose notes send this kind of work elsewhere.
4. Confirm with the shelflist (shelflist_in_class): what has LC actually shelved at this number? If the
   works there are of a different kind, reconsider. The shelflist also shows the Cutter LC already uses
   for a topic, place, or person under an A-Z caption.
5. Optionally consult similar_records (LC records with similar descriptions). Treat them as evidence of
   practice, not as the answer: verify any number they suggest in the schedules before using it.
6. Call submit once, with the class number exactly as it should appear in 050 $a (class letters, number,
   and any subject/geographic/author Cutter the schedule requires, e.g. 'PR6045.O72', 'TX749.5.B43',
   'HV1478.V4'), the caption path you found it under, the schedule notes/tables that applied, and the
   subject headings that fit. No book number, no date: the shelflist prints complete call numbers
   ('QA76.76.C65 S65 2011'), whose last Cutter and date belong to that one book, not to the class.

Literary works: an individual author's number lives in the national literature by period (PR, PS, PQ ...),
with the author Cutter from the A-Z list or, for a new author, from the Cutter table; say
needs_new_number=true when the schedule does not print the author. Biography: see F 275 for where the
biography of a person classes. When the schedule provides an A-Z list and the needed Cutter is not
printed, construct it from the Cutter table (G 063) and mark needs_new_number.

Be economical: the first message already contains schedule search hits for the title and subjects
(and similar LC records when available). Usually two or three lookups settle it. The CSM chapters on
classification follow; more are available with load_rule."""


@dataclass
class Classification:
    class_number: str | None
    caption_path: str
    schedule_basis: str
    subject_headings: list[str]
    reasoning: str
    alternatives: list[dict]
    confidence: str
    needs_new_number: bool
    open_questions: list[str]
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    lambda_candidates: list[dict] | None = None


def _tools(use_lambda: bool) -> list[ToolSpec]:
    S = {"type": "string"}
    t = [
        ToolSpec("schedule_search", "Full-text search over the 2024 LC classification schedules: captions (with their "
                 "ancestor path and notes) and the schedules' index terms. Returns class numbers with captions. Use several "
                 "phrasings; optionally restrict to a subclass (e.g. 'PR', 'TX').",
                 {"type": "object", "properties": {"query": S, "section": {**S, "description": "optional subclass letters"},
                                                   "limit": {"type": "integer"}}, "required": ["query"]}),
        ToolSpec("schedule_lookup", "The schedule entry for a class number: caption path with notes, entries printed under "
                 "or around it, referenced tables. Use it to browse from a search hit to the specific number.",
                 {"type": "object", "properties": {"class_number": S}, "required": ["class_number"]}),
        ToolSpec("schedule_table", "An LCC table by code (e.g. 'P-PZ40', 'H77a', 'E1', 'KF23') with its rows; large tables are "
                 "truncated, so pass find=<words> to filter rows.",
                 {"type": "object", "properties": {"code": S, "find": S}, "required": ["code"]}),
        ToolSpec("shelflist_in_class", "What LC has actually shelved at a class number (call numbers with the heading and "
                 "title of each work). Confirms the number is used for this kind of work and shows Cutters already assigned.",
                 {"type": "object", "properties": {"class_number": S}, "required": ["class_number"]}),
    ]
    t.append(ToolSpec("load_rule", "Load another CSM classification chapter by id (see the list in the system prompt).",
                      {"type": "object", "properties": {"card_id": S}, "required": ["card_id"]}))
    if use_lambda:
        t.append(ToolSpec("similar_records", "LC catalog records whose descriptions are most similar to this resource "
                          "(vector search), with their class numbers and subject headings. Evidence of LC practice; verify "
                          "in the schedules before relying on it.", {"type": "object", "properties": {}}))
    t.append(ToolSpec("submit", "Submit the classification decision. Call exactly once.",
                      {"type": "object", "properties": {
                          "class_number": {**S, "description": "e.g. 'PR6045.O72' - class part only, no book number or date"},
                          "caption_path": {**S, "description": "schedule caption path, root to leaf"},
                          "schedule_basis": {**S, "description": "the schedule notes, tables, or A-Z entries that determined the number"},
                          "subject_headings": {"type": "array", "items": S},
                          "reasoning": S,
                          "alternatives": {"type": "array", "items": {"type": "object", "properties": {"class_number": S, "why_not": S},
                                                                       "required": ["class_number", "why_not"]}},
                          "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                          "needs_new_number": {"type": "boolean"},
                          "cutter_is_class": {"type": "boolean", "description": "Normally false. True only if the schedule itself "
                                              "prints the last Cutter as part of the class number and a submit was rejected for it."},
                          "open_questions": {"type": "array", "items": S}},
                       "required": ["class_number", "caption_path", "schedule_basis", "subject_headings", "reasoning",
                                    "alternatives", "confidence", "needs_new_number", "cutter_is_class", "open_questions"],
                       "additionalProperties": False}, strict=True))
    return t


class Classifier:
    def __init__(self, provider: Provider, shelflist: ShelflistClient | None = None, use_lambda: bool = True):
        self.provider = provider
        self.shelflist = shelflist or ShelflistClient()
        self.use_lambda = use_lambda

    def _run_tool(self, name: str, inp: dict, ctx: dict) -> str:
        try:
            sch = get_schedules()
            if name == "schedule_search":
                return sch.search_text(inp["query"], int(inp.get("limit") or 12), inp.get("section") or None) if sch else "(schedules not available)"
            if name == "schedule_lookup":
                return sch.lookup(inp["class_number"]).text(max_children=25) if sch else "(schedules not available)"
            if name == "schedule_table":
                t = sch.table(inp["code"]) if sch else None
                return f"Table {t.code}: {t.title}\n{t.text(max_rows=40, find=inp.get('find'))}" if t else f"no table {inp['code']}"
            if name == "load_rule":
                c = rules.load_cards().get(inp["card_id"].replace(" ", "").upper())
                return c.text() if c else f"no card {inp['card_id']}"
            if name == "shelflist_in_class":
                es = self.shelflist.in_class(inp["class_number"])
                if not es:
                    return "(nothing shelflisted at this number)"
                lines = [e.short() for e in es[:40]]
                if len(es) > 40:
                    lines.append(f"... {len(es) - 40} more")
                return "\n".join(lines)
            if name == "similar_records":
                if ctx.get("lambda") is None:
                    r = ctx["resource"]
                    ctx["lambda"] = lambda_classify(r.title, creator=r.creator, summary=r.summary, content=r.contents)
                return ctx["lambda"].text()
            return f"unknown tool {name}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"

    def check_submit(self, inp: dict) -> str | None:
        """Reject a 'class number' that is really a complete call number (book Cutter and/or date included)."""
        cn = str(inp.get("class_number") or "").strip()
        c = callnum.try_parse(cn)
        if c is None:
            return f"'{cn}' is not a parseable LC class number; resubmit in 050 $a form, e.g. 'PR6045.O72' or 'TX749.5.B43'."
        if c.cutters and c.date:
            return (f"'{cn}' ends with a date. A date after a Cutter is part of the book number, which is assigned later. "
                    "Resubmit the class part only.")
        if c.cutters and not inp.get("cutter_is_class"):
            try:
                same = [e for e in self.shelflist.in_class(cn)
                        if (pe := callnum.try_parse(e.term)) and pe.cutters == c.cutters]
            except Exception:  # noqa: BLE001
                same = []
            if same:
                e = same[0]
                return (f"'{cn}' looks like a complete call number, not a class number: LC shelves '{e.term}' ({e.heading()}) "
                        f"exactly there, so '{c.cutters[-1]}' is that book's own Cutter. Resubmit the class part only "
                        "(what 050 $a holds). If the schedule genuinely prints this Cutter as part of the class number, "
                        "resubmit the same value with cutter_is_class=true.")
        return None

    def classify(self, resource, *, extra_context: str | None = None, max_turns: int = 14) -> Classification:
        cards = rules.load_cards()
        system_blocks = [SYSTEM + "\n\n" + "\n\n".join(cards[c].text() for c in CLASSIFY_CARDS if c in cards),
                         "Other CSM chapters (load_rule by id): " + "; ".join(f"{c} {cards[c].title}" for c in EXTRA_CARDS if c in cards)]
        user = f"Resource metadata:\n{resource.text()}\n"
        if resource.subjects:
            user += "\n(Subject headings were supplied; use them as strong evidence of the topic.)\n"
        # prefetch: schedule search on the title and subjects; lambda candidates if enabled
        sch = get_schedules()
        if sch:
            q = resource.title + " " + " ".join(resource.subjects[:4])
            user += f"\n{sch.search_text(q, limit=10)}\n"
        ctx = {"resource": resource, "lambda": None}
        if self.use_lambda:
            try:
                ctx["lambda"] = lambda_classify(resource.title, creator=resource.creator, summary=resource.summary, content=resource.contents)
                user += f"\n{ctx['lambda'].text()}\n"
            except Exception as e:  # noqa: BLE001
                user += f"\n(similar_records unavailable: {e})\n"
        if extra_context:
            user += f"\nAdditional context from the requester:\n{extra_context}\n"
        user += "\nFind the class number in the schedules, confirm it against the shelflist, then call submit."
        lr = self.provider.tool_loop(system_blocks, user, _tools(self.use_lambda),
                                     lambda n, i: self._run_tool(n, i, ctx), submit_tool="submit", max_turns=max_turns,
                                     check_submit=self.check_submit)
        s = lr.submitted or {}
        lam = ctx["lambda"]
        return Classification(
            class_number=s.get("class_number"), caption_path=s.get("caption_path", ""), schedule_basis=s.get("schedule_basis", ""),
            subject_headings=list(s.get("subject_headings") or []), reasoning=s.get("reasoning", "" if s else "model did not submit"),
            alternatives=list(s.get("alternatives") or []), confidence=s.get("confidence", "low"),
            needs_new_number=bool(s.get("needs_new_number")), open_questions=list(s.get("open_questions") or []),
            trace=lr.trace, usage=lr.usage,
            lambda_candidates=[c.__dict__ for c in lam.candidates] if lam else None)
