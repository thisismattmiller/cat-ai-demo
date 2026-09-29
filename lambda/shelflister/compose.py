"""Step 4: compose the call number. A tool-using LLM call that has the routed rule
cards in its system prompt and mechanical helpers + the LC shelflist as tools."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import callnum, cutter, dates, rules, tables
from .filing import files_before, filing_key, filing_word
from .llm import Provider, ToolSpec
from .profile import Profile
from .schedule import get_schedules
from .shelflist_api import Entry, ShelflistClient
from .validate import validate

MAX_TURNS = 12

SYSTEM_HEAD = """You are an experienced Library of Congress shelflisting technician. Given a resource's
metadata, the class number chosen for it, and a profile of the resource, complete the
LC call number: add the book number (Cutter or Cutters), the date, and any other element (volume
designation, work letter) exactly as the Classification and Shelflisting Manual (CSM) directs.

Order of authority (G 053, G 058): (1) explicit instructions in the classification schedule for this
number, (2) the CSM rules given below, (3) works already shelflisted in the same class number. Fit the
new number so it files correctly among the existing entries and leaves room for future ones.

Method (most cases need one or two tool calls: the mechanical suggestions below the metadata are
usually right; verify them against the neighbours, run validate_call_number once, and submit):
1. Read the schedule entry (if given), the profile, and the shelflist entries already in this class number. Identify the situation
   (single Cutter for main entry; second Cutter after a subject/geographic Cutter; biography table;
   literary author table; date-only class; reserved .A ranges; etc.).
2. Decide the Cutter basis and where the new entry files among its neighbours (G 100 filing rules).
   The position is fixed by the HEADINGS of the existing entries, read alphabetically, not by the
   table: find the entry whose heading files immediately before yours and the one immediately after,
   then choose a Cutter strictly between their Cutters (fit_cutter). Use cutter_table only as the
   starting suggestion and filing_compare to check order. Do not copy another work's Cutter just
   because it shares the author, unless that Cutter also sits in the right alphabetical position
   (LC sometimes has anomalies; a new number must still file correctly). Prefer at least two
   digits after the letter so later entries can be interfiled. A title is filed up to its first
   period or slash, so a subtitle after a colon counts.
3. Determine the date per G 140 (work letters when the same Cutter+date exists).
4. Run validate_call_number on your proposal and fix anything it reports. Its "files before/after ...
   but heading sorts ..." warnings mean the Cutter is in the wrong alphabetical slot: move it, or if
   you are certain the neighbour itself is the anomaly, say so in open_questions and set
   needs_review.
5. Call submit exactly once with the final call number, your reasoning, and the CSM sections you relied on.

Be precise about punctuation and spacing: class number, then '.' + first Cutter, then a space and the
second Cutter if any, then a space and the date (e.g. 'PR6045.O72 R66 2026', 'TX749.5.B43 A142 2006',
'HN670.3.Z9 C67'). If the existing shelflist entries under this class disagree with the table (older
Cutters), follow the shelflist: the new number must file correctly relative to what is there.
Under a literary author number: a work BY the author takes the title Cutter in the separate-works span of
the author table; criticism of ONE of the author's works files WITH that work, its title Cutter extended
by digits for the critic (G 340, F 632), not in the general biography/criticism span; general
biography and criticism take the critic's Cutter in the .Z5-.Z999 (or .xZ5-) span. Under a number
subarranged "by translator" or "by date" the shelflist shows the pattern LC actually follows: match it.
If the schedule almost certainly carries an instruction you cannot see (e.g. an author table in P), say
so in open_questions and make the best choice consistent with the shelflist entries.

The rule cards follow. Each is one CSM chapter reduced to its rules and examples."""


@dataclass
class ComposeResult:
    call_number: str | None
    reasoning: str
    citations: list[str]
    confidence: str
    open_questions: list[str]
    needs_review: bool
    validation: dict
    trace: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    cards: list[str] = field(default_factory=list)


def _tool_defs() -> list[ToolSpec]:
    S = {"type": "string"}
    raw = [
        {"name": "shelflist_in_class", "description": "All call numbers already shelflisted under a class number at LC "
         "(from the id.loc.gov shelflist browse), with the heading each is based on. Use it to see the neighbours you must "
         "file among. Copy statements are removed.", "input_schema": {"type": "object", "properties": {
             "class_number": {**S, "description": "e.g. 'PR6045.O72' or 'TX749.5.B43'"}}, "required": ["class_number"]}},
        {"name": "shelflist_browse", "description": "Window of the LC shelflist around any call number (about 100 entries "
         "before and after). Use to look at a wider neighbourhood or a different class.", "input_schema": {"type": "object",
         "properties": {"call_number": S, "count": {"type": "integer", "default": 60}}, "required": ["call_number"]}},
        {"name": "cutter_table", "description": "The Cutter the G 063 table suggests for a word (surname, first filing word). "
         "This is a starting point; it must be adjusted to the shelflist.", "input_schema": {"type": "object",
         "properties": {"word": S, "digits": {"type": "integer", "default": 2}}, "required": ["word"]}},
        {"name": "fit_cutter", "description": "Choose a Cutter with the given initial letter that files strictly after `prev` "
         "and strictly before `next` (existing Cutters with the same letter; omit one for an open end), preferring "
         "`suggested` when it fits and never ending in 0 or 1.", "input_schema": {"type": "object", "properties": {
             "letter": S, "prev": S, "next": S, "suggested": S}, "required": ["letter"]}},
        {"name": "filing_compare", "description": "Apply the G 100 filing rules to two headings: reports which files first. "
         "Set names=true for personal/place names (initial articles kept).", "input_schema": {"type": "object",
         "properties": {"a": S, "b": S, "names": {"type": "boolean", "default": False}}, "required": ["a", "b"]}},
        {"name": "regions_table", "description": "G 300 Regions and Countries Table lookup (Cutter for a country/region).",
         "input_schema": {"type": "object", "properties": {"place": S}, "required": ["place"]}},
        {"name": "states_table", "description": "G 302 U.S. states and Canadian provinces table lookup.",
         "input_schema": {"type": "object", "properties": {"place": S}, "required": ["place"]}},
        {"name": "biography_table", "description": "The G 320 Biography Table (.xA2 collected works ... .xA6-Z biography by main entry).",
         "input_schema": {"type": "object", "properties": {}}},
        {"name": "call_number_date", "description": "Reduce a 264/260 $c string to the G 140 call-number date.",
         "input_schema": {"type": "object", "properties": {"raw": S, "corporate_body": {"type": "boolean", "default": False}},
                          "required": ["raw"]}},
        {"name": "schedule_lookup", "description": "Look up a class number in the 2024 LC classification schedules: the "
         "printed entry or containing range, its ancestors with their notes (schedule instructions override the CSM), the "
         "entries printed under/around it, and any table it references. Use it to check neighbouring numbers or a "
         "different class.", "input_schema": {"type": "object", "properties": {"class_number": S}, "required": ["class_number"]}},
        {"name": "schedule_table", "description": "Fetch an LCC table by code (e.g. 'P-PZ40', 'H77a', 'E1', 'N6', 'KF23') "
         "with its rows; large tables are truncated, so pass find=<words> to filter rows (e.g. a committee name).",
         "input_schema": {"type": "object", "properties": {"code": S, "find": S}, "required": ["code"]}},
        {"name": "load_rule", "description": "Load the full text of a CSM rule card that is not already in your context "
         "(see the index of cards).", "input_schema": {"type": "object", "properties": {"card_id": {**S, "description": "e.g. 'G230'"}},
                                                        "required": ["card_id"]}},
        {"name": "validate_call_number", "description": "Mechanical checks on a proposed call number: parseable, class number "
         "preserved, Cutter form, date form, not a duplicate, files consistently with neighbours' headings.",
         "input_schema": {"type": "object", "properties": {"call_number": S}, "required": ["call_number"]}},
        {"name": "submit", "description": "Submit the final call number. Call exactly once, at the end.",
         "input_schema": {"type": "object", "properties": {
             "call_number": {**S, "description": "Complete call number in LC display form, e.g. 'PR6045.O72 R66 2026'"},
             "reasoning": {**S, "description": "How the number was built, step by step, in plain language"},
             "citations": {"type": "array", "items": S, "description": "CSM sections relied on, e.g. 'G 063 sec. 2', 'G 320 table'"},
             "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
             "open_questions": {"type": "array", "items": S},
             "needs_review": {"type": "boolean", "description": "True if a human should check before use"}},
             "required": ["call_number", "reasoning", "citations", "confidence", "open_questions", "needs_review"],
             "additionalProperties": False},
         "strict": True},
    ]
    return [ToolSpec(t["name"], t["description"], t["input_schema"], bool(t.get("strict"))) for t in raw]


def _fmt_entries(entries: list[Entry], limit: int = 120) -> str:
    if not entries:
        return "(nothing shelflisted here yet)"
    rows = [e.short() for e in entries[:limit]]
    if len(entries) > limit:
        rows.append(f"... {len(entries) - limit} more")
    return "\n".join(rows)


def _neighbourhood(entries: list[Entry], class_number: str, letter: str | None, *, head: int = 6, around: int = 22) -> str:
    """The part of the class the new number will file in: the first entries (reserved .A
    ranges live there), then the block of entries whose book-number Cutter starts with
    `letter`, with a few entries either side. Falls back to the whole list when small."""
    if not entries:
        return "(nothing shelflisted here yet)"
    if len(entries) <= head + 2 * around or not letter:
        return _fmt_entries(entries, head + 2 * around)
    want = callnum.parse(class_number)
    k = len(want.cutters)
    # a thin sample across the whole class so the subarrangement pattern (by title, by
    # translator, by date, digit-added ...) is visible even outside the letter block
    step = max(1, len(entries) // 10)
    spread = "\n".join(entries[i].short() for i in range(head, len(entries), step))

    def book_letter(e: Entry) -> str:
        p = e.parsed
        return p.cutters[k][0] if p and len(p.cutters) > k else ""
    idx = [i for i, e in enumerate(entries) if book_letter(e) == letter]
    if not idx:
        # nothing with this letter yet: show where it would go
        before = [i for i, e in enumerate(entries) if book_letter(e) and book_letter(e) < letter]
        pos = (before[-1] + 1) if before else head
        lo, hi = max(0, pos - around // 2), min(len(entries), pos + around // 2)
        rows = [e.short() for e in entries[:head]] + [f"... (no entries Cuttered {letter}* yet; it would file here)"] + [e.short() for e in entries[lo:hi]]
        rows.append("Sample across the whole class (every %dth entry):\n%s" % (step, spread))
    else:
        lo, hi = max(0, idx[0] - 4), min(len(entries), idx[-1] + 5)
        block = entries[lo:hi]
        if len(block) > 2 * around:
            block = block[:around] + [None] + block[-around:]
        rows = [e.short() for e in entries[:head]] + [f"... ({lo - head} entries omitted)"] + \
               [e.short() if e else "... (middle of the letter block omitted)" for e in block]
        if hi < len(entries):
            rows.append(f"... ({len(entries) - hi} more entries after)")
        rows.append("Sample across the whole class (every %dth entry):\n%s" % (step, spread))
    return "\n".join(rows)


def _suggestions(profile: Profile, entries: list[Entry], class_number: str) -> str:
    """Mechanical starting points: table Cutter, neighbours by heading, fitted Cutter, date."""
    out = []
    basis = profile.main_entry.cutter_basis
    try:
        tc = cutter.table_cutter(basis)
        out.append(f"G 063 table Cutter for {basis!r}: {tc}")
    except Exception as e:  # noqa: BLE001
        tc = None
        out.append(f"(no table Cutter for {basis!r}: {e})")
    d = dates.call_number_date(profile.publication_date_raw or "", corporate_body=(profile.main_entry.entry_type == "corporate")) if profile.publication_date_raw else None
    out.append(f"G 140 date from {profile.publication_date_raw!r}: {d or '(none)'}")
    if tc and entries:
        want = callnum.parse(class_number)
        k = len(want.cutters)
        same = [e for e in entries if e.parsed and len(e.parsed.cutters) == k + 1 and e.parsed.cutters[k][0] == tc[0]]
        names = not profile.main_entry.entry_type in ("title", "uniform_title")
        me = filing_key(profile.main_entry.filing_string, ignore_article=not names)
        prev = nxt = None
        for e in same:
            hk = filing_key(e.heading(), ignore_article=not e.creator)
            if hk < me:
                prev = e
            elif hk > me and nxt is None:
                nxt = e
        pc = prev.parsed.cutters[k] if prev else None
        nc = nxt.parsed.cutters[k] if nxt else None
        try:
            fit = cutter.fit_between(tc[0], pc, nc, tc)
            out.append(f"By heading order among the {tc[0]}* entries: after {prev.term + ' (' + prev.heading()[:40] + ')' if prev else 'start of block'}, "
                       f"before {nxt.term + ' (' + nxt.heading()[:40] + ')' if nxt else 'end of block'} -> fit_cutter suggests {fit}")
            if d:
                dup = [e for e in entries if e.parsed and e.parsed.cutters == want.cutters + [fit] and e.parsed.date and e.parsed.date.startswith(d)]
                if dup:
                    out.append(f"{fit} {d} already exists ({dup[0].term}); G 140 work letters would apply if it is the same Cutter")
        except Exception as e:  # noqa: BLE001
            out.append(f"(fit_cutter: {e})")
    return "\n".join(out)


class Composer:
    def __init__(self, provider: Provider, shelflist: ShelflistClient | None = None):
        self.provider = provider
        self.shelflist = shelflist or ShelflistClient()

    # -- tools ----------------------------------------------------------
    def _run_tool(self, name: str, inp: dict, ctx: dict) -> str:
        try:
            if name == "shelflist_in_class":
                es = self.shelflist.in_class(inp["class_number"])
                ctx["neighbours"] = es
                return _fmt_entries(es, 60)
            if name == "shelflist_browse":
                es = self.shelflist.browse(inp["call_number"], int(inp.get("count", 60)))
                return _fmt_entries(es)
            if name == "cutter_table":
                return cutter.table_cutter(inp["word"], int(inp.get("digits", 2)))
            if name == "fit_cutter":
                return cutter.fit_between(inp["letter"].upper(), inp.get("prev"), inp.get("next"), inp.get("suggested"))
            if name == "filing_compare":
                names = bool(inp.get("names", False))
                a, b = inp["a"], inp["b"]
                first = "a" if files_before(a, b, ignore_article=not names) else ("same" if filing_key(a, ignore_article=not names) == filing_key(b, ignore_article=not names) else "b")
                return json.dumps({"files_first": first, "key_a": str(filing_key(a, ignore_article=not names)),
                                   "key_b": str(filing_key(b, ignore_article=not names)),
                                   "filing_word_a": filing_word(a, ignore_article=not names), "filing_word_b": filing_word(b, ignore_article=not names)})
            if name == "regions_table":
                return tables.format_rows(tables.regions_countries(inp["place"]))
            if name == "states_table":
                return tables.format_rows(tables.states_provinces(inp["place"]))
            if name == "biography_table":
                return "\n".join(f".x{r['suffix']}: {r['meaning']}  {r['notes']}" for r in tables.biography_table())
            if name == "call_number_date":
                return dates.call_number_date(inp["raw"], corporate_body=bool(inp.get("corporate_body"))) or "(no date found)"
            if name == "schedule_lookup":
                sch = get_schedules()
                return sch.lookup(inp["class_number"]).text(max_children=25) if sch else "(schedules not available)"
            if name == "schedule_table":
                sch = get_schedules()
                t = sch.table(inp["code"]) if sch else None
                return f"Table {t.code}: {t.title}\n{t.text(max_rows=40, find=inp.get('find'))}" if t else f"no table {inp['code']}"
            if name == "load_rule":
                cid = inp["card_id"].replace(" ", "").upper()
                c = rules.load_cards().get(cid)
                return c.text() if c else f"no card {cid}; available: {', '.join(rules.load_cards())}"
            if name == "validate_call_number":
                v = validate(inp["call_number"], ctx["class_number"], filing_string=ctx.get("filing_string"),
                             entry_type=ctx.get("entry_type", "title"), neighbours=ctx.get("neighbours"),
                             expect_date=ctx.get("expect_date", True), check_filing=ctx.get("check_filing", True))
                return json.dumps({"ok": v.ok, "errors": v.errors, "warnings": v.warnings})
            return f"unknown tool {name}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"

    # -- main -----------------------------------------------------------
    def compose(self, resource_text: str, class_number: str, profile: Profile, *, cards: list[rules.Card] | None = None,
                extra_context: str | None = None, schedule_text: str | None = None) -> ComposeResult:
        cards = cards if cards is not None else rules.route(set(profile.tags), class_number)
        core = [c for c in cards if c.always_load]
        cond = [c for c in cards if not c.always_load]
        system_blocks = [SYSTEM_HEAD + "\n\n" + "\n\n".join(c.text() for c in core)]
        if cond:
            system_blocks.append("\n\n".join(c.text() for c in cond))
        system_blocks.append("Other rule cards (load_rule by id if relevant): " + rules.short_index({c.id for c in cards}))

        try:
            neighbours = self.shelflist.in_class(class_number)
        except Exception as e:  # noqa: BLE001
            neighbours = []
        expect_date = "class:by_date" not in profile.tags
        # heading-order checks against neighbours only make sense when every entry in the class is
        # Cuttered for its main entry; author tables and biography numbers mix title and critic Cutters
        check_filing = not ({"form:literary_author", "class:biography_number", "class:table_instruction"} & set(profile.tags))
        ctx = {"class_number": class_number, "filing_string": profile.main_entry.filing_string,
               "entry_type": profile.main_entry.entry_type, "neighbours": neighbours, "expect_date": expect_date,
               "check_filing": check_filing}

        try:
            letter = cutter.table_cutter(profile.main_entry.cutter_basis)[0]
        except Exception:  # noqa: BLE001
            letter = None
        shelf_text = _neighbourhood(neighbours, class_number, letter)
        user = (f"Class number (050 $a): {class_number}\n\n"
                + (f"LC classification schedule entry (2024 edition; its notes and tables take precedence over the CSM):\n{schedule_text}\n\n" if schedule_text else "")
                + f"Resource profile (from step 1):\n{profile.model_dump_json(indent=1, exclude_none=True)}\n\n"
                f"Resource metadata:\n{resource_text}\n\n"
                f"Mechanical suggestions (verify against the neighbours; schedule tables override):\n{_suggestions(profile, neighbours, class_number)}\n\n"
                f"LC shelflist under {class_number} ({len(neighbours)} entries; the relevant part):\n{shelf_text}\n")
        if extra_context:
            user += f"\nAdditional context from the requester:\n{extra_context}\n"
        user += "\nBuild the call number. Finish by calling submit."

        lr = self.provider.tool_loop(system_blocks, user, _tool_defs(), lambda name, inp: self._run_tool(name, inp, ctx),
                                     submit_tool="submit", max_turns=MAX_TURNS)
        submitted, trace, usage = lr.submitted, lr.trace, lr.usage
        if not submitted:
            return ComposeResult(None, "no submission", [], "low", ["model did not submit"], True, {}, trace, usage, [c.id for c in cards])
        v = validate(submitted["call_number"], class_number, filing_string=profile.main_entry.filing_string,
                     entry_type=profile.main_entry.entry_type, neighbours=ctx.get("neighbours"), expect_date=expect_date,
                     check_filing=check_filing)
        return ComposeResult(
            call_number=submitted["call_number"], reasoning=submitted.get("reasoning", ""), citations=list(submitted.get("citations") or []),
            confidence=submitted.get("confidence", "low"), open_questions=list(submitted.get("open_questions") or []),
            needs_review=bool(submitted.get("needs_review")) or not v.ok,
            validation={"ok": v.ok, "errors": v.errors, "warnings": v.warnings}, trace=trace, usage=usage,
            cards=[c.id for c in cards])
