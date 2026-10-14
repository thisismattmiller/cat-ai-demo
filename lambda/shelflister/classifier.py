"""Classification as a schedule-first browsing loop.

The model must find the number in the LC classification schedules (full-text search over
captions and index terms, then browsing the hierarchy) and confirm it against what LC
has actually shelved there (the shelflist). Similar-record retrieval from the catalog
(the embedded-catalog lambda) is available as one more piece of evidence, and can be
switched off. The CSM classification chapters are in context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from . import callnum, rules, cutter, fusion
from .filing import filing_word, normalize_letters
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
   number for the work as a whole, not for one chapter). Settle the DISCIPLINE first -- the class
   letters follow from it, and most wrong answers are a discipline slip: a study of Islamic economics
   is religion (BP173.75), not fiqh (KBP); investigating a crime is criminology (HV), not law (K);
   a self-help book on feelings is psychology (BF), not sociology (HM); anatomy (QM) is not
   physiology (QP). Ask what kind of author writes this book for which readers. When LC subject headings are supplied, the
   FIRST heading names the primary topic and the class number normally matches it (its main heading
   and the first subdivision, e.g. "Investments--West Bank" -> the Investments number for Palestine);
   later headings are secondary aspects.
2. Search the schedules (schedule_search) with topical words, and with the subject-heading vocabulary
   LC uses. Try several phrasings; add the discipline or country when results are scattered
   (e.g. "women authors English literature"). Index hits point to the number or range for a topic.
3. Browse (schedule_lookup) the candidate numbers: read the caption path, the notes ("Class here",
   "Including", "For ... see", "Cf."), the A-Z lists and tables. Follow "see" references. Prefer the
   most specific caption that describes the whole work; do not stop at a general number when a specific
   one exists, and do not use a number whose notes send this kind of work elsewhere. When sibling A-Z
   captions differ by kind (regions vs cities, persons vs topics, a period), pick by what the entity is.
4. Confirm with the shelflist (shelflist_in_class): what has LC actually shelved at this number? If the
   works there are of a different kind, reconsider. LC's placement of closely similar works OUTRANKS a
   caption that merely contains a word from the title: when a number already holds books like this one,
   that is where this one goes, even if another caption reads more literally (BP173.75 holds the
   Islamic market-price books, so a new one goes there, not under the fiqh caption "Prices"). A caption
   match whose shelflist is empty or holds a different kind of work is weak evidence. Under an A-Z caption (by country, topic, person ...)
   LC's existing Cutter for that entity MUST be reused: call shelflist_in_class with find=<the entity>
   before constructing one. A work ABOUT a specific work or person that LC shelves at X.Cutter (the
   Cutter of the original author, e.g. ML189.T55 for al-Tīfāshī's treatise) takes that same Cutter as
   part of its class number, 050 $a = X.Cutter, and gets its own main-entry Cutter later; never submit
   the bare X for it. Also browse the neighbourhood (shelflist_browse): LC's practice includes numbers
   and decimal extensions the printed schedule does not show (HG5710.5 for the West Bank beside
   HG5710 Israel); the subjects of the works shelved there tell you what each number means.
5. Optionally consult similar_records (LC records with similar descriptions). Treat them as evidence of
   practice, not as the answer: verify any number they suggest in the schedules before using it.
6. Call submit once, with the class number exactly as it should appear in 050 $a (class letters, number,
   and any subject/geographic/author Cutter the schedule requires, e.g. 'PR6045.O72', 'TX749.5.B43',
   'HV1478.V4'), the caption path you found it under, the schedule notes/tables that applied, and the
   subject headings that fit. No book number, no date: the shelflist prints complete call numbers
   ('QA76.76.C65 S65 2011'), whose last Cutter and date belong to that one book, not to the class.

Fiction and other belles lettres class with the adult literature (PR, PS, PQ ...) unless the RECORD says the
work is for children or young adults (a "Juvenile fiction" / "Juvenile literature" subdivision, an audience
note, a juvenile audience code); never infer a juvenile audience from the author's fame, the plot, or a
bookseller category. Only then do PZ5-PZ10.3 apply (F 615).
Literary works: an individual author's number lives in the national literature by period (PR, PS, PQ ...),
with the author Cutter from the A-Z list or, for a new author, from the Cutter table; say
needs_new_number=true when the schedule does not print the author. Biography: see F 275 for where the
biography of a person classes. When the schedule provides an A-Z list and the needed Cutter is not
printed, construct it from the Cutter table (G 063) with the cutter_table tool and mark needs_new_number.
The record may carry a "name_authority" line from the LC Name Authority File for the main entry: an LC
class number established for the heading (053) is the author's own number -- use it for a literary work
by that author (and for criticism of the author) unless the work is juvenile and a PZ number is listed;
the places, field of activity and source notes settle the author's nationality, which decides the
national literature (PR vs PS vs PR9199 ...), and the subject of a biography. "place_of_publication"
is the strongest cue for the jurisdiction of a law book and a hint for a national literature or history.
Law (K classes): the law of one country classes in that country's subclass (KHD Brazil, KKY Ukraine,
KNW Indonesia ...), whose printed numbers come from a form table (Table KDZ-KH2, KJ-KKZ1, KL-KWX4 ...).
schedule_search with section=<the country's letters>, or with the country named in the query, searches
that table and returns the expanded numbers (marked "Table ... for <country>"); schedule_lookup resolves
them. Do not class a country's own law in the general K or KZ numbers.

Be economical: the first message already contains a candidate board (numbers proposed by the schedule
search, the vector index, similar LC records and the name authority, fused: a number several sources
agree on is strong evidence; each comes with its caption and what LC shelves there), then raw schedule
search hits (and similar LC records when available). Start from the board: verify the top candidates
(schedule_lookup for the notes, shelflist_in_class for the kind of works there) and pick between them.
The board can miss the right number: when none of its entries fits the whole work, search further.
Usually two or three lookups settle it. The CSM chapters on
classification follow; more are available with load_rule."""


SYSTEM_FORCED = """You are an LC subject cataloger. Given a resource's metadata and the schedule/shelflist evidence
already gathered, choose the single best LC class number (050 $a form: class letters, number, and any
subject/geographic/author Cutter the schedule requires; no book Cutter, no date). Prefer a number whose
caption path was seen in the evidence; say so in schedule_basis. Confidence is low unless the evidence is clear."""


class ForcedDecision(BaseModel):
    class_number: str = Field(description="050 $a form, e.g. 'HG5710.5.A3'")
    caption_path: str = ""
    schedule_basis: str = ""
    reasoning: str = ""
    confidence: str = Field(default="low", description="high | medium | low")


class AltFraming(BaseModel):
    discipline: str = Field(description="the other discipline, as a cataloger names it")
    why: str = Field(description="one line: what about the work would put it there")
    search_queries: list[str] = Field(description="2 or 3 caption-style English search strings for this framing")


class Frame(BaseModel):
    """A cataloger's first read of the record, made before touching the schedules."""
    discipline: str = Field(description="the field whose literature this belongs to, as a cataloger names it: "
                            "'Islamic religion', 'criminology', 'Brazilian law', 'Urdu literature', 'public health' ...")
    form: str = Field(description="monograph | textbook | fiction | poetry | biography | memoir | law code | legislation | "
                      "conference | exhibition catalog | juvenile | serial | other")
    topic: str = Field(description="the subject in one English line, as an LC subject heading would put it")
    place: str = Field(default="", description="country or region the work is about or whose law/literature it belongs to, else ''")
    period: str = Field(default="", description="period covered or the author's period, else ''")
    alternatives: list["AltFraming"] = Field(default_factory=list, description="one or two OTHER disciplines a cataloger "
                                            "might class this work under (criminology vs law, psychology vs sociology, religion "
                                            "vs religious law, history vs politics, literature vs biography), each with its own "
                                            "search strings; empty only when no other discipline is plausible")
    search_queries: list[str] = Field(description="4 to 6 short English search strings phrased the way LC classification "
                                      "captions and index entries are phrased (2-6 words each): the topic as a noun phrase "
                                      "('Sports facilities', 'Online social networks'), the topic with its aspect as a caption "
                                      "would put it ('Sports facilities economic aspects', 'Transportation environmental aspects'), "
                                      "the discipline's broad class term + topic, and the place + topic for a regional work")


SYSTEM_FRAME = """You are an LC subject cataloger reading a record before opening the schedules. From the metadata
(title, creator, summary or contents, notes, name authority data, place of publication) state the discipline,
the form, the topic in English, the place and period, and 3-5 short English search strings in the vocabulary
of LC classification captions and their index entries -- short noun phrases, the aspect after the topic ("Bicycles
transportation", "Jesus Christ person and offices"), not sentences or bookseller categories. The title may be
in another language: translate the topic, do not copy words. Then name one or two OTHER disciplines a
cataloger might reasonably class the work under, with why and their own search strings: most wrong class
numbers come from settling the discipline too early, so the alternatives are searched too.
This is a first read, not the decision; keep every field short."""


def prefetch_searches(sch, resource, extra_queries: list[str] | None = None) -> str:
    """Schedule searches to seed the first message: one per leading subject heading (the class usually
    matches the first heading), then the title when it is in Latin-script English-like text (a
    romanized Arabic or Indonesian title finds nothing in English captions)."""
    blocks, seen = [], set()
    queries = []
    for h in resource.subjects[:3]:
        main = " ".join(x.strip() for x in h.split("--")[:2])
        if main.lower() not in seen:
            seen.add(main.lower()); queries.append((main, 6))
    for q in extra_queries or []:
        if q and q.lower() not in seen:
            seen.add(q.lower()); queries.append((q[:120], 6))
    title = resource.title or ""
    alpha = [c for c in title if c.isalpha()]
    ascii_ratio = sum(1 for c in alpha if c.isascii()) / len(alpha) if alpha else 0
    if title and ascii_ratio >= 0.9 and title.lower() not in seen and not extra_queries:
        queries.append((title[:120], 8))      # the first-read queries say the topic in English; a romanized title does not
    if not queries and title:
        queries.append((title[:120], 8))
    for q, limit in queries[:5]:
        blocks.append(sch.search_text(q, limit=limit))
    return "\n".join(blocks)


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
    frame: dict | None = None
    lambda_dropped: list | None = None
    board: list[dict] = field(default_factory=list)


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
        ToolSpec("cutter_table", "The Cutter the G 063 Cutter table gives for a word (an author's surname for a literary "
                 "author number, a place or topic under an A-Z caption). The schedule_table tool does not know this table.",
                 {"type": "object", "properties": {"word": S, "digits": {"type": "integer"}}, "required": ["word"]}),
        ToolSpec("shelflist_in_class", "What LC has actually shelved at a class number (call numbers with the heading and "
                 "title of each work). Confirms the number is used for this kind of work and shows Cutters already assigned. "
                 "A large class is summarised as a map of its Cutters (Cutter, count, sample subject/heading); pass "
                 "find=<words> (a country, topic, person ...) to list the entries that mention them, e.g. the Cutter LC "
                 "already uses for Indonesia under an A-Z caption.",
                 {"type": "object", "properties": {"class_number": S, "find": S}, "required": ["class_number"]}),
        ToolSpec("shelflist_browse", "A window of the LC shelflist around any call number, in shelf order, with each work's "
                 "subject: the numbers just before and after, including decimal extensions and Cutters the printed schedule "
                 "does not show but LC uses in practice (e.g. HG5710.5 for the West Bank next to HG5710 Israel).",
                 {"type": "object", "properties": {"call_number": S, "count": {"type": "integer", "description": "entries in the window, default 40"}},
                  "required": ["call_number"]}),
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


def shelflist_view(es, class_number: str, find: str | None = None, *, max_lines: int = 40) -> str:
    """Entries at a class number for the model: all of them when few; with `find`, the entries mentioning
    the words; otherwise a map of the class-level Cutters (so an existing Cutter for a country, topic or
    person under an A-Z caption is visible even in a class of hundreds)."""
    if not es:
        return "(nothing shelflisted at this number)"
    if find:
        words = [normalize_letters(w).lower() for w in find.split() if w.strip()]
        hits = [e for e in es if all(w in normalize_letters(f"{e.creator} {e.uniformtitle} {e.title} {e.subject}").lower() for w in words)]
        if not hits:
            return f"{len(es)} entries at {class_number}; none mention {find!r}. Entries look like:\n" + "\n".join(e.short() for e in es[:8])
        lines = [f"{len(hits)} of {len(es)} entries at {class_number} mention {find!r}:"] + [e.short() + (f"  [{e.subject[:60]}]" if e.subject else "") for e in hits[:max_lines]]
        if len(hits) > max_lines:
            lines.append(f"... {len(hits) - max_lines} more")
        return "\n".join(lines)
    if len(es) <= max_lines:
        return "\n".join(e.short() for e in es)
    want = callnum.try_parse(class_number)
    k = len(want.cutters) if want else 0
    groups: dict[str, list] = {}
    for e in es:
        p = e.parsed
        key = p.cutters[k] if p and len(p.cutters) > k else "(no further Cutter)"
        groups.setdefault(key, []).append(e)
    lines = [f"{len(es)} entries at {class_number}, {len(groups)} distinct Cutters after it. Map (Cutter: count, sample subject | heading | title); "
             f"use find=<words> to list the entries for a country, topic or person:"]
    shown = 0
    for key, grp in groups.items():
        e = grp[0]
        sample = e.subject or ""
        lines.append(f"  {key:<8} {len(grp):>3}  {sample[:50]:<50} | {e.heading()[:30]} | {e.title[:30]}")
        shown += 1
        if shown >= 60:
            lines.append(f"  ... {len(groups) - shown} more Cutters")
            break
    return "\n".join(lines)


class Classifier:
    def __init__(self, provider: Provider, shelflist: ShelflistClient | None = None, use_lambda: bool = True):
        self.provider = provider
        self.shelflist = shelflist or ShelflistClient()
        self.use_lambda = use_lambda
        self._resource = None
        self._confirmed: set[str] = set()
        self._board: list[dict] = []
        self._examined: set[str] = set()
        self._compared = False

    def _run_tool(self, name: str, inp: dict, ctx: dict) -> str:
        try:
            sch = get_schedules()
            if name == "schedule_search":
                return sch.search_text(inp["query"], int(inp.get("limit") or 12), inp.get("section") or None) if sch else "(schedules not available)"
            if name in ("schedule_lookup", "shelflist_in_class", "shelflist_browse"):
                self._examined.add(str(inp.get("class_number") or inp.get("call_number") or "").split(" ")[0].upper())
            if name == "schedule_lookup":
                return sch.lookup(inp["class_number"]).text(max_children=25) if sch else "(schedules not available)"
            if name == "schedule_table":
                t = sch.table(inp["code"]) if sch else None
                return f"Table {t.code}: {t.title}\n{t.text(max_rows=40, find=inp.get('find'))}" if t else f"no table {inp['code']}"
            if name == "cutter_table":
                return cutter.table_cutter(inp["word"], int(inp.get("digits") or 2))
            if name == "load_rule":
                c = rules.load_cards().get(inp["card_id"].replace(" ", "").upper())
                return c.text() if c else f"no card {inp['card_id']}"
            if name == "shelflist_in_class":
                return shelflist_view(self.shelflist.in_class(inp["class_number"]), inp["class_number"], inp.get("find"))
            if name == "shelflist_browse":
                es = self.shelflist.browse(inp["call_number"], min(int(inp.get("count") or 40), 120))
                if not es:
                    return "(nothing in the shelflist there)"
                return "\n".join(e.short() + (f"  [{e.subject[:60]}]" if e.subject else "") for e in es)
            if name == "similar_records":
                if ctx.get("lambda") is None:
                    r = ctx["resource"]
                    ctx["lambda"] = lambda_classify(r.title, creator=r.creator, summary=r.summary, content=r.contents,
                                                    exclude=ctx.get("exclude"))
                return ctx["lambda"].text()
            return f"unknown tool {name}"
        except Exception as e:  # noqa: BLE001
            return f"ERROR: {type(e).__name__}: {e}"

    def check_submit(self, inp: dict, resource=None) -> str | None:
        """Reject a 'class number' that is really a complete call number: it ends with a date, or LC shelves a
        work at exactly those Cutters whose main entry is this resource's own (another edition, or the record
        itself), so the last Cutter is the book Cutter. A Cutter that stands for someone else (the author
        being studied, a topic, a place) is part of the class and passes."""
        resource = resource or self._resource
        cn = str(inp.get("class_number") or "").strip()
        c = callnum.try_parse(cn)
        if c is None:
            return f"'{cn}' is not a parseable LC class number; resubmit in 050 $a form, e.g. 'PR6045.O72' or 'TX749.5.B43'."
        if c.cutters and c.date:
            return (f"'{cn}' ends with a date. A date after a Cutter is part of the book number, which is assigned later. "
                    "Resubmit the class part only.")
        if c.cutters and not inp.get("cutter_is_class"):
            try:
                # a bare term equal to the class number (no date) is a class-level entry (PG3366.A15 Tolstoy
                # collections; a record listed with only its 050 $a), not evidence of a book Cutter
                same = [e for e in self.shelflist.in_class(cn)
                        if (pe := callnum.try_parse(e.term)) and pe.cutters == c.cutters and pe.date]
            except Exception:  # noqa: BLE001
                same = []
            own = filing_word(resource.creator or resource.title or "") if resource else ""
            mine = [e for e in same if own and filing_word(e.heading() or "") == own]
            if mine:
                e = mine[0]
                return (f"'{cn}' looks like a complete call number, not a class number: LC shelves '{e.term}' ({e.heading()}: "
                        f"{e.title[:50]}) exactly there, under this work's own main entry, so '{c.cutters[-1]}' is a book Cutter. "
                        "Resubmit the class part only (what 050 $a holds). If that Cutter really stands for a person, topic or "
                        "place that is part of the class (e.g. the author this work is about), resubmit it with cutter_is_class=true.")
            # An invented book Cutter: starts with the main entry's own letter, the schedule does not print that Cutter,
            # and nobody else is shelved at it (an established class Cutter would have other works under it).
            last = c.cutters[-1]
            if own and last[0].upper() == own[0].upper() and not self._schedule_prints(c):
                others = [e for e in same if filing_word(e.heading() or "") != own]
                if not others:
                    return (f"'{cn}': the last Cutter '{last}' begins with the same letter as this work's main entry "
                            f"('{own}'), the schedule does not print it, and no other work is shelved at it, so it looks like "
                            "a book Cutter you derived from the main entry. The book Cutter is assigned later: resubmit the "
                            "class part only (what 050 $a holds). If the Cutter really stands for a topic, place or person that "
                            "is part of the class and only happens to share the letter, resubmit it with cutter_is_class=true.")
        # (4) Rivals on the board the model never looked at: compare before deciding (once).
        if self._board and not self._compared:
            fam = lambda b: {k.split(":")[0] for k in b.get("sources", {})}
            chosen = next((b for b in self._board if b["class_number"] == cn), None)
            chosen_rank = self._board.index(chosen) if chosen else len(self._board)
            top = [b for b in self._board[:4] if b["class_number"] != cn and b.get("shelved") and self._board.index(b) < chosen_rank]
            rivals = [b for b in top if (not chosen or len(fam(b)) >= len(fam(chosen)))
                      and not any(x and (x.startswith(b["class_number"].upper()) or b["class_number"].upper().startswith(x)) for x in self._examined)]
            if rivals:
                self._compared = True
                lines = [f"  {b['class_number']:<14} {b.get('caption', '')[:70]} -- LC shelves {b.get('shelved')} work(s): "
                         + " | ".join(b.get("samples") or []) for b in rivals[:2]]
                return (f"Before deciding on '{cn}': the candidate board ranks number(s) above yours that you have not examined, "
                        "proposed by as many sources:\n" + "\n".join(lines) + "\nLook at each (shelflist_in_class / schedule_lookup) and "
                        "then resubmit, keeping your number or taking the rival; say in reasoning why the other does not fit.")
        # (8) The entity the Cutter stands for may already have a Cutter under this number (from the caption path).
        if c.cutters and inp.get("caption_path") and f"entity:{cn}" not in self._confirmed:
            entity = re.split(r"\s*(?:--|>)\s*", str(inp["caption_path"]))[-1].strip()
            m_par = re.search(r"\(([^()]+)\)\s*$", entity)
            if m_par:
                entity = m_par.group(1).strip()           # "By region or state, A-Z (West Virginia)" -> "West Virginia"
            stop = {"general", "works", "other", "special", "topics", "region", "regions", "state", "states", "country",
                    "countries", "local", "individual", "particular", "subject", "subjects", "place", "places", "period"}
            words = [w for w in re.findall(r"[A-Za-z]{4,}", normalize_letters(entity).lower()) if w not in stop]
            if words:
                bare = c.format()[: c.format().rfind(c.cutters[-1])].rstrip(". ")
                k = len(c.cutters) - 1
                # the schedule itself may print a Cutter for this entity under the bare number (G156.5.E26 Environmental aspects)
                try:
                    printed = get_schedules().lookup(bare)
                    kids = (printed.children or printed.siblings) if printed else []
                except Exception:  # noqa: BLE001
                    kids = []
                own_printed = next((kid for kid in kids if kid.number and (pk := callnum.try_parse(f"{kid.section}{kid.number}"))
                                    and pk.cutters == c.cutters), None)
                own_ok = own_printed is not None and set(words) & set(re.findall(r"[a-z]{4,}", normalize_letters(own_printed.caption).lower()))
                for kid in ([] if own_ok else kids):      # the submitted Cutter is printed for this very entity: fine
                    pk = callnum.try_parse(f"{kid.section}{kid.number}") if kid.number else None
                    if pk and len(pk.cutters) == k + 1 and pk.cutters[:k] == c.cutters[:k] and pk.cutters[k] != c.cutters[-1]:
                        capw = set(re.findall(r"[a-z]{4,}", normalize_letters(kid.caption).lower()))
                        if capw and set(words) <= capw:
                            self._confirmed.add(f"entity:{cn}")
                            return (f"The schedule prints {kid.section}{kid.number} \"{kid.caption}\" under {bare}: that is the Cutter for "
                                    f"'{entity}', not '{c.cutters[-1]}'. Resubmit with the printed Cutter, or the same number again if the "
                                    "entity is genuinely different.")
                try:
                    es = self.shelflist.in_class(bare, count=1000)
                except Exception:  # noqa: BLE001
                    es = []
                groups: dict[str, int] = {}
                for e in ([] if own_ok else es):
                    pe = e.parsed
                    if pe and len(pe.cutters) > k and all(w in normalize_letters(f"{e.subject} {e.title}").lower() for w in words):
                        groups[pe.cutters[k]] = groups.get(pe.cutters[k], 0) + 1
                if groups:
                    best, n = max(groups.items(), key=lambda x: x[1])
                    if best != c.cutters[-1] and n >= 2 and groups.get(c.cutters[-1], 0) == 0:
                        self._confirmed.add(f"entity:{cn}")
                        return (f"Under {bare}, LC shelves {n} work(s) whose subject mentions '{entity}' at Cutter {best}, none at "
                                f"'{c.cutters[-1]}'. LC's established Cutter for the entity must be reused (G 063): resubmit with {best}, "
                                "or resubmit the same number again if the entity is genuinely different.")
        # A class Cutter nobody is shelved under, while LC already uses other Cutters with the same letter at that
        # number: the entity may already have a Cutter (I46 for Indonesia, not I5). Bounce once with the map.
        if c.cutters and not self._schedule_prints(c) and cn not in self._confirmed:
            try:
                at = self.shelflist.in_class(cn)
            except Exception:  # noqa: BLE001
                at = None
            if at is not None and not at:
                bare = c.format()[: c.format().rfind(c.cutters[-1])].rstrip(". ")
                letter = c.cutters[-1][0].upper()
                k = len(c.cutters) - 1
                try:      # the shelf order around the number shows every same-letter Cutter; a class list is capped
                    es = self.shelflist.browse(cn, 80) + self.shelflist.in_class(bare, count=1000)
                except Exception:  # noqa: BLE001
                    es = []
                groups: dict[str, list] = {}
                seen_terms = set()
                for e in es:
                    pe = e.parsed
                    if e.term in seen_terms or not pe or len(pe.cutters) <= k:
                        continue
                    seen_terms.add(e.term)
                    same_bare = f"{pe.letters}{pe.number}{'.' + pe.decimal if pe.decimal else ''}" + "".join("." + x if i == 0 else " " + x for i, x in enumerate(pe.cutters[:k]))
                    if same_bare.replace(" ", "") == bare.replace(" ", "") and pe.cutters[k].upper().startswith(letter):
                        groups.setdefault(pe.cutters[k], []).append(e)
                if groups:
                    self._confirmed.add(cn)
                    rows = [f"  {cu:<8} {len(g):>3}  {(g[0].subject or '')[:45]:<45} | {g[0].heading()[:28]} | {g[0].title[:28]}"
                            for cu, g in sorted(groups.items(), key=lambda x: -len(x[1]))[:12]]
                    return (f"No work is shelved at '{cn}'. Under {bare} LC already uses these Cutters beginning with '{letter}' "
                            f"(Cutter: count, sample subject | heading | title):\n" + "\n".join(rows) +
                            f"\nIf one of them stands for the same entity (country, topic, person) as '{c.cutters[-1]}', resubmit with "
                            "that Cutter: LC's established Cutter must be reused. If none does, resubmit the same number again to confirm it.")
        return None

    @staticmethod
    def _schedule_prints(c: "callnum.CallNumber") -> bool:
        """True when the 2024 schedule has a node for exactly this Cuttered number (HV6250.4.W65 Women)."""
        try:
            sch = get_schedules()
            node = sch.lookup(c.format()).exact if sch else None
        except Exception:  # noqa: BLE001
            return False
        if not node or not node.number:
            return False
        pn = callnum.try_parse(c.letters + node.number)
        return bool(pn and pn.cutters == c.cutters)

    def _forced_decision(self, resource, trace: list[dict]) -> dict:
        """The loop ended without a submit: decide from what was gathered (one structured call, no tools)."""
        parts = []
        for t in trace[-14:]:
            if t.get("tool") and t["tool"] != "submit":
                parts.append(f"> {t['tool']}({json.dumps(t.get('input'), ensure_ascii=False)})\n{str(t.get('output', ''))[:700]}")
            elif t.get("text"):
                parts.append(f"(model) {t['text'][:400]}")
        user = (f"Resource metadata:\n{resource.text()}\n\nEvidence gathered from the schedules and shelflist:\n\n"
                + "\n\n".join(parts) + "\n\nThe browsing budget is exhausted. Decide the class number now from this evidence.")
        try:
            obj, _ = self.provider.structured(SYSTEM_FORCED, user, ForcedDecision)
        except Exception as e:  # noqa: BLE001
            return {}
        d = obj.model_dump()
        d.setdefault("subject_headings", []); d.setdefault("alternatives", []); d.setdefault("open_questions", [])
        d["needs_new_number"] = False
        return d

    def _frame(self, resource) -> tuple[dict | None, dict]:
        """One cheap structured call: discipline, form, topic, place and English search strings (also the
        seed searches for a record with no subject headings and a non-English title)."""
        try:
            obj, usage = self.provider.structured(SYSTEM_FRAME, f"Resource metadata:\n{resource.text()}", Frame, max_tokens=1500)
            return obj.model_dump(), usage
        except Exception:  # noqa: BLE001
            return None, {}

    def classify(self, resource, *, extra_context: str | None = None, max_turns: int = 14,
                 exclude: dict | None = None) -> Classification:
        self._resource = resource
        self._confirmed = set()
        cards = rules.load_cards()
        system_blocks = [SYSTEM + "\n\n" + "\n\n".join(cards[c].text() for c in CLASSIFY_CARDS if c in cards),
                         "Other CSM chapters (load_rule by id): " + "; ".join(f"{c} {cards[c].title}" for c in EXTRA_CARDS if c in cards)]
        user = f"Resource metadata:\n{resource.text()}\n"
        if resource.subjects:
            user += "\n(Subject headings were supplied; use them as strong evidence of the topic.)\n"
        frame, frame_usage = self._frame(resource)
        if frame:
            user += ("\nFirst read (your own preliminary analysis; verify it in the schedules):\n"
                     f"  discipline: {frame['discipline']} | form: {frame['form']} | topic: {frame['topic']}"
                     + (f" | place: {frame['place']}" if frame.get('place') else "")
                     + (f" | period: {frame['period']}" if frame.get('period') else "") + "\n")
            for alt in frame.get("alternatives") or []:
                user += f"  could also be classed under: {alt.get('discipline')} -- {alt.get('why')}\n"
        # prefetch: schedule search on the subjects, the first-read queries and the title; lambda candidates if enabled
        sch = get_schedules()
        ctx = {"resource": resource, "lambda": None, "exclude": exclude}
        if self.use_lambda:
            try:
                ctx["lambda"] = lambda_classify(resource.title, creator=resource.creator, summary=resource.summary,
                                                content=resource.contents, exclude=exclude)
            except Exception as e:  # noqa: BLE001
                user += f"\n(similar_records unavailable: {e})\n"
        # candidate board: every proposer fused into one ranked, enriched list
        queries = [" ".join(x.strip() for x in h.split("--")[:2]) for h in resource.subjects[:2]] + list((frame or {}).get("search_queries") or [])
        for alt in (frame or {}).get("alternatives") or []:
            queries += list(alt.get("search_queries") or [])[:3]
        if not queries and resource.title:
            queries = [resource.title[:120]]
        board = []
        try:
            auth = [c for c in (resource.extra.get("authority_class_numbers") or []) if isinstance(c, str)]
            board = fusion.build_board(sch, self.shelflist, queries=queries, lambda_result=ctx["lambda"], authority_numbers=auth)
        except Exception as e:  # noqa: BLE001
            user += f"\n(candidate board unavailable: {e})\n"
        self._board = board
        self._examined = set()
        self._compared = False
        if board:
            user += "\n" + fusion.board_text(board) + "\n"
        if sch:
            user += "\n" + prefetch_searches(sch, resource, (frame or {}).get("search_queries")) + "\n"
        if ctx["lambda"] is not None:
            user += f"\n{ctx['lambda'].text()}\n"
        if extra_context:
            user += f"\nAdditional context from the requester:\n{extra_context}\n"
        user += "\nFind the class number in the schedules, confirm it against the shelflist, then call submit."
        lr = self.provider.tool_loop(system_blocks, user, _tools(self.use_lambda),
                                     lambda n, i: self._run_tool(n, i, ctx), submit_tool="submit", max_turns=max_turns,
                                     check_submit=self.check_submit)
        s = lr.submitted or {}
        forced = False
        if not s and lr.trace:
            s, forced = self._forced_decision(resource, lr.trace), True
        lam = ctx["lambda"]
        if forced and s:
            s["open_questions"] = ["classifier ran out of turns without submitting; this decision was forced from the "
                                   "evidence gathered, check it"] + list(s.get("open_questions") or [])
        elif s and lr.usage.get("turns", 0) >= 10:
            s["open_questions"] = [f"classifier needed {lr.usage['turns']} turns to decide: the choice was not clear-cut, "
                                   "review it"] + list(s.get("open_questions") or [])
        for k, v in frame_usage.items():
            if isinstance(v, (int, float)):
                lr.usage[k] = lr.usage.get(k, 0) + v
        if frame_usage:
            lr.usage["turns"] = lr.usage.get("turns", 0) + 1
        return Classification(
            class_number=s.get("class_number"), caption_path=s.get("caption_path", ""), schedule_basis=s.get("schedule_basis", ""),
            subject_headings=list(s.get("subject_headings") or []), reasoning=s.get("reasoning", "" if s else "model did not submit"),
            alternatives=list(s.get("alternatives") or []), confidence=s.get("confidence", "low"),
            needs_new_number=bool(s.get("needs_new_number")), open_questions=list(s.get("open_questions") or []),
            trace=lr.trace, usage=lr.usage,
            lambda_candidates=[c.__dict__ for c in lam.candidates] if lam else None, frame=frame,
            lambda_dropped=list(lam.dropped) if lam else None, board=board)
