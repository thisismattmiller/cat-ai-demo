"""Candidate board: one ranked list fused from every proposer the classifier has.

Proposers: the hybrid schedule search (full-text + vector index) run on the first-read queries and the
subject headings, the embedded-catalog neighbours' class numbers, and the LC class number(s) the Name
Authority File establishes for the main entry (053). Reciprocal rank fusion puts a number that several
sources (or several queries) propose above one that a single source proposes strongly. The top entries
are enriched with what the model would otherwise spend turns fetching: the caption path with its
instruction notes, and how many works LC has shelved there with sample headings.
"""
from __future__ import annotations

import re

from . import callnum

RRF_K = 20


def normalize(cn: str | None) -> str | None:
    """A candidate in 050 $a form, or None for ranges, MLC numbers and unparseable strings."""
    if not cn:
        return None
    cn = cn.strip().rstrip("+")
    if "-" in cn or cn.upper().startswith("MLC") or not callnum.try_parse(cn):
        return None
    c = callnum.parse(cn)
    if c.date and len(c.cutters) >= 2:   # a complete call number from a neighbour: class + book Cutter + date
        cn = f"{c.letters}{c.number}{'.' + c.decimal if c.decimal else ''}.{c.cutters[0]}" + \
             ("".join(" " + x for x in c.cutters[1:-1]))
    elif c.date:                         # one Cutter: class Cutter or book Cutter, keep it and drop the date
        cn = f"{c.letters}{c.number}{'.' + c.decimal if c.decimal else ''}" + (f".{c.cutters[0]}" if c.cutters else "")
    return re.sub(r"\s+", " ", cn)


def rrf(lists: dict[str, list[str]], k: int = RRF_K) -> list[dict]:
    """lists: source name -> ranked class numbers. Returns [{class_number, score, sources: {name: rank}}] by score."""
    scores: dict[str, float] = {}
    sources: dict[str, dict[str, int]] = {}
    for name, ranked in lists.items():
        seen = set()
        for rank, cn in enumerate(ranked, 1):
            cn = normalize(cn)
            if not cn or cn in seen:
                continue
            seen.add(cn)
            scores[cn] = scores.get(cn, 0.0) + 1.0 / (k + rank)
            sources.setdefault(cn, {})[name] = rank
    out = [{"class_number": cn, "score": round(s, 4), "sources": sources[cn]} for cn, s in scores.items()]
    # agreement across distinct source families outranks many hits from one family
    fam = lambda src: src.split(":")[0]
    out.sort(key=lambda d: (-len({fam(x) for x in d["sources"]}), -d["score"]))
    return out


def build_board(sch, shelflist, *, queries: list[str], lambda_result=None, authority_numbers: list[str] | None = None,
                limit: int = 8, per_query: int = 10) -> list[dict]:
    lists: dict[str, list[str]] = {}
    for i, q in enumerate(queries[:6]):
        try:
            hits = sch.search(q, per_query) if sch else []
        except Exception:  # noqa: BLE001
            hits = []
        lists[f"search:{i + 1}"] = [h["class_number"] for h in hits]
    if lambda_result is not None:
        cands = sorted(lambda_result.candidates, key=lambda c: (not c.recommended, -c.support, -c.best_score))
        lists["catalog:candidates"] = [c.class_number for c in cands]
        lists["catalog:neighbours"] = [n.class_number for n in lambda_result.neighbours if n.class_number]
    if authority_numbers:
        lists["authority:053"] = list(authority_numbers)
    board = rrf(lists)[:limit]
    for b in board:
        b["proposed_by"] = _describe_sources(b["sources"], queries)
        try:
            L = sch.lookup(b["class_number"]) if sch else None
            n = L.narrowest if L else None
            if n:
                path = [a.caption for a in L.ancestors[-3:]] + [n.caption]
                b["caption"] = " > ".join(path)
                b["exact"] = L.exact is not None
                notes = [x for x in L.instructions() if re.search(r"class here|including|for .* see|cf\.", x, re.I)]
                b["notes"] = notes[-1][:160] if notes else ""
                near = [x for x in (L.children or L.siblings) if x.number and x.number != n.number][:4]
                b["nearby"] = [f"{x.section}{x.number} {x.caption[:38]}" for x in near]
            else:
                b["caption"], b["exact"], b["notes"] = "(not in the 2024 schedules)", False, ""
        except Exception:  # noqa: BLE001
            b["caption"], b["exact"], b["notes"] = "", False, ""
        try:
            es = shelflist.in_class(b["class_number"]) if shelflist else []
            b["shelved"] = len(es)
            b["samples"] = [f"{(e.heading() or '')[:28]}: {(e.title or '')[:36]}" for e in es[:3]]
        except Exception:  # noqa: BLE001
            b["shelved"], b["samples"] = None, []
    return board


def _describe_sources(sources: dict[str, int], queries: list[str]) -> str:
    parts = []
    qs = [k for k in sources if k.startswith("search:")]
    if qs:
        parts.append(f"schedule search ({len(qs)} of the queries)")
    if "catalog:candidates" in sources or "catalog:neighbours" in sources:
        parts.append("similar LC records")
    if "authority:053" in sources:
        parts.append("name authority 053")
    return ", ".join(parts)


def board_text(board: list[dict]) -> str:
    if not board:
        return ""
    lines = ["Candidate board (fused from the schedule search, the vector index, similar LC records and the name "
             "authority; numbers several sources agree on come first; verify before choosing, and search further "
             "when none fits):"]
    for i, b in enumerate(board, 1):
        shelf = (f"LC shelves {b['shelved']} work(s) here" + (": " + " | ".join(b["samples"]) if b.get("samples") else "")) \
            if b.get("shelved") is not None else "shelflist not checked"
        lines.append(f"  {i}. {b['class_number']:<16} proposed by {b['proposed_by']}")
        lines.append(f"       {'caption' if b.get('exact') else 'within'}: {b.get('caption', '')}")
        if b.get("notes"):
            lines.append(f"       note: {b['notes']}")
        if b.get("nearby"):
            lines.append(f"       printed nearby: {' | '.join(b['nearby'])}")
        lines.append(f"       {shelf}")
    return "\n".join(lines)
