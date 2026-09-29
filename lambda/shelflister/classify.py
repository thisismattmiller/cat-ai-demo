"""Classification front end: Matt's embedded-catalog lambda.

`classify` embeds a title/creator/summary/contents description, finds the nearest LC
records, and (unless ids_only) asks an LLM which of their subject headings and class
numbers fit. We use it for candidate class numbers + subject headings + neighbour
records, then adjudicate with the schedules.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx

LAMBDA_URL = (os.environ.get("SUBJECT_SUGGEST_URL") or os.environ.get("SHELFLISTER_CLASSIFY_URL")
              or "https://abeniabvmaysz2npcr3sr47fxq0xgoes.lambda-url.us-east-1.on.aws/")


@dataclass
class Neighbour:
    lc_001: str
    score: float
    class_number: str | None
    title: str
    creator: str
    subjects: list[str]
    summary: str | None = None
    lcc_path: str | None = None


@dataclass
class Candidate:
    class_number: str
    hierarchy: list[str]
    recommended: bool
    support: int            # how many neighbours carry this class number
    best_score: float


@dataclass
class ClassifyResult:
    candidates: list[Candidate]
    subjects: list[dict]                 # {label, recommended}
    recommended_subjects: list[str]
    neighbours: list[Neighbour]
    raw: dict = field(default_factory=dict)

    def text(self) -> str:
        lines = ["Candidate class numbers from LC records with similar descriptions (vector search over LC's catalog):"]
        for c in self.candidates:
            flag = "RECOMMENDED" if c.recommended else ""
            lines.append(f"  {c.class_number:<14} used by {c.support} neighbour(s), best similarity {c.best_score:.3f} {flag}")
            if c.hierarchy:
                lines.append(f"      {' > '.join(c.hierarchy)}")
        lines.append("Subject headings the neighbours carry (recommended ones marked):")
        for s in self.subjects[:25]:
            lines.append(f"  {'*' if s.get('recommended') else ' '} {s['label']}")
        lines.append("Nearest LC records:")
        for n in self.neighbours:
            lines.append(f"  {n.score:.3f} {n.class_number or '-':<14} {n.creator[:35]:<35} {n.title[:60]}")
            if n.subjects:
                lines.append(f"        subjects: {'; '.join(n.subjects[:5])}")
        return "\n".join(lines)


def classify(title: str, *, creator: str | None = None, summary: str | None = None, content: str | None = None,
             top_k: int = 10, filter: dict | None = None, auto_filter: bool = False, timeout: float = 90.0) -> ClassifyResult:
    body = {"action": "classify", "title": title, "top_k": top_k}
    if creator:
        body["creator"] = creator
    if summary:
        body["summary"] = summary
    if content:
        body["content"] = content
    if filter:
        body["filter"] = filter
    if auto_filter:
        body["auto_filter"] = True
    r = httpx.post(LAMBDA_URL, json=body, headers={"Origin": "https://bibframe.org"}, timeout=timeout)
    r.raise_for_status()
    d = r.json()
    if "error" in d and "search_results" not in d:
        raise RuntimeError(f"classify lambda: {d['error']}")
    enrich = d.get("enrichment", {})
    neighbours = []
    for s in d.get("search_results", []):
        m = s.get("metadata", {})
        e = enrich.get(s["lc_001"], {})
        cls = (e.get("classifications") or [m.get("LCCCode")] or [None])[0]
        neighbours.append(Neighbour(lc_001=s["lc_001"], score=float(s.get("score", 0)), class_number=cls,
                                    title=m.get("Title", ""), creator=m.get("Creator", ""),
                                    subjects=e.get("subjects") or m.get("Subjects") or [], summary=m.get("Summary"),
                                    lcc_path=m.get("LCC")))
    support: dict[str, int] = {}
    best: dict[str, float] = {}
    for n in neighbours:
        for c in (enrich.get(n.lc_001, {}).get("classifications") or ([n.class_number] if n.class_number else [])):
            support[c] = support.get(c, 0) + 1
            best[c] = max(best.get(c, 0.0), n.score)
    cands = []
    seen = set()
    for u in d.get("unique_classifications", []):
        p = u.get("portion")
        if not p or p in seen:
            continue
        seen.add(p)
        cands.append(Candidate(p, u.get("hierarchy") or [], bool(u.get("recommended")), support.get(p, 0), best.get(p, 0.0)))
    for p in support:
        if p not in seen:
            cands.append(Candidate(p, [], False, support[p], best[p]))
    cands.sort(key=lambda c: (not c.recommended, -c.support, -c.best_score))
    subjects = [{"label": s["label"], "recommended": bool(s.get("recommended"))} for s in d.get("unique_subjects", [])]
    return ClassifyResult(cands, subjects, d.get("recommended_subjects", []), neighbours, raw=d)
