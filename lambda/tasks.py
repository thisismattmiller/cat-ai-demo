"""The tasks that run for an LCCN, each in its own Lambda invocation, each streaming events
through an Emitter (ws.py):

  record     the LC record from id.loc.gov (BIBFRAME), ISBNdb-enriched: what the other tasks see
  subjects   the subject-suggest lambda (embedded-catalog `classify` action): LCSH + LCC candidates
  shelflist  the shelflister: LC class number from the schedules, then Cutter + date from the CSM
  names      nar-auto-reconcile: LCNAF (then Wikidata) links for the record's unlinked contributors

Every task fetches the record itself (id.loc.gov is fast and cached per container), so the
tasks have no ordering dependency and can be invoked in parallel.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import asdict

import httpx

from ws import Emitter

SUBJECT_SUGGEST_URL = os.environ.get("SUBJECT_SUGGEST_URL") or "https://abeniabvmaysz2npcr3sr47fxq0xgoes.lambda-url.us-east-1.on.aws/"
TASKS = ["record", "subjects", "shelflist", "names"]


# --------------------------------------------------------------------------- the record
def load_record(lccn: str, *, use_isbndb: bool = True):
    from shelflister import idloc
    return idloc.from_lccn(lccn, use_isbndb=use_isbndb)


def record_dict(resource, rec) -> dict:
    r = asdict(resource)
    r.pop("identifiers", None)
    r.pop("extra", None)
    return {
        "lccn": rec.lccn,
        "instance_url": rec.instance_url,
        "bibid": rec.bibid,
        **r,
        "isbns": rec.isbns,
        "contributors": rec.contributors,
        "lc_call_numbers": [{"class_number": a, "item": b, "call_number": f"{a} {b}".strip()} for a, b in rec.lcc],
        "isbndb_added": rec.isbndb_added,
        "extra": {k: v for k, v in resource.extra.items() if k != "isbndb_subjects"},
        "isbndb_subjects": resource.extra.get("isbndb_subjects"),
    }


def run_record(lccn: str, options: dict, em: Emitter) -> dict:
    em.progress("resolving the LCCN at id.loc.gov")
    resource, rec = load_record(lccn, use_isbndb=options.get("isbndb", True))
    em.progress(f"parsed {rec.instance_url}" + (f"; ISBNdb added {', '.join(rec.isbndb_added)}" if rec.isbndb_added else ""))
    return record_dict(resource, rec)


# --------------------------------------------------------------------------- subject suggest
def run_subjects(lccn: str, options: dict, em: Emitter) -> dict:
    em.progress("fetching the record from id.loc.gov")
    resource, rec = load_record(lccn, use_isbndb=options.get("isbndb", True))
    body = {"action": "classify", "title": resource.title, "top_k": int(options.get("top_k", 10))}
    if resource.creator:
        body["creator"] = resource.creator
    if resource.summary:
        body["summary"] = resource.summary
    if resource.contents:
        body["content"] = resource.contents
    if options.get("auto_filter"):
        body["auto_filter"] = True
    em.progress("querying the subject-suggest lambda (vector search over LC's catalog, then an LLM pick)",
                fields=[k for k in ("title", "creator", "summary", "content") if k in body])
    r = httpx.post(SUBJECT_SUGGEST_URL, json=body, headers={"Origin": "https://bibframe.org"},
                   timeout=float(options.get("timeout", 240)))
    r.raise_for_status()
    d = r.json()
    if "error" in d and "search_results" not in d:
        raise RuntimeError(f"subject-suggest lambda: {d['error']}")
    enrich = d.get("enrichment", {}) or {}
    neighbours = []
    for s in d.get("search_results", []) or []:
        m = s.get("metadata", {}) or {}
        e = enrich.get(s.get("lc_001"), {}) or {}
        neighbours.append({
            "lc_001": s.get("lc_001"), "score": round(float(s.get("score", 0)), 3),
            "title": m.get("TitleEN") or m.get("Title", ""), "creator": m.get("Creator", ""),
            "class_number": (e.get("classifications") or [m.get("LCCCode")] or [None])[0],
            "subjects": (e.get("subjects") or m.get("Subjects") or [])[:6],
        })
    lc_subjects = list(resource.subjects)
    lc_class = [a for a, _ in rec.lcc]
    subjects = [{"label": s.get("label"), "recommended": bool(s.get("recommended")),
                 "on_lc_record": s.get("label") in lc_subjects} for s in d.get("unique_subjects", []) or []]
    classifications = [{"class_number": c.get("portion"), "hierarchy": c.get("hierarchy") or [],
                        "recommended": bool(c.get("recommended")), "on_lc_record": c.get("portion") in lc_class}
                       for c in d.get("unique_classifications", []) or []]
    return {
        "recommended_subjects": d.get("recommended_subjects", []) or [],
        "recommended_classifications": d.get("recommended_classifications", []) or [],
        "subjects": subjects,
        "classifications": classifications,
        "neighbours": neighbours,
        "lc_subjects": lc_subjects,
        "lc_genre_form": list(resource.genre_form),
        "lc_classifications": lc_class,
        "auto_filter": d.get("auto_filter"),
        "performance": d.get("performance"),
        "request": {k: (v if k != "summary" else v[:200] + ("…" if len(v) > 200 else "")) for k, v in body.items()},
    }


# --------------------------------------------------------------------------- shelflisting
def _short(v, n=90) -> str:
    s = v if isinstance(v, str) else str(v)
    return s if len(s) <= n else s[: n - 1] + "…"


def _tool_message(ev: dict) -> str:
    if "tool" in ev:
        inp = ev.get("input") or {}
        arg = inp.get("query") or inp.get("class_number") or inp.get("call_number") or inp.get("code") \
            or inp.get("rule") or inp.get("name") or inp.get("heading") or next(iter(inp.values()), "")
        return f"{ev['tool']}({_short(arg)})"
    return _short(ev.get("text", ""), 200)


def run_shelflist(lccn: str, options: dict, em: Emitter) -> dict:
    from shelflister.pipeline import Pipeline
    hide_class = bool(options.get("hide_class", True))
    hide_subjects = bool(options.get("hide_subjects", False))
    em.progress("fetching the record from id.loc.gov")
    resource, rec = load_record(lccn, use_isbndb=options.get("isbndb", True))
    lc_own = [{"class_number": a, "item": b} for a, b in rec.lcc]
    class_number = None
    if rec.lcc and not hide_class:
        class_number = rec.lcc[0][0]
        em.progress(f"using LC's class number {class_number}; building only the book number")
    elif rec.lcc:
        em.progress(f"LC already classified this record; hiding its call number and shelflist entry so the "
                    f"classification is done from scratch")
    if hide_subjects:
        resource.subjects, resource.genre_form = [], []

    pipe = Pipeline(model=options.get("model") or os.environ.get("SHELFLISTER_MODEL"),
                    effort=options.get("effort"), use_lambda=bool(options.get("use_lambda", True)))
    phase = {"name": "classify" if class_number is None else "compose"}

    def on_event(ev: dict) -> None:
        kind = "tool" if "tool" in ev else "text"
        em.progress(_tool_message(ev), phase=phase["name"], kind=kind,
                    **({"tool": ev["tool"], "input": ev.get("input"), "output": ev.get("output")} if kind == "tool" else {}))

    def on_stage(name: str, info: dict) -> None:
        if name == "classified":
            phase["name"] = "compose"
            em.partial({"stage": "classified", **info},
                       message=f"class number {info.get('class_number') or '(none)'} ({info.get('confidence')})")
        elif name == "profiled":
            em.progress(f"profiled: {', '.join(info.get('tags', []))}; rule cards {', '.join(info.get('cards', []))}",
                        phase="compose", stage="profiled", **info)

    pipe.provider.on_event = on_event
    em.progress(f"model {pipe.provider.model}: " + ("searching the LC schedules for the class number" if class_number is None
                                                     else "profiling the resource for the CSM rules"), phase=phase["name"])
    res = pipe.run(resource, class_number, exclude_bibids={rec.bibid} if rec.bibid else None, on_stage=on_stage)

    out = {
        "class_number": res.class_number,
        "book_number": res.book_number,
        "call_number": res.call_number,
        "marc_050": {"a": res.marc_050[0], "b": res.marc_050[1]} if res.marc_050 else None,
        "lc_call_numbers": lc_own,
        "hidden": {"class": hide_class and bool(rec.lcc), "subjects": hide_subjects},
        "model": res.model,
        "timing": res.timing,
        "usage": res.usage,
        "cards": res.cards,
        "profile": res.profile,
    }
    if res.classification:
        c = res.classification
        out["classification"] = {k: c.get(k) for k in ("class_number", "caption_path", "schedule_basis", "subject_headings",
                                                       "reasoning", "alternatives", "confidence", "needs_new_number",
                                                       "open_questions")}
        out["classification"]["tool_calls"] = sum(1 for t in c.get("trace", []) if "tool" in t)
    if res.compose:
        cp = res.compose
        out["compose"] = {k: cp.get(k) for k in ("call_number", "reasoning", "citations", "confidence", "open_questions",
                                                 "needs_review", "validation")}
        out["compose"]["tool_calls"] = sum(1 for t in cp.get("trace", []) if "tool" in t)
    if lc_own:
        from shelflister import callnum
        lc_class = lc_own[0]["class_number"]
        lc_full = f"{lc_class} {lc_own[0]['item']}".strip()
        ours = callnum.try_parse(res.call_number) if res.call_number else None
        theirs = callnum.try_parse(lc_full)
        out["comparison"] = {
            "lc_call_number": lc_full,
            "class_match": (res.class_number or "").replace(" ", "") == lc_class.replace(" ", ""),
            "exact_match": bool(ours and theirs and ours.format() == theirs.format()),
        }
    return out


# --------------------------------------------------------------------------- name reconciliation
class _LogRelay(logging.Handler):
    def __init__(self, em: Emitter):
        super().__init__(level=logging.INFO)
        self.em = em

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.em.progress(record.getMessage(), level=record.levelname.lower())
        except Exception:  # noqa: BLE001
            pass


def _contributor_dict(r) -> dict:
    d = r.model_dump(exclude_none=True)
    c = d.get("contributor", {})
    dec, wd = d.get("decision"), d.get("wikidata_decision")
    return {
        "label": c.get("label"), "roles": c.get("roles", []), "agent_type": c.get("agent_type"), "primary": c.get("primary"),
        "name_analysis": (d.get("search_plan") or {}).get("name_analysis"),
        "queries_run": len(d.get("queries_run", [])),
        "hits": [{"uri": h.get("uri"), "label": h.get("label"), "matched_variant": h.get("matched_variant")}
                 for h in d.get("hits", [])[:12]],
        "hits_total": len(d.get("hits", [])),
        "candidates_examined": len(d.get("candidates_examined", [])),
        "decision": dec and {k: dec.get(k) for k in ("decision", "matched_uri", "confidence", "reasoning", "runner_ups")},
        "wikidata": {
            "queries_run": len(d.get("wikidata_queries_run", [])),
            "hits": [{"qid": h.get("qid"), "label": h.get("label"), "description": h.get("description")}
                     for h in d.get("wikidata_hits", [])[:10]],
            "decision": wd and {k: wd.get(k) for k in ("decision", "matched_qid", "confidence", "reasoning", "runner_ups")},
            "name_guard": d.get("wikidata_name_guard"),
        } if (d.get("wikidata_queries_run") or wd) else None,
        "recommended_authority_uri": d.get("recommended_authority_uri"),
        "recommended_rwo_uri": d.get("recommended_rwo_uri"),
        "recommended_label": d.get("recommended_label"),
        "recommended_wikidata_qid": d.get("recommended_wikidata_qid"),
        "recommended_wikidata_uri": d.get("recommended_wikidata_uri"),
        "recommended_wikidata_label": d.get("recommended_wikidata_label"),
        "recommendation_source": d.get("recommendation_source"),
        "error": d.get("error"),
    }


def run_names(lccn: str, options: dict, em: Emitter) -> dict:
    from nar_reconcile.http import Http
    from nar_reconcile.llm import get_provider
    from nar_reconcile.pipeline import Options, reconcile_lccn

    provider_name = options.get("provider") or os.environ.get("NAR_PROVIDER", "claude")
    model = options.get("model") or os.environ.get("NAR_MODEL") or None
    kw = {"effort": options.get("effort", "medium")} if provider_name in ("claude", "anthropic", "sonnet") else {}
    prov = get_provider(provider_name, model, **kw)
    opts = Options(**{k: v for k, v in (options.get("nar") or {}).items() if k in Options.__dataclass_fields__})
    http = Http(cache_dir=None if os.environ.get("AWS_LAMBDA_FUNCTION_NAME") else os.environ.get("NAR_HTTP_CACHE"))

    relay = _LogRelay(em)
    log = logging.getLogger("nar_reconcile")
    log.addHandler(relay)
    prev_level = log.level
    if log.level == logging.NOTSET or log.level > logging.INFO:
        log.setLevel(logging.INFO)

    def on_book(book) -> None:
        contribs = [{"label": c.label, "roles": c.roles, "linked": c.reconciled, "agent_uri": c.agent_uri,
                     "agent_type": c.agent_type, "primary": c.primary} for c in book.contributors]
        n_un = len(book.unreconciled)
        em.partial({"stage": "book", "contributors": contribs, "unreconciled": n_un},
                   message=f"{len(contribs)} contributor(s) on the record, {n_un} without an LCNAF link")

    def on_contributor(r) -> None:
        em.partial({"stage": "contributor", "result": _contributor_dict(r)},
                   message=f"{r.contributor.label}: " + (
                       f"recommend {r.recommended_label} ({r.recommendation_source})" if r.recommended_authority_uri
                       else f"recommend Wikidata {r.recommended_wikidata_qid}" if r.recommended_wikidata_qid
                       else f"error: {r.error}" if r.error else "no link recommended"))

    em.progress(f"model {prov.model}: fetching the record and its contributors")
    try:
        result = reconcile_lccn(lccn, provider=prov, http=http, options=opts, on_book=on_book, on_contributor=on_contributor)
    finally:
        log.removeHandler(relay)
        log.setLevel(prev_level)
    if result.error:
        raise RuntimeError(result.error)
    book = result.book
    return {
        "provider": result.provider,
        "model": result.model,
        "instance_uri": result.instance_uri,
        "contributors": [{"label": c.label, "roles": c.roles, "linked": c.reconciled, "agent_uri": c.agent_uri,
                          "agent_type": c.agent_type, "primary": c.primary} for c in (book.contributors if book else [])],
        "results": [_contributor_dict(r) for r in result.results],
        "isbndb_used": bool(result.isbndb),
        "usage": prov.usage.as_dict(),
    }


RUNNERS = {"record": run_record, "subjects": run_subjects, "shelflist": run_shelflist, "names": run_names}


def normalize_lccn(v: str) -> str | None:
    """Accept '2025947561', ' 2025-947561', 'n 2024054501'-style or an in000… bib id; None if it isn't one."""
    s = re.sub(r"[\s\-]", "", (v or "")).lower()
    if re.fullmatch(r"in\d{5,}", s) or re.fullmatch(r"[a-z]{0,3}\d{8,10}", s):
        return s
    return None
