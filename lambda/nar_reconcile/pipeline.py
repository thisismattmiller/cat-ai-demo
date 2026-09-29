"""The reconciliation pipeline.

    result = reconcile_lccn("2025947561", provider="claude")

For every contributor on the work that has no agent URI:
  1. build mechanical name variants (names.py) and ask the LLM for more (SearchPlan)
  2. run each query through suggest2 (left-anchored + keyword), collect and de-dupe hits
  3. if too many hits, ask the LLM to shortlist from the suggest2 summaries
  4. fetch the full authority record + contributorTo/subjectOf works for each shortlisted hit
  5. ask the LLM for a match / no_match decision with confidence and reasoning
  6. apply guards (URI must be a real candidate; confidence threshold) and emit a recommendation
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from . import loc, prompts, wikidata
from .http import Http
from .isbndb import fetch_isbndb
from .llm import LLMError, Provider, get_provider
from .models import (
    BookRecord,
    Candidate,
    Contributor,
    ContributorResult,
    Decision,
    IsbndbBook,
    ReconcileResult,
    SearchPlan,
    Shortlist,
    SuggestHit,
    WikidataCandidate,
    WikidataDecision,
)
from .names import deterministic_variants

log = logging.getLogger(__name__)


@dataclass
class Options:
    isbndb: str = "auto"  # auto | always | never
    max_candidates: int = 12  # dossiers to fetch and show the decision model
    suggest_count: int = 20  # hits per suggest2 query
    min_confidence: float = 0.75  # below this a `match` is reported but not recommended
    llm_queries: bool = True  # ask the LLM for extra search strings
    only: list[str] | None = None  # restrict to contributors whose label contains one of these
    workers: int = 6
    wikidata: bool = True  # fall back to Wikidata when no LCNAF match is recommended
    wikidata_limit: int = 7  # hits per wbsearchentities query
    wikidata_max_candidates: int = 10


# --------------------------------------------------------------------------- #


def _isbndb_wanted(book: BookRecord, mode: str) -> bool:
    if mode == "never" or not book.isbns:
        return False
    if mode == "always":
        return True
    sparse = not book.summary and len(book.subjects) < 2
    return sparse


def _merge_hits(into: dict[str, SuggestHit], hits: list[SuggestHit]) -> None:
    for h in hits:
        if h.uri in into:
            for tag in h.found_by:
                if tag not in into[h.uri].found_by:
                    into[h.uri].found_by.append(tag)
            if h.matched_variant and not into[h.uri].matched_variant:
                into[h.uri].matched_variant = h.matched_variant
        else:
            into[h.uri] = h


def _run_queries(
    http: Http, queries: list[tuple[str, str]], rdftype: str | None, count: int, workers: int
) -> tuple[dict[str, SuggestHit], list[str]]:
    """queries: list of (q, mode). Returns hits by URI and the list of query tags run."""
    seen: set[tuple[str, str]] = set()
    todo = []
    for q, mode in queries:
        key = (q.strip().lower(), mode)
        if q.strip() and key not in seen:
            seen.add(key)
            todo.append((q.strip(), mode))

    def one(item):
        q, mode = item
        return loc.suggest2(http, q, keyword=(mode == "keyword"), rdftype=rdftype, count=count)

    hits: dict[str, SuggestHit] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(one, todo):
            _merge_hits(hits, res)
    return hits, [f"{m}:{q}" for q, m in todo]


def _decision_problem(d: Decision, by_token: dict[str, Candidate]) -> str | None:
    """Return a description of why a Decision is unusable, or None if it looks sound."""
    if len(d.reasoning.strip()) < 40:
        return f"reasoning too short: {d.reasoning!r}"
    if d.decision == "match":
        if loc.uri_token(d.matched_uri) not in by_token:
            return f"matched_uri {d.matched_uri!r} is not one of the candidates"
        if d.confidence <= 0.0:
            return "match with zero confidence"
    if d.decision == "no_match" and d.matched_uri:
        return "no_match but matched_uri is set"
    return None


def reconcile_contributor(
    http: Http,
    provider: Provider,
    book: BookRecord,
    contributor: Contributor,
    isbndb: IsbndbBook | None,
    opts: Options,
) -> ContributorResult:
    res = ContributorResult(contributor=contributor)
    book_ctx = prompts.render_book(book, isbndb)
    variants = deterministic_variants(contributor.label, contributor.marc_key)
    contrib_ctx = prompts.render_contributor(contributor, variants)
    _lcnaf_step(http, provider, book_ctx, contrib_ctx, contributor, variants, res, opts)
    if res.recommended_authority_uri:
        res.recommendation_source = "lcnaf"
    elif opts.wikidata:
        try:
            _wikidata_step(http, provider, book_ctx, contrib_ctx, contributor, variants, res.search_plan, res, opts)
        except Exception as e:  # never let the fallback break the LCNAF result
            log.exception("wikidata step failed for %s", contributor.label)
            res.error = (res.error + "; " if res.error else "") + f"wikidata: {type(e).__name__}: {e}"
    return res


def _lcnaf_step(
    http: Http,
    provider: Provider,
    book_ctx: str,
    contrib_ctx: str,
    contributor: Contributor,
    variants: list[str],
    res: ContributorResult,
    opts: Options,
) -> None:
    rdftype = loc.AGENT_TYPE_TO_RDFTYPE.get(contributor.agent_type)

    # ---- 1. queries
    queries: list[tuple[str, str]] = []
    for v in variants:
        queries.append((v, "left"))
    for v in variants[:3]:
        queries.append((v, "keyword"))
    if opts.llm_queries:
        try:
            plan = provider.generate(prompts.SEARCH_PLAN_SYSTEM, book_ctx + "\n" + contrib_ctx, SearchPlan)
            res.search_plan = plan
            for sq in plan.queries[:8]:
                queries.append((sq.q, sq.mode))
        except LLMError as e:
            log.warning("search-plan step failed for %s: %s", contributor.label, e)
            res.error = f"search plan: {e}"

    # ---- 2. suggest2
    hits, res.queries_run = _run_queries(http, queries, rdftype, opts.suggest_count, opts.workers)
    res.hits = list(hits.values())
    log.info("%s: %d queries -> %d distinct hits", contributor.label, len(res.queries_run), len(hits))
    if not hits:
        res.decision = Decision(
            decision="no_match",
            matched_uri=None,
            confidence=1.0,
            reasoning="No LCNAF headings were returned by suggest2 for any query form.",
            runner_ups=[],
        )
        return

    # ---- 3. shortlist when there are too many
    ordered = list(hits.values())
    if len(ordered) > opts.max_candidates:
        summaries = "\n".join(prompts.render_hit(h) for h in ordered)
        user = (
            f"{book_ctx}\n{contrib_ctx}\n## Search hits ({len(ordered)}; keep at most {opts.max_candidates})\n{summaries}"
        )
        try:
            sl = provider.generate(prompts.SHORTLIST_SYSTEM, user, Shortlist)
            keep_tokens = {loc.uri_token(u) for u in sl.keep_uris}
            keep = [h for h in ordered if loc.uri_token(h.uri) in keep_tokens]
            if keep:
                ordered = keep[: opts.max_candidates]
            else:
                log.warning("shortlist kept nothing (%s); falling back to first %d hits", sl.reasoning[:200], opts.max_candidates)
                ordered = ordered[: opts.max_candidates]
        except LLMError as e:
            log.warning("shortlist step failed for %s: %s", contributor.label, e)
            ordered = ordered[: opts.max_candidates]

    # ---- 4. dossiers
    with ThreadPoolExecutor(max_workers=opts.workers) as ex:
        cands = [c for c in ex.map(lambda h: loc.build_candidate(http, h), ordered) if c]
    res.candidates_examined = [c.uri for c in cands]
    if not cands:
        res.error = (res.error + "; " if res.error else "") + "could not fetch any candidate authority records"
        return

    # ---- 5. decision (one retry if the output is a stub or names a URI that is not a candidate)
    dossiers = "".join(prompts.render_candidate(c, i + 1) for i, c in enumerate(cands))
    user = f"{book_ctx}\n{contrib_ctx}\n## LCNAF candidates ({len(cands)})\n{dossiers}"
    by_token: dict[str, Candidate] = {loc.uri_token(c.uri): c for c in cands}
    decision = None
    for attempt in range(2):
        try:
            decision = provider.generate(prompts.DECISION_SYSTEM, user, Decision)
        except LLMError as e:
            res.error = (res.error + "; " if res.error else "") + f"decision: {e}"
            return
        problem = _decision_problem(decision, by_token)
        if not problem:
            break
        log.warning("decision for %s rejected (%s); %s", contributor.label, problem, "retrying" if attempt == 0 else "giving up")
        if attempt == 0:
            user += (
                "\n\n## Note\nA previous attempt produced an unusable answer (" + problem + "). "
                "Write the full reasoning first, then decide. matched_uri must be exactly one of: "
                + ", ".join(c.uri for c in cands) + " or null."
            )
    decision.confidence = max(0.0, min(1.0, decision.confidence))

    # ---- 6. guards
    decision.runner_ups = [
        by_token[loc.uri_token(u)].uri for u in decision.runner_ups if loc.uri_token(u) in by_token
    ]
    if decision.decision == "match":
        cand = by_token.get(loc.uri_token(decision.matched_uri))
        if cand is None:
            decision.reasoning += f" [GUARD: model returned URI {decision.matched_uri!r} that was not among the candidates; treated as no_match]"
            decision.decision = "no_match"
            decision.matched_uri = None
        else:
            decision.matched_uri = cand.uri
            if decision.confidence >= opts.min_confidence:
                res.recommended_authority_uri = cand.uri
                res.recommended_rwo_uri = cand.uri.replace("/authorities/names/", "/rwo/agents/")
                res.recommended_label = cand.label
            else:
                decision.reasoning += f" [confidence {decision.confidence:.2f} below threshold {opts.min_confidence}; not recommended]"
    res.decision = decision


def _wikidata_step(
    http: Http,
    provider: Provider,
    book_ctx: str,
    contrib_ctx: str,
    contributor: Contributor,
    variants: list[str],
    plan: SearchPlan | None,
    res: ContributorResult,
    opts: Options,
) -> None:
    """Search Wikidata for the contributor and, if the LLM accepts a candidate, recommend it.
    If the accepted item carries an LCNAF id (P244), recover the LCNAF URI from it."""
    queries: list[str] = []

    def add(q: str | None):
        q = (q or "").strip()
        if q and q.lower() not in {x.lower() for x in queries}:
            queries.append(q)

    if plan:
        for f in plan.direct_order_forms[:3]:
            add(f)
        for f in plan.native_script_forms[:3]:
            add(f)
    if contributor.agent_type == "Person":
        for v in variants[:3]:
            add(wikidata.direct_order(v))
    else:
        for v in variants[:2]:
            add(v)
    queries = queries[:6]

    hits: dict[str, WikidataCandidate] = {}
    for q in queries:
        for h in wikidata.search(http, q, limit=opts.wikidata_limit):
            if h.qid in hits:
                hits[h.qid].found_by += [t for t in h.found_by if t not in hits[h.qid].found_by]
            else:
                hits[h.qid] = h
    res.wikidata_queries_run = queries
    if not hits:
        res.wikidata_hits = []
        log.info("%s: no Wikidata hits for %s", contributor.label, queries)
        return
    cands = wikidata.enrich(http, list(hits.values())[: opts.wikidata_max_candidates])
    # drop obvious non-agents (disambiguation pages, works) when typing is available
    cands = [c for c in cands if not any(t in ("Wikimedia disambiguation page",) for t in c.instance_of)]
    res.wikidata_hits = cands
    if not cands:
        return

    by_qid = {c.qid: c for c in cands}
    dossiers = "".join(prompts.render_wikidata_candidate(c, i + 1) for i, c in enumerate(cands))
    user = f"{book_ctx}\n{contrib_ctx}\n## Wikidata candidates ({len(cands)})\n{dossiers}"
    decision = None
    for attempt in range(2):
        try:
            decision = provider.generate(prompts.WIKIDATA_DECISION_SYSTEM, user, WikidataDecision)
        except LLMError as e:
            res.error = (res.error + "; " if res.error else "") + f"wikidata decision: {e}"
            return
        problem = None
        if len(decision.reasoning.strip()) < 40:
            problem = "reasoning too short"
        elif decision.decision == "match" and (decision.matched_qid or "").upper() not in by_qid:
            problem = f"matched_qid {decision.matched_qid!r} is not one of the candidates"
        elif decision.decision == "match" and decision.confidence <= 0:
            problem = "match with zero confidence"
        if not problem:
            break
        log.warning("wikidata decision for %s rejected (%s)", contributor.label, problem)
        if attempt == 0:
            user += f"\n\n## Note\nA previous attempt produced an unusable answer ({problem}). matched_qid must be one of: {', '.join(by_qid)} or null."
    decision.confidence = max(0.0, min(1.0, decision.confidence))
    decision.runner_ups = [q for q in decision.runner_ups if q in by_qid]
    if decision.decision == "match" and (decision.matched_qid or "").upper() not in by_qid:
        decision.decision, decision.matched_qid = "no_match", None
    res.wikidata_decision = decision
    if decision.decision != "match":
        return

    cand = by_qid[decision.matched_qid.upper()]
    decision.matched_qid = cand.qid
    res.wikidata_name_guard = wikidata.name_guard(contributor.label, cand.label, cand.aliases)
    if decision.confidence < opts.min_confidence:
        decision.reasoning += f" [confidence {decision.confidence:.2f} below threshold {opts.min_confidence}; not recommended]"
        return
    if res.wikidata_name_guard == "weak":
        decision.reasoning += " [name guard: matched label shares no name tokens with the record's form; review before use]"

    res.recommended_wikidata_qid = cand.qid
    res.recommended_wikidata_uri = cand.uri
    res.recommended_wikidata_label = cand.label
    res.recommendation_source = "wikidata"
    if cand.lcnaf_id:
        # Wikidata says this person has an LCNAF record suggest2 did not surface: verify and use it
        uri = f"http://id.loc.gov/authorities/names/{cand.lcnaf_id.strip()}"
        auth = loc.fetch_authority(http, uri)
        if auth is not None:
            res.recommended_authority_uri = auth.uri
            res.recommended_rwo_uri = auth.uri.replace("/authorities/names/", "/rwo/agents/")
            res.recommended_label = auth.label
            res.recommendation_source = "wikidata-p244"
            res.candidates_examined.append(auth.uri)
        else:
            decision.reasoning += f" [P244 {cand.lcnaf_id} did not resolve at id.loc.gov]"


def reconcile_lccn(
    lccn: str,
    provider: str | Provider = "claude",
    model: str | None = None,
    http: Http | None = None,
    options: Options | None = None,
    on_book=None,
    on_contributor=None,
) -> ReconcileResult:
    """`on_book(book)` fires once the record is parsed; `on_contributor(result)` after each contributor."""
    opts = options or Options()
    http = http or Http()
    prov = provider if isinstance(provider, Provider) else get_provider(provider, model)
    result = ReconcileResult(lccn=loc.normalize_lccn(lccn), provider=prov.name, model=prov.model)

    try:
        book = loc.fetch_book(http, lccn)
    except Exception as e:  # network / parse
        result.error = f"failed to fetch record: {e}"
        return result
    if book is None:
        result.error = "LCCN did not resolve at id.loc.gov"
        return result
    result.book = book
    result.instance_uri = book.instance_uri
    if on_book:
        on_book(book)

    if _isbndb_wanted(book, opts.isbndb):
        for isbn in book.isbns:
            try:
                result.isbndb = fetch_isbndb(http, isbn)
            except Exception as e:
                log.warning("ISBNdb failed for %s: %s", isbn, e)
            if result.isbndb:
                break

    targets = book.unreconciled
    if opts.only:
        targets = [c for c in targets if any(o.lower() in c.label.lower() for o in opts.only)]
    log.info("%s: %d contributors, %d unreconciled, %d targeted", book.lccn, len(book.contributors), len(book.unreconciled), len(targets))

    for c in targets:
        try:
            result.results.append(reconcile_contributor(http, prov, book, c, result.isbndb, opts))
        except Exception as e:
            log.exception("contributor %s failed", c.label)
            result.results.append(ContributorResult(contributor=c, error=f"{type(e).__name__}: {e}"))
        if on_contributor:
            on_contributor(result.results[-1])
    return result
