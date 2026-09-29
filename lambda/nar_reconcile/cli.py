"""Command line: nar-reconcile 2025947561 [more LCCNs] --provider claude|gemini"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from .http import Http
from .llm import dump_trace, get_provider
from .pipeline import Options, reconcile_lccn


def _print_summary(result) -> None:
    print(f"\n=== LCCN {result.lccn}  [{result.provider} / {result.model}]")
    if result.error:
        print(f"  ERROR: {result.error}")
        return
    b = result.book
    print(f"  {b.title}{' : ' + b.subtitle if b.subtitle else ''}  ({b.publication.statement or b.publication.date})")
    print(f"  instance: {b.instance_uri}")
    for c in b.contributors:
        print(f"    {'LINKED  ' if c.reconciled else 'UNLINKED'} {c.label}  [{', '.join(c.roles)}]  {c.authority_uri or ''}")
    if result.isbndb:
        print(f"  ISBNdb: {result.isbndb.title_long or result.isbndb.title} / authors: {', '.join(result.isbndb.authors)}")
    for r in result.results:
        print(f"\n  -> {r.contributor.label}")
        if r.search_plan:
            print(f"     name analysis: {r.search_plan.name_analysis}")
        print(f"     queries run: {len(r.queries_run)}; distinct hits: {len(r.hits)}; dossiers examined: {len(r.candidates_examined)}")
        if r.error:
            print(f"     error: {r.error}")
        d = r.decision
        if d:
            print(f"     decision: {d.decision}  uri: {d.matched_uri or '-'}  confidence: {d.confidence:.2f}")
            print(f"     reasoning: {d.reasoning}")
            if d.runner_ups:
                print(f"     runner-ups: {', '.join(d.runner_ups)}")
        if r.wikidata_queries_run:
            print(f"     wikidata: {len(r.wikidata_queries_run)} queries; {len(r.wikidata_hits)} hits")
        wd = r.wikidata_decision
        if wd:
            print(f"     wikidata decision: {wd.decision}  qid: {wd.matched_qid or '-'}  confidence: {wd.confidence:.2f}  name guard: {r.wikidata_name_guard or '-'}")
            print(f"     wikidata reasoning: {wd.reasoning}")
        if r.recommended_authority_uri:
            print(f"     RECOMMEND [{r.recommendation_source}]: {r.recommended_label}  {r.recommended_authority_uri}")
        elif r.recommended_wikidata_qid:
            print(f"     RECOMMEND [wikidata]: {r.recommended_wikidata_label}  {r.recommended_wikidata_uri}")
        else:
            print("     RECOMMEND: no link")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reconcile unlinked contributors on an LC record to LCNAF URIs.")
    ap.add_argument("lccn", nargs="+", help="One or more LCCNs (e.g. 2025947561)")
    ap.add_argument("--provider", "-p", default="claude", choices=["claude", "gemini"])
    ap.add_argument("--model", "-m", default=None, help="Override model ID")
    ap.add_argument("--effort", default="medium", help="Claude effort: low|medium|high|xhigh|max")
    ap.add_argument("--isbndb", default="auto", choices=["auto", "always", "never"])
    ap.add_argument("--max-candidates", type=int, default=12)
    ap.add_argument("--min-confidence", type=float, default=0.75)
    ap.add_argument("--no-llm-queries", action="store_true", help="Only use mechanical name variants for searching")
    ap.add_argument("--no-wikidata", action="store_true", help="Skip the Wikidata fallback when no LCNAF match is found")
    ap.add_argument("--only", action="append", help="Only reconcile contributors whose label contains this text (repeatable)")
    ap.add_argument("--out", "-o", default="out", help="Directory for JSON results (and LLM traces)")
    ap.add_argument("--trace", action="store_true", help="Also write the full LLM prompts/responses next to the result")
    ap.add_argument("--cache", default=None, help="Directory for an HTTP GET cache (or set NAR_HTTP_CACHE)")
    ap.add_argument("-v", "--verbose", action="count", default=0)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose > 1 else logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # AFC advisory is irrelevant here

    http = Http(cache_dir=args.cache)
    kw = {"effort": args.effort} if args.provider == "claude" else {}
    provider = get_provider(args.provider, args.model, **kw)
    opts = Options(
        isbndb=args.isbndb,
        max_candidates=args.max_candidates,
        min_confidence=args.min_confidence,
        llm_queries=not args.no_llm_queries,
        wikidata=not args.no_wikidata,
        only=args.only,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rc = 0
    for lccn in args.lccn:
        result = reconcile_lccn(lccn, provider=provider, http=http, options=opts)
        _print_summary(result)
        path = out / f"{result.lccn}.{provider.name}.json"
        path.write_text(result.model_dump_json(indent=2, exclude_none=True))
        if args.trace:
            dump_trace(provider, str(out / f"{result.lccn}.{provider.name}.trace.json"))
            provider.trace.clear()
        print(f"  wrote {path}")
        if result.error:
            rc = 1
    print(f"\nLLM usage: {json.dumps(provider.usage.as_dict())}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
