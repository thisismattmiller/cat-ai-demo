"""shelflist: recommend an LC shelfmark (class number + Cutter + date) from metadata.

  shelflist --title "A room of one's own" --creator "Woolf, Virginia, 1882-1941" --summary "..." --date 2026
  shelflist --title ... --class PR6045.O72          # class number known: only the book number is built
  shelflist --bibid 8258583 [--hide-class]           # LC record as input (for testing; hides its 050)
  shelflist --schedule HV1478 | --search "beef cooking"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import idloc, marc
from .pipeline import Pipeline
from .resource import Resource


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="shelflist", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--title")
    src.add_argument("--lccn", help="LC control number: the record is pulled from id.loc.gov (BIBFRAME), plus ISBNdb")
    src.add_argument("--bibid", help="LC bib id (8258583 or in00024328432), likewise from id.loc.gov")
    src.add_argument("--file", help="MARC XML / JSON / binary file")
    src.add_argument("--schedule", metavar="CLASS", help="print the LC schedule entry for a class number and exit")
    src.add_argument("--search", metavar="WORDS", help="search the LC schedules and exit")
    ap.add_argument("--creator", help="author/creator, surname first")
    ap.add_argument("--summary")
    ap.add_argument("--contents", help="table of contents / chapter list")
    ap.add_argument("--date", help="publication date")
    ap.add_argument("--publisher")
    ap.add_argument("--edition")
    ap.add_argument("--subject", action="append", help="LC subject heading (repeatable)")
    ap.add_argument("--class", dest="class_number", help="class number if already decided; skips classification")
    ap.add_argument("--context", help="extra notes for the model")
    ap.add_argument("--hide-class", action="store_true", help="with a record: ignore its 050 and classify from scratch")
    ap.add_argument("--no-lambda", action="store_true", help="do not consult the similar-records lambda")
    ap.add_argument("--no-isbndb", action="store_true", help="with --lccn/--bibid: do not enrich from ISBNdb")
    ap.add_argument("--sru", action="store_true", help="with --lccn/--bibid: fetch MARC from LC's SRU server instead of id.loc.gov")
    ap.add_argument("--model", help="gemini-3.8-flash (default), sonnet (claude-sonnet-5-5), opus, or any full model id")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="Claude models only")
    ap.add_argument("--out", help="write the full result JSON here")
    ap.add_argument("--trace", action="store_true", help="print the tool traces")
    a = ap.parse_args(argv)

    if a.schedule or a.search:
        from .schedule import get_schedules
        sch = get_schedules()
        if not sch:
            print("schedules not available (set SHELFLISTER_LCC_DIR)", file=sys.stderr)
            return 2
        print(sch.lookup(a.schedule).text() if a.schedule else sch.search_text(a.search, limit=25))
        return 0

    class_number = a.class_number
    exclude = set()
    if a.title:
        res_in = Resource(title=a.title, creator=a.creator, summary=a.summary, contents=a.contents, date=a.date,
                          publisher=a.publisher, edition=a.edition, subjects=a.subject or [])
    elif (a.lccn or a.bibid) and not a.sru:
        try:
            res_in, idrec = idloc.from_lccn(a.lccn, bibid=a.bibid, use_isbndb=not a.no_isbndb)
        except LookupError as e:
            print(f"{e}; trying LC's SRU server", file=sys.stderr)
            rec = marc.fetch_lc(lccn=a.lccn, bibid=a.bibid)
            res_in, idrec = Resource.from_marc(rec), None
        if idrec:
            print(f"id.loc.gov    : {idrec.instance_url}")
            if idrec.contributors:
                print(f"contributors  : {'; '.join(idrec.contributors)}")
            if idrec.isbndb_added:
                print(f"ISBNdb added  : {', '.join(idrec.isbndb_added)}")
            if idrec.lcc:
                print(f"LC's own 050  : {' | '.join(f'$a {x} $b {y}' if y else f'$a {x}' for x, y in idrec.lcc)}")
            if class_number is None and not a.hide_class and idrec.lcc:
                class_number = idrec.lcc[0][0]
            exclude = {idrec.bibid}
        else:
            given = marc.call_numbers_050(rec)
            if class_number is None and not a.hide_class and given:
                class_number = given[0][1]
            exclude = {rec["001"].data.strip()} if rec.get("001") else set()
    else:
        rec = marc.fetch_lc(lccn=a.lccn) if a.lccn else marc.fetch_lc(bibid=a.bibid) if a.bibid else marc.load(a.file)
        res_in = Resource.from_marc(rec)
        given = marc.call_numbers_050(rec)
        if given:
            print(f"LC's own 050  : {' | '.join(f'$a {g[1]} $b {g[2]}' for g in given)}")
        if class_number is None and not a.hide_class and given:
            class_number = given[0][1]
        f001 = rec.get("001")
        if f001:
            exclude = {f001.data.strip()}

    try:
        pipe = Pipeline(model=a.model, effort=a.effort, use_lambda=not a.no_lambda)
        res = pipe.run(res_in, class_number, extra_context=a.context, exclude_bibids=exclude)
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        if "credential" in msg.lower() or "API_KEY" in msg or "GOOGLE_AI" in msg or type(e).__name__.startswith(("Auth", "WorkloadIdentity")):
            print(f"credentials problem: {msg}", file=sys.stderr)
            return 2
        raise

    if res.classification:
        c = res.classification
        print(f"class number  : {c['class_number']}   ({c['confidence']}{'; new number' if c['needs_new_number'] else ''})")
        print(f"  caption     : {c['caption_path']}")
        print(f"  basis       : {c['schedule_basis']}")
        print(f"  subjects    : {'; '.join(c['subject_headings'])}")
        for alt in c["alternatives"]:
            print(f"  not {alt['class_number']}: {alt['why_not']}")
        print(f"  reasoning   : {c['reasoning']}")
        if c["open_questions"]:
            print("  open        : " + " | ".join(c["open_questions"]))
        if a.trace:
            _trace(c["trace"])
        print()
    else:
        print(f"class number  : {res.class_number}")
    if res.compose:
        cp = res.compose
        print(f"call number   : {res.call_number}")
        print(f"  050         : $a {res.marc_050[0]} $b {res.marc_050[1]}" if res.marc_050 else "  (no call number)")
        print(f"  confidence  : {cp['confidence']}   needs review: {cp['needs_review']}")
        v = cp["validation"]
        w = v.get("warnings") or []
        wtxt = ("  " + " | ".join(w[:4]) + (f" | +{len(w) - 4} more" if len(w) > 4 else "")) if w else ""
        print(f"  validation  : {'ok' if v.get('ok') else v.get('errors')}{wtxt}")
        print(f"  citations   : {'; '.join(cp['citations'])}")
        print(f"  cards       : {', '.join(res.cards)}")
        print(f"  reasoning   : {cp['reasoning']}")
        if cp["open_questions"]:
            print("  open        : " + " | ".join(cp["open_questions"]))
        if a.trace:
            _trace(cp["trace"])
    print(f"\nmodel {res.model}  usage {res.usage}  timing {res.timing}")
    if a.out:
        Path(a.out).write_text(res.to_json())
        print(f"wrote {a.out}")
    return 0


def _trace(trace):
    print("  trace:")
    for t in trace:
        if "tool" in t:
            print(f"    > {t['tool']}({t['input']})")
            print("      " + t["output"][:400].replace("\n", "\n      "))
        elif "text" in t:
            print(f"    [text] {t['text'][:300]}")


if __name__ == "__main__":
    sys.exit(main())
