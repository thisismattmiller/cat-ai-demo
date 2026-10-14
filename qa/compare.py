"""Tomorrow's pass: re-fetch each collected record from id.loc.gov, keep the ones that now carry the human-assigned
shelfmark, run them through the shelflister (with the call number hidden and the record excluded from the
shelflist, plus the name-reconciliation gate), and score the result against LC's.

  uv run qa/compare.py --limit 20 --workers 3            # writes qa/checked/<lccn>.cbd.rdf, qa/results.jsonl
  uv run qa/compare.py --report                          # summary table from qa/results.jsonl

Resumable: LCCNs already in qa/results.jsonl are skipped. The shelflister reads the record from id.loc.gov itself
(that is what the demo does), so qa/checked/ is the archival copy of what the human assigned.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lambda"))
from app import run_all  # noqa: E402
from shelflister import callnum, idloc  # noqa: E402
from ws import PrintConnection  # noqa: E402

TO_CHECK = ROOT / "qa" / "to_check"
CHECKED = ROOT / "qa" / "checked"
RESULTS = ROOT / "qa" / "results.jsonl"


def score(ours: str | None, theirs: str) -> dict:
    o, t = (callnum.try_parse(ours) if ours else None), callnum.try_parse(theirs)
    if not (o and t):
        return {"class": False, "exact": False, "book": False, "date": False}
    oa, ob = o.marc_050()
    ta, tb = t.marc_050()
    return {"class": oa.replace(" ", "") == ta.replace(" ", ""), "exact": o.format() == t.format(),
            "book": ob == tb, "date": (o.date or "") == (t.date or ""),
            "class_letters": o.letters == t.letters}


def one(m: dict, options: dict) -> dict:
    lccn = m["lccn"]
    out = {"lccn": lccn, "title": m.get("title"), "creator": m.get("creator"), "creator_linked_yesterday": bool(m.get("creator_uri")),
           "summary_source": m.get("summary_source")}
    try:
        xml = idloc.fetch_cbd(m["instance_url"], cache_dir=None)
        resource, rec = idloc.parse_cbd(xml)
    except Exception as e:  # noqa: BLE001
        return {**out, "outcome": "fetch_error", "error": f"{type(e).__name__}: {e}"}
    if not (rec.lcc and rec.lcc[0][1]):
        return {**out, "outcome": "still_no_shelfmark", "lcc_now": rec.lcc}
    (CHECKED / f"{lccn}.cbd.rdf").write_bytes(xml)
    theirs = f"{rec.lcc[0][0]} {rec.lcc[0][1]}".strip()
    out.update({"outcome": "compared", "lc_call_number": theirs, "creator_now": resource.creator, "creator_linked_now": bool(rec.creator_uri)})
    log = io.StringIO()
    t0 = time.time()
    res = run_all(["shelflist", "names"], lccn, {"hide_class": True, **options}, PrintConnection(log))
    sl = res.get("shelflist")
    out["seconds"] = round(time.time() - t0, 1)
    out["events"] = log.getvalue().count("\n")
    if not sl:
        out["outcome"] = "shelflist_failed"
        out["error"] = next((l for l in log.getvalue().splitlines() if '"type": "error"' in l), None)
        return out
    out.update({"our_call_number": sl.get("call_number"), "main_entry": sl.get("main_entry"),
                "classification_confidence": (sl.get("classification") or {}).get("confidence"),
                "compose_confidence": (sl.get("compose") or {}).get("confidence"),
                "needs_review": (sl.get("compose") or {}).get("needs_review"),
                "caption_path": (sl.get("classification") or {}).get("caption_path"),
                "timing": sl.get("timing"), "model": sl.get("model"), "score": score(sl.get("call_number"), theirs)})
    return out


def report() -> None:
    rows = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    c = Counter(r["outcome"] for r in rows)
    print(f"{len(rows)} records: {dict(c)}")
    cmp = [r for r in rows if r["outcome"] == "compared"]
    if not cmp:
        return
    n = len(cmp)
    for k, label in (("class_letters", "class letters"), ("class", "class number (050 $a)"), ("book", "book number (050 $b)"),
                     ("date", "date"), ("exact", "whole call number")):
        hit = sum(1 for r in cmp if r["score"].get(k))
        print(f"  {label:<26} {hit:>4}/{n}  {100 * hit / n:5.1f}%")
    print(f"  mean seconds/record        {sum(r['seconds'] for r in cmp) / n:6.1f}")
    gated = [r for r in cmp if r.get("main_entry", {}).get("source") not in (None, "record")]
    if gated:
        print(f"  main entry from LCNAF gate {len(gated)} (class match {sum(1 for r in gated if r['score']['class'])})")
    print("\nmisses (class number):")
    for r in cmp:
        if not r["score"]["class"]:
            print(f"  {r['lccn']}  ours {r['our_call_number']!s:<26} LC {r['lc_call_number']:<26} {(r.get('title') or '')[:50]}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many newly compared records (0 = all)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--options", default="{}", help="extra shelflist options JSON, e.g. '{\"model\": \"sonnet\"}'")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--require-summary", action="store_true", help="only records with an LC or ISBNdb summary (manifest summary_source)")
    a = ap.parse_args()
    if a.report:
        report(); return 0
    CHECKED.mkdir(exist_ok=True)
    manifest = json.load(open(TO_CHECK / "manifest.json"))
    todo = [m for m in manifest["records"] if m.get("file") and (m.get("summary_source") or not a.require_summary)]
    done = {json.loads(l)["lccn"] for l in RESULTS.read_text().splitlines() if l.strip()} if RESULTS.exists() else set()
    todo = [m for m in todo if m["lccn"] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} records to check ({len(done)} already done)", file=sys.stderr)
    options = json.loads(a.options)
    with ThreadPoolExecutor(max_workers=a.workers) as ex, open(RESULTS, "a") as f:
        for r in ex.map(lambda m: one(m, options), todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n"); f.flush()
            s = r.get("score") or {}
            print(f"{r['outcome']:<20} {r['lccn']:<12} ours {str(r.get('our_call_number')):<24} LC {str(r.get('lc_call_number')):<24} "
                  f"{'CLASS ' if s.get('class') else '      '}{'EXACT' if s.get('exact') else ''}", file=sys.stderr)
    report()
    return 0


if __name__ == "__main__":
    sys.exit(main())
