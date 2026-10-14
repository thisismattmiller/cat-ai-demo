"""Collect records cataloged today whose id.loc.gov copy has no shelfmark yet, so tomorrow's copy (with the
human-assigned 050 $b) can be compared against the shelflister.

  uv run qa/collect.py --since 2026-09-29 --want 200          # writes qa/to_check/<lccn>.cbd.rdf + manifest.json

Reads qa/production.json (the editor's publish log; `time` is UTC), keeps published Monographs with an LCCN from
`--since` (local midnight, America/New_York) onward, oldest first, and for each: resolves the LCCN at id.loc.gov,
fetches the instance CBD, and looks for bf:ClassificationLcc. Records with no call number, or a class number
without an item portion, are saved. Records already carrying a full call number, or not yet on id.loc.gov, are
noted in the manifest but not saved. Stops after `--want` saved records.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "lambda"))
from shelflister import idloc  # noqa: E402

OUT = ROOT / "qa" / "to_check"


def check(rec: dict) -> dict:
    lccn = rec["lccn"].strip()
    info = {"lccn": lccn, "eid": rec.get("eid"), "title": rec.get("title"), "contributor": rec.get("contributor"),
            "user": rec.get("user"), "time_utc": rec.get("time"), "typeid": rec.get("typeid")}
    try:
        url = idloc.resolve_lccn(lccn)
        xml = idloc.fetch_cbd(url, cache_dir=None)
        resource, idrec = idloc.parse_cbd(xml)
    except LookupError as e:
        return {**info, "outcome": "not_on_id", "error": str(e)}
    except (httpx.HTTPError, ValueError) as e:
        return {**info, "outcome": "error", "error": f"{type(e).__name__}: {e}"}
    info.update({"instance_url": idrec.instance_url, "bibid": idrec.bibid, "creator": resource.creator,
                 "creator_uri": idrec.creator_uri, "lcc_today": [{"class_number": a, "item": b} for a, b in idrec.lcc]})
    if idrec.lcc and idrec.lcc[0][1]:
        info["outcome"] = "has_shelfmark"
    else:
        info["outcome"] = "no_shelfmark" if not idrec.lcc else "class_only"
        (OUT / f"{lccn}.cbd.rdf").write_bytes(xml)
        info["file"] = f"qa/to_check/{lccn}.cbd.rdf"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", default=datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d"), help="local date (default today)")
    ap.add_argument("--want", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--log", default="qa/production.json")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    since = datetime.strptime(a.since, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/New_York")).timestamp()
    recs = [r for r in json.load(open(ROOT / a.log)).values()
            if r.get("status") == "published" and r.get("typeid") == "Monograph" and r.get("lccn")
            and r.get("timestamp", 0) >= since]
    recs.sort(key=lambda r: r["timestamp"])
    seen, queue = set(), []
    for r in recs:
        if r["lccn"] not in seen:
            seen.add(r["lccn"]); queue.append(r)
    print(f"{len(queue)} published monographs with an LCCN since {a.since} local; checking id.loc.gov ...", file=sys.stderr)

    manifest_path = OUT / "manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else {"since": a.since, "collected_at": None, "records": []}
    done = {m["lccn"] for m in manifest["records"]}
    saved = sum(1 for m in manifest["records"] if m.get("file"))
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        batch = [r for r in queue if r["lccn"] not in done]
        for i in range(0, len(batch), a.workers * 4):
            if saved >= a.want:
                break
            chunk = batch[i:i + a.workers * 4]
            for info in ex.map(check, chunk):
                manifest["records"].append(info)
                if info.get("file"):
                    saved += 1
                print(f"{info['outcome']:<14} {info['lccn']:<12} {(info.get('title') or '')[:60]}", file=sys.stderr)
            manifest["collected_at"] = datetime.now(ZoneInfo("America/New_York")).isoformat(timespec="seconds")
            manifest_path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    from collections import Counter
    c = Counter(m["outcome"] for m in manifest["records"])
    print(f"\nchecked {len(manifest['records'])} in {time.time() - t0:.0f}s: {dict(c)}; saved {saved} CBDs to {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
