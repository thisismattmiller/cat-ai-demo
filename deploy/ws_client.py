"""Test client: connect to the deployed WebSocket, run an LCCN, print events until every task finishes.

  uv run deploy/ws_client.py 2025947561 [--tasks record,subjects] [--ws wss://...]
  uv run deploy/ws_client.py 2025947561 --out out/2025947561.ws.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import websockets

ROOT = Path(__file__).resolve().parent.parent


def default_ws() -> str | None:
    cfg = ROOT / "docs" / "config.js"
    if cfg.exists():
        m = re.search(r'"(wss://[^"]+)"', cfg.read_text())
        if m:
            return m.group(1)
    return None


async def run(ws_url: str, lccn: str, tasks: list[str] | None, options: dict, out: str | None) -> int:
    results, errors = {}, {}
    t0 = time.time()
    async with websockets.connect(ws_url, max_size=2**21) as ws:
        msg = {"action": "run", "lccn": lccn, "options": options}
        if tasks:
            msg["tasks"] = tasks
        await ws.send(json.dumps(msg))
        pending: set[str] | None = None
        while pending is None or pending:
            try:
                raw = await asyncio.wait_for(ws.recv(), timeout=600)
            except asyncio.TimeoutError:
                print("timed out waiting for events", file=sys.stderr)
                return 1
            ev = json.loads(raw)
            t = f"{time.time() - t0:6.1f}s"
            typ, task = ev.get("type"), ev.get("task", "-")
            if typ is None and "message" in ev:
                # API Gateway's own notice: it stops waiting for the integration after 29 s; the function keeps running
                print(f"{t}  (api gateway: {ev['message']})")
                continue
            if pending is None and typ in ("result", "error", "progress", "partial"):
                pending = set()
            if typ == "started":
                pending = set(ev["tasks"])
                print(f"{t}  started {ev['lccn']}: {', '.join(ev['tasks'])}")
            elif typ == "progress":
                print(f"{t}  [{task}] {ev.get('message', '')[:160]}")
            elif typ == "partial":
                print(f"{t}  [{task}] ~ {ev.get('message', '')[:160]}")
            elif typ == "result":
                results[task] = ev["data"]
                pending.discard(task)
                print(f"{t}  [{task}] RESULT ({len(raw)} bytes) after {ev.get('t')}s")
            elif typ == "error":
                errors[task] = ev.get("error")
                if pending is not None:
                    pending.discard(task)
                print(f"{t}  [{task}] ERROR {ev.get('error')}")
                if pending is None:
                    return 1
            else:
                print(f"{t}  {raw[:200]}")
    print(f"\ndone in {time.time() - t0:.1f}s; results: {sorted(results)}; errors: {errors or 'none'}")
    for task, data in results.items():
        if task == "shelflist":
            print(f"  call number : {data.get('call_number')}   LC: {(data.get('comparison') or {}).get('lc_call_number')}")
        if task == "subjects":
            print(f"  subjects    : {'; '.join(data.get('recommended_subjects', [])[:6])}")
        if task == "names":
            for r in data.get("results", []):
                print(f"  name        : {r['label']} -> {r.get('recommended_authority_uri') or r.get('recommended_wikidata_uri') or 'no link'}")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps({"results": results, "errors": errors}, indent=1, ensure_ascii=False))
        print(f"wrote {out}")
    return 0 if not errors else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lccn")
    ap.add_argument("--ws", default=default_ws(), help="wss:// URL (default: from docs/config.js)")
    ap.add_argument("--tasks", help="comma-separated subset of record,subjects,shelflist,names")
    ap.add_argument("--options", default="{}")
    ap.add_argument("--out")
    a = ap.parse_args()
    if not a.ws:
        print("no WebSocket URL: pass --ws or run deploy/setup_aws.sh", file=sys.stderr)
        return 2
    return asyncio.run(run(a.ws, a.lccn, a.tasks.split(",") if a.tasks else None, json.loads(a.options), a.out))


if __name__ == "__main__":
    sys.exit(main())
