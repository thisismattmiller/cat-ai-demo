"""Local stand-in for the deployed WebSocket: replays recorded events (local_run.py output) to the demo page.

  uv run local_run.py 2025947561 > out/2025947561.events.jsonl
  uv run deploy/mock_ws.py out/2025947561.events.jsonl [--port 8765] [--speed 10]
  open docs/index.html?ws=ws://localhost:8765     (or serve docs/ with `python -m http.server`)

Every "run" message from the page replays the file, sleeping between events according to their `t`
stamps divided by --speed. Useful for working on the page without touching AWS.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import websockets


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("events", nargs="+", help="JSONL event files (one per task, or one for all)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--speed", type=float, default=10.0, help="replay speed factor")
    a = ap.parse_args()
    events = []
    for f in a.events:
        for line in Path(f).read_text().splitlines():
            if line.startswith("{"):
                events.append(json.loads(line))
    tasks = sorted({e["task"] for e in events if "task" in e}, key=["record", "subjects", "shelflist", "names"].index)
    print(f"{len(events)} events for tasks {tasks}", file=sys.stderr)

    async def replay(ws, ev, offset):
        await asyncio.sleep(offset / a.speed)
        await ws.send(json.dumps(ev))

    async def handler(ws):
        async for raw in ws:
            msg = json.loads(raw)
            if msg.get("action") == "ping":
                await ws.send(json.dumps({"type": "pong"}))
                continue
            want = msg.get("tasks") or tasks
            lccn = msg.get("lccn", "")
            await ws.send(json.dumps({"type": "started", "lccn": lccn, "tasks": [t for t in tasks if t in want]}))
            await asyncio.gather(*(replay(ws, {**e, "lccn": lccn}, e.get("t") or 0) for e in events if e.get("task") in want))

    async with websockets.serve(handler, "localhost", a.port):
        print(f"ws://localhost:{a.port}", file=sys.stderr)
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
