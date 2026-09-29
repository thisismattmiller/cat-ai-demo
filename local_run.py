"""Run the Lambda's tasks locally, concurrently, printing the same events the WebSocket would carry.

  uv run local_run.py 2025947561                       # all four tasks
  uv run local_run.py 2025947561 --tasks record,subjects
  uv run local_run.py 2025947561 --tasks shelflist --options '{"hide_class": true}'
  uv run local_run.py 2025947561 --out out/2025947561.json

Keys come from the environment (CLAUDE_PAID_API, GOOGLE_AI, ISBNDB_API_KEY); SUBJECT_SUGGEST_URL overrides
the subject-suggest lambda. HTTP caches land in lambda/.cache (shelflister) and .http-cache (nar) unless
SHELFLISTER_CACHE_DIR / NAR_HTTP_CACHE say otherwise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "lambda"))
os.environ.setdefault("NAR_HTTP_CACHE", str(ROOT / ".http-cache"))

from app import run_all  # noqa: E402
from tasks import TASKS, normalize_lccn  # noqa: E402
from ws import PrintConnection  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lccn")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--options", default="{}", help="JSON options passed to every task")
    ap.add_argument("--out", help="write the results here")
    a = ap.parse_args()
    lccn = normalize_lccn(a.lccn)
    if not lccn:
        print(f"not an LCCN: {a.lccn!r}", file=sys.stderr)
        return 2
    tasks = [t for t in a.tasks.split(",") if t in TASKS]
    options = json.loads(a.options)
    conn = PrintConnection()
    t0 = time.time()
    results = run_all(tasks, lccn, options, conn)
    print(f"# done in {time.time() - t0:.1f}s: " + ", ".join(f"{t}={'ok' if r else 'FAILED'}" for t, r in results.items()),
          file=sys.stderr)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1, ensure_ascii=False, default=str))
        print(f"# wrote {a.out}", file=sys.stderr)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
