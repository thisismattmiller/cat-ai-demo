"""Lambda entry point.

WebSocket (API Gateway, routes $connect / $disconnect / $default):
    client sends  {"action": "run", "lccn": "2025947561", "tasks": ["record","subjects","shelflist","names"], "options": {...}}
    the handler validates, replies {"type": "started", ...} on the socket, then runs the tasks concurrently, each
    posting progress / partial / result / error events to the connection as it goes (see ws.py, tasks.py).

    TASK_FANOUT=threads (default): the tasks run as threads of this one invocation, which lasts until the slowest
        finishes (API Gateway stops waiting for the integration after 29 s, but the invocation keeps running and
        keeps posting). Needs only logs + execute-api:ManageConnections.
    TASK_FANOUT=lambda: the function invokes itself asynchronously once per task (event {"task", "lccn", ...}).
        Needs lambda:InvokeFunction on itself, which the deploying user could not grant here.

Direct invocation with {"lccn": "..."} (no connection) runs every task in-process and returns their results,
which is handy for `aws lambda invoke` smoke tests.

Environment: CLAUDE_PAID_API or ANTHROPIC_API_KEY, GOOGLE_AI (Gemini), ISBNDB_API_KEY, SUBJECT_SUGGEST_URL,
             SHELFLISTER_CACHE_DIR=/tmp/cache, SHELFLISTER_LCC_DB (defaults to ./build/lcc.sqlite),
             SHELFLISTER_MODEL / NAR_PROVIDER / NAR_MODEL to change the defaults (gemini-3.8-flash / claude).
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tasks import RUNNERS, TASKS, normalize_lccn  # noqa: E402
from ws import Connection, Emitter, PrintConnection  # noqa: E402

FUNCTION_NAME = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
FANOUT = os.environ.get("TASK_FANOUT", "threads")


def _options(v) -> dict:
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            v = {}
    return v if isinstance(v, dict) else {}


def _tasks(v) -> list[str]:
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",")]
    if not isinstance(v, list) or not v:
        return list(TASKS)
    return [t for t in TASKS if t in v]


# --------------------------------------------------------------------------- websocket dispatcher
def _ws(event: dict, rc: dict, context) -> dict:
    route = rc.get("routeKey")
    cid = rc["connectionId"]
    if route in ("$connect", "$disconnect"):
        return {"statusCode": 200}
    try:
        body = json.loads(event.get("body") or "{}")
    except ValueError:
        body = {}
    endpoint = f"https://{rc['domainName']}/{rc['stage']}"
    conn = Connection(endpoint, cid)
    action = body.get("action", "run")
    if action == "ping":
        conn.send({"type": "pong"})
        return {"statusCode": 200}
    if action != "run":
        conn.send({"type": "error", "error": f"unknown action {action!r}"})
        return {"statusCode": 400}
    lccn = normalize_lccn(str(body.get("lccn", "")))
    if not lccn:
        conn.send({"type": "error", "error": "lccn is required (e.g. 2025947561)"})
        return {"statusCode": 400}
    tasks = _tasks(body.get("tasks"))
    options = _options(body.get("options"))
    run_id = body.get("run_id") or f"{lccn}-{int(time.time())}"

    if FANOUT == "lambda":
        import boto3
        lam = boto3.client("lambda")
        launched = []
        for task in tasks:
            payload = {"task": task, "lccn": lccn, "options": _task_options(options, task),
                       "connectionId": cid, "endpoint": endpoint, "run_id": run_id}
            try:
                lam.invoke(FunctionName=FUNCTION_NAME, InvocationType="Event", Payload=json.dumps(payload).encode())
                launched.append(task)
            except Exception as e:  # noqa: BLE001
                conn.send({"type": "error", "task": task, "lccn": lccn, "error": f"could not start: {type(e).__name__}: {e}"})
        conn.send({"type": "started", "lccn": lccn, "tasks": launched, "run_id": run_id})
        return {"statusCode": 200}

    conn.send({"type": "started", "lccn": lccn, "tasks": tasks, "run_id": run_id})
    run_all(tasks, lccn, options, conn)
    return {"statusCode": 200}


def _task_options(options: dict, task: str) -> dict:
    """Options may be flat ({"hide_class": true}) or per task ({"shelflist": {...}, "names": {...}})."""
    per = options.get(task)
    if isinstance(per, dict):
        return {**{k: v for k, v in options.items() if k not in TASKS}, **per}
    return {k: v for k, v in options.items() if k not in TASKS}


def run_all(tasks: list[str], lccn: str, options: dict, conn) -> dict:
    """Run the tasks concurrently (threads) on one connection; returns {task: result or None}."""
    with ThreadPoolExecutor(max_workers=max(1, len(tasks)), thread_name_prefix="task") as ex:
        futs = {t: ex.submit(run_task, t, lccn, _task_options(options, t), conn) for t in tasks}
        return {t: f.result() for t, f in futs.items()}


# --------------------------------------------------------------------------- worker
def run_task(task: str, lccn: str, options: dict, conn) -> dict | None:
    em = Emitter(conn, task, lccn)
    runner = RUNNERS.get(task)
    if not runner:
        em.error(f"unknown task {task!r}")
        return None
    em.progress("started")
    try:
        data = runner(lccn, options or {}, em)
        em.result(data)
        return data
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        em.error(f"{type(e).__name__}: {e}")
        return None


def _worker(event: dict, context) -> dict:
    conn = Connection(event["endpoint"], event["connectionId"]) if event.get("connectionId") else PrintConnection()
    data = run_task(event["task"], event["lccn"], event.get("options") or {}, conn)
    return {"ok": data is not None, "task": event["task"], "lccn": event["lccn"]}


# --------------------------------------------------------------------------- entry
def handler(event, context=None):
    rc = event.get("requestContext") if isinstance(event, dict) else None
    if rc and rc.get("connectionId"):
        return _ws(event, rc, context)
    if isinstance(event, dict) and event.get("task"):
        return _worker(event, context)
    if isinstance(event, dict) and event.get("lccn"):
        # direct synchronous invocation: run the requested tasks in-process (smoke test)
        lccn = normalize_lccn(str(event["lccn"]))
        if not lccn:
            return {"error": "bad lccn"}
        return run_all(_tasks(event.get("tasks")), lccn, _options(event.get("options")), PrintConnection(sys.stderr))
    return {"statusCode": 400, "body": "expected a WebSocket event, a worker event, or {lccn}"}
