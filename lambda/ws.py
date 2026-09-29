"""Sending events to a WebSocket client (API Gateway management API), with a print fallback for local runs.

API Gateway WebSocket frames are capped at 128 KB; anything bigger is shrunk (traces dropped first,
then the payload is replaced by an error) rather than failing the whole task.
"""
from __future__ import annotations

import json
import sys
import threading
import time

MAX_BYTES = 120_000


def _dumps(obj) -> bytes:
    return json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")


def _strip_traces(obj):
    if isinstance(obj, dict):
        return {k: _strip_traces(v) for k, v in obj.items() if k not in ("trace", "raw")}
    if isinstance(obj, list):
        return [_strip_traces(x) for x in obj]
    return obj


def fit(event: dict) -> bytes:
    data = _dumps(event)
    if len(data) <= MAX_BYTES:
        return data
    slim = _strip_traces(event)
    data = _dumps(slim)
    if len(data) <= MAX_BYTES:
        return data
    return _dumps({**{k: v for k, v in event.items() if k in ("type", "task", "t", "lccn")},
                   "type": "error", "error": f"result too large for the WebSocket ({len(data)} bytes)"})


class Connection:
    """post_to_connection wrapper. `gone` flips when the client disconnected."""

    def __init__(self, endpoint: str, connection_id: str):
        import boto3
        self.client = boto3.client("apigatewaymanagementapi", endpoint_url=endpoint)
        self.connection_id = connection_id
        self.gone = False
        self._lock = threading.Lock()

    def send(self, event: dict) -> bool:
        if self.gone:
            return False
        data = fit(event)
        with self._lock:
            try:
                self.client.post_to_connection(ConnectionId=self.connection_id, Data=data)
                return True
            except (self.client.exceptions.GoneException, self.client.exceptions.ForbiddenException):
                self.gone = True          # client disconnected (410), or the connection id is no longer valid (403)
            except Exception as e:  # noqa: BLE001
                print(f"post_to_connection failed: {type(e).__name__}: {e}", file=sys.stderr)
            return False


class PrintConnection:
    """Local stand-in: prints every event as one JSON line."""

    def __init__(self, out=None):
        self.out = out or sys.stdout
        self.gone = False
        self._lock = threading.Lock()

    def send(self, event: dict) -> bool:
        with self._lock:
            self.out.write(fit(event).decode("utf-8") + "\n")
            self.out.flush()
        return True


class Emitter:
    """Task-scoped event emitter: stamps task name, LCCN and elapsed seconds on every event."""

    def __init__(self, conn, task: str, lccn: str):
        self.conn, self.task, self.lccn = conn, task, lccn
        self.t0 = time.time()

    def _send(self, type_: str, **fields) -> None:
        self.conn.send({"type": type_, "task": self.task, "lccn": self.lccn, "t": round(time.time() - self.t0, 1), **fields})

    def progress(self, message: str, **fields) -> None:
        self._send("progress", message=message, **fields)

    def partial(self, data: dict, **fields) -> None:
        self._send("partial", data=data, **fields)

    def result(self, data: dict) -> None:
        self._send("result", data=data)

    def error(self, error: str, **fields) -> None:
        self._send("error", error=error, **fields)
