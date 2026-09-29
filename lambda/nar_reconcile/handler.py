"""AWS Lambda entry point.

Event shapes accepted:
  {"lccn": "2025947561", "provider": "claude", "options": {"isbndb": "auto", ...}}
  API Gateway proxy: body (JSON string) with the same keys, or queryStringParameters.lccn

Environment: CLAUDE_PAID_API / ANTHROPIC_API_KEY, GOOGLE_AI, ISBNDB_API_KEY,
             NAR_PROVIDER (default provider), NAR_MODEL (default model).
Note: leave NAR_HTTP_CACHE unset in Lambda (read-only filesystem apart from /tmp).
"""

from __future__ import annotations

import json
import logging
import os

from .http import Http
from .pipeline import Options, reconcile_lccn

log = logging.getLogger()
log.setLevel(logging.INFO)

_http: Http | None = None


def _parse_event(event: dict) -> dict:
    if not isinstance(event, dict):
        return {}
    if "body" in event and event.get("body"):
        body = event["body"]
        try:
            return json.loads(body) if isinstance(body, str) else dict(body)
        except ValueError:
            return {}
    if event.get("queryStringParameters"):
        return dict(event["queryStringParameters"])
    return event


def handler(event, context=None):
    global _http
    params = _parse_event(event)
    lccn = params.get("lccn")
    if not lccn:
        return {"statusCode": 400, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"error": "lccn is required"})}
    provider = params.get("provider") or os.environ.get("NAR_PROVIDER", "claude")
    model = params.get("model") or os.environ.get("NAR_MODEL") or None
    opt_kw = params.get("options") or {}
    if isinstance(opt_kw, str):
        opt_kw = json.loads(opt_kw)
    opts = Options(**{k: v for k, v in opt_kw.items() if k in Options.__dataclass_fields__})

    if _http is None:
        _http = Http(cache_dir=None)
    result = reconcile_lccn(lccn, provider=provider, model=model, http=_http, options=opts)
    status = 200 if not result.error else 502
    return {
        "statusCode": status,
        "headers": {"Content-Type": "application/json"},
        "body": result.model_dump_json(exclude_none=True),
    }
