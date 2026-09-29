"""Shared HTTP session: user agent, retries, and an optional on-disk GET cache.

The cache is for local development only (be kind to id.loc.gov while iterating).
Set NAR_HTTP_CACHE=/some/dir to enable it; leave unset in Lambda.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

log = logging.getLogger(__name__)

USER_AGENT = os.environ.get(
    "NAR_USER_AGENT", "nar-auto-reconcile/0.1 (+https://github.com/thisismattmiller)"
)
DEFAULT_TIMEOUT = float(os.environ.get("NAR_HTTP_TIMEOUT", "30"))


class CachedResponse:
    """Minimal stand-in for requests.Response when served from disk cache."""

    def __init__(self, status_code: int, headers: dict, content: bytes):
        self.status_code = status_code
        self.headers = headers
        self.content = content

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self):
        return json.loads(self.text)

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 400


def _session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = USER_AGENT
    retry = Retry(
        total=4,
        backoff_factor=0.8,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=16, pool_maxsize=16)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


class Http:
    def __init__(self, cache_dir: str | None = None):
        cache_dir = cache_dir if cache_dir is not None else os.environ.get("NAR_HTTP_CACHE")
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self.session = _session()

    def _cache_path(self, url: str, headers: dict | None) -> Path:
        key = url + "|" + json.dumps(headers or {}, sort_keys=True)
        return self.cache / (hashlib.sha256(key.encode()).hexdigest() + ".json")

    def get(
        self,
        url: str,
        *,
        headers: dict | None = None,
        allow_redirects: bool = True,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> requests.Response | CachedResponse:
        cache_path = self._cache_path(url, headers) if self.cache else None
        if cache_path and cache_path.exists():
            data = json.loads(cache_path.read_text())
            return CachedResponse(data["status"], data["headers"], bytes.fromhex(data["body"]))

        t0 = time.time()
        resp = self.session.get(
            url, headers=headers, allow_redirects=allow_redirects, timeout=timeout
        )
        log.debug("GET %s -> %s (%.2fs)", url, resp.status_code, time.time() - t0)

        if cache_path and resp.status_code in (200, 302, 404):
            cache_path.write_text(
                json.dumps(
                    {
                        "url": url,
                        "status": resp.status_code,
                        "headers": {k.lower(): v for k, v in resp.headers.items()},
                        "body": resp.content.hex(),
                    }
                )
            )
        return resp
