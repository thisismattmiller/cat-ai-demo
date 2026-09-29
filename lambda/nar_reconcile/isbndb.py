"""ISBNdb lookup (https://isbndb.com/apidocs/v2). Key from ISBNDB_API_KEY."""

from __future__ import annotations

import logging
import os
import re

from .http import Http
from .models import IsbndbBook

log = logging.getLogger(__name__)
BASE = "https://api2.isbndb.com"


def _clean_html(s: str | None) -> str | None:
    if not s:
        return None
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s).strip() or None


def fetch_isbndb(http: Http, isbn: str, api_key: str | None = None) -> IsbndbBook | None:
    key = api_key or os.environ.get("ISBNDB_API_KEY")
    if not key:
        log.info("ISBNDB_API_KEY not set; skipping ISBNdb")
        return None
    resp = http.get(f"{BASE}/book/{isbn}", headers={"Authorization": key})
    if resp.status_code != 200:
        log.info("ISBNdb %s -> %s", isbn, resp.status_code)
        return None
    b = resp.json().get("book") or {}
    return IsbndbBook(
        isbn=isbn,
        title=b.get("title"),
        title_long=b.get("title_long"),
        authors=b.get("authors") or [],
        publisher=b.get("publisher"),
        date_published=str(b.get("date_published")) if b.get("date_published") else None,
        synopsis=_clean_html(b.get("synopsis")),
        subjects=b.get("subjects") or [],
        pages=b.get("pages"),
        language=b.get("language"),
    )
