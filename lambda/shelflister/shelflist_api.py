"""Client for the id.loc.gov shelflist browse endpoint.

  https://id.loc.gov/controllers/xqapi-shelflist.xqy?bq=<callnum>&browse-order=ascending&browse=class&count=201&mime=json

Returns ~count entries centred on the query call number (about half before, half
after). Each entry: term (call number as displayed), creator, uniformtitle, title,
pubdate, subject, bibid, sort (LC's normalized sort key).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

import httpx

from . import callnum

BASE = "https://id.loc.gov/controllers/xqapi-shelflist.xqy"
import os
CACHE_DIR = Path(os.environ.get("SHELFLISTER_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache")) / "shelflist"


@dataclass
class Entry:
    term: str
    creator: str
    uniformtitle: str
    title: str
    pubdate: str
    subject: str
    bibid: str
    sort: str

    @property
    def parsed(self) -> callnum.CallNumber | None:
        return callnum.try_parse(self.term)

    def heading(self) -> str:
        """The string the book number is (usually) based on: 1XX, else 245."""
        return self.creator or self.uniformtitle or self.title

    def short(self) -> str:
        h = self.heading()
        t = self.title if self.creator else ""
        return f"{self.term:<32} {h[:40]:<40} {t[:40]}"


class ShelflistClient:
    def __init__(self, cache_dir: Path | None = CACHE_DIR, timeout: float = 60.0, exclude_bibids: set[str] | None = None):
        self.exclude_bibids = set(exclude_bibids or ())   # hide these records (the item being evaluated)
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self.http = httpx.Client(timeout=timeout, headers={"User-Agent": "shelflister/0.1"})

    def browse(self, call_number: str, count: int = 201) -> list[Entry]:
        q = re.sub(r"\s+", " ", call_number.strip())
        key = hashlib.sha1(f"{q}|{count}".encode()).hexdigest()
        cache = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache and cache.exists():
            data = json.loads(cache.read_text())
        else:
            url = f"{BASE}?bq={quote(q)}&browse-order=ascending&browse=class&count={count}&mime=json"
            r = self.http.get(url)
            r.raise_for_status()
            data = r.json()
            if cache:
                cache.write_text(json.dumps(data))
        out = []
        for d in data:
            if d.get("creator") == "Class would appear here.":
                continue
            if d.get("bibid") in self.exclude_bibids:
                continue
            out.append(Entry(term=d.get("term", ""), creator=d.get("creator", ""),
                             uniformtitle=d.get("uniformtitle", ""), title=d.get("title", ""),
                             pubdate=d.get("pubdate", ""), subject=d.get("subject", ""),
                             bibid=d.get("bibid", ""), sort=d.get("sort", "")))
        return out

    def in_class(self, class_number: str, count: int = 201) -> list[Entry]:
        """Entries whose class number (letters+number+decimal) equals `class_number`.
        Fetches a window around the class and filters; widens once if the window is full
        on one side."""
        target = callnum.parse(class_number)
        want = (target.letters, target.number, target.decimal)

        def same_class(p: callnum.CallNumber) -> bool:
            # letters/number/decimal equal, and any Cutters in the query are a prefix of the
            # entry's Cutters (so 'PR6045.O72' returns only the .O72 entries)
            if (p.letters, p.number, p.decimal) != want:
                return False
            if target.class_date and p.class_date != target.class_date:
                return False
            return p.cutters[: len(target.cutters)] == target.cutters

        entries = self.browse(class_number, count)
        keep = [e for e in entries if e.parsed and same_class(e.parsed)]
        # the window is centred on the query; if the class continues past either edge, extend
        seen = {e.term for e in keep}
        for _ in range(8):
            if not (entries and keep and entries[-1] is keep[-1]):
                break
            entries = self.browse(keep[-1].term, count)
            added = 0
            for e in entries:
                if e.parsed and same_class(e.parsed) and e.term not in seen:
                    keep.append(e); seen.add(e.term); added += 1
            if not added:
                break
        for _ in range(8):
            if not (entries and keep and entries[0] is keep[0]):
                break
            entries = self.browse(keep[0].term, count)
            front = [e for e in entries if e.parsed and same_class(e.parsed) and e.term not in seen]
            if not front:
                break
            keep = front + keep; seen.update(e.term for e in front)
        # drop copy statements
        return [e for e in keep if not re.search(r"\bCopy \d+$", e.term)]

    def neighbours(self, call_number: str, n: int = 8) -> tuple[list[Entry], list[Entry]]:
        """Entries immediately before and after where `call_number` would file."""
        entries = self.browse(call_number)
        target = callnum.parse(call_number).sort_key()
        before = [e for e in entries if e.parsed and e.parsed.sort_key() < target]
        after = [e for e in entries if e.parsed and e.parsed.sort_key() >= target]
        return before[-n:], after[:n]
