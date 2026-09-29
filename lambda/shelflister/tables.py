"""Lookup tables from the manual: G 300 regions/countries, G 302 states/provinces,
G 320 biography table, G 330 artists tables."""
from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path

from .filing import normalize_letters

DATA = Path(__file__).resolve().parent / "data"


@lru_cache(maxsize=None)
def _load(name: str):
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def _norm(s: str) -> str:
    return normalize_letters(s).lower().strip()


def _lookup(rows: list[dict], place: str, n: int = 5) -> list[dict]:
    q = _norm(place)
    exact = [r for r in rows if _norm(r["name"]) == q]
    if exact:
        return exact
    starts = [r for r in rows if _norm(r["name"]).startswith(q) or q.startswith(_norm(r["name"]))]
    if starts:
        return starts[:n]
    names = [_norm(r["name"]) for r in rows]
    close = difflib.get_close_matches(q, names, n=n, cutoff=0.6)
    return [r for r in rows if _norm(r["name"]) in close]


def regions_countries(place: str) -> list[dict]:
    """G 300 Regions and Countries Table lookup. Returns matching rows; 'see' rows point
    to the entry that carries the Cutter."""
    rows = _load("regions_countries.json")
    hits = _lookup(rows, place)
    out = []
    for r in hits:
        out.append(r)
        if r.get("see"):
            out.extend(x for x in rows if _norm(x["name"]) == _norm(r["see"]))
    return out


def states_provinces(place: str) -> list[dict]:
    """G 302 U.S. states and Canadian provinces lookup."""
    rows = _load("states_provinces.json")
    hits = _lookup(rows, place)
    out = []
    for r in hits:
        out.append(r)
        if r.get("see"):
            out.extend(x for x in rows if _norm(x["name"]) == _norm(r["see"]))
    return out


def biography_table() -> list[dict]:
    return _load("biography_table.json")


def artists_tables() -> dict:
    return _load("artists_table.json")


def format_rows(rows: list[dict]) -> str:
    if not rows:
        return "(no match)"
    lines = []
    for r in rows:
        c = r.get("cutter") or ("see " + r["see"] if r.get("see") else "?")
        extra = f"  [{r['notes']}]" if r.get("notes") else ""
        kind = f" ({r['kind']})" if r.get("kind") else ""
        lines.append(f"{r['name']}{kind}: {c}{extra}")
    return "\n".join(lines)
