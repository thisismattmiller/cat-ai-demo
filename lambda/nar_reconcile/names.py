"""Deterministic name-form manipulation for building suggest2 queries.

The LLM adds script/transliteration-aware variants; this module covers the
mechanical cases (misplaced Jr., trailing punctuation, direct-order names,
diacritics, initials) so we get hits even if the LLM step is skipped.
"""

from __future__ import annotations

import re
import unicodedata

SUFFIXES = {"jr", "jr.", "sr", "sr.", "ii", "iii", "iv", "v"}
_PUNCT_END = re.compile(r"[\s,;:/]+$")
_ABBREV_END = re.compile(r"(?:\b[A-Za-z]|\b(?:Jr|Sr|St|Mrs|Mr|Dr|Ph\.D))\.$")


def parse_marc_key(marc_key: str | None) -> dict[str, list[str]]:
    """Split a bflc:marcKey like '7001 $aCarroll, Jr., Leon,$eauthor.' into subfields."""
    out: dict[str, list[str]] = {}
    if not marc_key:
        return out
    body = marc_key[4:] if len(marc_key) > 4 else marc_key
    for m in re.finditer(r"\$([a-z0-9])([^$]*)", body):
        out.setdefault(m.group(1), []).append(m.group(2).strip())
    return out


def strip_trailing_punct(s: str) -> str:
    """Strip trailing commas/spaces and a final period, but keep the period of an abbreviation (Jr., L.)."""
    s = _PUNCT_END.sub("", s.strip())
    while s.endswith(".") and not _ABBREV_END.search(s):
        s = _PUNCT_END.sub("", s[:-1])
    return s


def strip_diacritics(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def _is_suffix(token: str) -> bool:
    return token.strip(" ,.").lower() in {x.strip(".") for x in SUFFIXES}


def normalize_suffix_position(name: str) -> tuple[str, str | None]:
    """'Carroll, Jr., Leon' -> ('Carroll, Leon', 'Jr.'); 'Smith, John, III' -> ('Smith, John', 'III')."""
    parts = [p.strip() for p in name.split(",")]
    suffix = None
    kept = []
    for p in parts:
        if p and _is_suffix(p) and suffix is None and len(parts) > 1:
            suffix = p
        else:
            kept.append(p)
    return ", ".join(p for p in kept if p), suffix


def strip_dates(name: str) -> str:
    """Remove a trailing LC date element like ', 1951-' or ', 1879-1966'."""
    return re.sub(r",\s*(?:\d{3,4}|b\.|d\.|ca\.|fl\.|active)[^,]*$", "", name).strip()


def invert_direct_order(name: str) -> str | None:
    """'Leon Carroll' -> 'Carroll, Leon'. Returns None if already inverted or single token."""
    if "," in name:
        return None
    toks = name.split()
    if len(toks) < 2:
        return None
    suffix = None
    if _is_suffix(toks[-1]) and len(toks) > 2:
        suffix = toks[-1]
        toks = toks[:-1]
    inv = f"{toks[-1]}, {' '.join(toks[:-1])}"
    if suffix:
        inv += f", {suffix}"
    return inv


def initials_form(name: str) -> str | None:
    """'Carroll, Leon' -> 'Carroll, L.'"""
    if "," not in name:
        return None
    surname, _, rest = name.partition(",")
    rest = rest.strip()
    if not rest:
        return None
    given = rest.split(",")[0].split()
    if not given:
        return None
    inits = " ".join(f"{g[0]}." for g in given if g and g[0].isalpha())
    return f"{surname.strip()}, {inits}" if inits else None


def deterministic_variants(label: str, marc_key: str | None = None) -> list[str]:
    """Ordered, de-duplicated list of query strings to try against suggest2."""
    out: list[str] = []

    def add(v: str | None):
        if not v:
            return
        v = strip_trailing_punct(v)
        v = re.sub(r"\s+", " ", v)
        if v and v not in out:
            out.append(v)

    sub = parse_marc_key(marc_key)
    base = strip_trailing_punct(sub["a"][0]) if sub.get("a") else strip_trailing_punct(label)

    # 1. base heading with suffix in LC position
    core, suffix = normalize_suffix_position(base)
    if suffix:
        add(f"{core}, {suffix}")
    add(core)
    add(base)

    # 2. with $q fuller form, $c terms, $d dates as LC would build them
    if sub.get("q"):
        add(f"{core} ({sub['q'][0].strip('()')})")
    if sub.get("d"):
        add(f"{core}, {sub['d'][0]}")
    if sub.get("c") and not suffix:
        add(f"{core}, {sub['c'][0]}")

    # 3. without dates, direct-order inversion, diacritics, initials
    core = strip_dates(core)
    add(core)
    add(invert_direct_order(core))
    add(strip_diacritics(core))
    add(initials_form(core))

    return out
