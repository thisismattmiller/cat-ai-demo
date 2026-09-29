"""Call-number dates (CSM G 140)."""
from __future__ import annotations

import re

ROMAN = {"M": 1000, "D": 500, "C": 100, "L": 50, "X": 10, "V": 5, "I": 1}


def roman_to_int(s: str) -> int | None:
    s = s.upper().replace(" ", "")
    if not s or any(ch not in ROMAN for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN[ch]
        total += -v if v < prev else v
        prev = max(prev, v)
    return total


def call_number_date(pub_date_264c: str, *, corporate_body: bool = False) -> str | None:
    """Reduce a 264 $c / 260 $c string to the date used in the call number (G 140 sec. 1
    and sec. 5).  Returns e.g. '2012', '1990z', or None if nothing usable.

    Rules: use the first (earliest) year in a range; brackets and '?' are ignored;
    'between X and Y' -> X; a decade/century only known ('197-', 'between 1990 and 1999')
    -> '1970z' unless the main entry is a corporate body, then '1970'."""
    s = pub_date_264c.strip()
    s = s.replace("©", "c").replace("℗", "p")
    # Roman numerals -> Arabic (M M X, MCMXCI-2010, MMI-MMII)
    def _rom(m):
        v = roman_to_int(m.group(0))
        return str(v) if v and 1000 <= v <= 2100 else m.group(0)
    s = re.sub(r"\b[MDCLXVI](?:\s?[MDCLXVI])+\b", _rom, s)
    # decade / century known only:  197-  19--  197-?  [199-]
    m = re.search(r"\b(\d{2,3})[-_?\]\s]*(?:\?)?", s)
    if m and re.search(r"\b\d{3}-|\b\d{2}--", s):
        stem = re.search(r"\b(\d{3})-|\b(\d{2})--", s)
        base = (stem.group(1) or stem.group(2)).ljust(4, "0")
        return base if corporate_body else base[:3] + "0z" if len(stem.group(1) or "") == 3 else (base if corporate_body else base[:2] + "00z")
    # between X and Y  (decade/century spans become 'z' dates)
    m = re.search(r"between\s+(?:\w+\s+\d{1,2},?\s+)?(\d{4})\s+and\s+(?:\w+\s+\d{1,2},?\s+)?(\d{4})", s, re.I)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        if a % 10 == 0 and (b - a) in (9, 99) and not corporate_body:
            return f"{a}z"
        if (b - a) >= 9 and a % 10 == 0 and not corporate_body:
            return f"{a}z"
        return str(a)
    m = re.search(r"not (?:before|after)\s+(?:\w+\s+\d{1,2},?\s+)?(\d{4})", s, re.I)
    if m:
        return m.group(1)
    # "1979 [i.e. 1978]" -> corrected date wins (sec. 5)
    m = re.search(r"\[?\s*i\.?\s*e\.?\s*(\d{4})", s, re.I)
    if m:
        return m.group(1)
    # "1977 (cover 1978)" -> cover date (sec. 5)
    m = re.search(r"cover\s+(\d{4})", s, re.I)
    if m:
        return m.group(1)
    # "1971, c1972" -> later copyright wins under AACR2 (sec. 5); "1981, c1980" -> 1981
    years = [int(y) for y in re.findall(r"(?<![\d])(\d{4})(?![\d])", s)]
    if not years:
        return None
    m = re.match(r"^\s*\[?(\d{4})\]?\s*,\s*c(\d{4})", s)
    if m:
        pub, cop = int(m.group(1)), int(m.group(2))
        return str(max(pub, cop)) if cop > pub else str(pub)
    return str(min(years[0], *years[:2]) if "-" in s and len(years) > 1 else years[0])


def next_work_letter(existing: list[str], base_year: str, *, first: str = "b") -> str:
    """Given call-number dates already in the shelflist for the same Cutter
    (e.g. ['1982', '1982b']), return the next free date+work letter for base_year
    (G 140 2.d/2.e: start work letters at 'b')."""
    used = {d for d in existing if d.startswith(base_year)}
    if base_year not in used:
        return base_year
    for ch in "bcdefghijklmnopqrstuvwxyz":
        if ch < first:
            continue
        if base_year + ch not in used:
            return base_year + ch
    raise ValueError("work letters exhausted")
