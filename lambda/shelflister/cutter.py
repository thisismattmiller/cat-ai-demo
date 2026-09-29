"""LC Cutter numbers (CSM G 063) and fitting a Cutter between shelflist neighbours.

The table gives a *starting point*. G 063 sec. 2: "In most cases, Cutters must be
adjusted to file an entry correctly and to allow room for later entries."  So the
real algorithm is:

  1. table_cutter(word)  -> the number the table suggests
  2. fit_between(prev, next, suggested) -> a Cutter that files strictly between the
     neighbours that already exist in the shelflist, preferring the suggestion.
"""
from __future__ import annotations

import re
from decimal import Decimal

from .filing import filing_word

VOWELS = "AEIOU"

# second letter -> digit, after an initial vowel
_AFTER_VOWEL = {"b": 2, "d": 3, "l": 4, "m": 4, "n": 5, "p": 6, "r": 7, "s": 8, "t": 8,
                "u": 9, "v": 9, "w": 9, "x": 9, "y": 9}
# second letter -> digit, after initial S
_AFTER_S = {"a": 2, "e": 4, "h": 5, "i": 5, "m": 6, "n": 6, "o": 6, "p": 6, "t": 7,
            "u": 8, "w": 9, "x": 9, "y": 9, "z": 9}   # "ch" handled separately -> 3
# third letter -> digit, after initial Qu
_AFTER_QU = {"a": 3, "e": 4, "i": 5, "o": 6, "r": 7, "t": 8, "y": 9}
# second letter -> digit, after other initial consonants
_AFTER_CONSONANT = {"a": 3, "e": 4, "i": 5, "o": 6, "r": 7, "u": 8, "y": 9}
# expansion digits for any further letter
_EXPANSION = {**{c: 3 for c in "abcd"}, **{c: 4 for c in "efgh"}, **{c: 5 for c in "ijkl"},
              **{c: 6 for c in "mno"}, **{c: 7 for c in "pqrs"}, **{c: 8 for c in "tuv"},
              **{c: 9 for c in "wxyz"}}

ALPHA = "abcdefghijklmnopqrstuvwxyz"


def _digit_with_gap(table: dict[str, int], letter: str) -> tuple[int, str]:
    """Digit for `letter` from `table`.  If the letter is not listed, use the digit of
    the nearest *preceding* listed letter and report 'unlisted' so the caller can push
    the expansion digit high (G 063 '**' examples: Chertok .C48, Clark .C58, Scanlon .S29).
    If the letter shares its digit with earlier letters of a range (e.g. 'm' in l-m),
    report 'range' (G 063 '*' examples: Import .I48, Singer .S57)."""
    if letter in table:
        d = table[letter]
        members = [c for c in ALPHA if table.get(c) == d]
        if len(members) > 1:
            return d, f"range:{members.index(letter)}/{len(members)}"
        return d, "exact"
    # unlisted: walk backwards to the nearest listed letter
    i = ALPHA.index(letter)
    while i >= 0 and ALPHA[i] not in table:
        i -= 1
    if i < 0:
        return 2, "unlisted"          # before every listed letter: lowest digit
    return table[ALPHA[i]], "unlisted"


def table_cutter(word: str, digits: int = 2) -> str:
    """Cutter suggested by the G 063 table for `word` (a filing word: surname, first
    word of a corporate name or title, place name ...).  Returns e.g. 'C36'.
    `digits` is the number of digits wanted (2 is the table's normal result, 3 adds
    one expansion digit)."""
    w = filing_word(word).lower()
    w = re.sub(r"[^a-z0-9]", "", w)
    if not w:
        raise ValueError(f"nothing to Cutter in {word!r}")
    if w[0].isdigit():
        return numeral_cutter(w)
    first = w[0].upper()
    rest = w[1:]
    out = [first]
    pos = 1                       # index into w of the next letter to encode
    if first in VOWELS:
        table = _AFTER_VOWEL
    elif first == "S":
        table = _AFTER_S
    elif first == "Q":
        if rest.startswith("u"):
            table = _AFTER_QU
            pos = 2
        elif rest[:1] > "u":                              # Qv-Qz: after all the Qu names
            out.append("9")
            pos = 2
            table = None
        else:
            # Qa-Qt use 2-29: spread the second letter across 2..29
            second = rest[:1] or "a"
            span = "abcdefghijklmnopqrst"
            idx = span.index(second) if second in span else len(span) - 1
            val = 2 + idx * 27 // len(span)          # 2..28
            out.append(str(val) if val >= 10 else str(val))
            pos = 2
            table = None
    else:
        table = _AFTER_CONSONANT

    if table is not None:
        if pos >= len(w):
            out.append("2")                              # single-letter word
        else:
            letter = w[pos]
            if table is _AFTER_S and w[pos:pos + 2] == "ch":
                d, kind = 3, "exact"
                pos += 1
            else:
                d, kind = _digit_with_gap(table, letter)
            out.append(str(d))
            pos += 1
            if kind == "unlisted":
                out.append("8")                          # push after the listed letter's names
            elif kind.startswith("range:"):
                idx, n = map(int, kind[6:].split("/"))   # spread members of a range across 2-9
                if idx > 0:                              # first member keeps the bare digit
                    out.append(str((3, 7)[idx] if n == 2 else 2 + (idx + 1) * 7 // (n + 1)))
    # expansion digits
    while len("".join(out)) - 1 < digits and pos < len(w):
        out.append(str(_EXPANSION.get(w[pos], 5)))
        pos += 1
    c = "".join(out)
    if len(c) > 2 and c[-1] in "01":
        c = c[:-1] + "2"
    return c


def numeral_cutter(word: str) -> str:
    """G 063 sec. 3 / G 100 sec. 14: entries beginning with a numeral Cutter in .A12-.A19,
    arranged by numeric value.  We map the leading number onto that span coarsely; the
    shelflist fit will refine it."""
    m = re.match(r"\d+", word)
    n = int(m.group()) if m else 0
    # map log-scale: 1..9 -> A13, 10..99 -> A14, 100..999 -> A15, 1000..9999 -> A16, more -> A17
    d = 3 + min(len(str(n)) - 1, 4) if n > 0 else 2
    return f"A1{d}"


# ---------------------------------------------------------------------------
# fitting between neighbours


def cutter_value(c: str) -> Decimal:
    """Decimal value of the digits of a Cutter (C36 -> 0.36)."""
    return Decimal("0." + c[1:]) if len(c) > 1 else Decimal(0)


def _from_value(letter: str, v: Decimal, max_digits: int = 6) -> str:
    s = format(v.normalize(), "f")
    s = s.split(".")[1] if "." in s else "0"
    s = s[:max_digits].rstrip("0") or "2"
    return letter + s


def fit_between(letter: str, prev: str | None, nxt: str | None, suggested: str | None = None) -> str:
    """Pick a Cutter with initial `letter` that sorts strictly after `prev` and strictly
    before `nxt` (both Cutters with the same letter, or None for open ends).
    Prefer `suggested` if it fits; otherwise choose a number in the gap, leaning toward
    the suggestion, and avoid ending in 0 or 1."""
    lo = cutter_value(prev) if prev and prev[0] == letter else Decimal(0)
    hi = cutter_value(nxt) if nxt and nxt[0] == letter else Decimal(1)
    if prev and prev[0] < letter:
        lo = Decimal(0)
    if nxt and nxt[0] > letter:
        hi = Decimal(1)
    if suggested and suggested[0] == letter:
        sv = cutter_value(suggested)
        if lo < sv < hi and suggested[-1] not in "01":
            return suggested
    # find the shortest decimal strictly inside (lo, hi), nudged toward `suggested`
    target = cutter_value(suggested) if suggested and suggested[0] == letter else (lo + hi) / 2
    target = min(max(target, lo), hi)
    for ndigits in range(2, 8):
        step = Decimal(1).scaleb(-ndigits)
        # candidates at this precision inside the open interval
        start = (lo / step).to_integral_value(rounding="ROUND_FLOOR") + 1
        end = (hi / step).to_integral_value(rounding="ROUND_CEILING") - 1
        if start > end:
            continue
        cands = []
        for k in range(int(start), int(end) + 1):
            v = k * step
            if not (lo < v < hi):
                continue
            s = _from_value(letter, v)
            if s[-1] in "01":
                continue
            cands.append((abs(v - target), s))
        if cands:
            cands.sort()
            return cands[0][1]
    raise ValueError(f"no room between {prev} and {nxt}")
