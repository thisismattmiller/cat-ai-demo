"""Parse, normalize, and sort Library of Congress call numbers.

A call number is:  <class letters><integer>[.<decimal>] [.<cutter1>][<cutter2>] [<date>[<work letter>]] [<extra>...]

Examples:  TX749.5.B43 A142 2006 | PR6045.O72 R66 2026 | HN670.3.Z9C6 | KFV2430 1950 .A243
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

CUTTER_RE = r"[A-Z][0-9]+"
_CLASS_RE = re.compile(
    r"^\s*(?P<letters>[A-Z]{1,3})\s*(?P<num>\d{1,4})(?:\.(?P<dec>\d+))?\s*"
    r"(?P<classdate>\d{4}[a-z]?)?\s*"           # e.g. GV722 1952 .W4 1981 (class arranged by date)
    r"(?P<rest>.*)$"
)
_CUTTER_TOKEN = re.compile(r"\.?([A-Z])(\d+)")
_DATE_TOKEN = re.compile(r"^(\d{4})([a-z])?$|^(\d{3})0z$")


@dataclass
class CallNumber:
    letters: str
    number: str                      # integer part as string (keeps leading form)
    decimal: str | None = None
    class_date: str | None = None    # a date that belongs to the class number (rare)
    cutters: list[str] = field(default_factory=list)   # e.g. ["B43", "A142"] without leading dot
    date: str | None = None          # e.g. "2006", "1998b", "1970z"
    extras: list[str] = field(default_factory=list)    # "vol.3", "Suppl.", "Copy 2", ...
    raw: str = ""

    # -- formatting -----------------------------------------------------
    @property
    def class_number(self) -> str:
        s = f"{self.letters}{self.number}"
        if self.decimal:
            s += f".{self.decimal}"
        if self.class_date:
            s += f" {self.class_date}"
        return s

    def cutter_string(self) -> str:
        if not self.cutters:
            return ""
        return "." + "".join(self.cutters)

    def format(self, style: str = "lc") -> str:
        """style 'lc' = the display form used by the LC shelflist browse
        (space between cutters, e.g. 'TX749.5.B43 A142 2006');
        style 'compact' = no space between cutters ('TX749.5.B43A142 2006')."""
        parts = [self.class_number]
        if self.cutters:
            if style == "lc" and len(self.cutters) == 2:
                parts[0] += "." + self.cutters[0]
                parts.append(self.cutters[1])
            else:
                parts[0] += self.cutter_string()
        if self.date:
            parts.append(self.date)
        parts.extend(self.extras)
        return " ".join(parts)

    def marc_050(self) -> tuple[str, str]:
        """Split into 050 $a and $b per G 070: $b begins with the last Cutter
        (the book number) when there is one; the date goes in $b."""
        if not self.cutters:
            a = self.class_number
            b = " ".join(x for x in [self.date, *self.extras] if x)
            return a, b
        if len(self.cutters) == 1:
            a = self.class_number
            b = "." + self.cutters[0]
        else:
            a = self.class_number + "." + "".join(self.cutters[:-1])
            b = self.cutters[-1]
        tail = " ".join(x for x in [self.date, *self.extras] if x)
        if tail:
            b = f"{b} {tail}"
        return a, b

    def __str__(self) -> str:
        return self.format()

    # -- sorting --------------------------------------------------------
    def sort_key(self) -> tuple:
        """A key that orders call numbers the way the shelflist does.
        Comparable to (but not byte-identical with) id.loc.gov's `sort` field."""
        dec = self.decimal or ""
        cutters = tuple((c[0], c[1:]) for c in self.cutters)  # digits compare as decimal fractions -> string compare works
        d = self.date or ""
        return (self.letters, int(self.number), dec, self.class_date or "", cutters, d, tuple(self.extras))


def parse(s: str) -> CallNumber:
    """Parse a call number in any of the common spellings:
    'TX749.5.B43 A142 2006', 'TX749.5 .B43 W35 2023', 'TX749.5.B43A142 2006',
    '$a TX749.5.B43 $b A142 2006', 'HN670.3.Z9C6', 'PR5551 1968b'."""
    raw = s
    s = re.sub(r"\$[ab]\s*", " ", s).strip()
    s = re.sub(r"\s+", " ", s)
    m = _CLASS_RE.match(s)
    if not m:
        raise ValueError(f"not an LC call number: {raw!r}")
    cn = CallNumber(letters=m["letters"], number=m["num"], decimal=m["dec"], raw=raw)
    rest = m["rest"].strip()
    if m["classdate"]:
        # Only treat as class date if cutters follow; 'PR5551 1968b' is a date-only book number.
        if re.match(r"^\.?[A-Z]\d", rest):
            cn.class_date = m["classdate"]
        else:
            rest = (m["classdate"] + " " + rest).strip()
    tokens = rest.split(" ") if rest else []
    i = 0
    # cutters: one or more tokens made only of cutter parts, may be glued (Z9C6) or dotted
    while i < len(tokens) and re.fullmatch(r"\.?(?:[A-Z]\d+)+", tokens[i]):
        cn.cutters.extend(a + b for a, b in _CUTTER_TOKEN.findall(tokens[i]))
        i += 1
    if i < len(tokens) and _DATE_TOKEN.match(tokens[i]):
        cn.date = tokens[i]
        i += 1
    cn.extras = tokens[i:]
    return cn


def try_parse(s: str | None) -> CallNumber | None:
    if not s:
        return None
    try:
        return parse(s)
    except ValueError:
        return None


def is_valid_cutter(c: str) -> list[str]:
    """Return a list of problems with a Cutter (empty = fine). Per G 063 a Cutter
    must not end in 0 or 1 (except the .A1x numeral range used for numerals)."""
    problems = []
    if not re.fullmatch(r"[A-Z]\d+", c):
        problems.append(f"Cutter {c!r} is not a letter followed by digits")
        return problems
    if c[-1] in "01" and not re.fullmatch(r"A1\d*", c):
        problems.append(f"Cutter {c!r} ends in {c[-1]} (G 063: do not end a Cutter with 0 or 1)")
    return problems
