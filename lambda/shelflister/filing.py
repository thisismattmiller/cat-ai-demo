"""LC filing rules (CSM G 100) reduced to a sort key.

`filing_key(heading)` returns a list of word tokens suitable for comparison so that
`filing_key(a) < filing_key(b)` means a files before b in the shelflist. It implements:

  sec. 1  file-as-is, word by word then character by character
  sec. 2  modified letters -> English equivalents (ä->a, ø->o, æ->ae, ð->d, þ->th ...)
  sec. 8  hyphens split words
  sec. 9  initial articles in the nominative case are ignored at the START of a main entry
  sec. 11 initials with punctuation are separate words; run-together acronyms are one word
  sec. 12/13 prefixes and suffixes are separate words when written separately
  sec. 14 numerals (digits) file before letters, by numeric value
  sec. 16 '&' has filing value after space and before digits/letters; other symbols ignored
  sec. 17 apostrophes: elided words are one word

Not implemented (needs human judgment): sec. 3 order of entry types with identical
leading elements, sec. 5 identical titles by date, sec. 15 chronological arrangements.
"""
from __future__ import annotations

import re
import unicodedata

from unidecode import unidecode

# MARC 21 Appendix F initial articles (nominative), a practical subset by language.
ARTICLES = {
    "a", "an", "the",                       # English
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",   # Spanish
    "le", "les", "l'", "une", "des",        # French (la/le/les/un/une shared)
    "der", "die", "das", "ein", "eine", "einen", "dem", "den",     # German
    "il", "gli", "i", "uno", "una", "un'",  # Italian (la/le shared)
    "o", "os", "as", "um", "uma",           # Portuguese (a shared)
    "de", "het", "een",                     # Dutch
    "en", "et", "ett", "den", "det",        # Scandinavian
    "al-", "el-", "ha-", "he-",             # Arabic/Hebrew (prefixed forms handled below)
}
_ELIDED_ARTICLE = re.compile(r"^(l'|d'|dell'|all'|un')", re.I)

_SPECIAL = {"æ": "ae", "Æ": "ae", "œ": "oe", "Œ": "oe", "ð": "d", "Ð": "d", "þ": "th", "Þ": "th",
            "ı": "i", "α": "a", "β": "b", "γ": "g", "ß": "ss", "ø": "o", "Ø": "o", "ł": "l", "Ł": "l"}


def normalize_letters(s: str) -> str:
    """G 100 sec. 2: strip diacritics, map special letters."""
    s = "".join(_SPECIAL.get(ch, ch) for ch in s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return unidecode(s)


def strip_initial_article(heading: str, lang: str | None = None) -> str:
    """G 100 sec. 9: ignore an initial nominative article at the start of a main
    entry. Personal/place names keep their articles (sec. 10) -- callers should not
    pass a 100/651 heading here."""
    h = heading.strip()
    m = _ELIDED_ARTICLE.match(h)
    if m:
        return h[m.end():].lstrip()
    parts = h.split(None, 1)
    if len(parts) == 2 and parts[0].lower().strip(".,:;") in ARTICLES:
        return parts[1]
    return h


def _tokenize(s: str) -> list[str]:
    s = normalize_letters(s).lower()
    s = s.replace("-", " ")                          # sec. 8 hyphenated words are separate
    s = s.replace("'", "")                           # sec. 17 elisions/possessives are one word
    s = re.sub(r"(?<=\w)\.(?=\w)", ". ", s)          # A.B.C. -> A. B. C. (sec. 11a)
    s = re.sub(r"[^\w&\s]", " ", s)                  # sec. 16 ignore other symbols
    return [t for t in s.split() if t]


def _token_key(tok: str) -> tuple:
    """Order: '&' < numerals (by value) < letters (sec. 14, 16)."""
    if tok == "&":
        return (0, 0, "")
    m = re.match(r"^(\d+)(.*)$", tok)
    if m:
        return (1, int(m.group(1)), m.group(2))
    return (2, 0, tok)


def filing_key(heading: str, *, ignore_article: bool = True) -> list[tuple]:
    """Sort key for a heading per G 100. Set ignore_article=False for personal and
    place names (sec. 10)."""
    h = strip_initial_article(heading) if ignore_article else heading
    return [_token_key(t) for t in _tokenize(h)]


def filing_word(heading: str, *, ignore_article: bool = True) -> str:
    """The first filing word of a heading -- the basis of the Cutter (G 053 2.a)."""
    h = strip_initial_article(heading) if ignore_article else heading
    toks = _tokenize(h)
    return toks[0] if toks else ""


def files_before(a: str, b: str, **kw) -> bool:
    return filing_key(a, **kw) < filing_key(b, **kw)
