"""Step 5: mechanical checks on a proposed call number."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import callnum
from .filing import filing_key
from .shelflist_api import Entry


@dataclass
class Validation:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def validate(proposed: str, class_number: str, *, filing_string: str | None = None,
             entry_type: str = "title", neighbours: list[Entry] | None = None,
             expect_date: bool = True, check_filing: bool = True) -> Validation:
    v = Validation(ok=True)
    try:
        cn = callnum.parse(proposed)
    except ValueError as e:
        return Validation(ok=False, errors=[str(e)])
    try:
        want = callnum.parse(class_number)
    except ValueError as e:
        return Validation(ok=False, errors=[f"class number given is not parseable: {e}"])

    # 1. class number preserved
    if (cn.letters, cn.number, cn.decimal) != (want.letters, want.number, want.decimal):
        v.errors.append(f"class number changed: {cn.class_number} vs given {want.class_number}")
    if want.cutters and cn.cutters[:len(want.cutters)] != want.cutters:
        v.errors.append(f"subject Cutter(s) {want.cutters} from the class number were altered: {cn.cutters}")

    # 2. cutter form
    for c in cn.cutters:
        v.errors.extend(callnum.is_valid_cutter(c))
    if len(cn.cutters) > 3:
        v.errors.append(f"{len(cn.cutters)} Cutters; LC call numbers outside G maps have at most 2 (3 only via digits)")

    # 3. date
    if cn.date:
        if not re.fullmatch(r"\d{4}[a-z]?|\d{3}0z", cn.date):
            v.errors.append(f"date {cn.date!r} is not a 4-digit year, optional work letter, or 'z' decade form")
    elif expect_date and not any(x.lower().startswith(("suppl", "index")) for x in cn.extras):
        v.warnings.append("no date in the call number (G 140: monographs shelflisted since 1982 carry the publication date)")

    # 4. neighbours: not a duplicate, and files where the heading says it should
    if neighbours:
        terms = {re.sub(r"\s+", " ", e.term) for e in neighbours}
        if cn.format() in terms:
            v.errors.append(f"{cn.format()} already exists in the shelflist (needs a work letter or a different Cutter)")
        if filing_string and check_filing:
            me = filing_key(filing_string, ignore_article=(entry_type in ("title", "uniform_title")))
            mykey = cn.sort_key()
            same_class = [e for e in neighbours if e.parsed and (e.parsed.letters, e.parsed.number, e.parsed.decimal) == (cn.letters, cn.number, cn.decimal)]
            for e in same_class:
                ek = e.parsed.sort_key()
                if ek == mykey or not e.heading():
                    continue
                # compare only when both have the same number of cutters and share the leading cutters
                if len(e.parsed.cutters) != len(cn.cutters) or e.parsed.cutters[:-1] != cn.cutters[:-1]:
                    continue
                theirs = filing_key(e.heading(), ignore_article=not e.creator)
                if ek < mykey and theirs > me:
                    v.warnings.append(f"files after {e.term} ({e.heading()[:40]}) but heading {filing_string[:40]!r} sorts before it")
                if ek > mykey and theirs < me:
                    v.warnings.append(f"files before {e.term} ({e.heading()[:40]}) but heading {filing_string[:40]!r} sorts after it")
    v.ok = not v.errors
    return v
