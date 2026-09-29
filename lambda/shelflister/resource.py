"""The input: book metadata as given (title, creator, summary, contents ...).
No MARC is built; MARC records (for evaluation) are reduced to the same shape."""
from __future__ import annotations

from dataclasses import dataclass, field

from pymarc import Record


@dataclass
class Resource:
    title: str
    creator: str | None = None          # authorized form if known, surname first: "Woolf, Virginia, 1882-1941"
    creator_type: str | None = None     # personal | corporate | conference | None (unknown)
    summary: str | None = None
    contents: str | None = None         # table of contents / chapter list
    date: str | None = None             # publication date as printed
    publisher: str | None = None
    edition: str | None = None
    language: str | None = None
    subjects: list[str] = field(default_factory=list)   # LC subject headings if known
    genre_form: list[str] = field(default_factory=list)
    uniform_title: str | None = None
    series: str | None = None
    notes: list[str] = field(default_factory=list)
    identifiers: dict = field(default_factory=dict)      # isbn, lccn, ...
    extra: dict = field(default_factory=dict)

    def text(self) -> str:
        """Labelled lines, in the order a cataloger reads them."""
        rows = [
            ("Title", self.title), ("Uniform title", self.uniform_title), ("Creator", self.creator),
            ("Creator type", self.creator_type), ("Edition", self.edition), ("Date", self.date),
            ("Publisher", self.publisher), ("Language", self.language), ("Series", self.series),
            ("Subjects", "; ".join(self.subjects) if self.subjects else None),
            ("Genre/form", "; ".join(self.genre_form) if self.genre_form else None),
            ("Summary", self.summary), ("Contents", self.contents),
        ]
        rows += [(f"Note", n) for n in self.notes]
        rows += [(k.upper(), str(v)) for k, v in self.identifiers.items()]
        rows += [(k, str(v)) for k, v in self.extra.items()]
        return "\n".join(f"{k}: {v}" for k, v in rows if v)

    def description_for_search(self) -> dict:
        return {"title": self.title, "creator": self.creator, "summary": self.summary, "content": self.contents}

    @classmethod
    def from_marc(cls, rec: Record) -> "Resource":
        def sub(tag, codes):
            f = rec.get(tag)
            return " ".join(sf.value for sf in f.subfields if sf.code in codes).strip() if f else None
        def subs(tag, codes, joiner=" -- "):
            out = []
            for f in rec.get_fields(tag):
                parts = [sf.value.strip(" .") for sf in f.subfields if sf.code in codes]
                if parts:
                    out.append(joiner.join(parts))
            return out
        creator, ctype = None, None
        for tag, t in (("100", "personal"), ("110", "corporate"), ("111", "conference")):
            v = sub(tag, "abcdnq")
            if v:
                creator, ctype = v.strip(" ,."), t
                break
        pub = rec.get("264") or rec.get("260")
        r = cls(
            # G 100 sec. 5: for filing the title runs to the first period or slash, so the
            # subtitle ($b) is part of it
            title=(sub("245", "abnp") or "").rstrip(" /:."),
            creator=creator, creator_type=ctype,
            summary=" ".join(subs("520", "a", " ")) or None,
            contents=" ".join(subs("505", "agtr", " ")) or None,
            date=" ".join(pub.get_subfields("c")).strip(" .") if pub else None,
            publisher=" ".join(pub.get_subfields("b")).rstrip(" ,") if pub else None,
            edition=sub("250", "ab"), language=None,
            subjects=[s for tag in ("600", "610", "611", "630", "650", "651") for s in subs(tag, "abcdvxyz")],
            genre_form=subs("655", "a"),
            uniform_title=sub("240", "a") or sub("130", "a"),
            series=sub("490", "av") or sub("830", "av"),
            notes=[n for n in subs("500", "a", " ")][:6],
        )
        f008 = rec.get("008")
        if f008 and len(f008.data) >= 38:
            r.language = f008.data[35:38].strip() or None
            if f008.data[22] == "j":
                r.extra["audience"] = "juvenile (008/22 = j)"
        f041 = rec.get("041")
        if f041 and f041.get_subfields("h"):
            r.extra["translated_from"] = " ".join(f041.get_subfields("h"))
        for tag, key in (("020", "isbn"), ("010", "lccn")):
            v = sub(tag, "a")
            if v:
                r.identifiers[key] = v.strip()
        return r
