"""LCCN -> id.loc.gov BIBFRAME instance (CBD RDF/XML) -> Resource, with optional ISBNdb enrichment.

  https://id.loc.gov/resources/instances/identifier/<LCCN>   302 -> .../instances/<id>
  https://id.loc.gov/resources/instances/<id>.cbd.rdf         the Instance plus its Work

The Work's PrimaryContribution is the main entry ("creator"). The Instance's ClassificationLcc, when
present, is LC's own call number (classificationPortion = 050 $a, itemPortion = 050 $b). The instance
id is also what the shelflist browse reports as `bibid` for that record, so it can be excluded.

ISBNdb (https://api2.isbndb.com/book/<isbn>, header Authorization: <ISBNDB_API_KEY>) fills in a
synopsis, publisher, date and BISAC-style subjects when LC's record is thin (CIP-level records).
"""
from __future__ import annotations

import html
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET

import httpx

from .resource import Resource

CACHE_DIR = Path(os.environ.get("SHELFLISTER_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache")) / "idloc"
ID_BASE = "https://id.loc.gov/resources/instances"
ISBNDB = "https://api2.isbndb.com/book/"

NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "bf": "http://id.loc.gov/ontologies/bibframe/",
    "bflc": "http://id.loc.gov/ontologies/bflc/",
    "madsrdf": "http://www.loc.gov/mads/rdf/v1#",
}
RDF_ABOUT = "{%s}about" % NS["rdf"]
RDF_RESOURCE = "{%s}resource" % NS["rdf"]
BF = "{%s}" % NS["bf"]
AGENT_TYPES = {"Person": "personal", "Family": "personal", "Organization": "corporate", "Jurisdiction": "corporate",
               "Meeting": "conference"}


@dataclass
class IdLocRecord:
    instance_id: str
    instance_url: str
    lccn: str | None = None
    isbns: list[str] = field(default_factory=list)
    lcc: list[tuple[str, str]] = field(default_factory=list)     # (classificationPortion, itemPortion) as LC assigned
    contributors: list[str] = field(default_factory=list)        # everyone, "Label (role)"
    isbndb: dict | None = None                                   # the ISBNdb book record, if fetched
    isbndb_added: list[str] = field(default_factory=list)        # which Resource fields ISBNdb filled

    @property
    def bibid(self) -> str:
        return self.instance_id


# ------------------------------------------------------------------ fetching
def resolve_lccn(lccn: str, timeout: float = 30.0) -> str:
    """LCCN (or an instance id / URL) -> instance URL."""
    v = lccn.strip().replace(" ", "")
    if v.startswith("http"):
        return v.split("?")[0].removesuffix(".cbd.rdf").removesuffix(".rdf")
    if re.fullmatch(r"in\d+", v):
        return f"{ID_BASE}/{v}"          # a FOLIO instance id, not an LCCN
    r = httpx.get(f"{ID_BASE}/identifier/{v}", timeout=timeout, follow_redirects=False)
    if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
        return r.headers["location"]
    if r.status_code == 404:
        raise LookupError(f"id.loc.gov has no instance for LCCN {lccn}")
    r.raise_for_status()
    raise LookupError(f"unexpected response {r.status_code} resolving LCCN {lccn}")


def instance_url(bibid: str) -> str:
    """LC bib id (old numeric or new in000...) -> instance URL."""
    return f"{ID_BASE}/{bibid.strip()}"


def fetch_cbd(instance_url: str, cache_dir: Path | None = CACHE_DIR, timeout: float = 60.0) -> bytes:
    iid = instance_url.rstrip("/").rsplit("/", 1)[-1]
    cache = cache_dir / f"{iid}.cbd.rdf" if cache_dir else None
    if cache and cache.exists():
        return cache.read_bytes()
    r = httpx.get(f"{instance_url}.cbd.rdf", timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    if cache:
        _write_atomic(cache, r.content)
    return r.content


def _write_atomic(path: Path, data: bytes) -> None:
    """Concurrent tasks fetch the same record: write to a temp file and rename, so a reader never sees a partial file."""
    import threading
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}-{threading.get_ident()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def isbndb_book(isbn: str, cache_dir: Path | None = CACHE_DIR, timeout: float = 30.0) -> dict | None:
    key = os.environ.get("ISBNDB_API_KEY")
    if not key:
        return None
    isbn = re.sub(r"[^0-9Xx]", "", isbn)
    cache = cache_dir / f"isbndb_{isbn}.json" if cache_dir else None
    if cache and cache.exists():
        return json.loads(cache.read_text()).get("book")
    r = httpx.get(ISBNDB + isbn, headers={"Authorization": key}, timeout=timeout)
    if r.status_code == 404:
        data = {}
    else:
        r.raise_for_status()
        data = r.json()
    if cache:
        _write_atomic(cache, json.dumps(data, ensure_ascii=False).encode("utf-8"))
    return data.get("book")


# ------------------------------------------------------------------ parsing
def _label(el) -> str | None:
    """rdfs:label or madsrdf:authoritativeLabel or rdf:value of an element, whitespace-normalized."""
    if el is None:
        return None
    for path in ("rdfs:label", "madsrdf:authoritativeLabel", "rdf:value"):
        t = el.find(path, NS)
        if t is not None and t.text and t.text.strip():
            return " ".join(t.text.split())
    return None


def _types(el) -> set[str]:
    out = set()
    for t in el.findall("rdf:type", NS):
        out.add(t.get(RDF_RESOURCE, "").rsplit("/", 1)[-1].rsplit("#", 1)[-1])
    return out


def _title(el) -> str | None:
    """bf:title/bf:Title -> 'Main title : subtitle' (subtitle counts for filing, G 100 sec. 5)."""
    for t in el.findall("bf:title/bf:Title", NS):
        if "VariantTitle" in _types(t) or "ParallelTitle" in _types(t):
            continue
        main = " ".join((t.findtext("bf:mainTitle", "", NS) or "").split())
        sub = " ".join((t.findtext("bf:subtitle", "", NS) or "").split())
        part = " ".join((t.findtext("bf:partName", "", NS) or "").split())
        if main:
            s = main + (f" : {sub}" if sub and sub not in main else "") + (f". {part}" if part else "")
            return s.rstrip(" /:.")
    return None


def parse_cbd(xml: bytes) -> tuple[Resource, IdLocRecord]:
    root = ET.fromstring(xml)
    inst = root.find("bf:Instance", NS)
    if inst is None:
        raise ValueError("no bf:Instance in the CBD")
    inst_url = inst.get(RDF_ABOUT, "")
    rec = IdLocRecord(instance_id=inst_url.rsplit("/", 1)[-1], instance_url=inst_url)
    io_el = inst.find("bf:instanceOf", NS)
    work_url = io_el.get(RDF_RESOURCE) if io_el is not None else None
    work = next((w for w in root.findall("bf:Work", NS) if w.get(RDF_ABOUT) == work_url), root.find("bf:Work", NS))

    # --- identifiers, classification
    for ident in inst.findall("bf:identifiedBy/*", NS):
        tag = ident.tag.replace(BF, "")
        val = (ident.findtext("rdf:value", "", NS) or "").strip()
        status = _label(ident.find("bf:status/bf:Status", NS)) or ""
        if tag == "Lccn" and val:
            rec.lccn = val
        elif tag == "Isbn" and val and "cancel" not in status.lower() and "invalid" not in status.lower():
            rec.isbns.append(val)
    for holder in ([work] if work is not None else []) + [inst]:
        for c in holder.findall("bf:classification/bf:ClassificationLcc", NS):
            a = (c.findtext("bf:classificationPortion", "", NS) or "").strip()
            b = (c.findtext("bf:itemPortion", "", NS) or "").strip()
            assigner = (c.find("bf:assigner/*", NS).get(RDF_ABOUT, "") if c.find("bf:assigner/*", NS) is not None else "")
            if a and (a, b) not in rec.lcc:
                rec.lcc.insert(0, (a, b)) if assigner.endswith("/dlc") else rec.lcc.append((a, b))

    # --- instance-level description
    r = Resource(title=_title(inst) or (_title(work) if work is not None else None) or "")
    r.edition = (inst.findtext("bf:editionStatement", None, NS) or "").strip() or None
    pa = inst.find("bf:provisionActivity/bf:ProvisionActivity", NS)
    if pa is not None:
        r.date = (pa.findtext("bflc:simpleDate", None, NS) or pa.findtext("bf:date", None, NS) or "").strip() or None
        r.publisher = (pa.findtext("bflc:simpleAgent", None, NS) or _label(pa.find("bf:agent/bf:Agent", NS)) or "").strip() or None
    if not r.date:
        stmt = inst.findtext("bf:publicationStatement", "", NS) or ""
        m = re.search(r"\b(1[5-9]\d\d|20\d\d)\b", stmt)
        r.date = m.group(1) if m else None
    r.series = (inst.findtext("bf:seriesStatement", None, NS) or "").strip() or None
    resp = (inst.findtext("bf:responsibilityStatement", None, NS) or "").strip()
    if resp:
        r.notes.append(f"Statement of responsibility: {resp}")
    for n in inst.findall("bf:note/bf:Note", NS):
        if any(t.get(RDF_RESOURCE, "").endswith("/internal") for t in n.findall("rdf:type", NS)):
            continue
        lab = _label(n)
        if lab and not re.match(r"^\d{3}\s", lab):
            r.notes.append(lab)
    for k in ("isbn", "lccn"):
        v = rec.isbns[0] if k == "isbn" and rec.isbns else rec.lccn if k == "lccn" else None
        if v:
            r.identifiers[k] = v
    r.identifiers["id.loc.gov"] = inst_url

    # --- work-level description
    if work is not None:
        for contrib in work.findall("bf:contribution/bf:Contribution", NS):
            agent = contrib.find("bf:agent/bf:Agent", NS)
            lab = _label(agent)
            if not lab:
                continue
            role = _label(contrib.find("bf:role/bf:Role", NS)) or ""
            atypes = _types(agent) if agent is not None else set()
            ctype = next((v for k, v in AGENT_TYPES.items() if k in atypes), None)
            rec.contributors.append(f"{lab}" + (f" ({role})" if role else ""))
            if "PrimaryContribution" in _types(contrib) and r.creator is None:
                r.creator, r.creator_type = lab.rstrip(" ,."), ctype
                if role:
                    r.extra["creator_role"] = role
        others = [c for c in rec.contributors if r.creator is None or not c.startswith(r.creator)]
        if others:
            r.notes.append("Other contributors: " + "; ".join(others))
        summ = [_label(s) for s in work.findall("bf:summary/bf:Summary", NS)]
        r.summary = " ".join(s for s in summ if s) or None
        toc = [_label(t) for t in work.findall("bf:tableOfContents/bf:TableOfContents", NS)]
        r.contents = " ".join(t for t in toc if t) or None
        seen = set()
        for s in work.findall("bf:subject/*", NS):
            lab = _label(s)
            if lab and lab not in seen:
                seen.add(lab)
                r.subjects.append(lab)
        for g in work.findall("bf:genreForm/*", NS):
            lab = _label(g)
            if lab and lab not in r.genre_form:
                r.genre_form.append(lab)
        langs = [l.findtext("bf:code", "", NS) or l.get(RDF_ABOUT, "").rsplit("/", 1)[-1] for l in work.findall("bf:language/bf:Language", NS)]
        r.language = ", ".join(x for x in langs if x) or None
        wt = _types(work) - {"Work"}
        if wt:
            r.extra["work_types"] = ", ".join(sorted(wt))
        aud = [_label(a) for a in work.findall("bf:intendedAudience/*", NS)]
        if any(aud):
            r.extra["audience"] = "; ".join(a for a in aud if a)
        od = (work.findtext("bf:originDate", None, NS) or "").strip()
        if od:
            r.extra["origin_date"] = od
        for rel in work.findall("bf:relation/bf:Relation", NS) + work.findall("bflc:relationship/bflc:Relationship", NS):
            rl = (_label(rel.find("bf:relation/bf:Relation", NS)) or _label(rel.find("bflc:relation/bflc:Relation", NS)) or "").lower()
            target = rel.find("bf:associatedResource/*", NS)
            if "translation" in rl and target is not None:
                r.extra["translation_of"] = _title(target) or _label(target) or "(see record)"
        if r.series is None:
            ser = work.find(".//bf:Series", NS)
            if ser is not None:
                r.series = _title(ser) or _label(ser)
        # a uniform / preferred title on the work, when it differs from the instance title
        for t in work.findall("bf:title/bf:Title", NS):
            if "PreferredTitle" in _types(t) or "UniformTitle" in _types(t):
                r.uniform_title = " ".join((t.findtext("bf:mainTitle", "", NS) or "").split()) or None
    r.notes = r.notes[:8]
    return r, rec


# ------------------------------------------------------------------ ISBNdb enrichment
def _strip_html(s: str) -> str:
    s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(html.unescape(s).split())


def enrich_from_isbndb(r: Resource, book: dict, rec: IdLocRecord) -> None:
    """Fill what LC's record lacks. LC data always wins where both exist."""
    added = []
    syn = _strip_html(book.get("synopsis") or book.get("overview") or "")
    if syn and (not r.summary or len(syn) > len(r.summary) + 80):
        if r.summary:
            r.notes.append("Publisher synopsis (ISBNdb): " + syn[:1500])
        else:
            r.summary = syn[:3000]
        added.append("summary")
    if not r.date and book.get("date_published"):
        m = re.search(r"\d{4}", str(book["date_published"]))
        if m:
            r.date, _ = m.group(0), added.append("date")
    if not r.publisher and book.get("publisher"):
        r.publisher, _ = book["publisher"], added.append("publisher")
    if not r.creator and book.get("authors"):
        r.notes.append("Authors (ISBNdb, as printed): " + "; ".join(book["authors"]))
        added.append("authors note")
    subj = [s for s in book.get("subjects") or [] if isinstance(s, str)]
    if subj:
        r.extra["isbndb_subjects"] = "; ".join(subj[:12])
        added.append("bookseller subjects")
    if book.get("pages") and "pages" not in r.extra:
        r.extra["pages"] = str(book["pages"])
    if book.get("edition") and not r.edition:
        r.edition, _ = str(book["edition"]), added.append("edition")
    r.notes = r.notes[:8]
    rec.isbndb, rec.isbndb_added = book, added


# ------------------------------------------------------------------ entry point
def from_lccn(lccn: str | None = None, *, bibid: str | None = None, use_isbndb: bool = True,
              cache_dir: Path | None = CACHE_DIR) -> tuple[Resource, IdLocRecord]:
    url = instance_url(bibid) if bibid else resolve_lccn(lccn)
    r, rec = parse_cbd(fetch_cbd(url, cache_dir))
    if use_isbndb:
        for isbn in rec.isbns:
            try:
                book = isbndb_book(isbn, cache_dir)
            except httpx.HTTPError as e:
                r.notes.append(f"(ISBNdb lookup failed for {isbn}: {e})")
                break
            if book:
                enrich_from_isbndb(r, book, rec)
                break
    return r, rec
