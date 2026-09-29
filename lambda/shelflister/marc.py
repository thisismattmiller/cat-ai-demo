"""Load MARC bibliographic records (from LC by LCCN / bib id, or local files) and
render them as readable text for the LLM."""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import time
from pathlib import Path

import httpx
import pymarc
from pymarc import Record

SRU = "http://lx2.loc.gov:210/LCDB"
CACHE_DIR = Path(os.environ.get("SHELFLISTER_CACHE_DIR", Path(__file__).resolve().parent.parent / ".cache")) / "marc"

# Fields shown to the LLM, in order. Everything else is noise for shelflisting.
SHOW_TAGS = ["010", "020", "041", "050", "051", "082", "100", "110", "111", "130", "240", "245", "246",
             "250", "260", "264", "300", "336", "337", "338", "362", "490", "500", "501", "502", "504",
             "505", "520", "530", "533", "534", "546", "580", "588", "600", "610", "611", "630", "648",
             "650", "651", "655", "700", "710", "711", "730", "740", "775", "776", "780", "785", "800",
             "810", "811", "830"]


def fetch_lc(*, lccn: str | None = None, bibid: str | None = None, cache_dir: Path | None = CACHE_DIR) -> Record:
    if not (lccn or bibid):
        raise ValueError("need lccn or bibid")
    query = f"bath.lccn={lccn.strip()}" if lccn else f"rec.id={bibid.strip()}"
    key = hashlib.sha1(query.encode()).hexdigest()
    cache = cache_dir / f"{key}.xml" if cache_dir else None
    if cache and cache.exists():
        xml = cache.read_bytes()
    else:
        xml = None
        for attempt in range(5):
            try:
                r = httpx.get(SRU, params={"version": "1.1", "operation": "searchRetrieve", "query": query,
                                           "maximumRecords": "1", "recordSchema": "marcxml"}, timeout=60)
                r.raise_for_status()
                xml = r.content
                break
            except (httpx.TransportError, httpx.HTTPStatusError) as e:   # LC's SRU server resets connections under load
                if attempt == 4:
                    raise
                time.sleep(1.5 * (attempt + 1))
        time.sleep(0.4)
        if b"<zs:numberOfRecords>0<" in xml:
            raise LookupError(f"no LC record for {query}")
        if cache:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(xml)
    # strip the SRU envelope: pymarc wants a <collection> or <record>
    m = re.search(rb"<record[ >].*?</record>", xml, re.S)
    if not m:
        raise LookupError(f"no MARC record in response for {query}")
    recs = pymarc.parse_xml_to_array(io.BytesIO(m.group(0)))
    return recs[0]


def load(path: str | Path) -> Record:
    p = Path(path)
    data = p.read_bytes()
    if data.lstrip().startswith(b"<"):
        return pymarc.parse_xml_to_array(io.BytesIO(data))[0]
    if data.lstrip().startswith(b"{"):
        return pymarc.parse_json_to_array(io.BytesIO(data))[0]
    return next(iter(pymarc.MARCReader(data)))


def _field_text(f: pymarc.Field) -> str:
    if f.is_control_field():
        return f.data or ""
    parts = [f"${sf.code} {sf.value}" for sf in f.subfields]
    return " ".join(parts)


def render(rec: Record, *, hide_050: bool = False, tags: list[str] | None = None) -> str:
    """Readable MARC-ish listing: 'TAG ind1ind2 $a ... $b ...' one field per line."""
    tags = tags or SHOW_TAGS
    lines = [f"LDR    {rec.leader}"]
    f008 = rec.get("008")
    if f008:
        lines.append(f"008    {f008.data}")
    for tag in tags:
        for f in rec.get_fields(tag):
            if hide_050 and tag in ("050", "051"):
                continue
            ind = "" if f.is_control_field() else f"{f.indicator1 or ' '}{f.indicator2 or ' '}"
            lines.append(f"{tag} {ind:2} {_field_text(f)}")
    return "\n".join(lines)


def call_numbers_050(rec: Record) -> list[tuple[str, str, str]]:
    """All 050 fields as (ind1, $a, $b) tuples."""
    out = []
    for f in rec.get_fields("050"):
        a = " ".join(f.get_subfields("a"))
        b = " ".join(f.get_subfields("b"))
        out.append((f.indicator1 or " ", a, b))
    return out


def main_entry(rec: Record) -> tuple[str, str] | None:
    """(tag, heading) for the 1XX main entry, or ('245', title) if there is none."""
    for tag in ("100", "110", "111", "130"):
        f = rec.get(tag)
        if f:
            return tag, " ".join(sf.value for sf in f.subfields if sf.code in "abcdnpqt")
    t = rec.get("245")
    if t:
        return "245", " ".join(sf.value for sf in t.subfields if sf.code in "anp")
    return None


def to_dict(rec: Record) -> dict:
    return json.loads(rec.as_json())
