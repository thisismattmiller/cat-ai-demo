"""Wikidata fallback: wbsearchentities candidates + wbgetentities enrichment.

Used when no LCNAF match was found. Follows the wikidata-reconcile playbook:
keep alias-match info, enrich with typing/dates/occupations/P244, pace requests.
"""

from __future__ import annotations

import logging
import re
import threading
import time
import unicodedata
from urllib.parse import urlencode

from .http import Http
from .models import WikidataCandidate

log = logging.getLogger(__name__)

API = "https://www.wikidata.org/w/api.php"
_lock = threading.Lock()
_last_call = 0.0
MIN_INTERVAL = 0.34  # ~3 req/s, per Wikimedia etiquette

# properties we surface to the adjudicator
PROPS = {
    "P31": "instance_of",
    "P106": "occupations",
    "P27": "citizenships",
    "P19": "birth_place",
    "P69": "educated_at",
    "P108": "employers",
    "P800": "notable_works",
    "P1412": "languages",
    "P166": "awards",
}
DATE_PROPS = {"P569": "birth_date", "P570": "death_date"}
ID_PROPS = {"P244": "lcnaf_id", "P214": "viaf_id", "P213": "isni", "P496": "orcid"}


def _paced_get(http: Http, url: str):
    global _last_call
    with _lock:
        wait = MIN_INTERVAL - (time.time() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()
    return http.get(url)


def detect_language(s: str) -> str:
    """Pick a wbsearchentities language from the script of the query string."""
    for ch in s:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        if name.startswith("HANGUL"):
            return "ko"
        if name.startswith("CJK") or name.startswith("HIRAGANA") or name.startswith("KATAKANA"):
            return "zh" if name.startswith("CJK") else "ja"
        if name.startswith("ARABIC"):
            return "ar"
        if name.startswith("CYRILLIC"):
            return "ru"
        if name.startswith("HEBREW"):
            return "he"
        if name.startswith("GREEK"):
            return "el"
        if name.startswith("DEVANAGARI"):
            return "hi"
        return "en"
    return "en"


def direct_order(lc_form: str) -> str:
    """'Carroll, Leon, Jr.' -> 'Leon Carroll Jr.'; 'Kim, Young-ha, 1968-' -> 'Young-ha Kim'."""
    s = re.sub(r",\s*(?:\d{3,4}|b\.|d\.|ca\.|fl\.|active)[^,]*$", "", lc_form).strip().rstrip(", ")
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) == 1:
        return parts[0]
    surname, given = parts[0], parts[1]
    rest = [p for p in parts[2:] if p]
    return " ".join([given, surname] + rest)


def search(http: Http, query: str, language: str | None = None, limit: int = 7) -> list[WikidataCandidate]:
    lang = language or detect_language(query)
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": lang,
        "uselang": "en",
        "format": "json",
        "limit": limit,
        "type": "item",
    }
    resp = _paced_get(http, f"{API}?{urlencode(params)}")
    if resp.status_code != 200:
        log.warning("wbsearchentities %r -> %s", query, resp.status_code)
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    out = []
    for r in data.get("search") or []:
        m = r.get("match") or {}
        out.append(
            WikidataCandidate(
                qid=r["id"],
                label=r.get("label") or r["id"],
                description=(r.get("display") or {}).get("description", {}).get("value") or r.get("description"),
                match_type=m.get("type"),
                match_text=m.get("text"),
                found_by=[f"{lang}:{query}"],
            )
        )
    return out


def _time_value(claims: list) -> str | None:
    for c in claims:
        try:
            t = c["mainsnak"]["datavalue"]["value"]["time"]  # +1985-03-23T00:00:00Z
            precision = c["mainsnak"]["datavalue"]["value"].get("precision", 11)
        except (KeyError, TypeError):
            continue
        t = t.lstrip("+")
        return t[:4] if precision <= 9 else t[:7] if precision == 10 else t[:10]
    return None


def _item_ids(claims: list, limit: int = 8) -> list[str]:
    ids = []
    for c in claims:
        try:
            v = c["mainsnak"]["datavalue"]["value"]
            if isinstance(v, dict) and v.get("id"):
                ids.append(v["id"])
        except (KeyError, TypeError):
            continue
        if len(ids) >= limit:
            break
    return ids


def _string_values(claims: list) -> list[str]:
    out = []
    for c in claims:
        try:
            v = c["mainsnak"]["datavalue"]["value"]
            if isinstance(v, str):
                out.append(v)
        except (KeyError, TypeError):
            continue
    return out


def _get_entities(http: Http, ids: list[str], props: str) -> dict:
    result = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        params = {"action": "wbgetentities", "ids": "|".join(chunk), "props": props, "languages": "en", "format": "json"}
        resp = _paced_get(http, f"{API}?{urlencode(params)}")
        if resp.status_code != 200:
            continue
        try:
            result.update(resp.json().get("entities") or {})
        except ValueError:
            continue
    return result


def enrich(http: Http, cands: list[WikidataCandidate]) -> list[WikidataCandidate]:
    """Fill dates, occupations, citizenship, notable works, identifiers, sitelinks for each candidate."""
    if not cands:
        return cands
    ents = _get_entities(http, [c.qid for c in cands], "labels|descriptions|aliases|claims|sitelinks")

    # collect referenced item ids so we can label them in one more batch
    refs: set[str] = set()
    per_cand: dict[str, dict[str, list[str]]] = {}
    for c in cands:
        e = ents.get(c.qid) or {}
        claims = e.get("claims") or {}
        bucket: dict[str, list[str]] = {}
        for pid, field in PROPS.items():
            ids = _item_ids(claims.get(pid) or [])
            bucket[field] = ids
            refs.update(ids)
        per_cand[c.qid] = bucket
    labels = {}
    if refs:
        ref_ents = _get_entities(http, sorted(refs), "labels")
        for qid, e in ref_ents.items():
            labels[qid] = ((e.get("labels") or {}).get("en") or {}).get("value") or qid

    for c in cands:
        e = ents.get(c.qid) or {}
        claims = e.get("claims") or {}
        c.label = ((e.get("labels") or {}).get("en") or {}).get("value") or c.label
        c.description = ((e.get("descriptions") or {}).get("en") or {}).get("value") or c.description
        c.aliases = [a["value"] for a in ((e.get("aliases") or {}).get("en") or [])][:10]
        c.sitelinks = len(e.get("sitelinks") or {})
        c.wikipedia = ((e.get("sitelinks") or {}).get("enwiki") or {}).get("title")
        for pid, field in DATE_PROPS.items():
            setattr(c, field, _time_value(claims.get(pid) or []))
        for pid, field in ID_PROPS.items():
            vals = _string_values(claims.get(pid) or [])
            setattr(c, field, vals[0] if vals else None)
        bucket = per_cand.get(c.qid, {})
        for field, ids in bucket.items():
            setattr(c, field, [labels.get(i, i) for i in ids])
    return cands


# --------------------------------------------------------------------------- #
# Name guard (mechanical sanity check on the LLM's pick; flags, never drops)
# --------------------------------------------------------------------------- #


def _norm(s: str) -> str:
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch)).lower()
    s = re.sub(r",\s*(?:\d{3,4}|b\.|d\.|ca\.|fl\.)[^,]*$", "", s)
    s = re.sub(r"^(the)\s+", "", s)
    return re.sub(r"[^\w\s]", " ", s).split()


def name_guard(surface: str, matched_label: str, aliases: list[str] | None = None) -> str:
    """'strong' if the matched label/aliases share the name tokens with the surface form, else 'weak'."""
    a = set(t for t in _norm(surface) if len(t) > 1)
    for lbl in [matched_label] + list(aliases or []):
        b = set(t for t in _norm(lbl) if len(t) > 1)
        if not a or not b:
            continue
        shared = a & b
        if a == b or (len(shared) >= 2) or (len(shared) >= 1 and min(len(a), len(b)) == 1):
            return "strong"
    return "weak"
