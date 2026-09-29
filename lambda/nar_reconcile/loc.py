"""id.loc.gov client: LCCN resolution, CBD RDF parsing, suggest2, MADS authority parsing,
and the works/relationships endpoints."""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from urllib.parse import quote, urlencode

from .http import Http
from .models import (
    BookRecord,
    Candidate,
    Citation,
    Contributor,
    Publication,
    SuggestHit,
)

log = logging.getLogger(__name__)

ID = "https://id.loc.gov"
NS = {
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "bf": "http://id.loc.gov/ontologies/bibframe/",
    "bflc": "http://id.loc.gov/ontologies/bflc/",
    "madsrdf": "http://www.loc.gov/mads/rdf/v1#",
    "ri": "http://id.loc.gov/ontologies/RecordInfo#",
    "xml": "http://www.w3.org/XML/1998/namespace",
}
RDF_ABOUT = f"{{{NS['rdf']}}}about"
RDF_RESOURCE = f"{{{NS['rdf']}}}resource"
XML_LANG = f"{{{NS['xml']}}}lang"

# bf:Agent rdf:type -> suggest2 rdftype filter
AGENT_TYPE_TO_RDFTYPE = {
    "Person": "PersonalName",
    "Organization": "CorporateName",
    "Jurisdiction": "CorporateName",
    "Meeting": "ConferenceName",
    "Family": "FamilyName",
}
NAME_RDFTYPES = {"PersonalName", "CorporateName", "ConferenceName", "FamilyName", "Geographic"}


def _q(tag: str) -> str:
    prefix, local = tag.split(":")
    return f"{{{NS[prefix]}}}{local}"


def _local(uri_or_tag: str | None) -> str | None:
    if not uri_or_tag:
        return None
    return re.split(r"[/#}]", uri_or_tag)[-1]


def _text(el: ET.Element | None) -> str | None:
    if el is None or el.text is None:
        return None
    t = re.sub(r"\s+", " ", el.text).strip()
    return t or None


def _find_text(el: ET.Element, path: str) -> str | None:
    return _text(el.find(path, NS))


def _labels(el: ET.Element, path: str) -> list[str]:
    out = []
    for e in el.findall(path, NS):
        t = _text(e)
        if t and t not in out:
            out.append(t)
    return out


def _pref_label(el: ET.Element, tag: str = "madsrdf:authoritativeLabel") -> str | None:
    """Prefer the label without an xml:lang (the romanized LC form)."""
    els = el.findall(tag, NS)
    for e in els:
        if XML_LANG not in e.attrib and _text(e):
            return _text(e)
    for e in els:
        if _text(e):
            return _text(e)
    return None


# --------------------------------------------------------------------------- #
# LCCN -> instance -> CBD RDF
# --------------------------------------------------------------------------- #


def normalize_lccn(lccn: str) -> str:
    return re.sub(r"\s+", "", lccn).strip()


def resolve_lccn(http: Http, lccn: str) -> str | None:
    """Follow the identifier redirect to the instance URI (https://id.loc.gov/resources/instances/...)."""
    url = f"{ID}/resources/instances/identifier/{quote(normalize_lccn(lccn))}"
    resp = http.get(url, allow_redirects=False)
    if resp.status_code in (301, 302, 303, 307, 308):
        loc = resp.headers.get("location")
        if loc:
            return loc if loc.startswith("http") else ID + loc
    if resp.status_code == 200 and resp.headers.get("x-uri"):
        return resp.headers["x-uri"].replace("http://", "https://")
    log.warning("LCCN %s did not resolve (%s)", lccn, resp.status_code)
    return None


def fetch_cbd(http: Http, instance_uri: str) -> str:
    uri = instance_uri.replace("http://id.loc.gov", ID)
    resp = http.get(f"{uri}.cbd.rdf")
    if resp.status_code != 200:
        raise RuntimeError(f"CBD fetch failed: {uri}.cbd.rdf -> {resp.status_code}")
    return resp.text


def _parse_contribution(c: ET.Element) -> Contributor | None:
    contribution = c.find("bf:Contribution", NS)
    if contribution is None:
        return None
    primary = any(
        _local(t.get(RDF_RESOURCE)) == "PrimaryContribution"
        for t in contribution.findall("rdf:type", NS)
    )
    agent_el = contribution.find("bf:agent", NS)
    if agent_el is None:
        return None
    agent = agent_el.find("bf:Agent", NS)
    if agent is None:
        # rare: <bf:agent rdf:resource="..."/> with no nested description
        uri = agent_el.get(RDF_RESOURCE)
        if not uri:
            return None
        return Contributor(label=uri, agent_uri=uri, primary=primary)

    types = [_local(t.get(RDF_RESOURCE)) for t in agent.findall("rdf:type", NS)]
    agent_type = next((t for t in types if t in AGENT_TYPE_TO_RDFTYPE), types[0] if types else "Person")
    roles, role_uris = [], []
    for r in contribution.findall("bf:role/bf:Role", NS):
        lbl = _find_text(r, "rdfs:label") or _find_text(r, "bf:code")
        if lbl:
            roles.append(lbl)
        if r.get(RDF_ABOUT):
            role_uris.append(r.get(RDF_ABOUT))
    label = _find_text(agent, "rdfs:label") or _find_text(agent, "bflc:name00MatchKey") or ""
    return Contributor(
        label=label,
        marc_key=_find_text(agent, "bflc:marcKey"),
        agent_uri=agent.get(RDF_ABOUT),
        agent_type=agent_type or "Person",
        roles=roles,
        role_uris=role_uris,
        primary=primary,
    )


def parse_cbd(xml_text: str, lccn: str | None, instance_uri: str) -> BookRecord:
    root = ET.fromstring(xml_text)
    inst = next(
        (e for e in root.findall("bf:Instance", NS) if e.get(RDF_ABOUT, "").endswith(_local(instance_uri) or "")),
        root.find("bf:Instance", NS),
    )
    if inst is None:
        raise RuntimeError("No bf:Instance in CBD")
    work_uri = None
    ref = inst.find("bf:instanceOf", NS)
    if ref is not None:
        work_uri = ref.get(RDF_RESOURCE)
    work = next(
        (w for w in root.findall("bf:Work", NS) if work_uri and w.get(RDF_ABOUT) == work_uri),
        root.find("bf:Work", NS),
    )

    rec = BookRecord(lccn=lccn or "", instance_uri=instance_uri, work_uri=work_uri)

    # ----- titles
    t = inst.find("bf:title/bf:Title", NS)
    if t is not None:
        rec.title = _find_text(t, "bf:mainTitle")
        rec.subtitle = _find_text(t, "bf:subtitle")
        rec.part_title = _labels(t, "bf:partName") + _labels(t, "bf:partNumber")
    for vt in inst.findall("bf:title/bf:VariantTitle", NS) + inst.findall("bf:title/bf:ParallelTitle", NS):
        lbl = _find_text(vt, "rdfs:label") or _find_text(vt, "bf:mainTitle")
        if lbl:
            rec.variant_titles.append(lbl)
    rec.responsibility_statement = _find_text(inst, "bf:responsibilityStatement")

    # ----- publication
    pub = Publication(statement=_find_text(inst, "bf:publicationStatement"))
    for pa in inst.findall("bf:provisionActivity/bf:ProvisionActivity", NS):
        kinds = [_local(x.get(RDF_RESOURCE)) for x in pa.findall("rdf:type", NS)]
        if kinds and "Publication" not in kinds:
            continue
        pub.place = pub.place or _find_text(pa, "bflc:simplePlace") or _find_text(pa, "bf:place/bf:Place/rdfs:label")
        pub.publisher = pub.publisher or _find_text(pa, "bflc:simpleAgent") or _find_text(pa, "bf:agent/bf:Agent/rdfs:label")
        pub.date = pub.date or _find_text(pa, "bflc:simpleDate") or _find_text(pa, "bf:date")
    rec.publication = pub

    # ----- identifiers
    for lc in inst.findall("bf:identifiedBy/bf:Lccn", NS):
        val = _find_text(lc, "rdf:value")
        if val and not lccn:
            rec.lccn = normalize_lccn(val)
            break
    for isbn in inst.findall("bf:identifiedBy/bf:Isbn", NS):
        status_el = isbn.find("bf:status/bf:Status", NS)
        status = _local(status_el.get(RDF_ABOUT)) if status_el is not None else None
        val = _find_text(isbn, "rdf:value")
        if val and status != "cancinv":
            rec.isbns.append(re.sub(r"[^0-9Xx]", "", val))

    # ----- notes / series / toc on the instance
    for n in inst.findall("bf:note/bf:Note", NS):
        kinds = [_local(x.get(RDF_RESOURCE)) for x in n.findall("rdf:type", NS)]
        lbl = _find_text(n, "rdfs:label")
        if lbl and "internal" not in kinds:
            rec.notes.append(lbl)
    rec.series += _labels(inst, "bf:seriesStatement")
    rec.series += _labels(inst, "bf:hasSeries/bf:Instance/bf:title/bf:Title/bf:mainTitle")

    # ----- work-level
    if work is not None:
        rec.languages = _labels(work, "bf:language/bf:Language/rdfs:label")
        for s in work.findall("bf:subject", NS):
            for child in s:
                lbl = _pref_label(child) or _find_text(child, "rdfs:label")
                if lbl:
                    rec.subjects.append(lbl)
        rec.genre_forms = _labels(work, "bf:genreForm/bf:GenreForm/rdfs:label") + _labels(
            work, "bf:genreForm/bf:GenreForm/madsrdf:authoritativeLabel"
        )
        for cl in work.findall("bf:classification/*", NS):
            portion = _find_text(cl, "bf:classificationPortion")
            item = _find_text(cl, "bf:itemPortion")
            if portion:
                rec.classification.append(f"{portion} {item or ''}".strip())
        rec.summary = _labels(work, "bf:summary/bf:Summary/rdfs:label")
        rec.table_of_contents = _labels(work, "bf:tableOfContents/bf:TableOfContents/rdfs:label")
        for n in work.findall("bf:note/bf:Note", NS):
            kinds = [_local(x.get(RDF_RESOURCE)) for x in n.findall("rdf:type", NS)]
            lbl = _find_text(n, "rdfs:label")
            if lbl and "internal" not in kinds:
                rec.notes.append(lbl)
        for c in work.findall("bf:contribution", NS):
            contrib = _parse_contribution(c)
            if contrib:
                rec.contributors.append(contrib)
    return rec


def fetch_book(http: Http, lccn: str) -> BookRecord | None:
    instance_uri = resolve_lccn(http, lccn)
    if not instance_uri:
        return None
    xml_text = fetch_cbd(http, instance_uri)
    return parse_cbd(xml_text, normalize_lccn(lccn), instance_uri)


# --------------------------------------------------------------------------- #
# suggest2
# --------------------------------------------------------------------------- #


def suggest2(
    http: Http,
    q: str,
    *,
    keyword: bool = False,
    rdftype: str | None = None,
    count: int = 20,
) -> list[SuggestHit]:
    params = {"q": q, "count": count}
    if keyword:
        params["searchtype"] = "keyword"
    if rdftype:
        params["rdftype"] = rdftype
    url = f"{ID}/authorities/names/suggest2/?{urlencode(params)}"
    resp = http.get(url)
    if resp.status_code != 200:
        log.warning("suggest2 %s -> %s", url, resp.status_code)
        return []
    try:
        data = resp.json()
    except ValueError:
        return []
    tag = f"{'keyword' if keyword else 'left'}:{q}"
    hits = []
    for h in data.get("hits") or []:
        more = h.get("more") or {}
        rdftypes = more.get("rdftypes") or []
        if "NameTitle" in rdftypes or "Title" in rdftypes:
            continue
        if rdftypes and not (set(rdftypes) & NAME_RDFTYPES):
            continue
        hits.append(
            SuggestHit(
                uri=h["uri"],
                token=h.get("token") or _local(h["uri"]),
                label=h.get("aLabel") or h.get("suggestLabel") or "",
                matched_variant=h.get("vLabel") or None,
                rdftypes=rdftypes,
                birth_dates=more.get("birthdates") or [],
                death_dates=more.get("deathdates") or [],
                occupations=list(dict.fromkeys(more.get("occupations") or [])),
                activity_fields=list(dict.fromkeys(more.get("activityfields") or [])),
                locales=more.get("locales") or [],
                nonlatin_labels=more.get("nonlatinLabels") or [],
                sources=more.get("sources") or [],
                marc_keys=more.get("marcKeys") or [],
                found_by=[tag],
            )
        )
    return hits


# --------------------------------------------------------------------------- #
# Authority (MADS/RDF) dossier
# --------------------------------------------------------------------------- #


def _iter_labels_under(el: ET.Element, path: str) -> list[str]:
    """Collect preferred labels of the objects found at `path` (e.g. madsrdf:occupation/*/madsrdf:occupation/*)."""
    out = []
    for obj in el.findall(path, NS):
        lbl = _pref_label(obj) or _find_text(obj, "rdfs:label") or _find_text(obj, "madsrdf:variantLabel")
        if not lbl and obj.get(RDF_RESOURCE):
            lbl = obj.get(RDF_RESOURCE)
        if lbl and lbl not in out:
            out.append(lbl)
    return out


def parse_authority(xml_text: str, uri: str) -> Candidate:
    root = ET.fromstring(xml_text)
    http_uri = uri.replace("https://", "http://")
    main = None
    for el in root:
        if el.get(RDF_ABOUT) in (uri, http_uri):
            main = el
            break
    if main is None:
        main = root[0]

    token = _local(uri) or uri
    cand = Candidate(uri=http_uri, token=token, label=_pref_label(main) or token)
    cand.rdf_types = [_local(main.tag) or ""] + [
        _local(t.get(RDF_RESOURCE)) or "" for t in main.findall("rdf:type", NS)
    ]
    cand.rdf_types = [t for t in dict.fromkeys(cand.rdf_types) if t and t in NAME_RDFTYPES | {"NameTitle", "Title", "Topic", "DeprecatedAuthority"}]

    # variants (see-from references)
    for v in main.findall("madsrdf:hasVariant/*", NS):
        lbl = _pref_label(v, "madsrdf:variantLabel")
        if lbl and lbl not in cand.variants:
            cand.variants.append(lbl)
    # non-Latin authoritative labels count as variants too
    for e in main.findall("madsrdf:authoritativeLabel", NS):
        if XML_LANG in e.attrib and _text(e) and _text(e) not in cand.variants:
            cand.variants.append(_text(e))

    for n in main.findall("madsrdf:note", NS):
        t = _text(n)
        if t:
            cand.notes.append(t)

    for s in main.findall("madsrdf:hasSource/madsrdf:Source", NS):
        cand.citations.append(
            Citation(
                status=_find_text(s, "madsrdf:citationStatus"),
                source=_find_text(s, "madsrdf:citationSource"),
                note=_find_text(s, "madsrdf:citationNote"),
            )
        )

    for rel in (
        "madsrdf:hasRelatedAuthority/*",
        "madsrdf:hasEarlierEstablishedForm/*",
        "madsrdf:hasLaterEstablishedForm/*",
        "madsrdf:see/*",
        "madsrdf:hasReciprocalAuthority/*",
        "madsrdf:hasBroaderAuthority/*",
    ):
        for lbl in _iter_labels_under(main, rel):
            kind = rel.split("/")[0].split(":")[1]
            cand.related_authorities.append(f"{lbl} [{kind}]")

    for ext in ("madsrdf:hasExactExternalAuthority", "madsrdf:hasCloseExternalAuthority"):
        for e in main.findall(ext, NS):
            r = e.get(RDF_RESOURCE)
            if r is None and len(e):
                r = e[0].get(RDF_ABOUT)
                lbl = _pref_label(e[0])
                if lbl:
                    r = f"{r} ({lbl})"
            if r:
                cand.external_authorities.append(r)

    # RWO: the real-world-object block carries the biographical data
    rwo = main.find("madsrdf:identifiesRWO/madsrdf:RWO", NS)
    if rwo is not None:
        cand.birth_date = _find_text(rwo, "madsrdf:birthDate")
        cand.death_date = _find_text(rwo, "madsrdf:deathDate")
        cand.gender = (
            _iter_labels_under(rwo, "madsrdf:gender/*") or [_find_text(rwo, "madsrdf:gender")]
        )[0]
        cand.fuller_name = _find_text(rwo, "madsrdf:fullerName/*/madsrdf:elementList/*/madsrdf:elementValue") or (
            _iter_labels_under(rwo, "madsrdf:fullerName/*") or [None]
        )[0]
        cand.languages = _iter_labels_under(rwo, "madsrdf:associatedLanguage/*")
        cand.locales = _iter_labels_under(rwo, "madsrdf:associatedLocale/*")
        cand.fields_of_activity = _iter_labels_under(
            rwo, "madsrdf:fieldOfActivity/*/madsrdf:fieldOfActivity/*"
        ) + _iter_labels_under(rwo, "madsrdf:fieldOfActivity/madsrdf:Topic")
        cand.occupations = _iter_labels_under(rwo, "madsrdf:occupation/*/madsrdf:occupation/*") + _iter_labels_under(
            rwo, "madsrdf:occupation/madsrdf:Occupation"
        )
        for aff in rwo.findall("madsrdf:hasAffiliation/madsrdf:Affiliation", NS):
            org = _iter_labels_under(aff, "madsrdf:organization/*")
            start = _find_text(aff, "madsrdf:affiliationStart")
            end = _find_text(aff, "madsrdf:affiliationEnd")
            for o in org:
                span = f" ({start or ''}-{end or ''})" if (start or end) else ""
                cand.affiliations.append(o + span)
        for d in rwo.findall("madsrdf:entityDescriptor/*", NS):
            lbl = _pref_label(d) or _find_text(d, "rdfs:label")
            if lbl:
                cand.notes.append(f"entity descriptor: {lbl}")
        for bp in _iter_labels_under(rwo, "madsrdf:birthPlace/*"):
            cand.notes.append(f"birth place: {bp}")
        for dp in _iter_labels_under(rwo, "madsrdf:deathPlace/*"):
            cand.notes.append(f"death place: {dp}")

    dates = []
    for ri in main.findall("madsrdf:adminMetadata/ri:RecordInfo", NS):
        d = _find_text(ri, "ri:recordChangeDate")
        status = _find_text(ri, "ri:recordStatus")
        if d:
            dates.append((d, status))
    if dates:
        dates.sort()
        cand.record_created = dates[0][0][:10]
        cand.record_changed = dates[-1][0][:10]
    return cand


def fetch_authority(http: Http, uri: str) -> Candidate | None:
    https_uri = uri.replace("http://id.loc.gov", ID)
    resp = http.get(f"{https_uri}.rdf")
    if resp.status_code != 200:
        log.warning("authority %s -> %s", uri, resp.status_code)
        return None
    try:
        return parse_authority(resp.text, https_uri)
    except ET.ParseError as e:
        log.warning("authority parse failed %s: %s", uri, e)
        return None


def parse_relationships(data: dict) -> tuple[list[str], int]:
    total = int((data.get("summary") or {}).get("total") or 0)
    results = data.get("results") or []
    if isinstance(results, dict):  # id.loc.gov returns a bare object when there is exactly one result
        results = [results]
    labels = [r.get("label") for r in results if isinstance(r, dict) and r.get("label")]
    return labels, total


def fetch_relationships(http: Http, uri: str, kind: str, page: int = 0) -> tuple[list[str], int]:
    """kind: 'contributorto' | 'subjectof'. Returns (labels, total)."""
    http_uri = uri.replace("https://", "http://")
    url = f"{ID}/resources/works/relationships/{kind}/?label={quote(http_uri, safe=':/')}&page={page}"
    resp = http.get(url)
    if resp.status_code != 200:
        return [], 0
    try:
        data = resp.json()
    except ValueError:
        return [], 0
    return parse_relationships(data)


def uri_token(uri: str | None) -> str:
    """'http://id.loc.gov/authorities/names/no2023135128' -> 'no2023135128' (tolerates https, trailing slash, .rdf)."""
    if not uri:
        return ""
    return re.sub(r"\.(rdf|json|html)$", "", uri.strip().rstrip("/").split("/")[-1]).lower()


def build_candidate(http: Http, hit: SuggestHit, max_related: int = 15) -> Candidate | None:
    cand = fetch_authority(http, hit.uri)
    if cand is None:
        return None
    cand.matched_variant = hit.matched_variant
    cand.found_by = hit.found_by
    contrib, ctotal = fetch_relationships(http, hit.uri, "contributorto")
    subj, stotal = fetch_relationships(http, hit.uri, "subjectof")
    cand.contributor_to, cand.contributor_to_total = contrib[:max_related], ctotal
    cand.subject_of, cand.subject_of_total = subj[:max_related], stotal
    return cand
