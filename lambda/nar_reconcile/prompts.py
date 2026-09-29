"""System prompts and context renderers for the three LLM steps."""

from __future__ import annotations

from .models import BookRecord, Candidate, Contributor, IsbndbBook, SuggestHit

# --------------------------------------------------------------------------- #
# Renderers: compact, labelled plain text (cheaper and clearer than JSON dumps)
# --------------------------------------------------------------------------- #


def _line(k: str, v) -> str:
    if v is None or v == "" or v == []:
        return ""
    if isinstance(v, list):
        v = "; ".join(str(x) for x in v)
    return f"{k}: {v}\n"


def render_book(book: BookRecord, isbndb: IsbndbBook | None = None) -> str:
    title = book.title or ""
    if book.subtitle:
        title += f" : {book.subtitle}"
    s = "## The book being cataloged\n"
    s += _line("LCCN", book.lccn)
    s += _line("Title", title)
    s += _line("Part", book.part_title)
    s += _line("Variant titles", book.variant_titles)
    s += _line("Statement of responsibility", book.responsibility_statement)
    p = book.publication
    s += _line("Published", p.statement or " ".join(x for x in [p.place, p.publisher, p.date] if x))
    s += _line("ISBNs", book.isbns)
    s += _line("Language", book.languages)
    s += _line("Subjects", book.subjects)
    s += _line("Genre/form", book.genre_forms)
    s += _line("Classification", book.classification)
    s += _line("Series", book.series)
    s += _line("Summary", book.summary)
    s += _line("Contents", book.table_of_contents)
    s += _line("Notes", book.notes)
    s += "Contributors on this record:\n"
    for c in book.contributors:
        status = f"LINKED -> {c.authority_uri}" if c.reconciled else "UNLINKED"
        roles = ", ".join(c.roles) or "role unspecified"
        s += f"  - {c.label} ({c.agent_type}; {roles}{'; primary' if c.primary else ''}) [{status}]\n"
    if isbndb:
        s += "\n## Additional metadata from ISBNdb\n"
        s += _line("Title", isbndb.title_long or isbndb.title)
        s += _line("Authors as printed", isbndb.authors)
        s += _line("Publisher", isbndb.publisher)
        s += _line("Date published", isbndb.date_published)
        s += _line("Subjects", isbndb.subjects)
        s += _line("Synopsis", isbndb.synopsis)
    return s


def render_contributor(c: Contributor, variants: list[str] | None = None) -> str:
    s = "## The contributor to reconcile\n"
    s += _line("Name as it appears in the record", c.label)
    s += _line("MARC key", c.marc_key)
    s += _line("Agent type", c.agent_type)
    s += _line("Role(s)", c.roles)
    s += _line("Primary contribution", "yes" if c.primary else "no")
    if variants:
        s += _line("Mechanical variants already being searched", variants)
    return s


def render_hit(h: SuggestHit) -> str:
    s = f"- {h.uri}\n"
    s += f"  heading: {h.label}\n"
    if h.matched_variant:
        s += f"  (matched via variant: {h.matched_variant})\n"
    if h.nonlatin_labels:
        s += f"  non-Latin forms: {'; '.join(h.nonlatin_labels)}\n"
    if h.birth_dates or h.death_dates:
        s += f"  dates: b. {' '.join(h.birth_dates) or '?'} d. {' '.join(h.death_dates) or '?'}\n"
    if h.occupations:
        s += f"  occupations: {'; '.join(h.occupations)}\n"
    if h.activity_fields:
        s += f"  fields: {'; '.join(h.activity_fields)}\n"
    if h.locales:
        s += f"  locales: {'; '.join(h.locales)}\n"
    for src in h.sources[:3]:
        s += f"  source: {src[:300]}\n"
    return s


def render_candidate(cand: Candidate, idx: int) -> str:
    s = f"### Candidate {idx}: {cand.uri}\n"
    s += _line("Authorized heading", cand.label)
    s += _line("Type", cand.rdf_types)
    s += _line("Variant / see-from forms", cand.variants)
    s += _line("Fuller name", cand.fuller_name)
    s += _line("Birth date", cand.birth_date)
    s += _line("Death date", cand.death_date)
    s += _line("Gender", cand.gender)
    s += _line("Languages", cand.languages)
    s += _line("Associated places", cand.locales)
    s += _line("Fields of activity", cand.fields_of_activity)
    s += _line("Occupations", cand.occupations)
    s += _line("Affiliations", cand.affiliations)
    s += _line("Related authorities", cand.related_authorities)
    s += _line("Notes", cand.notes)
    if cand.citations:
        s += "Sources cited in the authority record:\n"
        for c in cand.citations:
            src = " ".join(x for x in [c.status, c.source] if x)
            s += f"  - {src}: {c.note or ''}\n"
    s += _line("External identifiers", cand.external_authorities)
    s += _line("Record created / last changed", f"{cand.record_created} / {cand.record_changed}")
    if cand.contributor_to_total:
        s += f"Works this heading contributed to ({cand.contributor_to_total} total; first {len(cand.contributor_to)} shown):\n"
        for w in cand.contributor_to:
            s += f"  - {w}\n"
    else:
        s += "Works this heading contributed to: none found in LC catalog\n"
    if cand.subject_of:
        s += f"Works about this heading ({cand.subject_of_total} total; first {len(cand.subject_of)} shown):\n"
        for w in cand.subject_of:
            s += f"  - {w}\n"
    elif cand.subject_of_total:
        s += f"Works about this heading: {cand.subject_of_total} in LC catalog (titles unavailable)\n"
    if cand.found_by:
        s += _line("Surfaced by queries", cand.found_by)
    return s + "\n"


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #

SEARCH_PLAN_SYSTEM = """You are an expert authority-control cataloger at a research library. You help find the
Library of Congress Name Authority File (LCNAF) record for a contributor named on a book record.

Your job in this step: produce search strings for the id.loc.gov `suggest2` service, which searches
LCNAF headings and their variant (see-from) forms.

Two search modes exist:
- `left`: left-anchored browse against the heading string. Works only if the query is a prefix of the
  authorized or a variant heading in LC form, i.e. "Surname, Forename" (with any qualifiers after:
  "Carroll, Leon, Jr." / "Kim, Young-ha, 1968-" / "Smith, John, 1950-"). A prefix like "Carroll, Leon" matches
  "Carroll, Leon, Jr.". Punctuation and order matter; a misplaced suffix ("Carroll, Jr., Leon") gets zero hits.
- `keyword`: keyword search over the same headings (a trailing wildcard is added automatically). Order does not
  matter; useful for direct-order names, initials, and when the surname is uncertain.

Think about how LC would establish this name (RDA / NACO practice):
- Western names: "Surname, Forename Middle", suffixes after the forenames ("..., Jr."), dates appended if used
  to break conflicts. Try forms with and without middle names / initials, and full forms of initials if you can
  infer them.
- Compound and particle surnames (de, van, von, del, al-, ben, Mac/Mc, Saint/St.): LC entry element varies by
  language of the person; try both "Van Damme, Jean" and "Damme, Jean van" style entries.
- Spanish/Portuguese double surnames: entry under first surname ("García Márquez, Gabriel").
- Hungarian, Chinese, Japanese, Korean, Vietnamese: surname is already first in native order. LCNAF romanizes with
  ALA-LC tables: Korean uses McCune-Reischauer with breves and apostrophes ("Kim, Yŏng-ha"; "Pak, Wan-sŏ";
  "Ch'oe, In-hun"), but many records also carry the person's own preferred romanization ("Kim, Young-ha") as
  the heading or as a variant, so try both; Chinese uses pinyin, surname then given name joined without hyphen
  ("Mo, Yan"; "Wang, Anyi"); Japanese uses modified Hepburn with macrons ("Murakami, Haruki"; "Ōe, Kenzaburō").
  Also try the unaccented form, since suggest2 matching is character-sensitive.
- Arabic, Persian, Hebrew, Russian, Greek, South Asian names: ALA-LC romanization; try the common
  English-press spelling too. Arabic names may be entered under the ism, nisba, or laqab.
- Names of organizations: try the LC corporate form ("United States. Navy") and the natural-language form.
- If the record supplies a non-Latin form in the MARC key (e.g. 880 linkage), search it as-is in keyword mode
  as well; suggest2 indexes non-Latin variant labels.
- Use the book's subject, language, and place of publication to guess the name's origin and script.

Return 3 to 8 queries, most promising first, without duplicating the mechanical variants listed as already
being searched. Do not include role words ("author", "editor") or dates the record does not supply."""

SHORTLIST_SYSTEM = """You are an expert authority-control cataloger. A search returned many LCNAF headings for a
contributor on a book record. Using only the brief summaries provided, pick the candidates worth investigating in
depth (fetching the full authority record and the works they are linked to).

Keep a candidate if the name is compatible with the contributor's name (allowing for variant spellings,
romanization differences, fuller forms, or missing middle names) AND nothing in the summary rules it out
(e.g. death before the book's publication, obviously different field). Prefer to keep too many rather than too
few; you may keep up to the limit stated in the request. Drop candidates whose names clearly denote a different
person (different surname, incompatible forename) or the wrong entity type."""

DECISION_SYSTEM = """You are an expert authority-control cataloger performing NACO-quality identity matching.
Decide whether one of the LCNAF candidate records represents the same person or body as the contributor on the
book record.

Weigh evidence like a cataloger would:
- Strong positive evidence: the candidate's authority record cites this very book or another work by the same
  co-authors/publisher/series; the candidate's linked works are on the same topic or in the same series; fields of
  activity, occupation, affiliation, or locale match what the book implies; dates compatible with authorship;
  the candidate's variant forms include the exact form on the book; ISBNdb author names or synopsis corroborate.
- Negative evidence: candidate died before the work was written; candidate works in an unrelated field with no
  overlap (e.g. a 19th-century botanist vs. a 2026 true-crime co-author); qualifiers (fuller names, middle
  initials, dates) that conflict with what the book supplies; the candidate is a different entity type.
- A bare name match with no corroborating evidence is weak, especially for common names; for rare names it is
  moderate. Undifferentiated/"multiple identities" records are not matches.
- The absence of the book among the candidate's linked works is not negative evidence (new books are not yet
  linked); the absence of ANY related work or biographical data is merely uninformative.
- Non-Western names: romanization differences (McCune-Reischauer vs. Revised Romanization, Wade-Giles vs.
  pinyin, presence/absence of diacritics or hyphens) are not evidence against a match. Korean and Chinese names are
  often shared by many people, so require topical or biographical corroboration before matching.

Output: `match` with the single best candidate URI and a calibrated probability that it is the same entity, or
`no_match` if none is adequately supported (then matched_uri is null and confidence expresses your probability
that the best candidate is NOT the person, i.e. your confidence in no_match). Cite the concrete fields that drove
the decision in `reasoning`. It is far better to return no_match than to link the wrong person: a wrong link
corrupts the catalog, while a no_match simply leaves the name for a human to handle."""


# --------------------------------------------------------------------------- #
# Wikidata fallback
# --------------------------------------------------------------------------- #


def render_wikidata_candidate(c, idx: int) -> str:
    s = f"### Candidate {idx}: {c.qid}\n"
    s += _line("Label", c.label)
    if c.match_type == "alias" and c.match_text and c.match_text != c.label:
        s += f"(matched via alias/former name: {c.match_text})\n"
    s += _line("Description", c.description)
    s += _line("Aliases", c.aliases)
    s += _line("Instance of", c.instance_of)
    s += _line("Birth date", c.birth_date)
    s += _line("Death date", c.death_date)
    s += _line("Birth place", c.birth_place)
    s += _line("Citizenship", c.citizenships)
    s += _line("Occupations", c.occupations)
    s += _line("Educated at", c.educated_at)
    s += _line("Employers", c.employers)
    s += _line("Notable works", c.notable_works)
    s += _line("Languages", c.languages)
    s += _line("Awards", c.awards)
    s += _line("LCNAF id (P244)", c.lcnaf_id)
    s += _line("VIAF id", c.viaf_id)
    s += _line("Wikipedia (en)", c.wikipedia)
    s += _line("Sitelinks (prominence)", c.sitelinks)
    s += _line("Surfaced by queries", c.found_by)
    return s + "\n"


WIKIDATA_DECISION_SYSTEM = """You are an expert authority-control cataloger. No Library of Congress Name Authority record was
found for a contributor on a book record. Decide whether one of these Wikidata items is the same person or body,
so the catalog can link to Wikidata instead.

Judge by identity, not by name similarity:
- A label that differs from the name on the book is not evidence against a match when the item matched via an
  alias or a former/married/romanized name; romanization and diacritic differences are never negative evidence.
- Strong positive evidence: occupation, field, nationality, dates, affiliations, or notable works consistent with
  the book (an author of a tennis memoir who is a tennis player; a novelist for a novel; a museum for an exhibition
  catalog); a Wikipedia article whose subject plainly wrote or produced works like this one; an LCNAF id on the item
  that is compatible with the name.
- Negative evidence: died before the work was written; a different kind of entity; an occupation or era that
  cannot have produced this book; a distinguishing qualifier on the book (birth year, fuller name) that
  conflicts with the item.
- For a common personal name with several plausible bearers and no contextual discriminator, do not guess:
  return no_match. Where exactly one bearer fits the context, prefer it, leaning toward the more prominent item
  (higher sitelink count) when two both fit and nothing separates them.
- A bare name coincidence on an item with no corroborating context is not a match.

Output: `match` with the single best Q-id and a calibrated probability, or `no_match` (matched_qid null,
confidence = your probability that no candidate is the person). Cite the concrete fields that drove the decision.
A wrong link is worse than no link."""
