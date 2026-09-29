"""Pydantic models shared across the pipeline (book record, candidates, LLM I/O)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    """Base for models used as LLM structured-output schemas (no extra keys)."""

    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Bibliographic record (from the id.loc.gov CBD RDF)
# --------------------------------------------------------------------------- #


class Contributor(BaseModel):
    label: str
    marc_key: str | None = None
    agent_uri: str | None = None  # http://id.loc.gov/rwo/agents/... when already reconciled
    agent_type: str = "Person"  # Person | Organization | Meeting | Family | Jurisdiction
    roles: list[str] = Field(default_factory=list)
    role_uris: list[str] = Field(default_factory=list)
    primary: bool = False

    @property
    def reconciled(self) -> bool:
        return bool(self.agent_uri)

    @property
    def authority_uri(self) -> str | None:
        if not self.agent_uri:
            return None
        return self.agent_uri.replace("/rwo/agents/", "/authorities/names/")


class Publication(BaseModel):
    place: str | None = None
    publisher: str | None = None
    date: str | None = None
    statement: str | None = None


class BookRecord(BaseModel):
    lccn: str
    instance_uri: str
    work_uri: str | None = None
    title: str | None = None
    subtitle: str | None = None
    part_title: list[str] = Field(default_factory=list)
    variant_titles: list[str] = Field(default_factory=list)
    responsibility_statement: str | None = None
    publication: Publication = Field(default_factory=Publication)
    isbns: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    subjects: list[str] = Field(default_factory=list)
    genre_forms: list[str] = Field(default_factory=list)
    classification: list[str] = Field(default_factory=list)
    summary: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    series: list[str] = Field(default_factory=list)
    table_of_contents: list[str] = Field(default_factory=list)
    contributors: list[Contributor] = Field(default_factory=list)

    @property
    def unreconciled(self) -> list[Contributor]:
        return [c for c in self.contributors if not c.reconciled]


class IsbndbBook(BaseModel):
    isbn: str
    title: str | None = None
    title_long: str | None = None
    authors: list[str] = Field(default_factory=list)
    publisher: str | None = None
    date_published: str | None = None
    synopsis: str | None = None
    subjects: list[str] = Field(default_factory=list)
    pages: int | None = None
    language: str | None = None


# --------------------------------------------------------------------------- #
# Authority candidates
# --------------------------------------------------------------------------- #


class SuggestHit(BaseModel):
    """One hit from suggest2, with the useful bits of the `more` block."""

    uri: str
    token: str
    label: str  # authorized label (aLabel)
    matched_variant: str | None = None  # vLabel when the query matched a see-from
    rdftypes: list[str] = Field(default_factory=list)
    birth_dates: list[str] = Field(default_factory=list)
    death_dates: list[str] = Field(default_factory=list)
    occupations: list[str] = Field(default_factory=list)
    activity_fields: list[str] = Field(default_factory=list)
    locales: list[str] = Field(default_factory=list)
    nonlatin_labels: list[str] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    marc_keys: list[str] = Field(default_factory=list)
    found_by: list[str] = Field(default_factory=list)  # which queries surfaced this hit


class Citation(BaseModel):
    status: str | None = None
    source: str | None = None
    note: str | None = None


class Candidate(BaseModel):
    """Full dossier for one LCNAF authority, assembled from the .rdf and relationships."""

    uri: str
    token: str
    label: str
    rdf_types: list[str] = Field(default_factory=list)
    variants: list[str] = Field(default_factory=list)
    fuller_name: str | None = None
    birth_date: str | None = None
    death_date: str | None = None
    gender: str | None = None
    languages: list[str] = Field(default_factory=list)
    fields_of_activity: list[str] = Field(default_factory=list)
    occupations: list[str] = Field(default_factory=list)
    locales: list[str] = Field(default_factory=list)
    affiliations: list[str] = Field(default_factory=list)
    related_authorities: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    external_authorities: list[str] = Field(default_factory=list)
    record_created: str | None = None
    record_changed: str | None = None
    contributor_to: list[str] = Field(default_factory=list)
    contributor_to_total: int | None = None
    subject_of: list[str] = Field(default_factory=list)
    subject_of_total: int | None = None
    matched_variant: str | None = None
    found_by: list[str] = Field(default_factory=list)


class WikidataCandidate(BaseModel):
    qid: str
    label: str
    description: str | None = None
    match_type: str | None = None  # label | alias
    match_text: str | None = None  # the string that matched (important when match_type == alias)
    aliases: list[str] = Field(default_factory=list)
    instance_of: list[str] = Field(default_factory=list)
    birth_date: str | None = None
    death_date: str | None = None
    occupations: list[str] = Field(default_factory=list)
    citizenships: list[str] = Field(default_factory=list)
    birth_place: list[str] = Field(default_factory=list)
    educated_at: list[str] = Field(default_factory=list)
    employers: list[str] = Field(default_factory=list)
    notable_works: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    awards: list[str] = Field(default_factory=list)
    lcnaf_id: str | None = None  # P244
    viaf_id: str | None = None
    isni: str | None = None
    orcid: str | None = None
    sitelinks: int = 0
    wikipedia: str | None = None
    found_by: list[str] = Field(default_factory=list)

    @property
    def uri(self) -> str:
        return f"http://www.wikidata.org/entity/{self.qid}"


# --------------------------------------------------------------------------- #
# LLM structured outputs
# --------------------------------------------------------------------------- #


class SearchQuery(Strict):
    q: str = Field(description="The exact string to send to suggest2 ?q=")
    mode: Literal["left", "keyword"] = Field(
        description="left = left-anchored heading browse (surname first, LC form); keyword = keyword search"
    )
    why: str = Field(description="One short clause on why this query might find the person")


class SearchPlan(Strict):
    # Field order matters: structured output is generated in schema order, so analysis comes before answers.
    name_analysis: str = Field(
        description="Brief analysis: name origin/script, likely LC heading form, ambiguities"
    )
    queries: list[SearchQuery] = Field(description="Ordered list of queries to run, most promising first")
    native_script_forms: list[str] = Field(
        description="The name in its original script if it can be inferred with confidence (e.g. 유이제, 赵丽, تركي بن دويس); empty list otherwise"
    )
    direct_order_forms: list[str] = Field(
        description="1-3 natural-order forms as the person would be named in prose or on Wikipedia (e.g. 'Leon Carroll Jr.', 'Bethanie Mattek-Sands')"
    )


class Shortlist(Strict):
    reasoning: str = Field(description="Which hits are compatible with the contributor and why, before listing them")
    keep_uris: list[str] = Field(description="Candidate URIs worth investigating in depth")


class Decision(Strict):
    # reasoning first so the model weighs the evidence before committing to a decision
    reasoning: str = Field(
        description="Evidence for and against each plausible candidate, citing specific fields, ending with the conclusion"
    )
    decision: Literal["match", "no_match"]
    matched_uri: str | None = Field(
        description="The id.loc.gov/authorities/names URI of the matching candidate (must be one of the listed candidates), or null"
    )
    confidence: float = Field(
        description="Number between 0 and 1: probability that matched_uri is the same person/body (or, for no_match, probability that no candidate is)"
    )
    runner_ups: list[str] = Field(description="Other candidate URIs that were plausible but rejected (may be empty)")


class WikidataDecision(Strict):
    reasoning: str = Field(
        description="Evidence for and against each plausible candidate (identity, not name similarity), ending with the conclusion"
    )
    decision: Literal["match", "no_match"]
    matched_qid: str | None = Field(description="The Q-id of the matching item (must be one of the candidates), or null")
    confidence: float = Field(description="Number between 0 and 1: probability that matched_qid is the same person/body")
    runner_ups: list[str] = Field(description="Other candidate Q-ids that were plausible but rejected (may be empty)")


# --------------------------------------------------------------------------- #
# Final output
# --------------------------------------------------------------------------- #


class ContributorResult(BaseModel):
    contributor: Contributor
    search_plan: SearchPlan | None = None
    queries_run: list[str] = Field(default_factory=list)
    hits: list[SuggestHit] = Field(default_factory=list)
    candidates_examined: list[str] = Field(default_factory=list)
    decision: Decision | None = None
    wikidata_queries_run: list[str] = Field(default_factory=list)
    wikidata_hits: list[WikidataCandidate] = Field(default_factory=list)
    wikidata_decision: WikidataDecision | None = None
    wikidata_name_guard: str | None = None  # strong | weak
    recommended_authority_uri: str | None = None
    recommended_rwo_uri: str | None = None
    recommended_label: str | None = None
    recommended_wikidata_qid: str | None = None
    recommended_wikidata_uri: str | None = None
    recommended_wikidata_label: str | None = None
    recommendation_source: str | None = None  # lcnaf | wikidata-p244 | wikidata
    error: str | None = None


class ReconcileResult(BaseModel):
    lccn: str
    instance_uri: str | None = None
    book: BookRecord | None = None
    isbndb: IsbndbBook | None = None
    provider: str
    model: str
    results: list[ContributorResult] = Field(default_factory=list)
    error: str | None = None
