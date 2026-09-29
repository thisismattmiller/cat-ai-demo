"""Step 1: profile the resource. One structured-output LLM call that extracts the
facts the shelflisting rules condition on."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from .llm import Provider

TAG_VOCABULARY = """
entry:personal entry:corporate entry:conference entry:uniform_title entry:title entry:numeral entry:initials entry:government_body
form:serial form:periodical form:yearbook form:congress form:society_publication form:collected_works form:selected_works
form:biography form:autobiography form:correspondence form:diary form:juvenile form:dissertation form:translation
form:parallel_text form:later_edition form:reprint form:facsimile form:supplement form:abridgment form:commentary
form:criticism form:index form:abstract form:software form:microform form:comic form:cartographic form:music
form:sound_recording form:rare form:law form:government_document form:legislative_hearing form:festschrift form:folklore
form:genealogy form:ethnic_group form:essays form:general_special form:literary_work form:literary_author
form:literary_collection form:bound_with form:multivolume form:archival_inventory form:discography form:teaching
form:foreign_relations form:historic_preservation form:local_court_records form:library_resources form:lc_publication
form:artist form:government_official form:time_period form:city_region form:washington_dc form:analytic_in_set
form:bibliography form:call_number_change
class:has_subject_cutter class:has_geographic_cutter class:reserved_a_range class:table_instruction class:by_date
class:biography_number class:double_cutter class:alternate_number class:obsolete_or_reserved class:incomplete_plus class:general_works_number
form:multi_topic
""".split()


class MainEntry(BaseModel):
    tag: Literal["100", "110", "111", "130", "245"] = Field(description="MARC tag the main entry would have: 100 personal, "
                                                           "110 corporate, 111 conference, 130 uniform title, 245 title (no creator)")
    heading: str = Field(description="The main entry heading, surname first for persons")
    entry_type: Literal["personal", "corporate", "conference", "uniform_title", "title"]
    cutter_basis: str = Field(description="The exact word the book-number Cutter is built from: a personal surname, "
                              "the first filing word of a corporate/conference name, or the first filing word of the "
                              "title with any initial article removed (G 100 sec. 9). Give the word only.")
    filing_string: str = Field(description="The full heading as it should be compared for filing position (article "
                               "stripped for titles; names as written, e.g. 'Woolf, Virginia, 1882-1941').")
    basis_reason: str = Field(description="One sentence: why this is the Cutter basis")


class ClassAnalysis(BaseModel):
    class_number: str = Field(description="The class number exactly as given (050 $a), e.g. 'PR6045.O72'")
    letters: str = Field(description="Class letters, e.g. 'PR'")
    existing_cutters: list[str] = Field(description="Cutters already part of the class number (subject/geographic/author "
                                        "Cutters printed in the schedule), e.g. ['O72']; [] if none")
    schedule_instruction: str | None = Field(description="The subarrangement instruction that applies to this number, taken "
                                             "from the schedule entry when it is given (e.g. 'Table P-PZ40', 'By date', "
                                             "'Subarrange each country by Table H77a'), or null if none")
    is_literary_author_number: bool = Field(description="True if this is an individual literary author's number in the P "
                                            "schedules (works by/about one author are subarranged under it)")
    likely_biography_number: bool = Field(description="True if the schedule caption for this number is 'Biography' / "
                                          "'Individual biography, A-Z' style (G 320 Biography Table applies)")
    reserved_a_range: bool = Field(description="True if the schedule is likely to reserve .A1-.A5 (or similar) Cutters "
                                   "under this number for periodicals/documents/general works")
    notes: str = Field(description="Anything else about the shape of this class number that affects the book number")


class Profile(BaseModel):
    main_entry: MainEntry
    title: str
    uniform_title: str | None
    edition_statement: str | None
    publication_date_raw: str = Field(description="The publication date exactly as given; empty string if none")
    call_number_date: str | None = Field(description="Your G 140 reduction of the date: e.g. '2006', '1990z'; null if none")
    language: str | None
    original_language: str | None = Field(description="If a translation, the language of the original")
    biographee: str | None = Field(description="If the work is a biography/criticism of one person, that person's heading, else null")
    geographic_focus: list[str] = Field(description="Places the work is about, most specific first")
    tags: list[str] = Field(description="Every applicable tag from the vocabulary; be generous, extra tags only load "
                            "extra rule cards")
    volume_designation: str | None = Field(description="If this is one part of a multipart item, its designation "
                                           "(e.g. 'vol. 3', 'pt. 2'); else null")
    class_analysis: ClassAnalysis
    summary: str = Field(description="Two sentences: what this resource is and what kind of shelflisting situation it presents")
    open_questions: list[str] = Field(description="Facts you could not determine from the record that would change the call number")


SYSTEM = f"""You are an experienced Library of Congress shelflisting technician. You are given the metadata of a
resource (title, creator, summary, contents, date, subjects ... whatever is known) and the class number
chosen for it. Extract the facts a shelflister needs before consulting the Classification and Shelflisting
Manual. Do not build the call number yet. Be literal about what the metadata says; do not guess dates or
headings that are not there. If the creator is not in authorized (surname-first) form, give the surname
as the Cutter basis anyway and say so in open_questions.

Tag vocabulary (use only these strings in `tags`):
{' '.join(TAG_VOCABULARY)}

Guidance for tags:
- entry:* describes the main entry. entry:numeral if the Cutter basis begins with digits; entry:initials if it is initials/acronym;
  entry:government_body if the corporate body is a government or government agency.
- form:literary_author when the class number is an individual literary author's number (P schedules); add form:literary_work
  for a work BY the author and form:criticism for a work ABOUT the author. Under an author number the book number
  is NOT the author's name: for a work BY the author the Cutter basis and filing_string are the TITLE of the work
  (initial article dropped); for a work ABOUT the author they are the critic's name (main entry) within the
  biography/criticism span of the author table. Likewise under any number already Cuttered for a person
  (biography numbers), Cutter for the main entry of the work about them per the biography table.
- form:biography for any biography of an individual (600 first subject with no $v that says otherwise); form:autobiography if by the subject.
- form:later_edition if the edition statement or notes show an edition other than the first.
- form:translation if the metadata says it is translated (translated_from, uniform title with language, notes).
- form:juvenile if the audience is juvenile or a subject has Juvenile ...
- class:has_subject_cutter if the class number already contains a Cutter (e.g. PR6045.O72, HV5824.C42).
- class:incomplete_plus if the 050 $a ends in '+' (cataloger left the number for the shelflister to complete, F 440).
"""


def build_prompt(resource_text: str, class_number: str, extra_context: str | None = None, schedule_text: str | None = None) -> str:
    p = f"Class number chosen for this resource: {class_number}\n\nResource metadata:\n{resource_text}\n"
    if schedule_text:
        p += f"\nThe LC classification schedule entry for this number (2024 edition):\n{schedule_text}\n"
    if extra_context:
        p += f"\nAdditional context from the requester:\n{extra_context}\n"
    return p


def profile_resource(resource_text: str, class_number: str, provider: Provider, *, extra_context: str | None = None,
                     schedule_text: str | None = None) -> tuple[Profile, dict]:
    prof, usage = provider.structured(SYSTEM, build_prompt(resource_text, class_number, extra_context, schedule_text), Profile)
    prof.tags = [t for t in prof.tags if t in TAG_VOCABULARY]
    return prof, usage
