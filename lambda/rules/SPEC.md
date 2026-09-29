# Rule card specification

Each chapter of the LC Classification and Shelflisting Manual (CSM) that affects how a
call number is built becomes one *rule card*: a markdown file `rules/<ID>.md` with YAML
frontmatter. Cards are loaded into an LLM's context selectively by a router, so the
frontmatter must be accurate and the body must be self-contained.

The audience is an LLM that has a bibliographic record and a class number and must
produce the complete LC call number (class number + Cutter(s) + date + other elements).
Anything about physical processing at LC (writing numbers in books, logging out,
routing to divisions, SLIC cards) is OUT of scope and must be dropped.

## Frontmatter

```yaml
---
id: G320                      # chapter code, no space
title: Biography              # chapter title as printed
source: "G 320, July 2013"    # code + date printed in the page header/footer
always_load: false            # true only for the handful of core chapters
triggers:                     # controlled vocabulary, see below. Card is loaded when
  - form:biography            #   ANY trigger matches the resource profile.
  - form:autobiography
class_prefixes: []            # e.g. ["P", "PZ", "K", "M", "N"]; card loads when the
                              #   class number starts with one of these. [] = any.
see_also: [F275, G058]        # other chapter ids referenced
summary: >-                   # 1-2 sentences, what the card decides. Used by the router
  How to build the Cutter(s) and date for biographies ...   #   and shown in an index.
---
```

### Trigger vocabulary

Main entry: `entry:personal`, `entry:corporate`, `entry:conference`, `entry:uniform_title`,
`entry:title`, `entry:numeral` (heading begins with a numeral), `entry:initials`,
`entry:government_body`.

Form / genre / situation tags: `form:serial`, `form:periodical`, `form:yearbook`,
`form:congress`, `form:society_publication`, `form:collected_works`, `form:selected_works`,
`form:biography`, `form:autobiography`, `form:correspondence`, `form:diary`,
`form:juvenile`, `form:dissertation`, `form:translation`, `form:parallel_text`,
`form:later_edition`, `form:reprint`, `form:facsimile`, `form:supplement`,
`form:abridgment`, `form:commentary`, `form:criticism`, `form:index`, `form:abstract`,
`form:software`, `form:microform`, `form:comic`, `form:cartographic`, `form:music`,
`form:sound_recording`, `form:rare`, `form:law`, `form:government_document`,
`form:legislative_hearing`, `form:festschrift`, `form:folklore`, `form:genealogy`,
`form:ethnic_group`, `form:essays`, `form:general_special`, `form:literary_work`,
`form:literary_author`, `form:literary_collection`, `form:bound_with`, `form:multivolume`,
`form:archival_inventory`, `form:discography`, `form:teaching`, `form:foreign_relations`,
`form:historic_preservation`, `form:local_court_records`, `form:library_resources`,
`form:lc_publication`, `form:artist`, `form:government_official`, `form:time_period`,
`form:city_region`, `form:washington_dc`, `form:analytic_in_set`, `form:bibliography`,
`form:call_number_change`.

Class-number shape: `class:has_subject_cutter`, `class:has_geographic_cutter`,
`class:reserved_a_range`, `class:table_instruction`, `class:by_date`,
`class:biography_number`, `class:double_cutter`, `class:alternate_number`,
`class:obsolete_or_reserved`.

Use `always_load: true` and an empty triggers list only for chapters every call number
needs (G053, G055, G058, G063, G065, G100, G140, G070).

If a chapter genuinely needs a tag that is not in the vocabulary, add it with the
same `prefix:snake_case` shape and list it at the bottom of the card under
`## New triggers proposed` so it can be added to the vocabulary.

## Body

Use exactly these sections, in this order. Omit a section only if the chapter has
nothing for it.

```
## Applies when
Bullet list of the concrete situations that make this chapter relevant, phrased as
tests an LLM can apply to a bibliographic record + class number.

## Rules
The procedure, as numbered steps or bullets. Preserve the manual's precision: exact
Cutter ranges (.A1-.A5), exact digits appended (.x2), exact date formats, exact
punctuation. Keep the manual's section numbers in brackets, e.g. [sec. 2.b], so the
LLM can cite them. Do NOT paraphrase loosely; do NOT add rules the manual doesn't state.
Where the chapter says schedule instructions override the manual, say so.

## Tables
Any table from the chapter, transcribed in full as a markdown table. If the PDF text
extraction scrambled a table, reconstruct it carefully from the row/column labels and
say "(reconstructed from scrambled extraction; verify against PDF)".

## Examples
EVERY example in the chapter, verbatim. Keep the 050 field strings exactly
(e.g. `050 00 $a TX749.5.B43 $b A142 2006`). Give each example the situation it
illustrates in one line above it. Examples are the most valuable part of the card.

## Pitfalls
Exceptions, "however" clauses, and things the chapter warns about.

## Cross-references
Which other chapters the text points to and for what.
```

## Style

- Be faithful. When the source is ambiguous, quote it rather than resolve it.
- Fix obvious PDF extraction artifacts (spaces inside words, broken lines) silently.
- Drop page headers/footers, "Continued" markers, and LC-internal procedure.
- Keep MARC field references (1XX, 245, 050 $a/$b) as-is; the LLM reads MARC.
- No commentary about the manual, no "note that this chapter...". Just the rules.
