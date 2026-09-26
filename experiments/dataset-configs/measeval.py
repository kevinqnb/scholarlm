"""
Dataset configuration for the measeval (MeasEval / SemEval-2021 Task 8) dataset.

Unlike pond/nfix/supermat, measeval has no fixed catalogue of measurable
attributes -- the goal is OPEN extraction of any directly reported numerical
measurement, whatever its subject matter (materials science, geology, biology,
oceanography, medicine, engineering, ...). `attribute_info_dict` collapses to a
single abstract bucket, "measurement", used only as a coarse per-document gate
("does this document report any direct numerical measurement at all"), which
every measeval paper trivially passes.

Subject-first design
---------------------
The entity is a general-purpose SUBJECT: the sample, specimen, site, material,
organism, structure, instrument, population, or similar concrete thing a
measurement is made on or of (`EntitySchema`: `name` + `identifiers`, the same
convention as pond/nfix/supermat -- `identifiers` is real-pipeline-only, see
EntitySchema's own comment below). What was measured (`property`) and under
what circumstances (`event_details`) are resolved afterwards, per subject, on
the measurement event -- a subject can have many distinct properties reported
(age, mass, temperature, ...), so event resolution enumerates all of them
rather than assuming exactly one. Value extraction then searches the located
page/table for the reported quantity matching a given (subject, property)
event, splitting it into `value` and `units` -- the normal MeasurementLM
value-extraction step, unmodified.

The entity field is `name` (not `subject`) and the attribute bucket's key is
the literal string `"measurement"` (not e.g. "quantity") because both match
`data/measeval/ground_truth.json`'s existing columns exactly: `name` holds
the MeasuredEntity span, `property` the MeasuredProperty span (or None when a
quantity attaches to its subject with no distinct property phrase, e.g.
"5318 participants"), `value`/`units` the parsed Quantity, and `attribute`
the constant "measurement". `analysis/ablation.py`'s `get_matching_rules`
measeval branch (imported by `analysis/baselines.py` too) reads this directly:
it fuzzy-matches `name` + `property` and strict-matches `document_id` +
`attribute` + `value` + `units`. measeval uses no judge pipeline -- ground
truth is matched directly (see data/measeval/README.md).

`DirectExtractionItemSchema` (used by Ablation 1 and the NuExtract3 baseline)
mirrors pond's field ordering: entity fields, then event fields, then
attribute/value/units. Every arm's raw output -- main pipeline, Ablation 1,
NuExtract3, Ablation 2, GLiNER, ChatExtract -- carries the same `name` /
`property` / `event_details` / `attribute` / `value` / `units` columns, so
`process_extraction_df` needs no measeval-specific branch. `identifiers` is
the one exception: the main (7-step) pipeline's own `EntitySchema` still
carries it, but no direct-extraction-style method does (removed 2026-09-25
from `DirectExtractionItemSchema`/`Ablation2EntitySchema`'s siblings across
every dataset, per instruction -- it's an alias-resolution aid for the real
pipeline's entity matching only, never something a baseline should
reproduce). `process_extraction_df` never reads `identifiers`, so this
doesn't need a measeval-specific branch either.

NuExtract3 needs nothing beyond what's already here: `run_baseline_nuextract3.py`
reads `entity_schema`, `attribute_info_dict`, `direct_extraction_schema`,
`direct_extraction_prompt`, and `nuextract_examples` straight off this config,
same as every other dataset. Ablation 2 (combined entity-attribute detection)
is a legitimate measurement here too, despite the single-bucket attribute
space: it tests whether asking the model to name the (trivial, constant)
attribute in the same step as the subject costs anything, versus the two-step
baseline; `attribute_terms` is always empty (`collect_attribute_terms=False`),
so it measures step-combination overhead only, not term-collection quality.

See data/measeval/README.md.
"""
from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel

from scholarlm.config import DatasetConfig, QUALIFIER_FIELD_NAMES


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class EntitySchema(BaseModel):
    """A general-purpose subject that at least one measured quantity is reported for."""

    name: str | None
    identifiers: str | None
    # ``identifiers`` is extracted by the real (7-step) pipeline only --
    # removed 2026-09-25 from every direct-extraction-style method (Ablation
    # 1, NuExtract, LangExtract; GLiNER already excluded it structurally,
    # never listed in gliner_field_descriptions, and ChatExtract's flat
    # schema never included it) per instruction: none of them should extract
    # it. It is also never shown to the judge (judge_filter_fields, below).


ENTITY_IDENTIFICATION_PROMPT = """You are an expert at identifying the subjects of reported measurements in scientific text. Given the provided text (including any tables), find every distinct subject that has at least one directly reported numerical measurement associated with it, and identify it by name.

A subject is any concrete thing a measurement is made on or of: a sample, specimen, site, material, compound, organism, structure, instrument, population, participant group, or similar. This text may come from any scientific discipline (e.g. materials science, geology, biology, oceanography, medicine, engineering) -- the subject can be anything reported on, not just one domain's typical entities.

Do NOT try to enumerate every property measured for a subject here, and do NOT copy out the numbers themselves -- that is resolved later, per subject. Your only job in this step is to name the distinct subjects that have measurements reported about them.


What counts as a subject?:
- Include any sample, specimen, site, material, organism, structure, instrument, or population that the text reports at least one direct numerical measurement for.
- A subject with many different properties measured (e.g. age, mass, temperature) is still ONE subject -- do not create a separate item per property.
- The same subject referred to by different phrasing elsewhere in the text (e.g. "the sample" and "Sample A" for the same physical object) is ONE subject, not two.
- If the text reports a bare, subject-less quantity with no identifiable subject at all (e.g. an isolated statistic with no named referent), do NOT invent a subject for it -- omit it here; it will be handled at extraction with a null subject.


Response schema:
For each distinct subject, output one item with the following fields:
- name: the subject's name or identifying description, copied verbatim from the text (e.g. "Sample A", "Lake Mendota sediment core", "the control group", "specimen MB-7"). Do NOT paraphrase or normalize wording.
- identifiers: every alternate short-form reference to this subject used in the text -- codes, abbreviations, or shortened versions of the name -- joined into a single string with semicolons separating each (e.g. "MB-7; Specimen 7"). Collect these whenever the text uses them for the same subject, even if the linkage is introduced only once. Do not include the primary name itself. If no alternatives exist, set to None.


Identification guidelines:
- Treat two mentions as the same subject unless the text clearly distinguishes them as physically or conceptually distinct (e.g. two different specimens, two different named sites).
- Do NOT create separate items for the same subject because it was measured under different conditions, at different times, or for different properties -- those distinctions are captured later, as measurement events.


Strict rules about missing information:
- Use ONLY the exact text explicitly present in the document.
- Do NOT infer, guess, or derive a subject from context.
- If the text reports no measurements with an identifiable subject at all, return an empty list.


Extraction procedure:
1. Scan the entire text, including tables, from beginning to end.
2. Each time you encounter a directly reported numerical measurement, determine what subject it is about.
3. Group measurements that share the same subject under a single item for that subject, recording any alternate short-form references in the identifiers field.
4. Output one JSON item per distinct subject.
5. Collect all items into a single JSON array under the key "items".


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "identifiers": "..."
    }
  ]
}
- If no subjects are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Attribute schema
# ---------------------------------------------------------------------------

# Single abstract bucket used only as a coarse per-document gate ("does this
# document report any direct numerical measurement at all"), which every
# measeval paper trivially passes. Key is "measurement", not "quantity", to
# match the constant `attribute` value in data/measeval/ground_truth.json.
_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
    "measurement": {
        "description": (
            "Any numerical quantity directly reported in the text for the given "
            "subject -- a count, physical property, concentration, dimension, rate, "
            "duration, statistical quantity (p-value, confidence interval bound, "
            "effect size, odds/hazard ratio), or other quantified characteristic, "
            "regardless of scientific domain or unit. This dataset has no fixed "
            "catalog of measurable properties: 'measurement' is a single umbrella "
            "bucket standing in for the fact that the document reports at least one "
            "such quantity at all -- it is not itself a property name. Search the "
            "text for the number associated with the given subject and event, and "
            "report its numeric part as the value and its unit as the units, exactly "
            "as written; if it carries no unit, leave the units empty."
        ),
        "units": [],
    },
}


# ---------------------------------------------------------------------------
# Measurement schema
# ---------------------------------------------------------------------------

class MeasurementEventSchema(BaseModel):
    """A single distinct property reported for an already-identified subject, and its context."""

    property: str | None
    event_details: str | None

_MEASUREMENT_EVENT_PROMPT = """EVENT FIELDS:
- property: the specific property, quantity type, or characteristic being measured for the given subject -- e.g. "mean annual temperature", "grain size", "paleolatitude", "mean age", "sedimentation rate". Copy the exact wording used in the text; do not paraphrase, abbreviate, or normalize it. This is open-ended -- there is no fixed list of allowed values. A property is NOT the number itself, and NOT a generic word like "measurement". If a quantity attaches directly to the subject with no distinct property phrase (e.g. a bare count like "5318 participants"), set this to None.
- event_details: the qualifying context for this specific measurement -- for example the date, method, location, treatment condition, comparison group, or circumstance under which it was measured (e.g. "at baseline in 1991", "under high pressure", "compared to the control group"). Copied or closely paraphrased from the text, and kept SHORT -- a phrase, not a sentence. Set to None if the text gives no distinguishing context.

Enumerate every distinct property reported for the given subject on this page. A subject commonly has MORE THAN ONE property measured (e.g. a specimen with both an age and a mass reported) -- output one item per distinct property, not one item total. Two properties that differ only in their qualifying context (e.g. the same property measured at two different times) are also two separate items. Only output an empty list if the given subject has no reported measurement at all on this page.
"""


# ---------------------------------------------------------------------------
# Ablation 1: direct extraction prompt
# ---------------------------------------------------------------------------


class DirectExtractionItemSchema(BaseModel):
    """Flat schema for Ablation 1: combines entity, event, value/units, and
    the qualifier/shape fields (the same shape
    MeasurementLM._parse_quantities() produces via a separate step -- see
    DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS). No ``identifiers`` field
    (removed 2026-09-25): that's an alias-resolution aid for the real
    pipeline's entity matching only, never something a direct-extraction-style
    method should reproduce -- see EntitySchema's own comment above."""

    # Entity fields
    name: str | None
    # Event fields
    property: str | None
    event_details: str | None
    # Measurement fields -- `attribute` is forced to the same single-value
    # bucket the main pipeline uses (see module docstring). It's a
    # Literal, not a free field: structured decoding leaves the model no
    # other valid value, so there's nothing to prompt it about.
    attribute: Literal["measurement"] = "measurement"
    value: str | None
    units: str | None
    # Qualifier/shape fields -- point_value/lower/upper/tolerance/
    # standard_deviation are str | float | None, matching MeasurementLM's
    # ParseQuantityResponse fix (see notes/scholarlm/experiments/
    # 2026-09-24-gptoss120b-parsequantity-schema-diag-{01,02}.md).
    qualifiers: list[str]
    point_value: str | float | None
    lower: str | float | None
    upper: str | float | None
    list_values: list[str] | None
    tolerance: str | float | None
    standard_deviation: str | float | None


class DirectExtractionItemSchemaNoQualifiers(BaseModel):
    """Ablation-1 no-qualifiers variant of DirectExtractionItemSchema: same
    entity/event/measurement fields, with the 7 qualifier/shape fields
    dropped entirely (params.include_qualifiers=false in run_ablation.py,
    run_baseline_nuextract3.py, or run_baseline_langextract.py -- see
    tests/test_ablation1_no_qualifiers.py for the field-set diff this must
    maintain)."""

    # Entity fields
    name: str | None
    # Event fields
    property: str | None
    event_details: str | None
    # Measurement fields
    attribute: Literal["measurement"] = "measurement"
    value: str | None
    units: str | None


# `from __future__ import annotations` defers every annotation to a string, and
# `load_dataset_config` (experiments/run_extraction.py) imports this file via
# importlib WITHOUT registering it in sys.modules -- so pydantic has no module
# namespace in which to resolve "Literal[...]" later, and any attempt to build
# a JSON schema from this model raises PydanticUserError ("Literal is not
# defined"). Rebuilding here, at module scope, resolves it against this file's
# own globals while they are still available. Without this, Ablation 1 and the
# NuExtract3 baseline fail at `create_model(...).model_json_schema()` before
# issuing a single request. No other config declares a Literal field, so these
# are the only two that need it.
DirectExtractionItemSchema.model_rebuild()
DirectExtractionItemSchemaNoQualifiers.model_rebuild()


_DIRECT_EXTRACTION_PROMPT = """Subject and Measurement Identification:
Find every distinct subject in the document that has at least one directly reported numerical measurement -- a sample, specimen, site, material, organism, structure, instrument, or population, whatever its subject matter (materials science, geology, biology, medicine, or any other domain). Then, for each subject, extract every distinct property reported for it, with its quantity.

Include statistical quantities -- p-values and significance thresholds, confidence and significance levels ("95%"), the numeric bounds of a reported confidence interval, correlations, effect sizes, odds and hazard ratios. Skip only numerals that are structural rather than measured: citation years, reference/figure/table/equation numbers, and page numbers.

For each (subject, property) combination found, output one record with:
- name: the subject's name or identifying description, copied verbatim from the text. Set to None only if the text reports a quantity with no identifiable subject at all.
- property: the specific property or quantity type being measured for that subject (e.g. "mean annual temperature", "grain size", "paleolatitude"), copied verbatim from the text -- not the number itself. If the quantity attaches directly to its subject with no distinct property phrase (e.g. "5318 participants"), set this to None.
- event_details: any qualifying context for this specific measurement (method, location, condition, date, comparison), copied or closely paraphrased from the text and kept to a short phrase. Set to None if not applicable.
- value: the reported quantity exactly as written, in full -- including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- units: the unit part of the reported quantity as written, or None if it is unitless (e.g. a plain count or dimensionless ratio).
- qualifiers: a list of zero or more tags describing the shape of the reported quantity. Use only tags from this set, and combine them freely when the text supports it (e.g. an approximate mean is ["IsApproximate", "IsMean"]); use an empty list for a plain, unhedged single value:
  - "IsCount": a count of discrete items, not a continuous measurement.
  - "IsApproximate": explicitly hedged, e.g. "~12", "about 50", "approximately".
  - "IsList": an enumerated list of separate values, not a single number or range.
  - "IsRange": a reported interval or one-sided bound, e.g. "3-7", "< 5", "at least 10".
  - "IsMean": an explicitly stated mean/average.
  - "IsMedian": an explicitly stated median.
  - "HasTolerance": an explicit +/- value or confidence interval is reported alongside the value.
  - "HasSD": an explicit standard deviation is reported alongside the value.
- point_value: the single central value, when one is directly reported -- a plain point value, or the stated mean/median/count. Leave null if no single central value is reported.
- lower / upper: the bounds of a reported range or one-sided inequality. For a two-sided range, populate both. For a one-sided bound, populate only the reported side. Leave both null if no range or bound is reported.
- list_values: the parsed items of an enumerated list, in the order reported. Leave null unless "IsList" applies.
- tolerance: the confidence interval or +/- value exactly as reported, as a freeform string. Leave null unless "HasTolerance" applies.
- standard_deviation: the standard deviation exactly as reported, as a freeform string. Leave null unless "HasSD" applies.

Rules:
- Output one record per distinct (subject, property) measurement. A subject with several properties measured produces several records. The same measurement restated in prose and in a table is one record.
- Do NOT infer, guess, or derive any field. Use ONLY information explicitly stated in the text.
- Do NOT extract vague, qualitative, or non-numeric statements.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "property": "...",
      "event_details": "...",
      "attribute": "measurement",
      "value": "...",
      "units": "...",
      "qualifiers": [...],
      "point_value": "...",
      "lower": "...",
      "upper": "...",
      "list_values": [...],
      "tolerance": "...",
      "standard_deviation": "..."
    }
  ]
}
- If no measurements are found, output exactly:
{ "items": [] }
"""


# Ablation-1 no-qualifiers variant of _DIRECT_EXTRACTION_PROMPT: identical
# except the JSON example under "Output format requirements" drops the 7
# qualifier/shape keys, matching DirectExtractionItemSchemaNoQualifiers. See
# tests/test_ablation1_no_qualifiers.py for the exact diff this must maintain.
_DIRECT_EXTRACTION_PROMPT_NO_QUALIFIERS = """Subject and Measurement Identification:
Find every distinct subject in the document that has at least one directly reported numerical measurement -- a sample, specimen, site, material, organism, structure, instrument, or population, whatever its subject matter (materials science, geology, biology, medicine, or any other domain). Then, for each subject, extract every distinct property reported for it, with its quantity.

Include statistical quantities -- p-values and significance thresholds, confidence and significance levels ("95%"), the numeric bounds of a reported confidence interval, correlations, effect sizes, odds and hazard ratios. Skip only numerals that are structural rather than measured: citation years, reference/figure/table/equation numbers, and page numbers.

For each (subject, property) combination found, output one record with:
- name: the subject's name or identifying description, copied verbatim from the text. Set to None only if the text reports a quantity with no identifiable subject at all.
- property: the specific property or quantity type being measured for that subject (e.g. "mean annual temperature", "grain size", "paleolatitude"), copied verbatim from the text -- not the number itself. If the quantity attaches directly to its subject with no distinct property phrase (e.g. "5318 participants"), set this to None.
- event_details: any qualifying context for this specific measurement (method, location, condition, date, comparison), copied or closely paraphrased from the text and kept to a short phrase. Set to None if not applicable.
- value: the reported quantity exactly as written, in full -- including any range, list, inequality, mean/median/count label, or uncertainty measure (± value, confidence interval, standard deviation) reported alongside it. Do not convert, round, drop, or otherwise modify any part of it.
- units: the unit part of the reported quantity as written, or None if it is unitless (e.g. a plain count or dimensionless ratio).

Rules:
- Output one record per distinct (subject, property) measurement. A subject with several properties measured produces several records. The same measurement restated in prose and in a table is one record.
- Do NOT infer, guess, or derive any field. Use ONLY information explicitly stated in the text.
- Do NOT extract vague, qualitative, or non-numeric statements.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "property": "...",
      "event_details": "...",
      "attribute": "measurement",
      "value": "...",
      "units": "..."
    }
  ]
}
- If no measurements are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# NuExtract3 baseline: few-shot synthetic examples
#
# Synthetic text, never overlapping with data/measeval papers. Every output
# value below is an exact substring of its own input text (NuExtract's
# verbatim-string fields are trained to copy spans, not paraphrase). The
# three examples span different scientific domains (biology, materials
# science, ecology) and each gives one subject multiple distinct properties,
# to demonstrate the multi-event case the event prompt above describes.
# Together they touch all of point_value, lower/upper, list_values,
# tolerance, and standard_deviation at least once. No `identifiers` key:
# direct_extraction_schema doesn't have one (removed 2026-09-25), so these
# examples never had it to begin with.
# ---------------------------------------------------------------------------

# _NUEXTRACT_QUANTITY_DEFAULTS_JSON is the qualifier/shape fields' JSON,
# plain-point-shaped, spliced into every item below; the 214-individuals count
# and the critical-current-density range each override it to demonstrate a
# non-plain-point shape -- this baseline's only real instruction channel
# (few-shot examples -- see module docstring in measurementlm_nuextract.py)
# must actually show the qualifier fields in use, not just plain points.
_NUEXTRACT_QUANTITY_DEFAULTS_JSON = (
    '"qualifiers": [], "point_value": "{point}", "lower": null, "upper": null, '
    '"list_values": null, "tolerance": null, "standard_deviation": null'
)

_NUEXTRACT_EXAMPLE_1_INPUT = (
    "Specimen MB-7, a juvenile coho salmon collected from the study reach, "
    "had a fork length of 8.4 cm and a body mass of 6.2 g at the time of "
    "capture in June 2019. A total of 214 individuals were sampled across "
    "the site."
)

_NUEXTRACT_EXAMPLE_1_OUTPUT = (
    '{"items": ['
    '{"name": "Specimen MB-7", "property": "fork length", "event_details": "at the time of capture in June 2019", "attribute": "measurement", "value": "8.4", "units": "cm", '
    + _NUEXTRACT_QUANTITY_DEFAULTS_JSON.format(point="8.4") + '}, '
    '{"name": "Specimen MB-7", "property": "body mass", "event_details": "at the time of capture in June 2019", "attribute": "measurement", "value": "6.2", "units": "g", '
    + _NUEXTRACT_QUANTITY_DEFAULTS_JSON.format(point="6.2") + '}, '
    '{"name": "the site", "property": null, "event_details": "sampled across the site", "attribute": "measurement", "value": "214", "units": null, '
    '"qualifiers": ["IsCount"], "point_value": "214", "lower": null, "upper": null, '
    '"list_values": null, "tolerance": null, "standard_deviation": null}'
    ']}'
)

_NUEXTRACT_EXAMPLE_2_INPUT = (
    "The Bi2212 crystal exhibited a superconducting transition temperature "
    "(Tc) of 84.2 K, with a critical current density measured between "
    "3.0 x 10^4 and 3.2 x 10^4 A/cm^2 at 77 K. Under 2 GPa of applied "
    "pressure, Tc increased to 91.5 K."
)

_NUEXTRACT_EXAMPLE_2_OUTPUT = (
    '{"items": ['
    '{"name": "Bi2212 crystal", "property": "superconducting transition temperature", "event_details": null, "attribute": "measurement", "value": "84.2", "units": "K", '
    + _NUEXTRACT_QUANTITY_DEFAULTS_JSON.format(point="84.2") + '}, '
    '{"name": "Bi2212 crystal", "property": "critical current density", "event_details": "measured at 77 K", "attribute": "measurement", "value": "3.0 x 10^4 and 3.2 x 10^4", "units": "A/cm^2", '
    '"qualifiers": ["IsRange"], "point_value": null, "lower": "3.0 x 10^4", "upper": "3.2 x 10^4", '
    '"list_values": null, "tolerance": null, "standard_deviation": null}, '
    '{"name": "Bi2212 crystal", "property": "superconducting transition temperature", "event_details": "Under 2 GPa of applied pressure", "attribute": "measurement", "value": "91.5", "units": "K", '
    + _NUEXTRACT_QUANTITY_DEFAULTS_JSON.format(point="91.5") + '}'
    ']}'
)

# Rounds out shape coverage with list_values, tolerance, and standard_deviation
# -- example 1 and 2 above only reach point_value, IsCount, and IsRange.
_NUEXTRACT_EXAMPLE_3_INPUT = (
    "A cohort of loggerhead sea turtles was tracked over three nesting "
    "seasons; clutch sizes were recorded as 92, 105, and 118 eggs. Curved "
    "carapace length was measured at 78.4 ± 1.2 cm across the tagged "
    "individuals. The mean incubation temperature was 29.1 (SD 0.6) °C "
    "across all monitored nests."
)

_NUEXTRACT_EXAMPLE_3_OUTPUT = (
    '{"items": ['
    '{"name": "loggerhead sea turtles", "property": "clutch size", "event_details": "over three nesting seasons", "attribute": "measurement", "value": "92, 105, and 118", "units": "eggs", '
    '"qualifiers": ["IsList"], "point_value": null, "lower": null, "upper": null, '
    '"list_values": ["92", "105", "118"], "tolerance": null, "standard_deviation": null}, '
    '{"name": "the tagged individuals", "property": "curved carapace length", "event_details": null, "attribute": "measurement", "value": "78.4 ± 1.2", "units": "cm", '
    '"qualifiers": ["HasTolerance"], "point_value": "78.4", "lower": null, "upper": null, '
    '"list_values": null, "tolerance": "± 1.2", "standard_deviation": null}, '
    '{"name": "all monitored nests", "property": "incubation temperature", "event_details": null, "attribute": "measurement", "value": "29.1 (SD 0.6)", "units": "°C", '
    '"qualifiers": ["IsMean", "HasSD"], "point_value": "29.1", "lower": null, "upper": null, '
    '"list_values": null, "tolerance": null, "standard_deviation": "0.6"}'
    ']}'
)

_NUEXTRACT_EXAMPLES = [
    {"input": _NUEXTRACT_EXAMPLE_1_INPUT, "output": _NUEXTRACT_EXAMPLE_1_OUTPUT},
    {"input": _NUEXTRACT_EXAMPLE_2_INPUT, "output": _NUEXTRACT_EXAMPLE_2_OUTPUT},
    {"input": _NUEXTRACT_EXAMPLE_3_INPUT, "output": _NUEXTRACT_EXAMPLE_3_OUTPUT},
]


def _drop_qualifiers(examples: list[dict]) -> list[dict]:
    """No-qualifiers counterpart of a `nuextract_examples`-shaped list: same
    inputs, with QUALIFIER_FIELD_NAMES removed from every output item --
    the few-shot counterpart of direct_extraction_schema_no_qualifiers, for
    the NuExtract3/LangExtract baselines' params.include_qualifiers=false.
    Derived from the qualifier-bearing examples (not hand-duplicated) so the
    two can never drift apart.
    """
    stripped = []
    for example in examples:
        items = json.loads(example["output"])["items"]
        new_items = [
            {k: v for k, v in item.items() if k not in QUALIFIER_FIELD_NAMES}
            for item in items
        ]
        stripped.append({"input": example["input"], "output": json.dumps({"items": new_items})})
    return stripped


_NUEXTRACT_EXAMPLES_NO_QUALIFIERS = _drop_qualifiers(_NUEXTRACT_EXAMPLES)


# ---------------------------------------------------------------------------
# Ablation 2: combined entity-attribute extraction prompt
# ---------------------------------------------------------------------------

class Ablation2EntitySchema(BaseModel):
    """Entity schema for Ablation 2: one item per (subject, attribute) pair."""

    name: str | None
    identifiers: str | None
    # Reserved fields required by Ablation 2 (see run_ablation.py's runtime check).
    attribute: str
    attribute_terms: list[str]


_ABLATION2_IDENTIFICATION_PROMPT = """You are an expert at identifying the subjects of reported measurements in scientific text, and at detecting which measurement attributes are reported for each subject. Given the provided text (including any tables), extract all distinct (subject, measured attribute) pairs for which a direct numerical measurement is reported.

A subject is any concrete thing a measurement is made on or of: a sample, specimen, site, material, organism, structure, instrument, or population. This dataset has only one attribute, "measurement" -- emit one item per subject that has at least one directly reported numerical measurement, paired with that constant attribute name.

IMPORTANT: Only emit a pair when a direct numerical measurement exists in the document for that subject. Do NOT emit pairs where the only data is qualitative, model parameters, or goodness-of-fit statistics with no reported subject.


Response schema:
For each (subject, attribute) pair, output one item with the following fields:
- name: the subject's name or identifying description, copied verbatim from the text.
- identifiers: every alternate short-form reference to this subject used in the text, joined into a single string with semicolons separating each. Do not include the primary name itself. If no alternatives exist, set to None.
- attribute: always the exact string "measurement" -- this dataset has only one attribute.
- attribute_terms: any terminology or abbreviations used in the document to describe the kind of measurement reported for this subject. This dataset has no fixed attribute vocabulary to collect terms for, so this should always be an empty list.


Identification guidelines:
- Treat two mentions as the same subject unless the text clearly distinguishes them as physically or conceptually distinct.
- Multiple measurements or properties reported for the same subject should produce only ONE (subject, attribute) pair -- not one per property or per measurement event.
- Do NOT infer, guess, or derive any identifying information. Use ONLY information explicitly stated in the text.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "identifiers": "...",
      "attribute": "measurement",
      "attribute_terms": []
    }
  ]
}
- If no (subject, attribute) pairs with direct numerical measurements are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# ChatExtract property phrase
#
# ChatExtract (measurementlm_chatextract.py) targets one known property per
# run and substitutes it into every prompt as "{prop}" (e.g. "What is the
# value of the {prop} in the following text?"). Left unset, that phrase falls
# back to the bare attribute_info_dict key, "measurement" -- grammatically
# degenerate ("a value of measurement"). This supplies a real noun phrase for
# that single bucket so the prompts read naturally.
#
# ChatExtract is the one arm this dataset's design does NOT reach in the same
# way as the others, and that is a property of the method rather than an
# oversight: it verifies a value against a property it was told about in
# advance, and has no step that discovers which property a number belongs to.
# `_make_record` in measurementlm_chatextract.py accordingly still emits
# `property: None` for every measeval record; matching against ground truth
# falls back to name + value + units alone (see analysis/ablation.py's
# `get_matching_rules` and data/measeval/README.md).
# ---------------------------------------------------------------------------

_CHATEXTRACT_PROPERTY_NAMES: dict[str, str] = {
    "measurement": "directly reported numerical measurement",
}

# ChatExtract's reference prompts ask about a "material"/"compound" -- measeval
# subjects span every scientific discipline, not just chemical compounds, so
# this replaces that wording with a generic noun for the measurement subject
# (see DatasetConfig.chatextract_entity_noun).
_CHATEXTRACT_ENTITY_NOUN = "subject"


# ---------------------------------------------------------------------------
# GLiNER2 baseline: per-field descriptions for event fields beyond the
# subject name (see DatasetConfig.gliner_field_descriptions). Copied verbatim
# from _DIRECT_EXTRACTION_PROMPT's own per-field bullets above -- GLiNER sees
# the same wording Ablation 1 already uses, not freshly authored text.
# No gliner_entity_description override is needed: the entity schema's `name`
# field already matches what GLiNER expects by default via
# entity_type_description. ``identifiers`` has no entry here: it's an
# alias-resolution aid for the real pipeline's entity matching, not reported
# content, so GLiNER never asks for it.
# ---------------------------------------------------------------------------

_GLINER_FIELD_DESCRIPTIONS: dict[str, str] = {
    "property": (
        'the specific property or quantity type being measured for that subject '
        '(e.g. "mean annual temperature", "grain size", "paleolatitude"), copied '
        "verbatim from the text -- not the number itself. If the quantity attaches "
        "directly to its subject with no distinct property phrase (e.g. \"5318 "
        "participants\"), set this to None."
    ),
    "event_details": (
        "any qualifying context for this specific measurement (method, location, "
        "condition, date, comparison), copied or closely paraphrased from the "
        "text and kept to a short phrase. Set to None if not applicable."
    ),
}


# ---------------------------------------------------------------------------
# Config instance
# ---------------------------------------------------------------------------

CONFIG = DatasetConfig(
    name="measeval",
    data_dir="data/measeval",
    metadata_file="data/measeval/directory.json",
    entity_schema=EntitySchema,
    entity_identification_prompt=ENTITY_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A general-purpose subject that at least one directly reported numerical "
        "measurement is associated with -- a sample, specimen, site, material, "
        "compound, organism, structure, instrument, or population."
    ),
    attribute_info_dict=_ATTRIBUTE_INFO_DICT,
    # See MeasurementLM's docstring: measeval's attribute space is a single
    # abstract bucket with no real terminology to ask for, so the detection
    # step's "terms" request is disabled here rather than left to dump
    # unrelated numeric values.
    collect_attribute_terms=False,
    measurement_event_schema=MeasurementEventSchema,
    measurement_event_prompt=_MEASUREMENT_EVENT_PROMPT,
    direct_extraction_schema=DirectExtractionItemSchema,
    direct_extraction_prompt=_DIRECT_EXTRACTION_PROMPT,
    direct_extraction_schema_no_qualifiers=DirectExtractionItemSchemaNoQualifiers,
    direct_extraction_prompt_no_qualifiers=_DIRECT_EXTRACTION_PROMPT_NO_QUALIFIERS,
    nuextract_examples=_NUEXTRACT_EXAMPLES,
    nuextract_examples_no_qualifiers=_NUEXTRACT_EXAMPLES_NO_QUALIFIERS,
    chatextract_property_names=_CHATEXTRACT_PROPERTY_NAMES,
    chatextract_entity_noun=_CHATEXTRACT_ENTITY_NOUN,
    gliner_field_descriptions=_GLINER_FIELD_DESCRIPTIONS,
    # baseline_filter_fields no longer needed for identifiers (removed
    # 2026-09-25): direct_extraction_schema doesn't have that field at all
    # any more, so there's nothing left for the NuExtract baselines to filter
    # out of it -- see EntitySchema's comment above.
    # paper_subset: set to a list of document_id codes to restrict the run.
    paper_subset=None,
    # paper_filter: None processes all three splits (train+trial+eval) by default.
    # Set to `lambda m: m["source_split"] == "eval"` to restrict to the official
    # held-out MeasEval test split for leaderboard-comparable evaluation -- see
    # data/measeval/README.md's "Train/trial/eval and comparability" section.
    paper_filter=None,
    paper_exclude=None,
    ablation2_entity_schema=Ablation2EntitySchema,
    ablation2_entity_identification_prompt=_ABLATION2_IDENTIFICATION_PROMPT,
    # measeval has no judge pipeline today (ground truth is matched directly --
    # see module docstring), so this is currently inert; set for parity with
    # the other three datasets in case that changes. identifiers is an
    # alias-resolution aid, not something to judge on; event_details is a
    # catch-all for distinguishing measurements from each other, not the main
    # extraction interest.
    judge_filter_fields=["identifiers", "event_details"],
    ground_truth_file="data/measeval/ground_truth.json",
    # Units are open free text (Quantity.other["unit"]), not drawn from a fixed
    # catalog, so there is nothing to convert here -- see data/measeval/README.md's
    # "Attribute is free text, not a closed catalog" section.
    unit_conversion_table={},
)
