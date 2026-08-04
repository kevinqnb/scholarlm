"""
Dataset configuration for the measeval (MeasEval / SemEval-2021 Task 8) dataset.

Unlike pond/nfix/supermat, measeval has no fixed catalogue of measurable
attributes -- the goal is OPEN extraction of any directly reported numerical
measurement, whatever its subject matter (materials science, geology, biology,
medicine, engineering, ...). This is a structural mismatch with the base
MeasurementLM pipeline, which loops over `attribute_info_dict` as a closed set
of named attributes (document-level detection, per-page provenance, per-
attribute value extraction all key off the literal attribute name).

Quantity-first design
---------------------
The unit of enumeration here is the QUANTITY, not the subject. `EntitySchema`
carries a single field, `quantity`: one item per directly reported number in
the text, copied verbatim (units included, as written). Everything that
describes that number -- what it was measured on (`name`), what property of it
was measured (`property`), and under what circumstances
(`additional_details`) -- is resolved afterwards, on the measurement event.
Value extraction then has nothing to search for: it splits the already-known
`quantity` span into its numeric `value` and its `units`.

This is a deliberate inversion of the previous design, where entities were
(subject, property) pairs and the quantity was whatever the value step found
for them. Two measurements on the ten-document dev subset motivated the
change (gemma-3-27b, `data/experiments/measeval/extraction/gemma-3-27b/2026_08_01_ten/`):

  - recovery 0.370, but a *value-only* ceiling of 0.446 -- i.e. 55% of gold
    rows were lost simply because their number never appeared in the output at
    all, before any subject/property/unit matching entered into it. Decomposing
    the strict-match criteria: quantity coverage cost 55 points, units ~5, and
    subject/property ~1. The bottleneck was never the subject.
  - only 50 entities enumerated across ten documents carrying 92 gold rows.

Enumerating (subject, property) pairs asks the model to solve the hard half of
the problem first, and any pair it fails to name takes all of that pair's
numbers down with it. Enumerating quantities first asks only "which tokens in
this text are reported numbers", then attaches meaning to each one. It also
aligns the pipeline's unit of enumeration with the ground truth's unit of
annotation: MeasEval's own rows are one per `annotSet`, and every `annotSet`
carries at most one Quantity (see data/measeval/README.md).

`quantity` deliberately holds the span WITH its units ("54.8 years", "5318",
"3.7 x 10^6 cells") rather than a bare number, because that is exactly what
MeasEval's `Quantity.text` is -- the entity field is gold-aligned, and the
value-extraction step already knows how to separate a number from its unit.

The attribute axis is unchanged: `attribute_info_dict` collapses to a single
abstract bucket, "measurement", used only as a coarse per-document gate ("does
this document report any direct numerical measurement at all"), which every
measeval paper trivially passes. Its description additionally tells the value
and standardization steps that when the entity carries a quantity, `value` and
`units` are that quantity split apart rather than a fresh search -- the
attribute description is the config-level hook that reaches those prompts.

The ground truth schema (data/measeval/preprocessing.py) mirrors this
directly: `quantity` is the raw MeasEval Quantity span, `value` its parsed
number and `units` its unit, `name` holds the MeasuredEntity span, `property`
the MeasuredProperty span (or None for the ~36% of rows where a quantity
attaches to its subject with no distinct property phrase, e.g. "5318
participants"), and `attribute` is the constant "measurement" (a trivial
strict-match, since both sides always agree). Ground truth was already built
one row per `annotSet`, so this change required only adding the raw `quantity`
span alongside the offsets already recorded -- no restructuring of the raw
annotation grouping.

The matching branch lives in analysis/ablation.py's `get_matching_rules`
(imported by analysis/baselines.py too) -- NOT analysis/calibration.py, which
is judge/probe-only and has no relevance here since measeval uses no judge
pipeline (ground truth is matched directly; see data/measeval/README.md).
It is deliberately UNCHANGED by this redesign, so runs before and after remain
directly comparable: `attribute` strict-matches as-is (constant on both sides),
and `name` + `property` stay in the fuzzy set. They are simply sourced from the
measurement event now rather than the entity; both are plain columns of the
final record either way, so no analysis code had to move with them.

DirectExtractionItemSchema below (used by Ablation 1 and the NuExtract
baseline) mirrors the same ordering: `quantity` is declared FIRST, because
structured decoding emits fields in declaration order -- the model commits to
the number before it has to say what the number is about, which is the whole
point of the flip. Every arm's raw output -- main pipeline, Ablation 1,
NuExtract -- therefore carries the same `attribute` / `name` / `property` /
`value` / `units` columns before it ever reaches analysis code;
`process_extraction_df` needs no measeval-specific branch. See
data/measeval/README.md.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from scholarlm.config import DatasetConfig


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class EntitySchema(BaseModel):
    """One directly reported numerical quantity, copied verbatim from the text."""

    quantity: str | None


# A single field by design -- see the module docstring. The subject and the
# property measured are NOT identified here; they are resolved per-quantity at
# the measurement-event step. Enumerating them here is what the quantity-first
# redesign moved away from.
ENTITY_IDENTIFICATION_PROMPT = """You are an expert at finding every reported measurement in scientific text. Given the provided text (including any tables), find every directly reported numerical quantity and copy each one out verbatim.

Your ONLY job in this step is to locate the numbers. Do not explain what they measure, what they were measured on, or why -- that is resolved later. Scan the text for numerals and copy out each reported quantity exactly as written.

This text may come from any scientific discipline (e.g. materials science, geology, biology, oceanography, medicine, engineering). A quantity is any directly reported number: a measured value, a count, a percentage, a concentration, a dimension, a rate, a duration, a date used as a measurement, or any other quantified characteristic -- with or without a unit.


What counts as an item?:
- Include EVERY directly reported numerical quantity that appears in the text, including quantities reported inside tables.
- Include the quantity's unit as part of the span when one is written with it (e.g. "54.8 years", "12.3 mg/L", "3.7 x 10^6 cells").
- Include bare counts and percentages with no unit (e.g. "5318", "31%").
- Include statistical quantities reported in the text: p-values and significance thresholds ("p < .05", ".001"), confidence and significance levels ("95%"), the numeric bounds of a reported confidence interval ("1.11", "1.62"), correlation and effect-size values, odds/hazard ratios, and similar. These ARE reported measurements for the purposes of this task.
- Do NOT include numbers that are structural rather than measured: citation years, reference numbers, equation numbers, figure/table/section numbers, and page numbers.
- Do NOT invent, compute, convert, or round any number. Copy only numbers that literally appear in the text.


Response schema:
For each quantity, output one record with the following field:
- quantity: the quantity exactly as it appears in the text -- copied character-for-character, including its unit when one is written with it. Do NOT paraphrase, normalize notation, convert units, or reformat the number.


Identification guidelines:
- Each distinct reported number in the text is its own item. Two different measurements that happen to share the same numeral (e.g. a depth of "5 m" and a count of "5") are TWO items, not one.
- The SAME reported measurement restated elsewhere in the text (e.g. once in prose and again in a table) is ONE item. Do not list it twice.
- Do NOT merge quantities that differ in any way -- different numbers, different units, or different reported measurements are always separate items.
- A range reported as two endpoints (e.g. "between 3 and 7 m") gives one item per endpoint that is itself a reported measurement.


Strict rules about missing information:
- Use ONLY the exact text explicitly present in the document.
- Do NOT infer, guess, or derive quantities from context.
- If the text reports no numerical quantities at all, return an empty list.


Extraction procedure:
1. Scan the entire text, including tables, from beginning to end.
2. Each time you encounter a numeral that is part of a directly reported measurement, copy out the quantity span verbatim, including its unit when written.
3. Skip numerals that are citation years, reference/figure/table/equation numbers, or statistical model parameters.
4. Output one JSON item per distinct reported quantity, in the order they appear in the text.
5. Collect all items into a single JSON array under the key "items".


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "quantity": "..."
    }
  ]
}
- If no quantities are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Attribute schema
# ---------------------------------------------------------------------------

# Single abstract bucket used only as a coarse per-document gate ("does this
# document report any direct numerical measurement at all"), which every
# measeval paper trivially passes.
#
# The second paragraph of the description exists for the VALUE and STANDARDIZE
# steps, not for the gate: `attribute_info_dict[attr]["description"]` is
# injected into `_extract_values_from_text`, `_extract_values_from_tables` and
# `_standardize` as "Attribute description: ...", and it is the only
# config-level hook that reaches them. Under the quantity-first design those
# steps are handed the quantity in the entity description and only have to
# split it, so they are told exactly that.
_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
    "measurement": {
        "description": (
            "Any numerical quantity directly reported in the text -- a count, "
            "physical property, concentration, dimension, rate, or other quantified "
            "characteristic, regardless of scientific domain or unit. This dataset has no "
            "fixed catalog of measurable properties: 'measurement' is a single umbrella "
            "bucket standing in for the fact that the document reports at least one such "
            "quantity at all -- it is not itself a property name. "
            "IMPORTANT: the entity description already contains the exact quantity span "
            "of interest, copied verbatim from this text. Do not search for a different "
            "number. Report that quantity's numeric part as the value and its unit as the "
            "units, exactly as written; if it carries no unit, leave the units empty."
        ),
        "units": [],
    },
}


# ---------------------------------------------------------------------------
# Measurement schema
# ---------------------------------------------------------------------------

class MeasurementEventSchema(BaseModel):
    """What a single already-identified quantity was measured on, and of what."""

    name: str | None
    property: str | None
    additional_details: str | None

_MEASUREMENT_EVENT_PROMPT = """EVENT FIELDS:
- name: the SUBJECT the quantity was measured on or of -- the sample, specimen, site, material, compound, organism, structure, instrument, population, or similar that this number describes. Copy the noun phrase that identifies it verbatim, character-for-character, from the text. Do NOT paraphrase, expand abbreviations, or normalize wording. Set to None only if the text attaches this quantity to no subject at all.
- property: the specific property, quantity type, or characteristic of that subject which this number reports -- e.g. "mean annual temperature", "grain size", "paleolatitude", "mean age", "sedimentation rate". Copy the exact wording used in the text; do not paraphrase, abbreviate, or normalize it. This is open-ended -- there is no fixed list of allowed values. A property is NOT the number itself, and NOT a generic word like "measurement". If the quantity attaches directly to its subject with no distinct property phrase (e.g. a bare count like "5318 participants"), set this to None.
- additional_details: the qualifying context for this measurement -- for example the date, method, location, treatment condition, comparison group, or circumstance under which it was measured (e.g. "at baseline in 1991", "under high pressure", "compared to the control group"). Copied or closely paraphrased from the text, and kept SHORT -- a phrase, not a sentence, and never an explanation of your reasoning. Set to None if the text gives no distinguishing context.

CRITICAL: Output EXACTLY ONE item. The quantity has already been identified for you -- it is a single number that was measured once, on one subject, of one property. This step is not an enumeration: you are describing that one number, not searching for more. Never output two items because a number could be read two ways; choose the reading the text supports and output that single item. Only output an empty list if the given quantity does not actually appear on this page at all.
"""


# ---------------------------------------------------------------------------
# Ablation 1: direct extraction prompt
# ---------------------------------------------------------------------------


class DirectExtractionItemSchema(BaseModel):
    """Flat schema for Ablation 1: combines entity, event, and value/units fields."""

    # `quantity` FIRST, deliberately: structured decoding emits fields in
    # declaration order, so the model commits to the number before it has to
    # say what the number is about -- the single-pass equivalent of the main
    # pipeline's quantity-first enumeration (see module docstring).
    quantity: str | None
    # Event fields -- what that quantity was measured on, and of what.
    name: str | None
    property: str | None
    additional_details: str | None
    # Measurement fields -- `attribute` is forced to the same single-value
    # bucket the main pipeline uses (see module docstring). It's a
    # Literal, not a free field: structured decoding leaves the model no
    # other valid value, so there's nothing to prompt it about.
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
# NuExtract baseline fail at `create_model(...).model_json_schema()` before
# issuing a single request. No other config declares a Literal field, so this
# is the only one that needs it.
DirectExtractionItemSchema.model_rebuild()


_DIRECT_EXTRACTION_PROMPT = """Quantity Identification:
Find EVERY directly reported numerical quantity in the document -- measured values, counts, percentages, concentrations, dimensions, rates, durations -- whatever its subject matter (materials science, geology, biology, medicine, or any other domain). Then, for each quantity, describe what it measures.

Do not start from the subjects being studied and look for their numbers; start from the numbers and work outwards. Include statistical quantities -- p-values and significance thresholds, confidence and significance levels ("95%"), the numeric bounds of a reported confidence interval, correlations, effect sizes, odds and hazard ratios. Skip only numerals that are structural rather than measured: citation years, reference/figure/table/equation numbers, and page numbers.

For each quantity found, output one record with:
- quantity: the quantity exactly as it appears in the text, copied character-for-character, including its unit when one is written with it (e.g. "54.8 years", "5318", "12.3 mg/L"). Do not paraphrase, convert, or reformat.
- name: the SUBJECT that quantity was measured on or of -- a sample, specimen, site, material, organism, structure, population, or other concrete thing -- copied verbatim from the text. Set to None if the text attaches the quantity to no subject at all.
- property: the specific property or quantity type being measured for that subject (e.g. "mean annual temperature", "grain size", "paleolatitude"), copied verbatim from the text -- not the number itself. If the quantity is a bare count or attaches directly to its subject with no distinct property phrase (e.g. "5318 participants"), set this to None.
- additional_details: any qualifying context for this specific measurement (method, location, condition, date, comparison), copied or closely paraphrased from the text and kept to a short phrase. Set to None if not applicable.
- value: the numeric part of the quantity, exactly as reported. Do not convert, round, or combine with uncertainty bounds.
- units: the unit part of the quantity as reported, or None if it is unitless (e.g. a plain count or dimensionless ratio).

Rules:
- Output one record per reported quantity. Two different measurements that share the same numeral are two records; the same measurement restated in prose and in a table is one record.
- Do NOT infer, guess, or derive any field. Use ONLY information explicitly stated in the text.
- Do NOT extract vague, qualitative, or non-numeric statements.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "quantity": "...",
      "name": "...",
      "property": "...",
      "additional_details": "...",
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
# ChatExtract property phrase
#
# ChatExtract (measurementlm_chatextract.py) targets one known property per
# run and substitutes it into every prompt as "{prop}" (e.g. "What is the
# value of the {prop} in the following text?"). Left unset, that phrase falls
# back to the bare attribute_info_dict key, "measurement" -- grammatically
# degenerate ("a value of measurement"). This supplies a real noun phrase for
# that single bucket so the prompts read naturally.
#
# ChatExtract is the one arm the quantity-first redesign does NOT reach, and
# that is a property of the method rather than an oversight: it verifies a
# value against a property it was told about in advance, and has no step that
# discovers which property a number belongs to. Making it quantity-first would
# mean not implementing ChatExtract. `_make_record` in
# measurementlm_chatextract.py accordingly still emits `property: None` for
# every measeval record; matching against ground truth falls back to
# name + value + units alone (see analysis/ablation.py's `get_matching_rules`
# and data/measeval/README.md).
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
# GLiNER entity-field description
#
# GLiNER2 builds its structure's subject field description from
# `entity_type_description` by default ("The name or identifier of {desc} for
# which the {property} is reported"). Under the quantity-first design that
# field describes a QUANTITY, which would ask GLiNER for "the name or
# identifier of a directly reported numerical quantity" -- nonsense. GLiNER is
# a flat span tagger with no pipeline to invert, so it stays subject-centric
# (name + value + units, same as ChatExtract) and gets a subject-level
# description here instead. See DatasetConfig.gliner_entity_description.
# ---------------------------------------------------------------------------

_GLINER_ENTITY_DESCRIPTION = (
    "the concrete subject a measurement is made on -- a sample, specimen, site, "
    "material, compound, organism, structure, instrument, or population"
)


# ---------------------------------------------------------------------------
# Config instance
#
# No nuextract_examples: NuExtract-2.0-8B is vision-only (run_baseline_nuextract.py
# requires {data_dir}/processed_pdfs/, rendered from source PDFs by
# experiments/process_pdfs.py) and measeval ships plain text with gold
# character-offset annotations directly -- there are no PDFs to render, so
# that baseline can never run on this dataset (see data/measeval/README.md).
# ---------------------------------------------------------------------------

CONFIG = DatasetConfig(
    name="measeval",
    data_dir="data/measeval",
    metadata_file="data/measeval/directory.json",
    entity_schema=EntitySchema,
    entity_identification_prompt=ENTITY_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A single directly reported numerical quantity -- a measured value, count, "
        "percentage, concentration, dimension, rate, or other quantified characteristic -- "
        "copied verbatim from the text, including its unit when one is written with it."
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
    chatextract_property_names=_CHATEXTRACT_PROPERTY_NAMES,
    chatextract_entity_noun=_CHATEXTRACT_ENTITY_NOUN,
    gliner_entity_description=_GLINER_ENTITY_DESCRIPTION,
    # paper_subset: set to a list of document_id codes to restrict the run.
    paper_subset=None,
    # paper_filter: None processes all three splits (train+trial+eval) by default.
    # Set to `lambda m: m["source_split"] == "eval"` to restrict to the official
    # held-out MeasEval test split for leaderboard-comparable evaluation -- see
    # data/measeval/README.md's "Train/trial/eval and comparability" section.
    paper_filter=None,
    paper_exclude=None,
    # Ablation 2 (combined entity-attribute detection) isn't meaningful here:
    # with a single "measurement" bucket there is no interesting choice among
    # attributes to ablate, so it's left disabled.
    ablation2_entity_schema=None,
    ablation2_entity_identification_prompt=None,
    judge_filter_fields=None,
    ground_truth_file="data/measeval/ground_truth.json",
    # Units are open free text (Quantity.other["unit"]), not drawn from a fixed
    # catalog, so there is nothing to convert here -- see data/measeval/README.md's
    # "Attribute is free text, not a closed catalog" section.
    unit_conversion_table={},
)
