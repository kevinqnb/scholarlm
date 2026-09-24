"""
Dataset configuration for the supermat (superconductivity) dataset.

This is the single source of truth for all supermat-specific values: entity schema,
attribute catalogue, entity identification prompt, and file paths. All pipeline
runners (run_extraction, run_judge, run_analysis) load this via importlib.

Mapping onto the pipeline's entity/attribute/event model
----------------------------------------------------------
entity  = the superconducting material/sample
attribute = "tc" (superconducting critical temperature)
event = event_details, a free-text catch-all covering pressure, measurement
method, and any other condition under which a given Tc was measured
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from scholarlm.config import DatasetConfig


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class EntitySchema(BaseModel):
    """Entity fields for a distinct superconducting material or sample."""

    name: str | None
    identifiers: str | None
    # ``sample_details`` was removed 2026-09-20: doping/form/substrate/growth
    # info now belongs in the event_details catch-all (see
    # MeasurementEventSchema) instead of its own entity field.
    # ``identifiers`` is extracted by the real pipeline and its ablations only
    # -- see DatasetConfig.baseline_filter_fields (below) for the NuExtract
    # baselines; GLiNER already excludes it structurally (never listed in
    # gliner_field_descriptions) and ChatExtract's flat schema never included
    # it. It is also never shown to the judge (judge_filter_fields, below).


ENTITY_IDENTIFICATION_PROMPT = """You are an expert in identifying superconducting materials referenced in scientific literature. Given the provided text (including any tables), extract all distinct superconducting materials.

A superconducting material is a specific compound, chemical formula, or elemental substance for which superconductivity (or its absence) is discussed. The same base compound measured at different doping levels, pressures, or via different measurement methods should be represented as a single material record — those distinctions will be captured separately as measurement events. A different doping level, form, or growth condition of the SAME base compound is a modifier of that material, not a new entity. Only chemically distinct compounds (a different formula or stoichiometric family) are separate entities.


Response schema:
For each distinct material, output one item with the following fields:
- name: the material's name or chemical formula, as given in the text (e.g. "YBa2Cu3O7-δ", "MgB2", "mercury"). Use whatever primary identifier the paper provides — a full formula, a common compositional name, or an element name.
- identifiers: every alternate short-form reference to this material used in the text — abbreviations, sample codes, or shortened names — joined into a single string with semicolons separating each (e.g. "YBCO; Y-123"). Collect these whenever the text uses them for the same material, even if the linkage is introduced only once (e.g. "YBa2Cu3O7-δ (YBCO)"). Do not include the primary name itself. If no alternatives exist, set to None.


Identification guidelines:
Treat materials with the same base formula as multiple separate items ONLY if they are clearly described as chemically distinct compounds (different stoichiometric family or composition). Do NOT create separate items for the same compound because it was measured at different doping levels, under different pressures, or via different measurement methods — those distinctions, along with doping level, form, substrate, and growth condition, will be captured separately as measurement-event details.


Strict rules about missing information:
- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If a field is not explicitly given, set its value to None.


Extraction procedure:
1. Scan the entire text, including tables, table captions, and table footnotes, for any mentions of superconducting materials.
2. Determine which mentions correspond to distinct materials using the identification guidelines above.
3. For each distinct material, actively scan the full text for any alternate short-form references (abbreviations, sample codes) that refer to it. Record all such identifiers in the identifiers field.
4. Output one JSON item per distinct material.
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
- If no distinct materials are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Attribute schema
# ---------------------------------------------------------------------------

_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
    "tc": {
        "description": (
            "Superconducting critical temperature (Tc) — the temperature at which the "
            "material becomes superconducting, as reported for a specific measurement "
            "(onset, midpoint of the resistive transition, or zero-resistance criterion). "
            "This is NOT other transition temperatures (e.g. structural, magnetic, "
            "Curie/Néel) unless explicitly tied to the onset or loss of superconductivity, "
            "and NOT a Debye temperature, melting point, or other unrelated temperature."
        ),
        "units": ["K", "mK"],
    },
}


# ---------------------------------------------------------------------------
# Measurement schema
# ---------------------------------------------------------------------------

class MeasurementEventSchema(BaseModel):
    """Event-level fields that distinguish individual Tc measurements for a material."""

    event_details: str | None


_MEASUREMENT_EVENT_PROMPT = """EVENT FIELDS:
- event_details: A catch-all for whatever distinguishes this Tc measurement from another one for the same material — most commonly the applied pressure and the measurement method, but also the criterion used to define the value or any other distinguishing condition. Include, whenever stated:
  - Pressure: the applied pressure under which this Tc measurement was taken. Use "ambient" if the text states ambient/atmospheric pressure or no pressure is mentioned as a variable; otherwise report the stated pressure with its unit (e.g. "2 GPa", "500 GPa"). Unstated pressure defaults to "ambient" rather than being omitted.
  - Measurement method: map to one of these four categories whenever the text supports it: "resistivity" (resistance, R-T curve, ρ(T)), "magnetic susceptibility" (susceptibility, magnetization, AC susceptibility, M(T)), "specific heat" (heat capacity, C(T)), "theoretical calculation" (predicted/calculated values, e.g. Eliashberg theory). If the method is stated but doesn't fit any category, report it as given.
  - The criterion used to define this Tc value, if stated — for example "onset", "midpoint of resistive transition", or "zero resistance".
  - Doping level or fraction, crystal form, substrate, growth/treatment qualifiers, or any other distinguishing context (e.g., increasing/decreasing Tc trend) not captured above.
  The goal is that two Tc measurements for the same material that are genuinely distinct (different pressure, method, criterion, or sample condition) end up with different event_details, while two reports of the literal same measurement do not. Keep this to one sentence or fewer, joining multiple pieces of information with semicolons. Set to None only if none of the above is stated.
"""


# ---------------------------------------------------------------------------
# Ablation 1: direct extraction prompt
# ---------------------------------------------------------------------------


class DirectExtractionItemSchema(BaseModel):
    """Flat schema for Ablation 1: combines entity, event, attribute, value,
    units, and the qualifier/shape fields (the same shape
    MeasurementLM._parse_quantities() produces via a separate step -- see
    DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS)."""

    # Entity fields
    name: str | None
    identifiers: str | None
    # Event fields
    event_details: str | None
    # Measurement fields
    attribute: str
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


_DIRECT_EXTRACTION_PROMPT = """Entity Identification:
Extract all distinct superconducting materials (compounds, chemical formulas, or elemental substances) mentioned in the document.

Entity fields:
- name: the material's name or chemical formula, as given in the text (e.g. "YBa2Cu3O7-δ", "MgB2", "mercury").
- identifiers: every alternate short-form reference to this material used in the text — abbreviations, sample codes, or shortened names — joined into a single string with semicolons separating each (e.g. "YBCO; Y-123"). Do not include the primary name itself. If no alternatives exist, set to None.

Entity identification rules:
- Treat materials as separate only if they are chemically distinct compounds (different formula or stoichiometric family).
- Do NOT create separate items for the same compound measured at different doping levels, pressures, or methods — those are captured in the measurement event's event_details.
- Do NOT infer, guess, or derive any field value. Use ONLY information explicitly stated in the text. If a field is not explicitly given, set it to None.


Measurement event fields:
For each material and each detected Tc measurement, also identify the measurement event context:
- event_details: A catch-all for whatever distinguishes this Tc measurement from another one for the same material. Include, whenever stated:
  - Pressure: the applied pressure for this measurement. Use "ambient" if ambient/atmospheric or unstated; otherwise report the stated pressure with its unit (e.g. "2 GPa").
  - Measurement method: mapped to "resistivity", "magnetic susceptibility", "specific heat", or "theoretical calculation" whenever the text supports it; otherwise report as given.
  - The Tc-defining criterion (onset, midpoint, zero resistance), if stated.
  - Doping level or fraction, crystal form, substrate, growth/treatment qualifiers, or any other distinguishing context.
  Two genuinely distinct measurements should end up with different event_details. Keep this to one sentence or fewer, joining multiple pieces of information with semicolons. Set to None only if none of the above is stated.


Attributes to extract:
For each (material, measurement event) combination, extract a value for the following attribute if directly measured and reported:

1. tc — Superconducting critical temperature (Tc). NOT other transition temperatures (structural, magnetic, Curie/Néel) unless explicitly tied to superconductivity, and NOT Debye temperature or melting point. Units: K or mK.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "identifiers": "...",
      "event_details": "...",
      "attribute": "...",
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


# ---------------------------------------------------------------------------
# NuExtract-2.0-8B baseline: few-shot synthetic examples
#
# NuExtract's calling convention has no field for freeform instructions,
# only a JSON template and optional few-shot examples.
# Every output value below (except attribute, a fixed enum, and event_details,
# which joins several pieces of information with semicolons) is an exact
# substring of its input text, since NuExtract's verbatim-string fields are
# trained to copy spans rather than paraphrase. Together the three examples
# cover multiple entities per passage, multiple measurement events for the
# same entity (ambient vs. high pressure), three of the four measurement-method
# categories (resistivity, magnetic susceptibility, theoretical calculation),
# and all of point_value, lower/upper, list_values, tolerance, and
# standard_deviation at least once. No `identifiers` key: see
# DatasetConfig.baseline_filter_fields below -- these examples are
# baseline-only, so they never show the field a baseline shouldn't reproduce.
# ---------------------------------------------------------------------------

_NUEXTRACT_EXAMPLE_1_INPUT = (
    "Magnesium diboride (MgB2) is a simple binary compound that exhibits "
    "superconductivity. In resistivity measurements taken at ambient pressure, "
    "an onset transition was observed at 39 K. In the same study, "
    "YBa2Cu3O7-δ (also known as YBCO), a well-known cuprate superconductor, "
    "was examined using magnetic susceptibility measurements, which placed "
    "the midpoint of the diamagnetic transition at 92 K, also at ambient "
    "pressure."
)

_NUEXTRACT_QUANTITY_DEFAULTS = {
    "qualifiers": [], "point_value": None, "lower": None, "upper": None,
    "list_values": None, "tolerance": None, "standard_deviation": None,
}

_NUEXTRACT_EXAMPLE_1_OUTPUT = json.dumps(
    {
        "items": [
            {
                "name": "Magnesium diboride",
                "event_details": "ambient pressure; resistivity; onset",
                "attribute": "tc", "value": "39", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {"point_value": "39"}),
            },
            {
                "name": "YBa2Cu3O7-δ",
                "event_details": "ambient pressure; magnetic susceptibility; midpoint of the diamagnetic transition",
                "attribute": "tc", "value": "92", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {"point_value": "92"}),
            },
        ]
    }
)

# The 203 K measurement's added tolerance demonstrates a non-plain-point
# shape, so this baseline's only real instruction channel (few-shot examples
# -- see module docstring in measurementlm_nuextract.py) actually shows the
# qualifier fields in use, not just plain points.
_NUEXTRACT_EXAMPLE_2_INPUT = (
    "Hydrogen sulfide (H3S, sample S-2) is a polycrystalline sample that "
    "becomes superconducting under extreme compression. At a pressure of "
    "155 GPa, resistivity measurements showed zero resistance at 203 ± 2 K. "
    "When the pressure was increased to 200 GPa, the zero-resistance "
    "criterion shifted to 178 K in the same sample. Separately, a "
    "theoretical calculation using Eliashberg theory predicts a Tc of "
    "235 K for LaH10 at a pressure of 170 GPa."
)

_NUEXTRACT_EXAMPLE_2_OUTPUT = json.dumps(
    {
        "items": [
            {
                "name": "Hydrogen sulfide",
                "event_details": "155 GPa; resistivity; polycrystalline; zero resistance",
                "attribute": "tc", "value": "203 ± 2", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["HasTolerance"], "point_value": "203", "tolerance": "± 2",
                }),
            },
            {
                "name": "Hydrogen sulfide",
                "event_details": "200 GPa; resistivity; polycrystalline; zero-resistance criterion",
                "attribute": "tc", "value": "178", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {"point_value": "178"}),
            },
            {
                "name": "LaH10",
                "event_details": "170 GPa; theoretical calculation",
                "attribute": "tc", "value": "235", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {"point_value": "235"}),
            },
        ]
    }
)

# Rounds out shape coverage with lower/upper, list_values, and
# standard_deviation -- example 1 and 2 above only reach point_value and
# HasTolerance.
_NUEXTRACT_EXAMPLE_3_INPUT = (
    "Iron selenide (FeSe, sample F-4) is a layered superconductor. At "
    "ambient pressure, susceptibility measurements across three separate "
    "crystal batches gave critical temperatures of 8.0, 8.5, and 9.1 K. "
    "Under applied pressure of 6 GPa, resistivity measurements on the same "
    "sample placed the transition between 12 and 14 K. A separate "
    "polycrystalline sample of niobium nitride (NbN) showed a mean "
    "transition temperature of 16.2 (SD 0.5) K across five specimens, "
    "measured at ambient pressure using resistivity."
)

_NUEXTRACT_EXAMPLE_3_OUTPUT = json.dumps(
    {
        "items": [
            {
                "name": "Iron selenide",
                "event_details": "ambient pressure; magnetic susceptibility",
                "attribute": "tc", "value": "8.0, 8.5, and 9.1", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["IsList"], "list_values": ["8.0", "8.5", "9.1"],
                }),
            },
            {
                "name": "Iron selenide",
                "event_details": "6 GPa; resistivity",
                "attribute": "tc", "value": "12 and 14", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["IsRange"], "lower": "12", "upper": "14",
                }),
            },
            {
                "name": "Niobium nitride",
                "event_details": "ambient pressure; resistivity; polycrystalline; across five specimens",
                "attribute": "tc", "value": "16.2 (SD 0.5)", "units": "K",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["IsMean", "HasSD"], "point_value": "16.2", "standard_deviation": "0.5",
                }),
            },
        ]
    }
)

_NUEXTRACT_EXAMPLES = [
    {"input": _NUEXTRACT_EXAMPLE_1_INPUT, "output": _NUEXTRACT_EXAMPLE_1_OUTPUT},
    {"input": _NUEXTRACT_EXAMPLE_2_INPUT, "output": _NUEXTRACT_EXAMPLE_2_OUTPUT},
    {"input": _NUEXTRACT_EXAMPLE_3_INPUT, "output": _NUEXTRACT_EXAMPLE_3_OUTPUT},
]


# ---------------------------------------------------------------------------
# Ablation 2: combined entity-attribute extraction prompt
# ---------------------------------------------------------------------------

class Ablation2ObservationSchema(BaseModel):
    """Entity schema for Ablation 2: one item per (material, attribute) pair."""

    # Entity fields (same as EntitySchema)
    name: str | None
    identifiers: str | None
    # Reserved fields required by Ablation 2
    attribute: str
    attribute_terms: list[str]


_ABLATION2_IDENTIFICATION_PROMPT = """You are an expert in identifying superconducting materials referenced in scientific literature, and in detecting which measurement attributes are reported for each material. Given the provided text (including any tables), extract all distinct (material, measured attribute) pairs for which a direct numerical measurement is reported.

A superconducting material is a specific compound, chemical formula, or elemental substance. Emit one item per (material, attribute) pair.

IMPORTANT: Only emit a pair when a direct numerical measurement exists in the document for that material and attribute. Do NOT emit pairs where the only data is qualitative, model parameters, or goodness-of-fit statistics.


Response schema:
For each (material, attribute) pair, output one item with the following fields:
- name: the material's name or chemical formula, as given in the text.
- identifiers: every alternate short-form reference to this material used in the text, joined into a single string with semicolons separating each. Do not include the primary name itself. If no alternatives exist, set to None.
- attribute: the exact attribute name from the list below.
- attribute_terms: any terminology or abbreviations used in the document to refer to that attribute (e.g. "Tc", "transition temperature"). Pay close attention to tables and figure captions. Do not infer, guess, or fabricate terms not explicitly present.


Attributes to detect (use these exact names in the attribute field):
1. tc — Superconducting critical temperature (Tc). NOT other transition temperatures (structural, magnetic, Curie/Néel) unless explicitly tied to superconductivity, and NOT Debye temperature or melting point.


Identification guidelines:
- Treat materials as separate only if they are chemically distinct compounds (different formula or stoichiometric family). Do NOT create separate items for the same compound measured at different doping levels, pressures, or methods.
- Multiple measurements of the same material for the same attribute should produce only one (material, attribute) pair — not one per measurement event.
- Do NOT infer, guess, or derive any identifying information. Use ONLY information explicitly stated in the text. If a field is not explicitly given, set its value to None.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "identifiers": "...",
      "attribute": "...",
      "attribute_terms": [...]
    }
  ]
}
- If no (material, attribute) pairs with direct numerical measurements are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# ChatExtract baseline: per-attribute <PROPERTY> phrase
#
# ChatExtract is a single-property method; each prompt reads "...a value of
# <PROPERTY>...". Left unset, this falls back to the bare attribute_info_dict
# key, "tc" -- grammatically degenerate ("a value of tc"). This supplies a
# real noun phrase, matching the lead clause of _ATTRIBUTE_INFO_DICT["tc"]'s
# own description.
# ---------------------------------------------------------------------------

_CHATEXTRACT_PROPERTY_NAMES: dict[str, str] = {
    "tc": "superconducting critical temperature",
}

# ChatExtract's reference prompts ask about a "material"/"compound" -- this
# happens to already match supermat's own entities, but is set explicitly
# here for consistency with the other dataset configs (see
# DatasetConfig.chatextract_entity_noun).
_CHATEXTRACT_ENTITY_NOUN = "material"

# ---------------------------------------------------------------------------
# GLiNER2 baseline: per-field descriptions for entity/event fields beyond the
# subject name (see DatasetConfig.gliner_field_descriptions). Copied verbatim
# from _DIRECT_EXTRACTION_PROMPT's own per-field bullets above -- GLiNER sees
# the same wording Ablation 1 already uses, not freshly authored text.
# ``identifiers`` has no entry here: it's an alias-resolution aid for the real
# pipeline's entity matching, not reported content, so GLiNER never asks for it.
# ---------------------------------------------------------------------------

_GLINER_FIELD_DESCRIPTIONS: dict[str, str] = {
    "event_details": (
        "A catch-all for whatever distinguishes this Tc measurement from "
        "another one for the same material -- most commonly the applied "
        'pressure (use "ambient" if ambient/atmospheric or unstated, otherwise '
        'the stated pressure with its unit, e.g. "2 GPa") and the measurement '
        'method (mapped to "resistivity", "magnetic susceptibility", "specific '
        'heat", or "theoretical calculation" whenever the text supports it), '
        "but also the Tc-defining criterion (onset, midpoint, zero resistance), "
        "doping level, crystal form, substrate, growth/treatment qualifiers, or "
        "any other distinguishing context. Two genuinely distinct measurements "
        "should end up with different event_details. One sentence or fewer, "
        "joining multiple pieces of information with semicolons. Set to None "
        "only if none of the above is stated."
    ),
}


# ---------------------------------------------------------------------------
# Config instance
# ---------------------------------------------------------------------------

# Note: raw_data.csv references 5 additional filename codes with no local PDF.
# 3 turned out to be duplicate-DOI registrations of a paper already present under
# a different code, and are merged into that document_id in preprocessing.py; the
# other 2 are genuinely missing and their rows are dropped. None of the 5 ever
# appear in directory.json or as an OCR/PDF filename, so there is nothing to
# exclude here -- see data/supermat/preprocessing.py for the full reconciliation.

CONFIG = DatasetConfig(
    name="supermat",
    data_dir="data/supermat",
    metadata_file="data/supermat/directory.json",
    entity_schema=EntitySchema,
    entity_identification_prompt=ENTITY_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A distinct superconducting material or sample — a specific compound, "
        "chemical formula, or elemental substance."
    ),
    attribute_info_dict=_ATTRIBUTE_INFO_DICT,
    measurement_event_schema=MeasurementEventSchema,
    measurement_event_prompt=_MEASUREMENT_EVENT_PROMPT,
    direct_extraction_schema=DirectExtractionItemSchema,
    direct_extraction_prompt=_DIRECT_EXTRACTION_PROMPT,
    nuextract_examples=_NUEXTRACT_EXAMPLES,
    chatextract_property_names=_CHATEXTRACT_PROPERTY_NAMES,
    chatextract_entity_noun=_CHATEXTRACT_ENTITY_NOUN,
    gliner_field_descriptions=_GLINER_FIELD_DESCRIPTIONS,
    # identifiers is extracted by the real pipeline and its ablations only --
    # see EntitySchema's comment above; excluded here from the NuExtract
    # baselines specifically (GLiNER/ChatExtract already never see it).
    baseline_filter_fields=["identifiers"],
    paper_subset=None,
    paper_filter=None,
    paper_exclude=None,
    ablation2_entity_schema=Ablation2ObservationSchema,
    ablation2_entity_identification_prompt=_ABLATION2_IDENTIFICATION_PROMPT,
    # Judge sees only: name (+ attribute, value, units). identifiers is an
    # alias-resolution aid, not something to judge on; event_details is a
    # catch-all for distinguishing measurements from each other, not the main
    # extraction interest, so it's kept out of the judge prompt too.
    judge_filter_fields=["identifiers", "event_details"],
    ground_truth_file="data/supermat/ground_truth.json",
    # Matching rules for the id-addressed evaluation path (analysis/match_cache.py,
    # analysis/recovery_validity.py) -- see DatasetConfig's docstring for scope
    # and why these are allowed to diverge from analysis/ablation.py's own
    # get_matching_rules (converted_value, not point_value; no legacy history
    # here to stay comparable with).
    strict_matching={
        "document_id": "document_id",
        "attribute": "attribute",
        "units": "units",
    },
    fuzzy_matching={
        "name": "name",
        "value": "value"
    },
    fuzzy_threshold=0.625,
)
