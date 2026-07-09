"""
Dataset configuration for the supermat (superconductivity) dataset.

This is the single source of truth for all supermat-specific values: entity schema,
attribute catalogue, entity identification prompt, and file paths. All pipeline
runners (run_extraction, run_judge, run_analysis) load this via importlib.

Mapping onto the pipeline's entity/attribute/event model
----------------------------------------------------------
SuperMat's own <material>/<class>/<tc>/<tcValue>/<pressure>/<me_method> tags don't
map 1:1 onto entity/attribute/event, but the reason a Tc value differs between two
rows for the same material is always "different pressure or measurement method" --
exactly what the event-resolution step is for:

    entity    = the superconducting material/sample
    attribute = "tc" (superconducting critical temperature) -- the only attribute
                in this dataset
    event     = pressure + measurement method (the conditions under which a given
                Tc was measured)
"""
from __future__ import annotations

from pydantic import BaseModel

from scholarlm.config import DatasetConfig


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class EntitySchema(BaseModel):
    """Entity fields for a distinct superconducting material or sample."""

    name: str | None
    identifiers: str | None
    sample_details: str | None


ENTITY_IDENTIFICATION_PROMPT = """You are an expert in identifying superconducting materials referenced in scientific literature. Given the provided text (including any tables), extract all distinct superconducting materials.

A superconducting material is a specific compound, chemical formula, or elemental substance for which superconductivity (or its absence) is discussed. The same base compound measured at different doping levels, pressures, or via different measurement methods should be represented as a single material record — those distinctions will be captured separately as measurement events. A different doping level, form, or growth condition of the SAME base compound is a modifier of that material, not a new entity. Only chemically distinct compounds (a different formula or stoichiometric family) are separate entities.


Response schema:
For each distinct material, output one item with the following fields:
- name: the material's name or chemical formula, as given in the text (e.g. "YBa2Cu3O7-δ", "MgB2", "mercury"). Use whatever primary identifier the paper provides — a full formula, a common compositional name, or an element name.
- identifiers: every alternate short-form reference to this material used in the text — abbreviations, sample codes, or shortened names — joined into a single string with semicolons separating each (e.g. "YBCO; Y-123"). Collect these whenever the text uses them for the same material, even if the linkage is introduced only once (e.g. "YBa2Cu3O7-δ (YBCO)"). Do not include the primary name itself. If no alternatives exist, set to None.
- sample_details: doping level or fraction (e.g. "x = 0.10", "optimally doped", "20%-doped"), crystal form (single crystal, polycrystalline, powder, thin film), substrate (e.g. "grown on MgO(100)"), and growth/treatment qualifiers (as-grown, annealed, untwinned) explicitly stated for this material. Set to None if no such details are given.


Identification guidelines:
Treat materials with the same base formula as multiple separate items ONLY if they are clearly described as chemically distinct compounds (different stoichiometric family or composition). Do NOT create separate items for the same compound because it was measured at different doping levels, under different pressures, or via different measurement methods — those distinctions will be captured separately as measurement events, and doping level belongs in sample_details, not as a new entity.


Strict rules about missing information:
- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If a field is not explicitly given, set its value to None.
- Do NOT infer sample_details from the material name alone.


Extraction procedure:
1. Scan the entire text, including tables, table captions, and table footnotes, for any mentions of superconducting materials.
2. Determine which mentions correspond to distinct materials using the identification guidelines above.
3. For each distinct material, actively scan the full text for any alternate short-form references (abbreviations, sample codes) that refer to it. Record all such identifiers in the identifiers field.
4. Record any doping, form, substrate, or growth details explicitly stated for the material in sample_details.
5. Output one JSON item per distinct material.
6. Collect all items into a single JSON array under the key "items".


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "identifiers": "...",
      "sample_details": "..."
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

    pressure: str | None
    me_method: str | None
    additional_details: str | None


_MEASUREMENT_EVENT_PROMPT = """EVENT FIELDS:
- pressure: The applied pressure under which this Tc measurement was taken. Use "ambient" if the text states ambient/atmospheric pressure or no pressure is mentioned as a variable. Otherwise report the stated pressure with its unit (e.g. "2 GPa", "500 GPa"). Set to None only if pressure is genuinely ambiguous (not simply unstated — unstated pressure defaults to "ambient").
- me_method: The measurement method used to determine this Tc value. Map to one of these four categories whenever the text supports it: "resistivity" (resistance, R-T curve, ρ(T)), "magnetic susceptibility" (susceptibility, magnetization, AC susceptibility, M(T)), "specific heat" (heat capacity, C(T)), "theoretical calculation" (predicted/calculated values, e.g. Eliashberg theory). If the method is stated but doesn't fit any category, report it as given. Set to None if not stated.
- additional_details: The criterion used to define this Tc value, if stated — for example "onset", "midpoint of resistive transition", or "zero resistance". Include any other distinguishing context not captured by the fields above (e.g., increasing/decreasing Tc trend). Keep this to one sentence or fewer. Set to None if not applicable.
"""


# ---------------------------------------------------------------------------
# Ablation 1: direct extraction prompt
# ---------------------------------------------------------------------------


class DirectExtractionItemSchema(BaseModel):
    """Flat schema for Ablation 1: combines entity, event, attribute, value, and units."""

    # Entity fields
    name: str | None
    identifiers: str | None
    sample_details: str | None
    # Event fields
    pressure: str | None
    me_method: str | None
    additional_details: str | None
    # Measurement fields
    attribute: str
    value: str | None
    units: str | None


_DIRECT_EXTRACTION_PROMPT = """Entity Identification:
Extract all distinct superconducting materials (compounds, chemical formulas, or elemental substances) mentioned in the document.

Entity fields:
- name: the material's name or chemical formula, as given in the text (e.g. "YBa2Cu3O7-δ", "MgB2", "mercury").
- identifiers: every alternate short-form reference to this material used in the text — abbreviations, sample codes, or shortened names — joined into a single string with semicolons separating each (e.g. "YBCO; Y-123"). Do not include the primary name itself. If no alternatives exist, set to None.
- sample_details: doping level or fraction, crystal form (single crystal, polycrystalline, powder, thin film), substrate, and growth/treatment qualifiers explicitly stated for this material. Set to None if not given.

Entity identification rules:
- Treat materials as separate only if they are chemically distinct compounds (different formula or stoichiometric family).
- Do NOT create separate items for the same compound measured at different doping levels, pressures, or methods — doping belongs in sample_details; pressure/method are captured as measurement events.
- Do NOT infer, guess, or derive any field value. Use ONLY information explicitly stated in the text. If a field is not explicitly given, set it to None.


Measurement event fields:
For each material and each detected Tc measurement, also identify the measurement event context:
- pressure: The applied pressure for this measurement. Use "ambient" if ambient/atmospheric or unstated; otherwise report the stated pressure with its unit (e.g. "2 GPa").
- me_method: The measurement method, mapped to "resistivity", "magnetic susceptibility", "specific heat", or "theoretical calculation" whenever the text supports it; otherwise report as given. Set to None if not stated.
- additional_details: The Tc-defining criterion (onset, midpoint, zero resistance) if stated, plus any other distinguishing context. One sentence or fewer. Set to None if not applicable.


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
      "sample_details": "...",
      "pressure": "...",
      "me_method": "...",
      "additional_details": "...",
      "attribute": "...",
      "value": "...",
      "units": "..."
    }
  ]
}
- If no measurements are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Ablation 2: combined entity-attribute extraction prompt
# ---------------------------------------------------------------------------

class Ablation2ObservationSchema(BaseModel):
    """Entity schema for Ablation 2: one item per (material, attribute) pair."""

    # Entity fields (same as EntitySchema)
    name: str | None
    identifiers: str | None
    sample_details: str | None
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
- sample_details: doping level or fraction, crystal form, substrate, and growth/treatment qualifiers explicitly stated for this material. Set to None if not given.
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
      "sample_details": "...",
      "attribute": "...",
      "attribute_terms": [...]
    }
  ]
}
- If no (material, attribute) pairs with direct numerical measurements are found, output exactly:
{ "items": [] }
"""


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
    paper_subset=None,
    paper_filter=None,
    paper_exclude=None,
    ablation2_entity_schema=Ablation2ObservationSchema,
    ablation2_entity_identification_prompt=_ABLATION2_IDENTIFICATION_PROMPT,
    judge_filter_fields=None,
    ground_truth_file="data/supermat/ground_truth.json",
)
