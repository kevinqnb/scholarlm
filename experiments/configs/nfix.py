"""
Dataset configuration for the nfix (aquatic dinitrogen fixation) dataset.

This is the single source of truth for all nfix-specific values: entity schema,
attribute catalogue, entity identification prompt, file paths, and paper filter.
All pipeline runners load this via importlib; the existing experiments/nfix/
scripts are unchanged.
"""
from __future__ import annotations

from pydantic import BaseModel

from scholarlm.config import DatasetConfig
from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class ObservationSchema(BaseModel):
    """Site-level entity fields for an aquatic dinitrogen fixation study."""

    name: str | None
    identifiers: str | None
    site_type: str | None
    location: str | None


class MeasurementEventSchema(BaseModel):
    """Event-level fields that distinguish individual dinitrogen fixation measurements."""

    date: str | None
    nfix_method: str | None
    substrate_type: str | None
    sample_depth: str | None
    additional_details: str | None


class DirectExtractionItemSchema(BaseModel):
    """Flat schema for Ablation 1: combines entity, event, attribute, value, and units."""

    # Entity fields
    name: str | None
    identifiers: str | None
    site_type: str | None
    location: str | None
    # Event fields
    date: str | None
    nfix_method: str | None
    substrate_type: str | None
    sample_depth: str | None
    additional_details: str | None
    # Measurement fields
    attribute: str
    value: str | None
    units: str | None


class Ablation2ObservationSchema(BaseModel):
    """Entity schema for Ablation 2: one item per (site, attribute) pair."""

    # Entity fields (same as ObservationSchema)
    name: str | None
    identifiers: str | None
    site_type: str | None
    location: str | None
    # Reserved fields required by Ablation 2
    attribute: str
    attribute_terms: list[str]


# ---------------------------------------------------------------------------
# Entity identification prompt
# ---------------------------------------------------------------------------

_IDENTIFICATION_PROMPT = """You are an expert in identifying and extracting information from scientific literature. Given the provided text (including any tables), extract identifying information for unique dinitrogen fixation measurement sites.

A dinitrogen fixation measurement site is a distinct physical location or ecosystem where dinitrogen fixation rates were measured. Multiple measurements at the same site (on different dates, using different methods, at different depths) should be represented as a single site record.


Response schema:
Site identifying information includes the following fields:
- name: the name of the site (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides (e.g. "Site 3", "L1") as the name.
- identifiers: every alternate short-form reference to this site used in the text — site codes, numeric tags, or shortened versions of the name — joined into a single string with semicolons separating each (e.g. "L1; Lake M.; Mend."). Collect these whenever the text uses them for the same site, even if the linkage is introduced only once (e.g. "Lake Mendota (LM)"). Do not include the primary name itself. If no alternatives exist, set to None.
- site_type: the type of site (e.g. continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy, etc.). This must be explicitly stated or clearly described in the text; do NOT infer it from the entity name alone.
- location: the general geographic location of the site.


Identification rules:
Treat sites with the same name as multiple separate items ONLY if their geographic location clearly differs. Do NOT create separate items for the same site because measurements were taken on different dates, using different methods, or at different depths — those distinctions will be captured separately as measurement events.


Strict rules about missing information:
- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If a field is not explicitly given, set its value to None.
- Do NOT infer site_type from the entity name.


Extraction procedure:
1. Scan the entire text, including all tables, table captions, and table footnotes, for any mentions of dinitrogen fixation measurement sites.
2. Determine which mentions correspond to distinct sites using the identification rules above.
3. For each distinct site, actively scan the full text — including table row and column headers, table captions, and table footnotes — for any alternate short-form references (codes, numeric tags, abbreviated names) that refer to it. Record all such identifiers in the identifiers field.
4. Output one JSON item per distinct site.
5. Collect all items into a single JSON array under the key "items".


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "identifiers": "...",
      "site_type": "...",
      "location": "..."
    }
  ]
}
- If no dinitrogen fixation measurement sites are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Measurement event prompt
# ---------------------------------------------------------------------------

_MEASUREMENT_EVENT_PROMPT = """Event fields:
- date: The date the measurement was taken. Use one of the following formats depending on available precision:
  - Full date: "dd-mm-yyyy"
  - Month and year only: "mm-yyyy"
  - Season and year: "Spring yyyy", "Summer yyyy", "Fall yyyy", or "Winter yyyy"
  - Year only: "yyyy"
  Set to None if no date is stated on this page.
- nfix_method: The method used to measure dinitrogen fixation (e.g., acetylene reduction assay, ARA, 15N2 incorporation). Set to None if not stated.
- substrate_type: The substrate on which the measurement was taken (e.g., water column, benthos). Set to None if not stated.
- sample_depth: The depth at which the sample was collected (e.g., "surface", "0-5 cm", "bottom", "0-10 m"). Set to None if not stated.
- additional_details: Any other distinguishing context not captured by the above fields (e.g., light vs. dark incubation, specific treatment condition). Keep this to one sentence or fewer. Set to None if not applicable.
"""


# ---------------------------------------------------------------------------
# Ablation 1: direct extraction prompt
# ---------------------------------------------------------------------------

_DIRECT_EXTRACTION_PROMPT = """Entity identification:
Extract all distinct dinitrogen fixation measurement sites mentioned in the document.

Entity fields:
- name: the name of the site (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides.
- identifiers: every alternate short-form reference to this site used in the text — site codes, numeric tags, or shortened versions of the name — joined into a single string with semicolons separating each (e.g. "L1; Lake M.; Mend."). Collect these whenever the text uses them for the same site, even if the linkage is introduced only once (e.g. "Lake Mendota (LM)"). Do not include the primary name itself. If no alternatives exist, set to None.
- site_type: the type of site (e.g., continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy). Must be explicitly stated; do NOT infer from the site name.
- location: the general geographic location of the site.

Entity identification rules:
- Treat sites as separate only if their geographic location clearly differs.
- Do NOT create separate items for the same site because measurements were taken on different dates, methods, or depths — those distinctions are captured as measurement events.
- Do NOT infer, guess, or derive any field value. Use ONLY information explicitly stated in the text. If a field is not explicitly given, set it to None.


Measurement event fields:
For each site and each detected attribute measurement, also identify the measurement event context:
- date: The date of the measurement. Formats: "dd-mm-yyyy", "mm-yyyy", "Spring/Summer/Fall/Winter yyyy", or "yyyy". Set to None if not stated.
- nfix_method: The method used to measure dinitrogen fixation (e.g., acetylene reduction assay, ARA, 15N2 incorporation). Set to None if not stated.
- substrate_type: The substrate on which the measurement was taken (e.g., water column, benthos). Set to None if not stated.
- sample_depth: The depth at which the sample was collected (e.g., "surface", "0-5 cm", "0-10 m"). Set to None if not stated.
- additional_details: Any other distinguishing context not captured above (e.g., light vs. dark incubation, specific treatment condition). One sentence or fewer. Set to None if not applicable.


Attributes to extract:
For each (site, measurement event) combination, extract values for any of the following attributes if directly measured and reported:

1. nfix_rate_mass — Rate of dinitrogen fixation per unit mass. NOT rates per area or volume. Units: nmol N g⁻¹ h⁻¹, nmol C2H4 g⁻¹ h⁻¹, nmol N2 g⁻¹ h⁻¹, µg N g⁻¹ d⁻¹, nmol N2 g⁻¹ d⁻¹, µmol N g⁻¹ d⁻¹, nmol C2H4 g⁻¹ d⁻¹, nmol N g⁻¹ d⁻¹, µg N g⁻¹ h⁻¹, µg N kg⁻¹ d⁻¹, µmol N g⁻¹ h⁻¹, fmol N g⁻¹ h⁻¹, ng N g⁻¹ d⁻¹, ng N g⁻¹ h⁻¹, nmol N kg⁻¹ h⁻¹, µmol C2H4 g⁻¹ d⁻¹, µmol N kg⁻¹ h⁻¹, µmol N2 g⁻¹ d⁻¹, or similar mass-normalized rate units.
2. nfix_rate_areal — Rate of dinitrogen fixation per unit area. NOT rates per mass or volume. Units: µmol N m⁻² h⁻¹, mg N m⁻² d⁻¹, µmol N m⁻² d⁻¹, µmol C2H4 m⁻² h⁻¹, nmol C2H4 cm⁻² h⁻¹, mmol N m⁻² d⁻¹, µg N m⁻² h⁻¹, mg N m⁻² h⁻¹, nmol C2H4 cm⁻² d⁻¹, nmol C2H4 m⁻² h⁻¹, µmol N2 m⁻² h⁻¹, g N m⁻² yr⁻¹, mmol N m⁻² h⁻¹, mmol N2 m⁻² d⁻¹, nmol N cm⁻² h⁻¹, µmol N2 m⁻² d⁻¹, kg N2 ha⁻¹ yr⁻¹, mg N m⁻² yr⁻¹, mg N2 m⁻² h⁻¹, ng N m⁻² h⁻¹, nmol C2H4 m⁻² d⁻¹, µg N cm⁻² h⁻¹, µg N2 m⁻² h⁻¹, or similar area-normalized rate units.
3. nfix_rate_volumetric — Rate of dinitrogen fixation per unit volume. NOT rates per mass or area. Units: nmol N L⁻¹ d⁻¹, nmol N L⁻¹ h⁻¹, nmol C2H4 L⁻¹ h⁻¹, µg N L⁻¹ h⁻¹, ng N L⁻¹ h⁻¹, mg N m⁻³ d⁻¹, nmol C2H4 cm⁻³ h⁻¹, nmol C2H4 mL⁻¹ h⁻¹, nmol N cm⁻³ d⁻¹, nmol N cm⁻³ h⁻¹, µg N m⁻³ h⁻¹, µmol N2 L⁻¹ d⁻¹, µmol N2 L⁻¹ h⁻¹, mmol C2H4 m⁻³ d⁻¹, nmol C2H4 cm⁻³ d⁻¹, nmol N m⁻³ h⁻¹, nmol N2 cm⁻³ d⁻¹, nmol N2 L⁻¹ d⁻¹, nmol N2 L⁻¹ h⁻¹, µg N L⁻¹ d⁻¹, µg N2 L⁻¹ h⁻¹, µg N2 m⁻³ d⁻¹, µmol C2H4 L⁻¹ d⁻¹, µmol C2H4 mL⁻¹ 3h⁻¹, µmol N L⁻¹ d⁻¹, µmol N L⁻¹ h⁻¹, or similar volume-normalized rate units.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "identifiers": "...",
      "site_type": "...",
      "location": "...",
      "date": "...",
      "nfix_method": "...",
      "substrate_type": "...",
      "sample_depth": "...",
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

_ABLATION2_IDENTIFICATION_PROMPT = """You are an expert in identifying dinitrogen fixation measurement sites referenced in scientific literature, and in detecting which measurement attributes are reported for each site. Given the provided text (including any tables), extract all distinct (measurement site, measured attribute) pairs for which a direct numerical measurement is reported.

A dinitrogen fixation measurement site is a distinct physical location or ecosystem where dinitrogen fixation rates were measured. Emit one item per (site, attribute) pair. If a site has multiple attributes measured, emit one item per attribute.

IMPORTANT: Only emit a pair when a direct numerical measurement exists in the document for that site and attribute. Do NOT emit pairs where the only data is qualitative, model parameters, or goodness-of-fit statistics.


Response schema:
For each (site, attribute) pair, output one item with the following fields:
- name: the name of the site (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides.
- identifiers: every alternate short-form reference to this site used in the text — site codes, numeric tags, or shortened versions of the name — joined into a single string with semicolons separating each (e.g. "L1; Lake M.; Mend."). Collect these whenever the text uses them for the same site, even if the linkage is introduced only once (e.g. "Lake Mendota (LM)"). Do not include the primary name itself. If no alternatives exist, set to None.
- site_type: the type of site (e.g., continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy). Must be explicitly stated; do NOT infer from the site name.
- location: the general geographic location of the site.
- attribute: the exact attribute name from the list below.
- attribute_terms: any terminology or abbreviations used in the document to refer to that attribute. Pay close attention to tables and figure captions. Do not infer, guess, or fabricate terms not explicitly present.


Attributes to detect (use these exact names in the attribute field):
1. nfix_rate_mass — Rate of dinitrogen fixation per unit mass. NOT rates per area or volume. Units normalized by substrate mass (e.g., nmol N g⁻¹ h⁻¹, µg N g⁻¹ d⁻¹, nmol C2H4 g⁻¹ h⁻¹, nmol N2 g⁻¹ h⁻¹, µmol N g⁻¹ h⁻¹, ng N g⁻¹ d⁻¹, fmol N g⁻¹ h⁻¹, nmol N kg⁻¹ h⁻¹, µmol C2H4 g⁻¹ d⁻¹, µmol N kg⁻¹ h⁻¹, µmol N2 g⁻¹ d⁻¹).
2. nfix_rate_areal — Rate of dinitrogen fixation per unit area. NOT rates per mass or volume. Units normalized by area (e.g., µmol N m⁻² h⁻¹, mg N m⁻² d⁻¹, nmol C2H4 cm⁻² h⁻¹, µmol C2H4 m⁻² h⁻¹, mmol N m⁻² d⁻¹, µg N m⁻² h⁻¹, nmol C2H4 m⁻² h⁻¹, µmol N2 m⁻² h⁻¹, g N m⁻² yr⁻¹, kg N2 ha⁻¹ yr⁻¹, ng N m⁻² h⁻¹).
3. nfix_rate_volumetric — Rate of dinitrogen fixation per unit volume. NOT rates per mass or area. Units normalized by water volume (e.g., nmol N L⁻¹ d⁻¹, nmol C2H4 L⁻¹ h⁻¹, µg N L⁻¹ h⁻¹, ng N L⁻¹ h⁻¹, mg N m⁻³ d⁻¹, nmol N cm⁻³ h⁻¹, µmol N2 L⁻¹ d⁻¹, mmol C2H4 m⁻³ d⁻¹, nmol N2 L⁻¹ d⁻¹, µmol C2H4 L⁻¹ d⁻¹, µmol N L⁻¹ h⁻¹).


Identification guidelines:
- Treat sites as separate only if their geographic location clearly differs (different coordinates or explicitly described as distinct locations). Do NOT create separate items for the same site because measurements were taken on different dates, methods, or depths.
- Multiple measurements of the same site for the same attribute should produce only one (site, attribute) pair — not one per measurement event.
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
      "site_type": "...",
      "location": "...",
      "attribute": "...",
      "attribute_terms": [...]
    }
  ]
}
- If no (site, attribute) pairs with direct numerical measurements are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Attribute catalogue
# ---------------------------------------------------------------------------

_MASS_UNITS = [
    "nmol N g⁻¹ h⁻¹", "nmol C2H4 g⁻¹ h⁻¹", "nmol N2 g⁻¹ h⁻¹", "µg N g⁻¹ d⁻¹",
    "nmol N2 g⁻¹ d⁻¹", "µmol N g⁻¹ d⁻¹", "nmol C2H4 g⁻¹ d⁻¹", "nmol N g⁻¹ d⁻¹",
    "µg N g⁻¹ h⁻¹", "µg N kg⁻¹ d⁻¹", "µmol N g⁻¹ h⁻¹", "fmol N g⁻¹ h⁻¹",
    "ng N g⁻¹ d⁻¹", "ng N g⁻¹ h⁻¹", "nmol N kg⁻¹ h⁻¹", "µmol C2H4 g⁻¹ d⁻¹",
    "µmol N kg⁻¹ h⁻¹", "µmol N2 g⁻¹ d⁻¹",
]

_AREAL_UNITS = [
    "µmol N m⁻² h⁻¹", "mg N m⁻² d⁻¹", "µmol N m⁻² d⁻¹", "µmol C2H4 m⁻² h⁻¹",
    "nmol C2H4 cm⁻² h⁻¹", "mmol N m⁻² d⁻¹", "µg N m⁻² h⁻¹", "mg N m⁻² h⁻¹",
    "nmol C2H4 cm⁻² d⁻¹", "nmol C2H4 m⁻² h⁻¹", "µmol N2 m⁻² h⁻¹", "g N m⁻² yr⁻¹",
    "mmol N m⁻² h⁻¹", "mmol N2 m⁻² d⁻¹", "nmol N cm⁻² h⁻¹", "µmol N2 m⁻² d⁻¹",
    "kg N2 ha⁻¹ yr⁻¹", "mg N m⁻² yr⁻¹", "mg N2 m⁻² h⁻¹", "ng N m⁻² h⁻¹",
    "nmol C2H4 m⁻² d⁻¹", "µg N cm⁻² h⁻¹", "µg N2 m⁻² h⁻¹",
]

_VOLUMETRIC_UNITS = [
    "nmol N L⁻¹ d⁻¹", "nmol N L⁻¹ h⁻¹", "nmol C2H4 L⁻¹ h⁻¹", "µg N L⁻¹ h⁻¹",
    "ng N L⁻¹ h⁻¹", "mg N m⁻³ d⁻¹", "nmol C2H4 cm⁻³ h⁻¹", "nmol C2H4 mL⁻¹ h⁻¹",
    "nmol N cm⁻³ d⁻¹", "nmol N cm⁻³ h⁻¹", "µg N m⁻³ h⁻¹", "µmol N2 L⁻¹ d⁻¹",
    "µmol N2 L⁻¹ h⁻¹", "mmol C2H4 m⁻³ d⁻¹", "nmol C2H4 cm⁻³ d⁻¹", "nmol N m⁻³ h⁻¹",
    "nmol N2 cm⁻³ d⁻¹", "nmol N2 L⁻¹ d⁻¹", "nmol N2 L⁻¹ h⁻¹", "µg N L⁻¹ d⁻¹",
    "µg N2 L⁻¹ h⁻¹", "µg N2 m⁻³ d⁻¹", "µmol C2H4 L⁻¹ d⁻¹", "µmol C2H4 mL⁻¹ h⁻¹",
    "µmol N L⁻¹ d⁻¹", "µmol N L⁻¹ h⁻¹",
]

_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
    "nfix_rate_mass": {
        "description": (
            "Rate of dinitrogen fixation per unit mass: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by substrate mass."
        ),
        "units": _MASS_UNITS,
    },
    "nfix_rate_areal": {
        "description": (
            "Rate of dinitrogen fixation per unit area: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by area."
        ),
        "units": _AREAL_UNITS,
    },
    "nfix_rate_volumetric": {
        "description": (
            "Rate of dinitrogen fixation per unit volume: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by water volume."
        ),
        "units": _VOLUMETRIC_UNITS,
    },
}


other_attributes = {
    "nfix_incubation_time": {
        "description": (
            "Duration of the experimental incubation for measuring dinitrogen fixation, "
            "from introduction of the tracer or substrate analog to termination and sampling."
        ),
        "units": ["minutes", "hours", "days"],
    },
    "nfix_incubation_temperature": {
        "description": (
            "Temperature at which the sample was held during the dinitrogen fixation incubation. "
            "Extract only if a specific numeric temperature is reported for the incubation itself. "
            "Do not extract in situ water temperatures unless the text explicitly states they equal "
            "the incubation temperature. If the text says only 'ambient temperature' or 'in situ "
            "temperature' without a numeric value, set to None."
        ),
        "units": ["°C", "K"],
    },
}


# ---------------------------------------------------------------------------
# Paper filter
# ---------------------------------------------------------------------------

# Excludes papers whose extraction_location indicates figures, supplements, archives, or author notes.
# This is a heuristic to exclude papers that are unlikely to contain extractable data.
def _nfix_paper_filter(metadata: dict) -> bool:
    """Exclude papers whose extraction_location indicates figures, supplements, archives, or author notes."""
    location = metadata.get("extraction_location", "")
    return not any(x in location for x in ["figure", "supplement", "archive", "author"])


# ---------------------------------------------------------------------------
# Config instance
# ---------------------------------------------------------------------------

_EXCLUDED_PAPERS = [
    "R95",  # Couldn't access 
    "R3", # Data in figures only
    "R51", # Data not found in paper
]

# Subset of 10 papers with the most data points:
_TOP_PAPERS = [
    "R163", "R164", "R172", "R248", "R124",
    "R51", "R59", "R114", "R43", "R103",
]

CONFIG = DatasetConfig(
    name="nfix",
    data_dir="data/nfix",
    metadata_file="data/nfix/directory.json",
    entity_schema=ObservationSchema,
    entity_identification_prompt=_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A distinct dinitrogen fixation measurement site — a specific ecosystem or location "
        "identified by name, type, and coordinates."
    ),
    attribute_info_dict=_ATTRIBUTE_INFO_DICT,
    measurement_event_schema=MeasurementEventSchema,
    measurement_event_prompt=_MEASUREMENT_EVENT_PROMPT,
    direct_extraction_schema=DirectExtractionItemSchema,
    direct_extraction_prompt=_DIRECT_EXTRACTION_PROMPT,
    # paper_subset: uncomment the line below to run only the 10-paper development set.
    # paper_subset=_DEV_SUBSET,
    paper_subset=None,
    paper_filter=_nfix_paper_filter,
    paper_exclude=_EXCLUDED_PAPERS,
    ablation2_entity_schema=Ablation2ObservationSchema,
    ablation2_entity_identification_prompt=_ABLATION2_IDENTIFICATION_PROMPT,
    judge_filter_fields=["location", "nfix_method", "substrate_type"],
    judge_instructions=JUDGE_INSTRUCTIONS,
    ground_truth_file="data/nfix/ground_truth_review.json",
)
