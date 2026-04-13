"""
Dataset configuration for the pond (aquatic ecosystem) dataset.

This is the single source of truth for all pond-specific values: entity schema,
attribute catalogue, entity identification prompt, and file paths.  All pipeline
runners (run_extraction, run_judge, run_analysis) load this via importlib; the
existing experiments/pond/ scripts are unchanged and continue to import their
own copies of the prompts from extract_prompts.py.
"""
from __future__ import annotations

from pydantic import BaseModel

from scholarlm.config import DatasetConfig


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class ObservationSchema(BaseModel):
    """Entity fields for a distinct pond / lake / wetland ecosystem."""

    name: str | None
    abbreviations: str | None
    location: str | None
    ecosystem: str | None


class MeasurementEventSchema(BaseModel):
    """Event-level fields that distinguish individual measurements within a pond ecosystem."""

    date: str | None
    additional_details: str | None


class DirectExtractionItemSchema(BaseModel):
    """Flat schema for Ablation 6: combines entity, event, attribute, value, and units."""

    # Entity fields
    name: str | None
    abbreviations: str | None
    location: str | None
    ecosystem: str | None
    # Event fields
    date: str | None
    additional_details: str | None
    # Measurement fields
    attribute: str
    value: str | None
    units: str | None


# ---------------------------------------------------------------------------
# Entity identification prompt
# ---------------------------------------------------------------------------

_IDENTIFICATION_PROMPT = """You are an expert in identifying ponds, lakes, and wetlands referenced in scientific literature. Given the provided text (including any tables), extract all distinct aquatic ecosystems.

An aquatic ecosystem is a specific pond, lake, wetland, or similar water body. Multiple measurements of the same ecosystem (on different dates, under different treatment conditions, at different sites within the ecosystem) should be represented as a single ecosystem record.


What counts as an aquatic ecosystem?:
- Include ponds, lakes, wetlands, and similar aquatic ecosystems.
- Marshes, bogs, fens, and swamps should all be considered as "wetland".
- If the ecosystem type is unclear, classify it as "other".


Response schema:
For each distinct ecosystem, output one item with the following fields:
- name: the name of the ecosystem (e.g. "Lake Mendota", "Beaver Pond"). If no full name is given, use whatever primary identifier the paper provides.
- abbreviations: any secondary codes or abbreviations used in the text to refer to the same ecosystem (e.g. "L1", "Lake M.", "Mend."). If the primary identifier is already a code with no alternatives, set to None.
- location: the general geographic location of the ecosystem (e.g. "central Wisconsin", "Ontario, Canada"), if explicitly stated.
- ecosystem: the ecosystem type (e.g. "pond", "lake", "wetland", "other").

NOTE: While an ecosystem might be introduced by its full name (e.g., "Lake Mendota"), many papers use numerical or coded identifiers and abbreviations (e.g. "L1", "Lake 1", "Lake M.", "Mend.") to refer to the same ecosystem later on. It is very important that these identifiers are collected and reported in the "abbreviations" field.

NOTE: Site names may appear in row or column headers in tables. Location metadata may be encoded in table captions or table footnotes. Check all of these when identifying ecosystems.


Identification guidelines:
Treat ecosystems with the same name as multiple separate items ONLY if they are clearly described as distinct physical water bodies (e.g., two ponds at different locations explicitly named separately). Do NOT create separate items for the same ecosystem because measurements were taken on different dates, at different sub-sites, or under different treatment conditions — those distinctions will be captured separately as measurement events.


Strict rules about missing information:
- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If a field is not explicitly given, set its value to None.
- Do NOT infer the ecosystem type from the entity name.
- Do NOT infer coordinates from general geographic descriptions.


Extraction procedure:
1. Scan the entire text, including tables, for any mentions of specific ponds, lakes, wetlands, or coded sites.
2. Determine which mentions correspond to distinct physical ecosystems using the identification guidelines above.
3. Output one JSON item per distinct ecosystem.
4. Collect all items into a single JSON array under the key "items".


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "abbreviations": "...",
      "location": "...",
      "ecosystem": "..."
    }
  ]
}
- If no distinct ecosystems are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Measurement event prompt
# ---------------------------------------------------------------------------

_MEASUREMENT_EVENT_PROMPT = """EVENT FIELDS:
- date: The date the measurement was taken. Use one of the following formats depending on available precision:
  - Full date: "dd-mm-yyyy"
  - Month and year only: "mm-yyyy"
  - Season and year: "Spring yyyy", "Summer yyyy", "Fall yyyy", or "Winter yyyy"
  - Year only: "yyyy"
  Set to None if no date is stated on this page.
- additional_details: Any other distinguishing context not captured by date — for example, treatment site or sub-site (e.g., "inlet zone", "P1"), treatment state (e.g., "restored", "control", "fertilized"), or other sampling conditions. Keep this to one sentence or fewer. Set to None if not applicable.
"""

# ---------------------------------------------------------------------------
# Attribute catalogue
# ---------------------------------------------------------------------------

_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
    "surface_area": {
        "description": (
            "Surface area of the water body itself, representing the horizontal area of open water "
            "or the stated ecosystem boundary at the time of measurement or description. This is NOT "
            "the same as watershed area, drainage basin area, catchment area, or littoral zone area."
        ),
        "units": ["km^2", "mi^2", "ha", "m^2", "acres"],
    },
    "max_depth": {
        "description": (
            "Maximum physical water depth of the ecosystem, defined as the deepest point of the water "
            "body at the time of measurement or as reported in the source. This is NOT the same as "
            "mean depth, average depth, or Secchi depth (water transparency depth)."
        ),
        "units": ["m", "km", "ft"],
    },
    "vegetation_cover": {
        "description": (
            "Fraction or percentage of the ecosystem surface area covered by aquatic macrophytes or "
            "other rooted/floating aquatic vegetation. This should refer to areal coverage, not biomass "
            "or volume. This is NOT the same as algal cover, periphyton cover, or phytoplankton density."
        ),
        "units": ["percent", "fraction"],
    },
    "ph": {
        "description": (
            "pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a "
            "dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH."
        ),
        "units": [],
    },
    "tn": {
        "description": (
            "Total nitrogen (TN) concentration in the water column, representing the sum of all nitrogen "
            "forms — both dissolved and particulate, including nitrate (NO3-), nitrite (NO2-), ammonium "
            "(NH3/NH4+), and organic nitrogen. This must be the aggregate 'total nitrogen' value as "
            "explicitly reported in the source. This is NOT the same as individual nitrogen species "
            "(e.g., NO3- alone, NO2- alone, NH3 alone, combined NO3-+NO2-, or particulate organic nitrogen "
            "[PON]) unless they are explicitly labeled as total nitrogen."
        ),
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"],
    },
    "tp": {
        "description": (
            "Total phosphorus (TP) concentration in the water column, representing the sum of all "
            "phosphorus forms — both dissolved and particulate. This must be the aggregate 'total "
            "phosphorus' value as explicitly reported in the source. This is NOT the same as individual "
            "phosphorus species (e.g., soluble reactive phosphorus [SRP], orthophosphate [PO4(3-)], "
            "dissolved reactive phosphorus [DRP], or particulate phosphorus [PP]) unless they are "
            "explicitly labeled as total phosphorus."
        ),
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"],
    },
    "chla": {
        "description": (
            "Chlorophyll-a (Chl-a) concentration in the water column, used as a proxy for phytoplankton "
            "biomass. This should refer to extracted or in situ chlorophyll-a measurements only. This is "
            "NOT the same as total chlorophyll, chlorophyll-b, chlorophyll-c, pheophytin, or other "
            "pigment measurements unless they are explicitly labeled as chlorophyll-a."
        ),
        "units": ["µg/L", "mg/L", "mg/m^3"],
    },
}

other_attributes = {
    "latitude": {
        "description": (
            "Geographic latitude of the ecosystem location, expressed in a standard geographic "
            "coordinate system (e.g., WGS84). This should refer to the centroid or stated reference "
            "point of the ecosystem, not a bounding box or region."
        ),
        "units": ["degrees", "radians"],
    },
    "longitude": {
        "description": (
            "Geographic longitude of the ecosystem location, expressed in a standard geographic "
            "coordinate system (e.g., WGS84). This should refer to the centroid or stated reference "
            "point of the ecosystem, not a bounding box or region."
        ),
        "units": ["degrees", "radians"],
    },
}

# ---------------------------------------------------------------------------
# Direct extraction prompt (Ablation 6)
# ---------------------------------------------------------------------------

_DIRECT_EXTRACTION_PROMPT = """Entity Identification:
Extract all distinct aquatic ecosystems (ponds, lakes, wetlands, and similar water bodies) mentioned in the document.

Entity fields:
- name: the name of the ecosystem (e.g. "Lake Mendota", "Beaver Pond"). If no full name is given, use whatever primary identifier the paper provides.
- abbreviations: any secondary codes or abbreviations used to refer to the same ecosystem (e.g. "L1", "Lake M.", "Mend."). Set to None if the primary identifier has no alternatives.
- location: the general geographic location of the ecosystem, if explicitly stated.
- ecosystem: the ecosystem type ("pond", "lake", "wetland", or "other").

Entity identification rules:
- Treat ecosystems as separate only if they are clearly distinct physical water bodies.
- Do NOT create separate items for the same ecosystem because measurements were taken on different dates or conditions — those distinctions are captured as measurement events.
- Do NOT infer, guess, or derive any field value. Use ONLY information explicitly stated in the text. If a field is not explicitly given, set it to None.


Measurement event fields:
For each ecosystem and each detected attribute measurement, also identify the measurement event context:
- date: The date of the measurement. Formats: "dd-mm-yyyy", "mm-yyyy", "Spring/Summer/Fall/Winter yyyy", or "yyyy". Set to None if not stated.
- additional_details: Any other distinguishing context not captured by date (e.g., treatment site, treatment state, sampling conditions). One sentence or fewer. Set to None if not applicable.


Attributes to extract:
For each (ecosystem, measurement event) combination, extract values for any of the following attributes if directly measured and reported:

1. surface_area — Surface area of the water body (NOT watershed, catchment, or littoral zone area). Units: km^2, mi^2, ha, m^2, or acres.
2. max_depth — Maximum physical water depth (NOT mean depth, average depth, or Secchi depth). Units: m, km, or ft.
3. vegetation_cover — Fraction or percentage of the ecosystem surface covered by aquatic macrophytes or rooted/floating vegetation (NOT algal cover, periphyton, or phytoplankton). Units: percent or fraction.
4. ph — pH of the water (dimensionless). NOT soil or sediment pH.
5. tn — Total nitrogen (TN): the aggregate sum of ALL nitrogen forms — dissolved and particulate. NOT individual species (NO3-, NO2-, NH3, etc.) unless explicitly labeled as total nitrogen. Units: µg/L, mg/L, μmol/L, ppm, or ppb.
6. tp — Total phosphorus (TP): the aggregate sum of ALL phosphorus forms. NOT individual species (SRP, PO4(3-), DRP, PP, etc.) unless explicitly labeled as total phosphorus. Units: µg/L, mg/L, μmol/L, ppm, or ppb.
7. chla — Chlorophyll-a (Chl-a) concentration. NOT total chlorophyll, chlorophyll-b, pheophytin, or other pigments unless explicitly labeled as chlorophyll-a. Units: µg/L, mg/L, or mg/m^3.


Output format requirements:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "abbreviations": "...",
      "location": "...",
      "ecosystem": "...",
      "date": "...",
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

other_things = """
1. latitude — Geographic latitude of the ecosystem location. Units: degrees or radians.
2. longitude — Geographic longitude of the ecosystem location. Units: degrees or radians.
"""

# ---------------------------------------------------------------------------
# Config instance
# ---------------------------------------------------------------------------

# Development subset used in early experiments
_DEV_SUBSET = [
    'physical_and_chemical_limnological',
    'physical-chemical_influences',
    'prairie_wetland',
    'net_heterotrophy',
    'habitat_characteristics',
    'biodiversity_of_constructed',
    'fish_production_in_lakes',
    'long-term_stability',
    'diversity_of_macroinvertebrates',
    'impact_of_macrophytes'
]

_TOP_PAPERS = [
    'classification_trees',
    'physical-chemical_influences',
    'habitat_characteristics',
    'physical_and_chemical_limnological',
    'prairie_wetland',
    'macroinvertebrate_size',
    'relationships_between_fish',
    'net_heterotrophy',
    'impact_of_macrophytes',
    'environmental_conditions',
]

CONFIG = DatasetConfig(
    name="pond",
    data_dir="data/pond",
    metadata_file="data/pond/directory.json",
    entity_schema=ObservationSchema,
    entity_identification_prompt=_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A distinct aquatic ecosystem — a specific pond, lake, wetland, or similar water body."
    ),
    attribute_info_dict=_ATTRIBUTE_INFO_DICT,
    measurement_event_schema=MeasurementEventSchema,
    measurement_event_prompt=_MEASUREMENT_EVENT_PROMPT,
    direct_extraction_schema=DirectExtractionItemSchema,
    direct_extraction_prompt=_DIRECT_EXTRACTION_PROMPT,
    # paper_subset: set to a list of paper codes to restrict the run, e.g.:
    #   paper_subset=["physical_and_chemical_limnological", "prairie_wetland"]
    paper_subset=None,
    paper_filter=None,
)
