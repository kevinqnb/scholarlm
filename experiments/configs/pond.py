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
    """Entity fields for a pond / lake / wetland observation."""

    name: str | None
    abbreviations: str | None
    location: str | None
    site: str | None
    state: str | None
    date: str | None
    ecosystem: str | None


# ---------------------------------------------------------------------------
# Entity identification prompt
# ---------------------------------------------------------------------------

_IDENTIFICATION_PROMPT = """You are an expert in identifying ponds, lakes, and wetlands referenced in scientific literature.

Given the provided text (including any tables), extract all distinct ecosystem observations.

An ecosystem observation is defined as a specific pond, lake, wetland, or other aquatic ecosystem.
The observation may be further identified by a specific treatment site within the ecosystem, a specific treatment state, and/or by a specific date of measurement.


WHAT COUNTS AS AN ECOSYSTEM:
- Include ponds, lakes, wetlands, and similar aquatic ecosystems.
- Marshes, bogs, fens, and swamp should all be considered as "wetland".
- If the ecosystem type is unclear, classify it as "other".


ATTRIBUTE SCHEMA:
For each distinct ecosystem observation, output one item with the following attributes:
- name
- abbreviations and/or codes for reference
- general location
- treatment site
- treatment state
- date of observation
- ecosystem type

NOTE: While an ecosystem might be introduced by its full name (e.g., "Lake Mendota"), many papers use numerical or coded identifiers and abbreviations (e.g. "L1", "Lake 1", "Lake M.", "Mend.") to refer to the same ecosystem later on. Therefore, it is very important that these identifiers are collected and reported in the "abbreviations and/or codes for reference" field.


IDENTIFICATION GUIDELINES:
Treat ecosystem observations with the same name as multiple separate items if ANY of the following differ:
- Site or sub-site identifier (e.g., different plots, basins, units, or coded sites such as "P1", "W2", etc.)
- Treatment state (e.g., restored vs unrestored, control vs treatment, fertilized vs unfertilized, etc.)
- Date of observation or sampling

However, if the same ecosystem is mentioned with the same site, same state, and same date, do not duplicate it.


STRICT RULES ABOUT MISSING INFORMATION:
- Do NOT infer, guess, or derive any attribute.
- Use ONLY information explicitly stated in the text.
- If an attribute is not explicitly given, set its value to None.


EXTRACTION PROCEDURE (FOLLOW IN ORDER):
1. Scan the entire text, including tables, for any mentions of specific ponds, lakes, wetlands, or coded sites.
2. Determine which mentions correspond to distinct ecosystem observations using the identity rules above.
3. Output one JSON item per distinct observation.
4. Collect all items into a single JSON array under the key "items".


OUTPUT FORMAT REQUIREMENTS:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "abbreviations": "...",
      "location": "...",
      "site": "...",
      "state": "...",
      "date": "...",
      "ecosystem": "..."
    }
  ]
}
- If no distinct ecosystems are found, output exactly:
{ "items": [] }
"""

# ---------------------------------------------------------------------------
# Attribute catalogue
# ---------------------------------------------------------------------------

_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
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
            "forms — both dissolved and particulate, including nitrate (NO₃), nitrite (NO₂), ammonium "
            "(NH₃/NH₄⁺), and organic nitrogen. This must be the aggregate 'total nitrogen' value as "
            "explicitly reported in the source. This is NOT the same as individual nitrogen species "
            "(e.g., NO₃ alone, NO₂ alone, NH₃ alone, combined NO₃+NO₂, or particulate organic nitrogen "
            "[PON]) unless they are explicitly labeled as total nitrogen."
        ),
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"],
    },
    "tp": {
        "description": (
            "Total phosphorus (TP) concentration in the water column, representing the sum of all "
            "phosphorus forms — both dissolved and particulate. This must be the aggregate 'total "
            "phosphorus' value as explicitly reported in the source. This is NOT the same as individual "
            "phosphorus species (e.g., soluble reactive phosphorus [SRP], orthophosphate [PO₄³⁻], "
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

CONFIG = DatasetConfig(
    name="pond",
    data_dir="data/pond",
    metadata_file="data/pond/directory.json",
    entity_schema=ObservationSchema,
    entity_identification_prompt=_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A distinct aquatic ecosystem observation — a specific pond, lake, wetland, or "
        "similar water body — potentially further identified by treatment site, treatment "
        "state, or date of measurement."
    ),
    attribute_info_dict=_ATTRIBUTE_INFO_DICT,
    # paper_subset: set to a list of paper codes to restrict the run, e.g.:
    #   paper_subset=["physical_and_chemical_limnological", "prairie_wetland"]
    paper_subset=None,
    paper_filter=None,
)
