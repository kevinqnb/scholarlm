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


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class ObservationSchema(BaseModel):
    """Site-level entity fields for an aquatic dinitrogen fixation study."""

    name: str | None
    abbreviations: str | None
    site_type: str | None
    latitude: float | None
    longitude: float | None


class MeasurementEventSchema(BaseModel):
    """Event-level fields that distinguish individual dinitrogen fixation measurements."""

    date: str | None
    nfix_method: str | None
    substrate_type: str | None
    sample_depth: str | None
    additional_details: str | None


# ---------------------------------------------------------------------------
# Entity identification prompt
# ---------------------------------------------------------------------------

_IDENTIFICATION_PROMPT = """You are an expert in identifying and extracting information from scientific literature. Given the provided text (including any tables), extract identifying information for unique dinitrogen fixation measurement sites.

A dinitrogen fixation measurement site is a distinct physical location or ecosystem where dinitrogen fixation rates were measured. Multiple measurements at the same site (on different dates, using different methods, at different depths) should be represented as a single site record.


RESPONSE SCHEMA:
Site identifying information includes the following fields:
- name: the name of the site (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides (e.g. "Site 3", "L1") as the name.
- abbreviations: any secondary numerical or coded identifiers and abbreviations used elsewhere in the text to refer to the same site (e.g. "L1", "Lake 1", "Lake M.", "Mend."). If the primary identifier is already a code and no alternatives are used, set this to None.
- site_type: the type of site (e.g. continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy, etc.). This must be explicitly stated or clearly described in the text; do NOT infer it from the entity name alone.
- latitude: the latitude of the site, reported exactly as stated in the text.
- longitude: the longitude of the site, reported exactly as stated in the text.

NOTE: While a site might be introduced by its full name (e.g., "Lake Mendota"), many papers use numerical or coded identifiers and abbreviations (e.g. "L1", "Lake 1", "Lake M.", "Mend.") to refer to the same site later on. It is very important that these secondary identifiers are collected and reported in the "abbreviations" field so that cross-references within the paper can be resolved.


TABLE HANDLING:
Site names may appear in row or column headers in tables. Location metadata may be encoded in table captions or table footnotes. Check all of these when identifying sites.


IDENTIFICATION GUIDELINES:
Treat sites with the same name as multiple separate items ONLY if their geographic location clearly differs (e.g., different latitude and longitude, or explicitly described as distinct locations). Do NOT create separate items for the same site because measurements were taken on different dates, using different methods, or at different depths — those distinctions will be captured separately as measurement events.


STRICT RULES ABOUT MISSING INFORMATION:
- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If a field is not explicitly given, set its value to None.
- Do NOT infer site_type from the entity name.
- Do NOT infer coordinates from general geographic descriptions.


EXTRACTION PROCEDURE (FOLLOW IN ORDER):
1. Scan the entire text, including all tables, table captions, and table footnotes, for any mentions of dinitrogen fixation measurement sites.
2. Resolve all abbreviations and coded identifiers back to their associated site.
3. Determine which mentions correspond to distinct sites using the identification guidelines above.
4. Output one JSON item per distinct site.
5. Collect all items into a single JSON array under the key "items".


OUTPUT FORMAT REQUIREMENTS:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- Latitude and longitude values must be numeric (not strings). All other values are strings or None.
- The top-level object must have this form:
{
  "items": [
    {
      "name": "...",
      "abbreviations": "...",
      "site_type": "...",
      "latitude": ...,
      "longitude": ...
    }
  ]
}
- If no dinitrogen fixation measurement sites are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Measurement event prompt
# ---------------------------------------------------------------------------

_MEASUREMENT_EVENT_PROMPT = """For this dataset, a measurement event is a distinct instance of a dinitrogen fixation measurement, characterized by the conditions under which it was taken.

EVENT FIELDS:
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

RULES:
- Each event item must be as complete as the page text allows. Populate every field that has a value explicitly stated on this page.
- Do NOT output multiple events that differ only by having different subsets of the same information. If the text supports identifying date + method + substrate for a measurement, output one event with all three fields populated — not separate events for each subset.
- Do NOT infer, guess, or derive field values. If a field is not explicitly stated on this page, set it to None.
"""


# ---------------------------------------------------------------------------
# Attribute catalogue
# ---------------------------------------------------------------------------

_MASS_UNITS = [
    "nmol-N g-1 h-1", "nmol-C2H4 g-1 h-1", "nmol-N2 g-1 h-1", "ug-N g-1 d-1",
    "nmol-N2 g-1 d-1", "umol-N g-1 d-1", "nmol-C2H4 g-1 d-1", "nmol-N g-1 d-1",
    "ug-N g-1 h-1", "ug-N kg-1 d-1", "umol-N g-1 h-1", "fmol-N g-1 h-1",
    "ng-N g-1 d-1", "ng-N g-1 h-1", "nmol-N kg-1 h-1", "umol-C2H4 g-1 d-1",
    "umol-N kg-1 h-1", "umol-N2 g-1 d-1",
]

_AREAL_UNITS = [
    "umol-N m-2 h-1", "mg-N m-2 d-1", "umol-N m-2 d-1", "umol-C2H4 m-2 h-1",
    "nmol-C2H4 cm-2 h-1", "mmol-N m-2 d-1", "ug-N m-2 h-1", "mg-N m-2 h-1",
    "nmol-C2H4 cm-2 d-1", "nmol-C2H4 m-2 h-1", "umol-N2 m-2 h-1", "g-N m-2 yr-1",
    "mmol-N m-2 h-1", "mmol-N2 m-2 d-1", "nmol-N cm-2 h-1", "umol-N2 m-2 d-1",
    "kg-N2 ha-1 yr-1", "mg-N m-2 yr-1", "mg-N2 m-2 h-1", "ng-N m-2 h-1",
    "nmol-C2H4 m-2 d-1", "ug-N cm-2 h-1", "ug-N2 m-2 h-1",
]

_VOLUMETRIC_UNITS = [
    "nmol-N L-1 d-1", "nmol-N L-1 h-1", "nmol-C2H4 L-1 h-1", "ug-N L-1 h-1",
    "ng-N L-1 h-1", "mg-N m-3 d-1", "nmol-C2H4 cm-3 h-1", "nmol-C2H4 mL-1 h-1",
    "nmol-N cm-3 d-1", "nmol-N cm-3 h-1", "ug-N m-3 h-1", "umol-N2 L-1 d-1",
    "umol-N2 L-1 h-1", "mmol-C2H4 m-3 d-1", "nmol-C2H4 cm-3 d-1", "nmol-N m-3 h-1",
    "nmol-N2 cm-3 d-1", "nmol-N2 L-1 d-1", "nmol-N2 L-1 h-1", "ug-N L-1 d-1",
    "ug-N2 L-1 h-1", "ug-N2 m-3 d-1", "umol-C2H4 L-1 d-1", "umol-C2H4 mL-1 3h-1",
    "umol-N L-1 d-1", "umol-N L-1 h-1",
]

_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
    "nfix_rate_mass": {
        "description": (
            "Rate of dinitrogen fixation per unit mass: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by substrate mass. Not equivalent to rates reported per unit area or volume."
        ),
        "units": _MASS_UNITS,
    },
    "nfix_rate_areal": {
        "description": (
            "Rate of dinitrogen fixation per unit area: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by area. Not equivalent to rates reported per unit mass or volume."
        ),
        "units": _AREAL_UNITS,
    },
    "nfix_rate_volumetric": {
        "description": (
            "Rate of dinitrogen fixation per unit volume: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by water volume. Not equivalent to rates reported per unit mass or area."
        ),
        "units": _VOLUMETRIC_UNITS,
    },
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
# Direct extraction prompt:
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Paper filter
# ---------------------------------------------------------------------------


def _nfix_paper_filter(metadata: dict) -> bool:
    """Exclude papers whose extraction_location indicates figures, supplements, archives, or author notes."""
    location = metadata.get("extraction_location", "")
    return not any(x in location for x in ["figure", "supplement", "archive", "author"])


# ---------------------------------------------------------------------------
# Config instance
# ---------------------------------------------------------------------------

# Development subset used in early experiments
_DEV_SUBSET = [
    "R163", "R164", "R172", "R248", "R124",
    "R51", "R59", "R114", "R43", "R103",
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
    # paper_subset: uncomment the line below to run only the 10-paper development set.
    # paper_subset=_DEV_SUBSET,
    paper_subset=None,
    paper_filter=_nfix_paper_filter,
)
