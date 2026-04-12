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
    """Entity fields for an aquatic dinitrogen fixation measurement event."""

    name: str | None
    abbreviations: str | None
    ecosystem_type: str | None
    latitude: float | None
    longitude: float | None
    date: str | None
    nfix_method: str | None
    substrate_type: str | None
    sample_depth: str | None


# ---------------------------------------------------------------------------
# Entity identification prompt
# ---------------------------------------------------------------------------

_IDENTIFICATION_PROMPT = """You are an expert in identifying and extracting information from scientific literature. Given the provided text (including any tables), extract identifying information for unique dinitrogen fixation measurements.

DINITROGEN FIXATION MEASUREMENTS:

A dinitrogen fixation measurement is any explicit report of a rate of dinitrogen fixation: the amount of nitrogen (or ethylene in acetylene reduction assays) per fixed unit of time. This is typically normalized by substrate mass, area, or water volume.

Information identifying a dinitrogen fixation measurement may include, but is not limited to, the following identifiers:

- name: the name of the ecosystem or site from which the measurement was taken (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides (e.g. "Site 3", "L1") as the name.
- abbreviations: any secondary numerical or coded identifiers and abbreviations used elsewhere in the text to refer to the same ecosystem (e.g. "L1", "Lake 1", "Lake M.", "Mend."). If the primary identifier is already a code and no alternatives are used, set this to None.
- ecosystem_type: the type of ecosystem from which the measurement was taken (e.g. continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy, etc.). This must be explicitly stated or clearly described in the text; do NOT infer it from the entity name alone (e.g. do not assume ecosystem_type is "lake" just because the name contains "Lake").
- latitude: the latitude of the location where the measurement was taken, reported exactly as stated in the text.
- longitude: the longitude of the location where the measurement was taken, reported exactly as stated in the text.
- date: the date when the measurement was taken, using one of the following formats depending on the precision available:
  - Full date: "dd-mm-yyyy"
  - Month and year only: "mm-yyyy"
  - Year only: "yyyy"
  - If no date information is available, set to None.
- nfix_method: the method used to measure dinitrogen fixation (e.g. acetylene reduction assay, ARA, or 15N2 incorporation)
- substrate_type: the type of substrate associated with the measurement (water column, benthos, or other)
- sample_depth: the depth at which the sample was collected (e.g. "surface", "0-5 cm", "bottom", "0-10 m", etc.)

NOTE: While an ecosystem might be introduced by its full name (e.g., "Lake Mendota"), many papers use numerical or coded identifiers and abbreviations (e.g. "L1", "Lake 1", "Lake M.", "Mend.") to refer to the same ecosystem later on. It is very important that these secondary identifiers are collected and reported in the "abbreviations" field so that cross-references within the paper can be resolved.

TABLE HANDLING:

Tables frequently contain dinitrogen fixation measurements. When extracting from tables, be aware of the following:
- Site or ecosystem names may appear in a row or column header.
- Method, date, or location metadata may be encoded in row or column headers, table captions, or table footnotes rather than in individual cells. Check all of these.
- If a table footnote defines an abbreviation (e.g. "* = acetylene reduction assay"), apply that definition to all relevant entries.

IDENTIFICATION GUIDELINES:

Treat dinitrogen fixation measurements with the same name as multiple separate items if ANY of the following differ:
- Location (e.g. different latitude and longitude)
- Date
- Method used to measure dinitrogen fixation (e.g. ARA vs 15N2 incorporation)
- Substrate type (e.g. water column vs benthos)
- Sample depth (e.g. surface vs 0-5 cm)

However, if the same ecosystem is referenced multiple times with the same identifying information, do not duplicate it.

Note: Some identifiers describe the site (name, ecosystem_type, latitude, longitude) while others describe the measurement event (nfix_method, date). A single site may have multiple distinct measurement events. Each unique combination of site + measurement event should be a separate item.

STRICT RULES ABOUT MISSING INFORMATION:

- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If an identifier is not explicitly given, set its value to None.
- Do NOT infer ecosystem_type from the entity name. For example, if a site is called "Lake Mendota" but the text never describes or categorizes it as a lake, set ecosystem_type to None.
- Do NOT infer coordinates from general geographic descriptions. If the text says "in central Wisconsin" but provides no latitude or longitude, set both to None.

EXTRACTION PROCEDURE (FOLLOW IN ORDER):

1. Scan the entire text, including all tables, table captions, and table footnotes, for any mentions of dinitrogen fixation measurements.
2. Resolve all abbreviations and coded identifiers back to their associated ecosystem or site.
3. Determine which mentions correspond to distinct measurement events using the identification guidelines above.
4. Output one JSON item per distinct measurement event.
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
      "ecosystem_type": "...",
      "latitude": ...,
      "longitude": ...,
      "date": "...",
      "nfix_method": "...",
      "substrate_type": "...",
      "sample_depth": "..."
    }
  ]
}

- If no dinitrogen fixation measurements are found, output exactly:
{ "items": [] }
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

CONFIG = DatasetConfig(
    name="nfix",
    data_dir="data/nfix",
    metadata_file="data/nfix/directory.json",
    entity_schema=ObservationSchema,
    entity_identification_prompt=_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A distinct dinitrogen fixation measurement event — a specific site and substrate "
        "identified by ecosystem type, method, date, sample depth, and coordinates."
    ),
    attribute_info_dict=_ATTRIBUTE_INFO_DICT,
    # paper_subset: uncomment the line below to run only the 10-paper development set.
    # paper_subset=_DEV_SUBSET,
    paper_subset=None,
    paper_filter=_nfix_paper_filter,
)
