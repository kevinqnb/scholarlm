"""
Dataset configuration for the nfix (aquatic dinitrogen fixation) dataset.

This is the single source of truth for all nfix-specific values: entity schema,
attribute catalogue, entity identification prompt, file paths, and paper filter.
All pipeline runners load this via importlib; the existing experiments/nfix/
scripts are unchanged.
"""
from __future__ import annotations

import json

from pydantic import BaseModel

from scholarlm.config import DatasetConfig
from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS


# ---------------------------------------------------------------------------
# Entity schema
# ---------------------------------------------------------------------------


class EntitySchema(BaseModel):
    """Site-level entity fields for an aquatic dinitrogen fixation study."""

    name: str | None
    identifiers: str | None
    ecosystem_type: str | None
    # ``location`` was removed 2026-09-20: it was never shown to the judge and
    # location is often not explicit in the text anyway.
    # ``identifiers`` is extracted by the real pipeline and its ablations only
    # -- see DatasetConfig.baseline_filter_fields (below) for the NuExtract
    # baselines; GLiNER already excludes it structurally (never listed in
    # gliner_field_descriptions) and ChatExtract's flat schema never included
    # it. It is also never shown to the judge (judge_filter_fields, below).


# This is a general prompt template for which we input
# entity instructions, type, and extraction fields.
# It's used as the first step of the pipeline for entity identification. 
ENTITY_IDENTIFICATION_PROMPT = """You are an expert in identifying and extracting information from scientific literature. Given the provided text (including any tables), extract identifying information for unique dinitrogen fixation measurement sites.

A dinitrogen fixation measurement site is a distinct physical location or ecosystem where dinitrogen fixation rates were measured. Multiple measurements at the same site (on different dates, using different methods, at different depths) should be represented as a single site record.


Response schema:
Site identifying information includes the following fields:
- name: the name of the site (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides (e.g. "Site 3", "L1") as the name.
- identifiers: every alternate short-form reference to this site used in the text — site codes, numeric tags, or shortened versions of the name — joined into a single string with semicolons separating each (e.g. "L1; Lake M.; Mend."). Collect these whenever the text uses them for the same site, even if the linkage is introduced only once (e.g. "Lake Mendota (LM)"). Do not include the primary name itself. If no alternatives exist, set to None.
- ecosystem_type: the type of site (e.g. continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy, etc.). This must be explicitly stated or clearly described in the text; do NOT infer it from the entity name alone.


Identification rules:
Treat sites with the same name as multiple separate items ONLY if their geographic location clearly differs. Do NOT create separate items for the same site because measurements were taken on different dates, using different methods, or at different depths — those distinctions will be captured separately as measurement events.


Strict rules about missing information:
- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If a field is not explicitly given, set its value to None.
- Do NOT infer ecosystem_type from the entity name.


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
      "ecosystem_type": "..."
    }
  ]
}
- If no dinitrogen fixation measurement sites are found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Attribute Schema
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


# ---------------------------------------------------------------------------
# Measurement Schema
# ---------------------------------------------------------------------------


class MeasurementEventSchema(BaseModel):
    """Event-level fields that distinguish individual dinitrogen fixation measurements."""

    date: str | None
    substrate_type: str | None
    event_details: str | None



_MEASUREMENT_EVENT_PROMPT = """Event fields:
- date: The date the measurement was taken. Use one of the following formats depending on available precision:
  - Full date: "dd-mm-yyyy"
  - Month and year only: "mm-yyyy"
  - Season and year: "Spring yyyy", "Summer yyyy", "Fall yyyy", or "Winter yyyy"
  - Year only: "yyyy"
  Set to None if no date is stated on this page.
- substrate_type: The physical substrate the fixation was measured in or on — where the sample was taken from, NOT how the reported rate is normalized (mass/area/volume is a separate choice, captured by the attribute itself, not this field). Must be exactly one of these three values — do not report any other wording:
  - "benthos": sediment, rock, microbial mat/biofilm, or other bottom/substrate material (e.g. "sediment cores were incubated", "microbial mats were sampled", "attached to cobble").
  - "water column": water samples, filtered seawater, or suspended particulates/plankton not tied to a specific host organism (e.g. "water samples were collected at 5 m", "surface water was filtered").
  - "other": fixation tied to a living plant, alga, or colonial organism rather than sediment or bulk water — e.g. seagrass or mangrove leaves/roots, marsh grass (Spartina) stems, macroalgae, epiphytes on a host surface, or a suspended colonial organism like Trichodesmium.
  Set to None only if the substrate is genuinely not stated; otherwise always classify into one of the three values above.
- event_details: A catch-all for any other distinguishing context not captured by date or substrate_type, whatever form it takes — for example, the dinitrogen-fixation measurement method (e.g., acetylene reduction assay, ARA, 15N2 incorporation), the sample depth (e.g., "surface", "0-5 cm", "bottom", "0-10 m"), light vs. dark incubation, or a specific treatment condition. The goal is that two measurements that are genuinely distinct (different method, depth, or condition) end up with different event_details, while two reports of the literal same measurement do not. Keep this to one sentence or fewer. Set to None if not applicable.
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
    ecosystem_type: str | None
    # Event fields
    date: str | None
    substrate_type: str | None
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



_DIRECT_EXTRACTION_PROMPT = """Entity identification:
Extract all distinct dinitrogen fixation measurement sites mentioned in the document.

Entity fields:
- name: the name of the site (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides.
- identifiers: every alternate short-form reference to this site used in the text — site codes, numeric tags, or shortened versions of the name — joined into a single string with semicolons separating each (e.g. "L1; Lake M.; Mend."). Collect these whenever the text uses them for the same site, even if the linkage is introduced only once (e.g. "Lake Mendota (LM)"). Do not include the primary name itself. If no alternatives exist, set to None.
- ecosystem_type: the type of site (e.g., continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy). Must be explicitly stated; do NOT infer from the site name.

Entity identification rules:
- Treat sites as separate only if their geographic location clearly differs.
- Do NOT create separate items for the same site because measurements were taken on different dates, methods, or depths — those distinctions are captured as measurement events.
- Do NOT infer, guess, or derive any field value. Use ONLY information explicitly stated in the text. If a field is not explicitly given, set it to None.


Measurement event fields:
For each site and each detected attribute measurement, also identify the measurement event context:
- date: The date of the measurement. Formats: "dd-mm-yyyy", "mm-yyyy", "Spring/Summer/Fall/Winter yyyy", or "yyyy". Set to None if not stated.
- substrate_type: The physical substrate the fixation was measured in or on — where the sample was taken from, NOT how the reported rate is normalized (mass/area/volume is a separate choice, captured by the attribute itself, not this field). Must be exactly one of these three values — do not report any other wording:
  - "benthos": sediment, rock, microbial mat/biofilm, or other bottom/substrate material.
  - "water column": water samples, filtered seawater, or suspended particulates/plankton not tied to a specific host organism.
  - "other": fixation tied to a living plant, alga, or colonial organism rather than sediment or bulk water — e.g. seagrass/mangrove leaves or roots, marsh grass (Spartina) stems, macroalgae, epiphytes, or a suspended colonial organism like Trichodesmium.
  Set to None only if the substrate is genuinely not stated; otherwise always classify into one of the three values above.
- event_details: A catch-all for any other distinguishing context not captured by date or substrate_type — for example, the dinitrogen-fixation measurement method (e.g., acetylene reduction assay, ARA, 15N2 incorporation), the sample depth (e.g., "surface", "0-5 cm", "0-10 m"), light vs. dark incubation, or a specific treatment condition. Two genuinely distinct measurements should end up with different event_details. One sentence or fewer. Set to None if not applicable.


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
      "ecosystem_type": "...",
      "date": "...",
      "substrate_type": "...",
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
# NuExtract-2.0-8B baseline: few-shot synthetic examples, since
# NuExtract's calling convention has no field for freeform instructions.
# Every output value below is
# an exact substring of its input text, since NuExtract's verbatim-string
# fields are trained to copy spans rather than paraphrase. Together the three
# examples touch all 3 nfix rate attributes and all of point_value,
# lower/upper, list_values, tolerance, and standard_deviation at least once.
# ---------------------------------------------------------------------------

_NUEXTRACT_EXAMPLE_1_INPUT = (
    "Tampa Bay Seagrass Site (TB-3) is a shallow seagrass meadow located in "
    "Tampa Bay, Florida. Sediment cores were incubated in August 2018 using "
    "the acetylene reduction assay. Samples were collected from the benthos "
    "at a depth of 0-5 cm during a light incubation. Dinitrogen fixation was "
    "measured at 4.2 nmol C2H4 g⁻¹ h⁻¹ in the sediment, "
    "while water column fixation reached 120 µmol N m⁻² d⁻¹ nearby."
)

_NUEXTRACT_QUANTITY_DEFAULTS = {
    "qualifiers": [], "point_value": None, "lower": None, "upper": None,
    "list_values": None, "tolerance": None, "standard_deviation": None,
}

_NUEXTRACT_EXAMPLE_1_OUTPUT = json.dumps(
    {
        "items": [
            {
                "name": "Tampa Bay Seagrass Site",
                "ecosystem_type": "seagrass meadow",
                "date": "August 2018",
                "substrate_type": "benthos",
                "event_details": "acetylene reduction assay; 0-5 cm depth; light incubation",
                "attribute": "nfix_rate_mass", "value": "4.2", "units": "nmol C2H4 g⁻¹ h⁻¹",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {"point_value": "4.2"}),
            },
            {
                "name": "Tampa Bay Seagrass Site",
                "ecosystem_type": "seagrass meadow",
                "date": "August 2018",
                "substrate_type": "benthos",
                "event_details": "acetylene reduction assay; 0-5 cm depth; light incubation",
                "attribute": "nfix_rate_areal", "value": "120", "units": "µmol N m⁻² d⁻¹",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {"point_value": "120"}),
            },
        ]
    }
)

# The volumetric rate's ranged phrasing demonstrates a non-plain-point shape,
# so this baseline's only real instruction channel (few-shot examples -- see
# module docstring in measurementlm_nuextract.py) actually shows the
# qualifier fields in use, not just plain points.
_NUEXTRACT_EXAMPLE_2_INPUT = (
    "The Chesapeake Bay Estuary Transect (CBET) is an estuary site in "
    "Chesapeake Bay. Samples of the water column from the surface (0 m) "
    "were incubated for 24 hours in March 2020 using 15N2 incorporation. "
    "Volumetric fixation rates ranging from 3.2 to 4.0 nmol N2 L⁻¹ h⁻¹ were "
    "recorded under dark conditions."
)

_NUEXTRACT_EXAMPLE_2_OUTPUT = json.dumps(
    {
        "items": [
            {
                "name": "Chesapeake Bay Estuary Transect",
                "ecosystem_type": "estuary",
                "date": "March 2020",
                "substrate_type": "water column",
                "event_details": "15N2 incorporation; surface; dark conditions",
                "attribute": "nfix_rate_volumetric", "value": "3.2 to 4.0", "units": "nmol N2 L⁻¹ h⁻¹",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["IsRange"], "lower": "3.2", "upper": "4.0",
                }),
            },
        ]
    }
)

# Rounds out shape coverage with list_values, tolerance, and standard_deviation
# -- example 1 and 2 above only reach point_value and IsRange.
_NUEXTRACT_EXAMPLE_3_INPUT = (
    "Baltic Sea Transect (BST) is an estuary site in the Baltic Sea. "
    "Sediment cores incubated in July 2019 using the acetylene reduction "
    "assay at a depth of 0-3 cm yielded fixation rates of 2.1, 2.8, and 3.4 "
    "nmol C2H4 g⁻¹ h⁻¹ across three replicates. Water column fixation was "
    "95 ± 12 µmol N m⁻² d⁻¹ under light conditions. Mean volumetric "
    "fixation was 3.6 (SD 0.4) nmol N2 L⁻¹ h⁻¹ at the surface."
)

_NUEXTRACT_EXAMPLE_3_OUTPUT = json.dumps(
    {
        "items": [
            {
                "name": "Baltic Sea Transect",
                "ecosystem_type": "estuary",
                "date": "July 2019",
                "substrate_type": "benthos",
                "event_details": "acetylene reduction assay; 0-3 cm depth",
                "attribute": "nfix_rate_mass", "value": "2.1, 2.8, and 3.4", "units": "nmol C2H4 g⁻¹ h⁻¹",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["IsList"], "list_values": ["2.1", "2.8", "3.4"],
                }),
            },
            {
                "name": "Baltic Sea Transect",
                "ecosystem_type": "estuary",
                "date": "July 2019",
                "substrate_type": "water column",
                "event_details": "acetylene reduction assay; under light conditions",
                "attribute": "nfix_rate_areal", "value": "95 ± 12", "units": "µmol N m⁻² d⁻¹",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["HasTolerance"], "point_value": "95", "tolerance": "± 12",
                }),
            },
            {
                "name": "Baltic Sea Transect",
                "ecosystem_type": "estuary",
                "date": "July 2019",
                "substrate_type": "water column",
                "event_details": "acetylene reduction assay; surface",
                "attribute": "nfix_rate_volumetric", "value": "3.6 (SD 0.4)", "units": "nmol N2 L⁻¹ h⁻¹",
                **(_NUEXTRACT_QUANTITY_DEFAULTS | {
                    "qualifiers": ["IsMean", "HasSD"], "point_value": "3.6", "standard_deviation": "0.4",
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
    """Entity schema for Ablation 2: one item per (site, attribute) pair."""

    # Entity fields (same as ObservationSchema)
    name: str | None
    identifiers: str | None
    ecosystem_type: str | None
    # Reserved fields required by Ablation 2
    attribute: str
    attribute_terms: list[str]


_ABLATION2_IDENTIFICATION_PROMPT = """You are an expert in identifying dinitrogen fixation measurement sites referenced in scientific literature, and in detecting which measurement attributes are reported for each site. Given the provided text (including any tables), extract all distinct (measurement site, measured attribute) pairs for which a direct numerical measurement is reported.

A dinitrogen fixation measurement site is a distinct physical location or ecosystem where dinitrogen fixation rates were measured. Emit one item per (site, attribute) pair. If a site has multiple attributes measured, emit one item per attribute.

IMPORTANT: Only emit a pair when a direct numerical measurement exists in the document for that site and attribute. Do NOT emit pairs where the only data is qualitative, model parameters, or goodness-of-fit statistics.


Response schema:
For each (site, attribute) pair, output one item with the following fields:
- name: the name of the site (e.g. "Lake Mendota", "Chesapeake Bay", "Plot A3"). If no full name is given, use whatever primary identifier the paper provides.
- identifiers: every alternate short-form reference to this site used in the text — site codes, numeric tags, or shortened versions of the name — joined into a single string with semicolons separating each (e.g. "L1; Lake M.; Mend."). Collect these whenever the text uses them for the same site, even if the linkage is introduced only once (e.g. "Lake Mendota (LM)"). Do not include the primary name itself. If no alternatives exist, set to None.
- ecosystem_type: the type of site (e.g., continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, cryptobiotic crust, tree canopy). Must be explicitly stated; do NOT infer from the site name.
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
      "ecosystem_type": "...",
      "attribute": "...",
      "attribute_terms": [...]
    }
  ]
}
- If no (site, attribute) pairs with direct numerical measurements are found, output exactly:
{ "items": [] }
"""


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

# ---------------------------------------------------------------------------
# ChatExtract baseline: per-attribute <PROPERTY> phrases
#
# ChatExtract is a single-property method; each prompt reads "...a value of
# <PROPERTY>...". The three nfix rate attributes differ only by normalization
# basis (mass / area / volume), which the phrase makes explicit so the model
# is steered toward the right unit family.
# ---------------------------------------------------------------------------

_CHATEXTRACT_PROPERTY_NAMES: dict[str, str] = {
    "nfix_rate_mass": "dinitrogen fixation rate per unit mass",
    "nfix_rate_areal": "dinitrogen fixation rate per unit area",
    "nfix_rate_volumetric": "dinitrogen fixation rate per unit volume",
}

# ChatExtract's reference prompts ask about a "material"/"compound" -- nfix
# entities are measurement sites, not chemical compounds, so this replaces
# that wording throughout (see DatasetConfig.chatextract_entity_noun).
_CHATEXTRACT_ENTITY_NOUN = "site"

# ---------------------------------------------------------------------------
# GLiNER2 baseline: per-field descriptions for entity/event fields beyond the
# subject name (see DatasetConfig.gliner_field_descriptions). Copied verbatim
# from _DIRECT_EXTRACTION_PROMPT's own per-field bullets above -- GLiNER sees
# the same wording Ablation 1 already uses, not freshly authored text.
# ``identifiers`` has no entry here: it's an alias-resolution aid for the real
# pipeline's entity matching, not reported content, so GLiNER never asks for it.
# ---------------------------------------------------------------------------

_GLINER_FIELD_DESCRIPTIONS: dict[str, str] = {
    "ecosystem_type": (
        "the type of site (e.g., continental shelf, estuary, lake, freshwater "
        "wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, soil, "
        "cryptobiotic crust, tree canopy). Must be explicitly stated; do NOT infer "
        "from the site name."
    ),
    "date": (
        'The date of the measurement. Formats: "dd-mm-yyyy", "mm-yyyy", '
        '"Spring/Summer/Fall/Winter yyyy", or "yyyy". Set to None if not stated.'
    ),
    "substrate_type": (
        "The physical substrate the fixation was measured in or on — where the "
        "sample was taken from, NOT how the reported rate is normalized "
        "(mass/area/volume is a separate choice, captured by the attribute "
        "itself, not this field). Must be exactly one of these three values — "
        "do not report any other wording: \"benthos\" (sediment, rock, microbial "
        "mat/biofilm, or other bottom/substrate material), \"water column\" "
        "(water samples, filtered seawater, or suspended particulates/plankton "
        "not tied to a specific host organism), or \"other\" (fixation tied to a "
        "living plant, alga, or colonial organism rather than sediment or bulk "
        "water — e.g. seagrass/mangrove leaves or roots, marsh grass stems, "
        "macroalgae, epiphytes, or a suspended colonial organism like "
        "Trichodesmium). Set to None only if the substrate is genuinely not "
        "stated; otherwise always classify into one of the three values above."
    ),
    "event_details": (
        "A catch-all for any other distinguishing context not captured above — "
        "including the dinitrogen-fixation measurement method (e.g., acetylene "
        "reduction assay, ARA, 15N2 incorporation) and the sample depth (e.g., "
        "\"surface\", \"0-5 cm\", \"0-10 m\"), plus anything else like light vs. "
        "dark incubation or a specific treatment condition. Two genuinely "
        "distinct measurements should end up with different event_details. One "
        "sentence or fewer. Set to None if not applicable."
    ),
}


CONFIG = DatasetConfig(
    name="nfix",
    data_dir="data/nfix",
    metadata_file="data/nfix/directory.json",
    entity_schema=EntitySchema,
    entity_identification_prompt=ENTITY_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A distinct dinitrogen fixation measurement site — a specific ecosystem or location "
        "identified by name, type, and coordinates."
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
    # paper_subset: uncomment the line below to run only the 10-paper development set.
    # paper_subset=_DEV_SUBSET,
    paper_subset=None,
    paper_filter=_nfix_paper_filter,
    paper_exclude=_EXCLUDED_PAPERS,
    ablation2_entity_schema=Ablation2ObservationSchema,
    ablation2_entity_identification_prompt=_ABLATION2_IDENTIFICATION_PROMPT,
    # Judge sees only: name, date, substrate_type (+ attribute, value, units).
    # identifiers are dropped so the judge evaluates the
    # minimal entity/event context; event_details is a catch-all for
    # distinguishing measurements from each other, not the main extraction
    # interest, so it's kept out of the judge prompt too. substrate_type is
    # now shown deliberately (2026-09-20) -- it's an important axis for
    # judging correctness; watch for the ground-truth-formatting-mismatch
    # risk noted historically for this field if judge numbers look off.
    judge_filter_fields=["identifiers", "event_details"],
    judge_instructions=JUDGE_INSTRUCTIONS,
    ground_truth_file="data/nfix/ground_truth_review.json",
    # Matching rules for the id-addressed evaluation path (analysis/match_cache.py,
    # analysis/recovery_validity.py) -- see DatasetConfig's docstring for scope
    # and why these are allowed to diverge from analysis/ablation.py's own
    # get_matching_rules (converted_value, not point_value; no legacy history
    # here to stay comparable with).
    strict_matching={
        "document_id": "document_id",
        "attribute": "attribute",
        "point_value": "point_value",
        "units": "units",
    },
    fuzzy_matching={
        "name": "name",
        "ecosystem_type": "ecosystem_type",
        "substrate_type": "substrate_type",
    },
    fuzzy_threshold=1 / 6,
    numeric_coerce=["point_value"],
)
