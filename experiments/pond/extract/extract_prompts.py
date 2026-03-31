POND_IDENTIFICATION_PROMPT = """You are an expert in identifying ponds, lakes, and wetlands referenced in scientific literature.

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



# --- Ablation 1: Combined (entity, attribute) identification prompt ---
#
# This prompt replaces the separate entity identification + attribute detection
# prompts from the baseline. The model is instructed to emit one item per
# (ecosystem observation, measurement attribute) pair, only when a direct
# numerical measurement for that attribute is reported for that entity.
POND_IDENTIFICATION_PROMPT_WITH_ATTRIBUTES = """You are an expert in identifying ponds, lakes, and wetlands referenced in scientific literature.

Given the provided text (including any tables), extract all distinct ecosystem observations.

An ecosystem observation is defined as a specific pond, lake, wetland, or other aquatic ecosystem.
The observation may be further identified by a specific treatment site within the ecosystem, a specific treatment state, a specific date of measurement, and/or by a specific measurement attribute type.


WHAT COUNTS AS AN ECOSYSTEM:
- Include ponds, lakes, wetlands, and similar aquatic ecosystems.
- Marshes, bogs, fens, and swamps should all be considered as "wetland".
- If the ecosystem type is unclear, classify it as "other".


IDENTIFICATION SCHEMA:
For each distinct ecosystem observation, output one item with the following fields:
- name: name of the ecosystem
- abbreviations: abbreviations and/or codes used to refer to the ecosystem
- location: general location
- site: treatment site or sub-site identifier (if applicable)
- state: treatment state (e.g., restored vs. unrestored, control vs. treatment)
- date: date of observation or sampling
- ecosystem: ecosystem type — one of: pond, lake, wetland, other
- attribute: the attribute for which a direct numerical measurement is reported for this ecosystem observation (must be one of: latitude, longitude, surface_area, max_depth, vegetation_cover, ph, tn, tp, chla)
- attribute_terms: a list of terminology, abbreviations, or column headers used in the document to refer to the attribute (e.g., ["TP", "total P", "total phosphorus"])

NOTE: A single ecosystem may appear as multiple items — once per distinct attribute. This is expected and correct.

NOTE: While an ecosystem might be introduced by its full name (e.g., "Lake Mendota"), many papers use numerical or coded identifiers and abbreviations (e.g. "L1", "Lake 1", "Lake M.", "Mend.") to refer to the same ecosystem later on. Therefore, it is very important that these identifiers are collected and reported in the "abbreviations" field.


IDENTIFICATION GUIDELINES:
Treat ecosystem observations as separate items if ANY of the following differ:
- Site or sub-site identifier (e.g., different plots, basins, units, or coded sites)
- Treatment state (e.g., restored vs unrestored, control vs treatment)
- Date of observation or sampling
- Attribute(s) for which a measurement is reported

However, if the same ecosystem is mentioned with the same site, state, date, and attribute, do not duplicate it.


AVAILABLE MEASUREMENT ATTRIBUTES:
Attributes should only be drawn from the following list, using the attribute name EXACTLY as it appears below. If none of these attributes are directly measured for a given ecosystem observation, then the observation is invalid and should not be reported. 

1. latitude: Geographic latitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.
2. longitude: Geographic longitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.
3. surface_area: Surface area of the water body itself, representing the horizontal area of open water. This is NOT the same as watershed area, drainage basin area, catchment area, or littoral zone area.
4. max_depth: Maximum physical water depth of the ecosystem, defined as the deepest point of the water body. This is NOT the same as mean depth, average depth, or Secchi depth.
5. vegetation_cover: Fraction or percentage of the ecosystem surface area covered by aquatic macrophytes or other rooted/floating aquatic vegetation. This is NOT the same as algal cover or phytoplankton density.
6. ph: pH of the water (dimensionless). Should refer to a measured water pH value, not soil or sediment pH.
7. tn: Total nitrogen (TN) concentration in the water column — the sum of ALL nitrogen forms. This is NOT the same as individual nitrogen species (e.g., NO₃ alone) unless they are explicitly labeled as total nitrogen.
8. tp: Total phosphorus (TP) concentration in the water column — the sum of ALL phosphorus forms. This is NOT the same as individual phosphorus species (e.g., SRP, orthophosphate) unless they are explicitly labeled as total phosphorus.
9. chla: Chlorophyll-a (Chl-a) concentration in the water column. This is NOT the same as total chlorophyll, chlorophyll-b, or pheophytin unless explicitly labeled as chlorophyll-a.

NOTE: It is very important that you only use the EXACT attribute names from the list above (latitude, longitude, surface_area, max_depth, vegetation_cover, ph, tn, tp, chla) when reporting attributes.


STRICT RULES ABOUT MISSING INFORMATION:
- Do NOT infer, guess, or derive any attribute.
- Use ONLY information explicitly stated in the text.
- If an entity field is not explicitly given, set its value to None.


OUTPUT FORMAT REQUIREMENTS:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- One item per (ecosystem observation, attribute) pair.
- If no qualifying pairs are found, output exactly: { "items": [] }
"""