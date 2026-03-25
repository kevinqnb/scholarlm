NFIX_IDENTIFICATION_PROMPT = """You are an expert in identifying and extracting information from scientific literature. Given the provided text (including any tables), extract identifying information for unique dinitrogen fixation measurements.

DINITROGEN FIXATION MEASUREMENTS:
A dinitrogen fixation measurement is any explicit report of a rate of dinitrogen fixation: the amount of nitrogen (or ethylene -- C2H4 -- for acetylene reduction assays) per fixed unit of time. This is typically normalized by substrate mass, area, or water volume.

Information identifying a dinitrogen fixation measurement may include, but is not limited to, the following identifiers:
- name: the name of the ecosystem from which the measurement was taken (e.g. "Lake Mendota", "Mendota Lake", "Mend. Lake", "L1", "Lake 1", etc.)
- abbreviations and/or codes for reference: any numerical or coded identifiers and abbreviations used in the text to refer to the same ecosystem (e.g. "L1", "Lake 1", "Lake M.", "Mend.", etc.)
- ecosystem_type: the type of ecosystem from which the measurement was taken (e.g. continental shelf, estuary, lake, freshwater wetland, salt marsh, mangrove, river, tidal flat, seagrass meadow, etc.)
- substrate_type: the type of substrate associated with the measurement (water column, benthos, or other)
- sample_depth: the depth at which the sample was collected (e.g. "surface", "0-5 cm", "bottom", "0-10 m", etc.)
- nfix_method: the method used to measure dinitrogen fixation (e.g. acetylene reduction assay -- ARA, 15N2 incorporation, dissolved N2:Ar ratio method, or other)
- latitude: the latitude of the location where the measurement was taken
- longitude: the longitude of the location where the measurement was taken
- year: the year when the measurement was taken
- month: the month when the measurement was taken
- day: the day when the measurement was taken
- hour_minute: the time of day when the measurement was taken (e.g. "14:30", "2:30 PM", etc.)
- season: the season when the measurement was taken (e.g. winter, spring, summer, fall)

NOTE: While an ecosystem might be introduced by its full name (e.g., "Lake Mendota"), many papers use numerical or coded identifiers and abbreviations (e.g. "L1", "Lake 1", "Lake M.", "Mend.") to refer to the same ecosystem later on. Therefore, it is very important that these identifiers are collected and reported in the "abbreviations and/or codes for reference" field.

IDENTIFICATION GUIDELINES:
Treat dinitrogen fixation measurements with the same name as multiple separate items if ANY of the following differ:
- Substrate type (e.g. water column vs benthos)
- Sample depth (e.g. surface vs 0-5 cm)
- Method used to measure dinitrogen fixation (e.g. ARA vs 15N2 incorporation)
- Location (e.g. different latitude and longitude)
- Date and time

However, if the same ecosystem is referenced multiple times with the same identifying information do not duplicate it.

STRICT RULES ABOUT MISSING INFORMATION:
- Do NOT infer, guess, or derive any identifying information.
- Use ONLY information explicitly stated in the text.
- If an identifier is not explicitly given, set its value to None.

EXTRACTION PROCEDURE (FOLLOW IN ORDER):
1. Scan the entire text, including tables, for any mentions of dinitrogen fixation measurements.
2. Determine which mentions correspond to distinct dinitrogen fixation measurements using the identification guidelines.
3. Output one JSON item per distinct dinitrogen fixation measurement.
4. Collect all items into a single JSON array under the key "items".

OUTPUT FORMAT REQUIREMENTS:
- Output must be valid, strictly parseable JSON.
- Do NOT include markdown, comments, or explanatory text.
- The top-level object must have this form:
{
  "items": [
    {
      "name": <string or null>,
      "abbreviations_and_codes": <string or null>,
      "ecosystem_type": <string or null>,
      "substrate_type": <string or null>,
      "sample_depth": <string or null>,
      "nfix_method": <string or null>,
      "latitude": <number or null>,
      "longitude": <number or null>,
      "year": <number or null>,
      "month": <number or null>,
      "day": <number or null>,
      "hour_minute": <string or null>,
      "season": <string or null>
    }
  ]
}
- If no distinct ecosystems are found, output exactly:
{ "items": [] }
"""