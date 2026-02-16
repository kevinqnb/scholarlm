import os
import re
import json
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from scholarlm import MeasurementLM
from scholarlm.measurementlm import NumpyEncoder
from scholarlm.utils import get_filenames_in_directory

#task_id = int(os.getenv('SGE_TASK_ID'))

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


main_directory = "data/pond"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output")
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

text_files = get_filenames_in_directory(ocr_directory, ignore = [".DS_Store"])
text_files.sort()

text_files = ['physical_and_chemical_limnological.txt']

text_filepaths = []
text_info = []
for f in text_files:
    paper_code = f.replace('.txt', '')
    filepath = os.path.join(ocr_directory, f)
    metadata = paper_info.get(paper_code, {})
    # ID Addition:
    metadata['paper_code'] = paper_code
    text_filepaths.append(filepath)
    text_info.append(metadata)


text = []
for filepath in text_filepaths:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        text.append(content)


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

class ObservationSchema(BaseModel):
    name: str | None
    abbreviations: str | None
    location: str | None
    site: str | None
    state: str | None
    date: str | None
    ecosystem: str | None


attribute_info_dict = {
    "latitude": {
        "description": "Geographic latitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.",
        "units": ["degrees", "radians"]
    },
    "longitude": {
        "description": "Geographic longitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.",
        "units": ["degrees", "radians"]
    },
    "surface_area": {
        "description": "Surface area of the water body itself (not the watershed or catchment area). This should represent the horizontal area of open water or the stated ecosystem boundary at the time of measurement or description.",
        "units": ["km^2", "mi^2", "ha", "m^2", "acres"]
    },
    "max_depth": {
        "description": "Maximum water depth of the ecosystem, defined as the deepest point of the water body at the time of measurement or as reported in the source. This is not the mean or average depth.",
        "units": ["m", "km", "ft"]
    },
    "vegetation_cover": {
        "description": "Fraction or percentage of the ecosystem surface area covered by aquatic macrophytes or other aquatic vegetation. This should refer to areal coverage, not biomass or volume.",
        "units": ["percent", "fraction"]
    },
    "ph": {
        "description": "pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.",
        "units": []
    },
    "tn": {
        "description": "Total nitrogen concentration in the water column, including both dissolved and particulate forms and all major species (e.g., nitrate, nitrite, ammonium, organic nitrogen), as explicitly reported in the source.",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "tp": {
        "description": "Total phosphorus concentration in the water column, including both dissolved and particulate forms, as explicitly reported in the source (i.e., not just soluble reactive phosphorus or orthophosphate).",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "chla": {
        "description": "Chlorophyll-a concentration in the water column, used as a proxy for phytoplankton biomass. This should refer to extracted or in situ chlorophyll-a measurements, not total chlorophyll or other pigments unless explicitly labeled as chlorophyll-a.",
        "units": ["µg/L", "mg/L", "mg/m^3"]
    },
}


measurementlm = MeasurementLM(
    model_name="gaunernst/gemma-3-27b-it-qat-autoawq",
    entity_identification_prompt=POND_IDENTIFICATION_PROMPT,
    entity_identification_schema=ObservationSchema,
    attribute_info_dict=attribute_info_dict,
    sampling_params={
        "temperature": 0.1,
        "top_p" : 0.95,
        "top_k" : 64,
        "max_tokens" : 8192,
        "seed": 342,
    },
)


def extract_entities(text, outfile):
    """Step 1: Entity extraction with table enrichment."""
    print("Extracting entities...")
    data = []
    for i, paper in enumerate(text):
        data.append({'document_id': i, 'context': paper})

    measurementlm.data = data
    data = measurementlm._extract_entities()

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def detect_attributes(infile, outfile):
    """Step 2: Attribute detection."""
    print("Detecting attributes...")
    with open(infile, 'r') as f:
        data = json.load(f)

    measurementlm.data = data
    data = measurementlm._detect_attributes()

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def extract_values(infile, outfile):
    """Steps 3+4: Extract values from text and tables."""
    print("Extracting values...")
    with open(infile, 'r') as f:
        data = json.load(f)

    entity_attribute_data = data

    measurementlm.data = entity_attribute_data
    text_values = measurementlm._extract_values_from_text()

    measurementlm.data = entity_attribute_data
    table_values = measurementlm._extract_values_from_tables()

    data = text_values + table_values

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def standardize_and_deduplicate(infile, outfile):
    """Step 5: Standardize and deduplicate."""
    print("Standardizing and deduplicating...")
    with open(infile, 'r') as f:
        data = json.load(f)

    measurementlm.data = data
    data = measurementlm._standardize_and_deduplicate()

    dataset = []
    for datapoint in data:
        document_id = datapoint['document_id']
        doc_metadata = text_info[document_id]
        dataset.append(
            doc_metadata | datapoint
        )

    with open(outfile, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)



outfile1 = "data/experiments/2026_02_18/ten_entities.json"
extract_entities(text, outfile1)

outfile2 = "data/experiments/2026_02_18/ten_attributes.json"
detect_attributes(outfile1, outfile2)

outfile3 = "data/experiments/2026_02_18/ten_values.json"
extract_values(outfile2, outfile3)

outfile4 = "data/experiments/2026_02_18/ten_final.json"
standardize_and_deduplicate(outfile3, outfile4)


'''
data = measurementlm.fit(text)

dataset = []
for datapoint in data:
    document_id = datapoint['document_id']
    doc_metadata = text_info[document_id]
    dataset.append(
        doc_metadata | datapoint
    )


outfile = "data/experiments/2026_02_16/pond.json"
with open(outfile, 'w') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
'''
