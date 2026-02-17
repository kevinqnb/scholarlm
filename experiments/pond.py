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

text_files = [
    'physical_and_chemical_limnological.txt',
    'physical-chemical_influences.txt',
    'prairie_wetland.txt',
    'net_heterotrophy.txt',
    'habitat_characteristics.txt',
    'biodiversity_of_constructed.txt',
    'fish_production_in_lakes.txt',
    'long-term_stability.txt',
    'diversity_of_macroinvertebrates.txt',
    'impact_of_macrophytes.txt'
]


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
- An ecosystem might be referred to by a specific name (e.g., "Lake Mendota"), a coded identifier (e.g., "P1", "W2", "Pond 1", "Pond A", etc.), or a general description (e.g., "the restored wetland", "the control pond", etc.). All of these should be extracted as distinct ecosystem observations if they meet the identity criteria below.


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
        "description": "Surface area of the water body itself, representing the horizontal area of open water or the stated ecosystem boundary at the time of measurement or description. This is NOT the same as watershed area, drainage basin area, catchment area, or littoral zone area.",
        "units": ["km^2", "mi^2", "ha", "m^2", "acres"]
    },
    "max_depth": {
        "description": "Maximum physical water depth of the ecosystem, defined as the deepest point of the water body at the time of measurement or as reported in the source. This is NOT the same as mean depth, average depth, or Secchi depth (water transparency depth).",
        "units": ["m", "km", "ft"]
    },
    "vegetation_cover": {
        "description": "Fraction or percentage of the ecosystem surface area covered by aquatic macrophytes or other rooted/floating aquatic vegetation. This should refer to areal coverage, not biomass or volume. This is NOT the same as algal cover, periphyton cover, or phytoplankton density.",
        "units": ["percent", "fraction"]
    },
    "ph": {
        "description": "pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.",
        "units": []
    },
    "tn": {
        "description": "Total nitrogen (TN) concentration in the water column, representing the sum of all nitrogen forms — both dissolved and particulate, including nitrate (NO₃), nitrite (NO₂), ammonium (NH₃/NH₄⁺), and organic nitrogen. This must be the aggregate 'total nitrogen' value as explicitly reported in the source. This is NOT the same as individual nitrogen species (e.g., NO₃ alone, NO₂ alone, NH₃ alone, combined NO₃+NO₂, or particulate organic nitrogen [PON]) unless they are explicitly labeled as total nitrogen.",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "tp": {
        "description": "Total phosphorus (TP) concentration in the water column, representing the sum of all phosphorus forms — both dissolved and particulate. This must be the aggregate 'total phosphorus' value as explicitly reported in the source. This is NOT the same as individual phosphorus species (e.g., soluble reactive phosphorus [SRP], orthophosphate [PO₄³⁻], dissolved reactive phosphorus [DRP], or particulate phosphorus [PP]) unless they are explicitly labeled as total phosphorus.",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "chla": {
        "description": "Chlorophyll-a (Chl-a) concentration in the water column, used as a proxy for phytoplankton biomass. This should refer to extracted or in situ chlorophyll-a measurements only. This is NOT the same as total chlorophyll, chlorophyll-b, chlorophyll-c, pheophytin, or other pigment measurements unless they are explicitly labeled as chlorophyll-a.",
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


def detect_attributes(text, outfile):
    """Step 2: Document-level attribute detection."""
    print("Detecting attributes...")
    data = []
    for i, paper in enumerate(text):
        data.append({'document_id': i, 'context': paper})

    measurementlm.data = data
    doc_attributes = measurementlm._detect_attributes()

    with open(outfile, 'w') as f:
        json.dump(doc_attributes, f, indent=4, ensure_ascii=False)


def combine_and_filter(entities_file, attributes_file, outfile):
    """Step 3: Cross-product entities x attributes, then filter pairs."""
    print("Combining and filtering entity-attribute pairs...")
    with open(entities_file, 'r') as f:
        entity_data = json.load(f)

    with open(attributes_file, 'r') as f:
        doc_attributes = json.load(f)

    # doc_attributes keys are strings after JSON round-trip
    entity_attribute_data = []
    for record in entity_data:
        doc_id = str(record['document_id'])
        for attr_name, terms in doc_attributes.get(doc_id, {}).items():
            entity_attribute_data.append(record | {
                'attribute': attr_name,
                'attribute_terms': terms,
            })

    measurementlm.data = entity_attribute_data
    data = measurementlm._filter_entity_attribute_pairs()

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
detect_attributes(text, outfile2)

outfile3 = "data/experiments/2026_02_18/ten_pairs.json"
combine_and_filter(outfile1, outfile2, outfile3)

outfile4 = "data/experiments/2026_02_18/ten_values.json"
extract_values(outfile3, outfile4)

outfile5 = "data/experiments/2026_02_18/ten_final.json"
standardize_and_deduplicate(outfile4, outfile5)


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
