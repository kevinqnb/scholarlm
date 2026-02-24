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

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


task_id = int(os.getenv('SGE_TASK_ID'))


main_directory = "data/pond"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output_cleaned_openai")
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

text_files = get_filenames_in_directory(ocr_directory, ignore = [".DS_Store", ".gitkeep"])
text_files.sort()

# split into 5 groups for 5 tasks; each task processes one group of files
files_per_task = len(text_files) // 5
start_index = (task_id - 1) * files_per_task
global_offset = start_index  # for tracking document_id across tasks
if task_id < 5:
    text_files = text_files[start_index : start_index + files_per_task]
else:
    text_files = text_files[start_index:]

'''
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
'''


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

    # Strip context before saving (re-injected from text[] in later steps)
    save_data = [{k: v for k, v in r.items() if k != 'context'} for r in data]
    with open(outfile, 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)


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


def _serialize_prov(prov_dict):
    """Serialize tuple-keyed provenance dict for JSON (keys become 'doc_id|item_id' strings)."""
    return {f"{k[0]}|{k[1]}": v for k, v in prov_dict.items()}


def _deserialize_prov(json_dict):
    """Deserialize JSON provenance dict back to tuple keys."""
    out = {}
    for k, v in json_dict.items():
        parts = k.split("|", 1)
        # doc_id is int, second part is entity_id or attr_name (string)
        try:
            doc_id = int(parts[0])
        except ValueError:
            doc_id = parts[0]
        out[(doc_id, parts[1])] = v
    return out


def entity_provenance(text, entities_file, outfile):
    """Step 2a: Entity provenance — locate pages/tables with data per entity."""
    print("Running entity provenance...")
    with open(entities_file, 'r') as f:
        entity_data = json.load(f)

    # Restore full document context into entity records
    for record in entity_data:
        doc_id = record['document_id']
        record['context'] = text[doc_id]

    prov = measurementlm._entity_provenance(entity_data)

    with open(outfile, 'w') as f:
        json.dump(_serialize_prov(prov), f, indent=4, ensure_ascii=False)


def attribute_provenance(text, attributes_file, outfile):
    """Step 2b: Attribute provenance — locate pages/tables with data per attribute."""
    print("Running attribute provenance...")
    with open(attributes_file, 'r') as f:
        doc_attributes = json.load(f)

    # doc_attributes keys are strings after JSON round-trip; _attribute_provenance handles that
    data = []
    for i, paper in enumerate(text):
        data.append({'document_id': i, 'context': paper})
    measurementlm.data = data

    prov = measurementlm._attribute_provenance(doc_attributes)

    with open(outfile, 'w') as f:
        json.dump(_serialize_prov(prov), f, indent=4, ensure_ascii=False)


def extract_values(text, entities_file, attributes_file, entity_prov_file, attr_prov_file, outfile):
    """Steps 5+6: Extract values from text and tables using provenance intersection."""
    print("Extracting values...")
    with open(entities_file, 'r') as f:
        entity_data = json.load(f)

    with open(attributes_file, 'r') as f:
        doc_attributes = json.load(f)

    with open(entity_prov_file, 'r') as f:
        entity_prov = _deserialize_prov(json.load(f))

    with open(attr_prov_file, 'r') as f:
        attr_prov = _deserialize_prov(json.load(f))

    # doc_attributes keys are strings after JSON; convert to int for consistency
    doc_attributes = {int(k): v for k, v in doc_attributes.items()}

    # Restore full document context into entity records
    for record in entity_data:
        doc_id = record['document_id']
        record['context'] = text[doc_id]

    text_values = measurementlm._extract_values_from_text(
        entity_data, doc_attributes, entity_prov, attr_prov
    )

    table_values = measurementlm._extract_values_from_tables(
        entity_data, doc_attributes, entity_prov, attr_prov
    )

    data = text_values + table_values

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def standardize_and_deduplicate(infile, outfile):
    """Step 7+8: Standardize then deduplicate."""
    print("Standardizing and deduplicating...")
    with open(infile, 'r') as f:
        data = json.load(f)

    measurementlm.data = data
    standardized = measurementlm._standardize()
    deduplicated = measurementlm._deduplicate(standardized)

    dataset = []
    for datapoint in deduplicated:
        document_id = datapoint['document_id']
        doc_metadata = text_info[document_id]
        dataset.append(
            doc_metadata | datapoint
        )

    with open(outfile, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


'''
outfile1 = "data/experiments/2026_02_18/prov_ten_entities.json"
extract_entities(text, outfile1)

outfile2 = "data/experiments/2026_02_18/prov_ten_attributes.json"
detect_attributes(text, outfile2)

outfile3a = "data/experiments/2026_02_18/prov_ten_entity_prov.json"
entity_provenance(text, outfile1, outfile3a)

outfile3b = "data/experiments/2026_02_18/prov_ten_attr_prov.json"
attribute_provenance(text, outfile2, outfile3b)

outfile4 = "data/experiments/2026_02_18/prov_ten_values.json"
extract_values(text, outfile1, outfile2, outfile3a, outfile3b, outfile4)

outfile5 = "data/experiments/2026_02_18/prov_ten_final.json"
standardize_and_deduplicate(outfile4, outfile5)
'''


data = measurementlm.fit(text)

dataset = []
for datapoint in data:
    document_id = datapoint['document_id']
    doc_metadata = text_info[document_id]
    datapoint['document_id'] = document_id + global_offset  # adjust for global document ID across tasks
    dataset.append(
        doc_metadata | datapoint
    )


outfile = f"data/experiments/2026_02_25/pond{task_id}_openai.json"
with open(outfile, 'w') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

