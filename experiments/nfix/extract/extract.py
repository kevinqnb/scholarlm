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

from extract_prompts import NFIX_IDENTIFICATION_PROMPT

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


#task_id = int(os.getenv('SGE_TASK_ID'))


main_directory = "data/nfix"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output_cleaned_vllm")
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)


def text_or_table_extraction(location):
    if 'figure' in location:
        return False
    if 'supplement' in location:
        return False
    if 'archive' in location:
        return False
    if 'author' in location:
        return False
    else:
        return True

registered_paper_info = {
    R: Rinfo for R,Rinfo in paper_info.items() if text_or_table_extraction(Rinfo['extraction_location']) 
}
registered_ids = list(registered_paper_info.keys())

text_files = get_filenames_in_directory(ocr_directory, ignore = [".DS_Store", ".gitkeep"])
text_files.sort()
text_files = [f for f in text_files if f.replace('.txt', '') in registered_ids]

text_files = [
    "R163.txt",
    "R164.txt",
    "R172.txt",
    "R248.txt",
    "R124.txt",
    "R51.txt",
    "R59.txt",
    "R114.txt",
    "R43.txt",
    "R103.txt"
]

'''
# split into 5 groups for 5 tasks; each task processes one group of files
files_per_task = len(text_files) // 5
start_index = (task_id - 1) * files_per_task
global_offset = start_index  # for tracking document_id across tasks
if task_id < 5:
    text_files = text_files[start_index : start_index + files_per_task]
else:
    text_files = text_files[start_index:]
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

class ObservationSchema(BaseModel):
    name: str | None
    abbreviations: str | None
    ecosystem_type: str | None
    latitude: float | None
    longitude: float | None
    date: str | None
    nfix_method: str | None
    substrate_type: str | None
    sample_depth: str | None

# Mass units
mass_units = [
    "nmol-N g-1 h-1",
    "nmol-C2H4 g-1 h-1",
    "nmol-N2 g-1 h-1",
    "ug-N g-1 d-1",
    "nmol-N2 g-1 d-1",
    "umol-N g-1 d-1",
    "nmol-C2H4 g-1 d-1",
    "nmol-N g-1 d-1",
    "ug-N g-1 h-1",
    "ug-N kg-1 d-1",
    "umol-N g-1 h-1",
    "fmol-N g-1 h-1",
    "ng-N g-1 d-1",
    "ng-N g-1 h-1",
    "nmol-N kg-1 h-1",
    "umol-C2H4 g-1 d-1",
    "umol-N kg-1 h-1",
    "umol-N2 g-1 d-1",
]

# Areal units
areal_units = [
    "umol-N m-2 h-1",
    "mg-N m-2 d-1",
    "umol-N m-2 d-1",
    "umol-C2H4 m-2 h-1",
    "nmol-C2H4 cm-2 h-1",
    "mmol-N m-2 d-1",
    "ug-N m-2 h-1",
    "mg-N m-2 h-1",
    "nmol-C2H4 cm-2 d-1",
    "nmol-C2H4 m-2 h-1",
    "umol-N2 m-2 h-1",
    "g-N m-2 yr-1",
    "mmol-N m-2 h-1",
    "mmol-N2 m-2 d-1",
    "nmol-N cm-2 h-1",
    "umol-N2 m-2 d-1",
    "kg-N2 ha-1 yr-1",
    "mg-N m-2 yr-1",
    "mg-N2 m-2 h-1",
    "ng-N m-2 h-1",
    "nmol-C2H4 m-2 d-1",
    "ug-N cm-2 h-1",
    "ug-N2 m-2 h-1"
]

# Volumetric units
volumetric_units = [
    "nmol-N L-1 d-1",
    "nmol-N L-1 h-1",
    "nmol-C2H4 L-1 h-1",
    "ug-N L-1 h-1",
    "ng-N L-1 h-1",
    "mg-N m-3 d-1",
    "nmol-C2H4 cm-3 h-1",
    "nmol-C2H4 mL-1 h-1",
    "nmol-N cm-3 d-1",
    "nmol-N cm-3 h-1",
    "ug-N m-3 h-1",
    "umol-N2 L-1 d-1",
    "umol-N2 L-1 h-1",
    "mmol-C2H4 m-3 d-1",
    "nmol-C2H4 cm-3 d-1",
    "nmol-N m-3 h-1",
    "nmol-N2 cm-3 d-1",
    "nmol-N2 L-1 d-1",
    "nmol-N2 L-1 h-1",
    "ug-N L-1 d-1",
    "ug-N2 L-1 h-1",
    "ug-N2 m-3 d-1",
    "umol-C2H4 L-1 d-1",
    "umol-C2H4 mL-1 3h-1",
    "umol-N L-1 d-1",
    "umol-N L-1 h-1"
]


attribute_info_dict = {
    # --- Rate attributes ---
    "nfix_rate_mass": {
        "description": (
            "Rate of dinitrogen fixation per unit mass: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by substrate mass. Not equivalent to rates reported per unit area or volume."
        ),
        "units": mass_units
    },
    "nfix_rate_areal": {
        "description": (
            "Rate of dinitrogen fixation per unit area: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by area. Not equivalent to rates reported per unit mass or volume."
        ),
        "units": areal_units
    },
    "nfix_rate_volumetric": {
        "description": (
            "Rate of dinitrogen fixation per unit volume: the amount of nitrogen "
            "(or ethylene in acetylene reduction assays) per fixed unit of time, "
            "normalized by water volume. Not equivalent to rates reported per unit mass or area."
        ),
        "units": volumetric_units
    },
    # --- Incubation conditions ---
    "nfix_incubation_time": {
        "description": (
            "Duration of the experimental incubation for measuring dinitrogen "
            "fixation, from introduction of the tracer or substrate analog to "
            "termination and sampling."
        ),
        "units": ["minutes", "hours", "days"]
    },
    "nfix_incubation_temperature": {
        "description": (
            "Temperature at which the sample was held during the dinitrogen "
            "fixation incubation. Extract only if a specific numeric temperature "
            "is reported for the incubation itself. Do not extract in situ water "
            "temperatures unless the text explicitly states they equal the "
            "incubation temperature. If the text says only 'ambient temperature' "
            "or 'in situ temperature' without a numeric value, set to None."
        ),
        "units": ["°C", "K"]
    },
}


measurementlm = MeasurementLM(
    model_name="gaunernst/gemma-3-27b-it-qat-autoawq",
    entity_identification_prompt=NFIX_IDENTIFICATION_PROMPT,
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
    for i, datapoint in enumerate(deduplicated):
        document_id = datapoint['document_id']
        doc_metadata = text_info[document_id]
        dataset.append(
            doc_metadata | datapoint | {'measurement_id': i}
        )

    with open(outfile, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)



outfile1 = "data/experiments/2026_04_01/ten_nfix_entities.json"
extract_entities(text, outfile1)

outfile2 = "data/experiments/2026_04_01/ten_nfix_attributes.json"
detect_attributes(text, outfile2)

outfile3a = "data/experiments/2026_04_01/ten_nfix_entity_prov.json"
entity_provenance(text, outfile1, outfile3a)

outfile3b = "data/experiments/2026_04_01/ten_nfix_attribute_prov.json"
attribute_provenance(text, outfile2, outfile3b)

outfile4 = "data/experiments/2026_04_01/ten_nfix_values.json"
extract_values(text, outfile1, outfile2, outfile3a, outfile3b, outfile4)

outfile5 = "data/experiments/2026_04_01/ten_nfix_final.json"
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


outfile = f"data/experiments/2026_03_04/pond{task_id}_vllm.json"
with open(outfile, 'w') as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
'''
