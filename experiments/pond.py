import os
import re
import json
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from scholarlm import DocumentLM2, MeasurementLM #, ContextLM 
from scholarlm.utils import get_filenames_in_directory

#task_id = int(os.getenv('SGE_TASK_ID'))

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


main_directory = os.getenv("POND_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH2")

with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

text_files = get_filenames_in_directory(text_directory, ignore = [".DS_Store"])
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
    filepath = os.path.join(text_directory, f)
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


identification_prompt = """You are an expert in identifying ponds, lakes, and wetlands referenced in text from scientific literature. Using the given context, find, identify, and list all individual pond, lake, or wetland ecosystems it mentions. 
* Any identified ecosystem must be a distinct entity, and not a general reference to or an aggregate collection of ecosystems. 
* For each ecosystem, provide the following identification attributes: name, date observed, geographic location, and ecosystem type (pond, lake, wetland, or other). If any one of these attributes is not explicitly mentioned in the text, respond with the value None for that attribute. 
* Each identified entity should be as specific as possible to the state it was measured in. If there are multiple dates observed for what is otherwise the same ecosystem, treat each as separate, identified items. If an ecosystem is measured in different physical states (e.g. after treatment or during different seasons), treat each as separate, identified items, and distinguish them in the name attribute. However, if any ecosystem is mentioned multiple times with identical attributes or states, only list it once. 
* If the text uses multiple names that refer to the same ecosystem, use only the most complete, full-form name that still uniquely identifies the ecosystem. It is also acceptable to use codes or abbreviations if that is the only form of the name given. 
* Make sure to check table rows or columns for any entities which may be listed there, but  are not mentioned elsewhere in the paper.
* Format the output as a JSON object with an array of items, where each item is an object containing the specified identification attributes. If no distinct ecosystems are found within the context, the items array should be empty.

Example response 1: {{'items': [{{'name': 'Pond A', 'date': '2020-05-01', 'location': 'Location A', 'ecosystem': 'pond'}}, {{'name': 'Pond A', 'date': '2021-05-01', 'location': 'Location A', 'ecosystem': 'pond'}}, {{'name': 'Wetland B', 'date': None, 'location': None, 'ecosystem': 'wetland'}}]}} 

Example response 2 (no distinct ecosystems are found): {{'items': []}}
"""

class IdentificationSchema(BaseModel):
    name: str | None
    date: str | None
    location: str | None
    ecosystem: str | None
    model_config = {
        'title': 'Ecosystem Identifier',
        'entity_description': 'ponds, lakes, or wetlands',
        'primary_identifier': 'name'
    }

class MeasurementSchema(BaseModel):
    latitude: float | None = Field(
        description="latitude",
        json_schema_extra={'units': ["degrees", "radians"]}
    )
    longitude: float | None = Field(
        description="longitude",
        json_schema_extra={'units': ["degrees", "radians"]}
    )
    surface_area: float | None = Field(
        description="surface area",
        json_schema_extra={'units': ["km^2", "mi^2", "ha", "m^2", "acres"]}
    )
    max_depth: float | None = Field(
        description="maximum depth",
        json_schema_extra={'units': ["m", "km", "ft"]}
    )
    vegetation_cover: float | None = Field(
        description="aquatic macrophyte percent coverage",
        json_schema_extra={'units': ["percent", "fraction"]}
    )
    ph: float | None = Field(
        description="pH level",
        json_schema_extra={'units': None}
    )
    tn: float | None = Field(
        description="total nitrogen concentration",
        json_schema_extra={'units': ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]}
    )
    tp: float | None = Field(
        description="total phosphorus concentration",
        json_schema_extra={'units': ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]}
    )
    chla: float | None = Field(
        description="chlorophyll-a concentration",
        json_schema_extra={'units': ["µg/L", "mg/L", "mg/m^3"]}
    )

measurementlm = MeasurementLM(
    model_name="gaunernst/gemma-3-27b-it-qat-autoawq",
    #model_name="cyankiwi/Olmo-3.1-32B-Instruct-AWQ-8bit",
    #model_name="Valdemardi/DeepSeek-R1-Distill-Qwen-32B-AWQ"
    identification_prompt=identification_prompt,
    identification_schema=IdentificationSchema,
    measurement_schema=MeasurementSchema,
    sampling_params={
        "temperature": 0.1,
        "top_p" : 0.95,
        "top_k" : 64,
        "max_tokens" : 8192,
        "seed": 342,
    },
    probe = False,
)


import math
import numpy as np
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Check for NaN
            if math.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def identify(text, outfile):
    print("Identifying...")
    data = []
    for i, paper in enumerate(text):
        data.append({'document_id': i, 'context' : paper})

    measurementlm.data = data
    data = measurementlm._identify_measurements()
    measurementlm.data = data
    data = measurementlm._identify_entities()

    # Save intermediate results
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def locate(infile, outfile):
    print("Locating...")
    with open(infile, 'r') as f:
        data = json.load(f)
    
    measurementlm.data = data
    data = measurementlm._page_locate()
    #measurementlm.data = data
    #data = measurementlm._in_table()
    measurementlm.data = data
    data = measurementlm._table_locate()

    if measurementlm.probe:
        # Save attention outputs separately
        attn_output_dir = "data/01_14_26/attention_outputs"
        filtered_data = []
        for entry in data:
            page_attn_output = entry.pop('page_attn_output', None)
            table_attn_output = entry.pop('table_attn_output', None)
            filtered_data.append(entry)

            measurement_id = entry['measurement_id']
            if page_attn_output is not None:
                attn_outfile = os.path.join(attn_output_dir, f"page_attn_{measurement_id}.npz")
                np.savez_compressed(attn_outfile, attn_output=page_attn_output)
            if table_attn_output is not None:
                attn_outfile = os.path.join(attn_output_dir, f"table_attn_{measurement_id}.npz")
                np.savez_compressed(attn_outfile, attn_output=table_attn_output)

        data = filtered_data

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def measure(infile, outfile):
    print("Measuring...")
    with open(infile, 'r') as f:
        data = json.load(f)
    
    measurementlm.data = data
    data = measurementlm._measure_vllm()
    measurementlm.data = data
    data = measurementlm._measure_vllm_rows()
    measurementlm.data = data
    data = measurementlm._measure_vllm_columns()
    measurementlm.data = data
    data = measurementlm._table_extract()

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def standardize(infile, outfile):
    print("Standardizing...")
    with open(infile, 'r') as f:
        data = json.load(f)

    measurementlm.data = data
    data = measurementlm._standardize_measurements()
    measurementlm.data = data
    data = measurementlm._standardize_units()

    dataset = []
    for datapoint in data:
        document_id = datapoint['document_id']
        doc_metadata = text_info[document_id]
        dataset.append(
            doc_metadata | datapoint
        )

    with open(outfile, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


#data = measurementlm.fit(text_chunks)

outfile1 = "data/01_14_26/ten_identify4.json"
identify(text, outfile1)

outfile2 = "data/01_14_26/ten_locate4.json"
locate(outfile1, outfile2)

outfile3 = "data/01_14_26/ten_measure4.json"
measure(outfile2, outfile3)

outfile4 = "data/01_14_26/ten_standardize4.json"
standardize(outfile3, outfile4)
