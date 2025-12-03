import os
import json
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from langchain_text_splitters import RecursiveCharacterTextSplitter
from scholarlm import DocumentLM, MeasurementLM #, ContextLM 
from scholarlm.utils import get_filenames_in_directory

#task_id = int(os.getenv('SGE_TASK_ID')) - 1
#available_jobs = [2,4,5,6,8,10]
#task_id = available_jobs[task_id - 1]

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


main_directory = os.getenv("POND_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH")

with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)


text_files = get_filenames_in_directory(text_directory, ignore = [".DS_Store"])
text_files.sort()
#text_files = text_files[:10]

#text_files = [f for i, f in enumerate(text_files) if i % 4 == task_id]

#text_files = [
#    'physical_and_chemical_limnological.json',
#]

text_filepaths = []
text_info = []
for f in text_files:
    paper_code = f.replace('.json', '')
    filepath = os.path.join(text_directory, f)
    metadata = paper_info.get(paper_code, {})
    # ID Addition:
    metadata['paper_code'] = paper_code
    text_filepaths.append(filepath)
    text_info.append(metadata)

text_splitter = RecursiveCharacterTextSplitter(
    separators = ["<table>"],
    chunk_size = 15000,
    chunk_overlap  = 100,
)
text_chunks = []
for filepath in text_filepaths:
    with open(filepath, 'r', encoding='utf-8') as file:
        doc_chunks = json.load(file)
        #text_chunks.append(doc_chunks)
        split_chunks = {}
        for page_id, chunk in doc_chunks.items():
            chunk_texts = text_splitter.split_text(chunk)
            split_chunks[page_id] = chunk_texts

        text_chunks.append(split_chunks)


identification_prompt = (
    "You are an expert in identifying ponds, lakes, and wetlands referenced in text from scientific literature. "
    "Using the given context, find, identify, and list all individual pond, lake, or wetland ecosystems it mentions. "
    "Any identified ecosystem must be a distinct entity, and not a general reference to or an aggregate collection of ecosystems. "
    "For each ecosystem, provide the following identification attributes: name, date observed, geographic location, and ecosystem type (pond, lake, wetland, or other). "
    "If any one of these attributes is not explicitly mentioned in the text, respond with the value None for that attribute. "
    "If there are multiple dates observed for what is otherwise the same ecosystem, treat each as separate, identified items. "
    "However, if any ecosystem is mentioned multiple times with identical attributes, only list it once.\n\n"
    "Format the output as a JSON object with an array of items, where each item is an object "
    "containing the specified identification attributes. For example:\n"
    "{{'items': [{{'name': 'Pond A', 'date': '2020-05-01', 'location': 'Location A', 'ecosystem': 'pond'}}, {{'name': 'Pond A', 'date': '2021-05-01', 'location': 'Location A', 'ecosystem': 'pond'}}, {{'name': 'Wetland B', 'date': None, 'location': None, 'ecosystem': 'wetland'}}]}}\n"
    "If no distinct ecosystems are found, respond with an empty list. For example:\n"
    "{{'items': []}}"
)


class IdentificationSchema(BaseModel):
    name: str | None
    date: str | None
    location: str | None
    ecosystem: str | None
    model_config = {
        'title': 'Ecosystem Identifier',
        'entity_description': 'ponds, lake, or wetland ecosystems',
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
    model_name="gaunernst/gemma-3-27b-it-int4-awq",
    identification_prompt=identification_prompt,
    identification_schema=IdentificationSchema,
    measurement_schema=MeasurementSchema,
    sampling_params={
        "temperature": 0.0,
        "top_p" : 0.95,
        "top_k" : 64,
        "max_tokens" : 4096,
        "seed": 342,
    },
    return_full_output=True, # THIS USES A SEPARATE INTERPRETABLE MODEL
    cache_dir="data/pond_page_chunks_fix"
)


def filter_identify(text_chunks, outfile):
    data = []
    for i in range(len(text_chunks)):
        for j, page_chunks in text_chunks[i].items():
            for k, chunk in enumerate(page_chunks):
                data.append({'document_id': i, 'page_id': j, 'chunk_id': k, 'context' : chunk})

    measurementlm.data = data
    data = measurementlm._filter()
    measurementlm.data = data
    data = measurementlm._identify()
    measurementlm.data = data
    data = measurementlm._measurements_filter()
    measurementlm.data = data

    # Save intermediate results
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def measure(infile, outfile):
    with open(infile, 'r') as f:
        data = json.load(f)
        
    measurementlm.data = data
    data = measurementlm._measure()

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def standardize_and_save(infile, outfile):
    with open(infile, 'r') as f:
        data = json.load(f)

    measurementlm.data = data
    data = measurementlm._standardize()

    dataset = []
    for datapoint in data:
        document_id = datapoint['document_id']
        doc_metadata = text_info[document_id]
        dataset.append(
            doc_metadata | datapoint
        )

    with open(outfile, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


#data = measurementlm.fit(text_chunks)

outfile1 = "data/12_03_25/pond_page_chunks_intermediate_fix.json"
filter_identify(text_chunks, outfile1)

outfile2 = "data/12_03_25/pond_page_chunks_measured_fix.json"
measure(outfile1, outfile2)

outfile3 = "data/12_03_25/pond_page_chunks_fix.json"
standardize_and_save(outfile2, outfile3)