import os
import re
import json
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scholarlm import DocumentLM, MeasurementLM, ContextLM 
from scholarlm.utils import get_filenames_in_directory

'''
# NDIF Setup
from nnsight import CONFIG
ndif_key = os.getenv("NDIF_KEY")
CONFIG.set_default_api_key(ndif_key)

# Hugging face setup
hf_key = os.getenv("HUGGINGFACE_KEY")
os.environ['HF_TOKEN'] = hf_key
'''

#task_id = int(os.getenv('SGE_TASK_ID'))

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
    identification_prompt=identification_prompt,
    identification_schema=IdentificationSchema,
    measurement_schema=MeasurementSchema,
    sampling_params={
        "temperature": 0.6,
        "top_p" : 0.95,
        "top_k" : 64,
        "max_tokens" : 8192,
        "seed": 342,
    },
    return_full_output=True,
    page_selection = False,
    cache_dir="data/12_10_25/pond_full_paper",
    use_llm = False
)

def measure(infile, outfile):
    with open(infile, 'r') as f:
        data = json.load(f)

    #import pdb; pdb.set_trace()
    
    measurementlm.data = data
    data = measurementlm._measure()

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)



outfile1 = "data/12_10_25/pond_full_paper_page_identified_gemma.json"
outfile2 = "data/12_10_25/pond_full_paper_measured_llama.json"
measure(outfile1, outfile2)
