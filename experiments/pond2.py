import os
import json
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

from scholarlm import DocumentLM, MeasurementLM #, ContextLM 
from scholarlm.utils import get_filenames_in_directory

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
text_files = text_files[:10]
text_filepaths = []
text_info = []
for f in text_files:
    filepath = os.path.join(text_directory, f)
    metadata = paper_info.get(f.replace('.json', ''), {})
    text_filepaths.append(filepath)
    text_info.append(metadata)


#doclm = DocumentLM(model = "allenai/olmOCR-2-7B-1025-FP8", ocr = False)
#text_chunks = doclm.fit(text_filepaths)
text_chunks = []
for filepath in text_filepaths:
    with open(filepath, 'r', encoding='utf-8') as file:
        doc_chunks = json.load(file)
        text_chunks.append(doc_chunks)


identification_prompt = (
    "You are an expert in identifying unique ponds, lakes, and wetlands referenced in text from scientific literature. "
    "Using the given context, find, identify, and list all individual pond, lake, or wetland ecosystems it mentions. "
    "Note that an ecosystem should be a distinct entity, and not a general reference to or an aggregate collection of ecosystems. "
    "For each ecosystem, provide the following identification attributes: name, date observed, geographic location, and ecosystem type (pond, lake, or wetland). "
    "If any attribute is not explicitly mentioned in the text, respond with the value None for that attribute. "
    "If there are multiple dates observed for what is otherwise the same ecosystem, treat each as separate, identified items. "
    "However, if any ecosystem is mentioned multiple times with identical attributes, only list it once.\n\n"
    "Format the output as a JSON object with an array of items, where each item is an object "
    "containing the specified identification attributes. For example:\n"
    "{{'items': [{{'name': 'Pond A', 'date': '2020-05-01', 'location': 'Location A', 'ecosystem': 'pond'}}, {{'name': 'Pond A', 'date': '2021-05-01', 'location': 'Location A', 'ecosystem': 'pond'}}, {{'name': 'Wetland B', 'date': None, 'location': None, 'ecosystem': 'wetland'}}]}}\n"
    "If no distinct ecosystems are found, respond with an empty list. For example:\n"
    "{{'items': []}}"
)

class Identifier(BaseModel):
    name: str | None
    date: str | None
    location: str | None
    ecosystem: str | None

class IdentificationSchema(BaseModel):
    items: list[Identifier]
    model_config = {
        'title': 'Identification Model',
        'prompt': identification_prompt
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
        json_schema_extra={'units': ["km^2", "mi^2", "ha", "m^2"]}
    )
    max_depth: float | None = Field(
        description="maximum depth",
        json_schema_extra={'units': ["m", "km", "ft"]}
    )
    vegetation_cover: float | None = Field(
        description="vegetation cover percentage",
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
        json_schema_extra={'units': ["µg/L", "mg/L"]}
    )

measurementlm = MeasurementLM(
    model_name="gaunernst/gemma-3-27b-it-int4-awq",
    item_description="ponds, lakes, or wetlands",
    identification_schema=IdentificationSchema,
    measurement_schema=MeasurementSchema,
    sampling_params={
        "temperature": 0.0,
        "top_p" : 0.95,
        "top_k" : 64,
        "max_tokens" : 4096,
        "seed": 342,
    },
    return_full_output=True
)


data = measurementlm.fit(text_chunks)
dataset = []
for datapoint in data:
    paper_id = datapoint['paper_id']
    doc_metadata = text_info[paper_id]
    dataset.append(
        doc_metadata | datapoint
    )


outfile = f"data/pond_results_10_papers_v1.json"
with open(outfile, 'w') as f:
    json.dump(dataset, f, indent=4)
#df = pd.DataFrame(dataset)
#df.to_csv(outfile, index=False)

