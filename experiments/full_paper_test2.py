import os
import re
import json
import pandas as pd
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scholarlm import DocumentLM, MeasurementLM #, ContextLM 
from scholarlm.utils import get_filenames_in_directory

#task_id = int(os.getenv('SGE_TASK_ID'))

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


main_directory = os.getenv("POND_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH2")

with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)


md_files = get_filenames_in_directory(md_directory, ignore = [".DS_Store"])
md_files.sort()

'''
text_files = [
    'physical_and_chemical_limnological.json',
    'physical-chemical_influences.json',
    'prairie_wetland.json',
    'net_heterotrophy.json',
    'habitat_characteristics.json',
    'biodiversity_of_constructed.json',
    'fish_production_in_lakes.json',
    'long-term_stability.json',
    'diversity_of_macroinvertebrates.json',
    'impact_of_macrophytes.json'
]
'''

md_filepaths = []
md_info = []
for f in md_files:
    paper_code = f.replace('.md', '')
    filepath = os.path.join(md_directory, f)
    metadata = paper_info.get(paper_code, {})
    # ID Addition:
    metadata['paper_code'] = paper_code
    md_filepaths.append(filepath)
    md_info.append(metadata)


def remove_references(text):
    """
    Remove everything from the beginning of any reference-like section.
    Supports:
      - Plain headings       (e.g., "References", "Bibliography:")
      - Markdown headings    (# References, ## Bibliography:)
      - Numbered headings    (7. References, 3.2 Bibliography:)
    """

    # All headings we want to detect
    headings = [
        "references",
        "bibliography",
        "works cited",
        "literature cited",
        "sources",
        "references and notes",
        "notes and references",
    ]

    # Build the alternation group for headings
    heading_group = "|".join(re.escape(h) for h in headings)

    # Build the full regex pattern
    # Explanation:
    #   ^\s*                  → start of line + optional whitespace
    #   (?:#+\s*)?            → optional markdown heading (e.g. "## ")
    #   (?:\d+(\.\d+)*\s*)?   → optional numbering (e.g. "7." or "3.2.1 ")
    #   (?:heading)           → one of our known headings
    #   \s*:?\s*$             → optional colon, trailing whitespace, end of line
    pattern = re.compile(
        rf"(?im)"                      # case-insensitive + multiline
        rf"^[ \t]*"                    # leading whitespace
        rf"(?:#+[ \t]*)?"              # optional markdown #
        rf"(?:\d+(?:\.\d+)*[ \t]*)?"   # optional number prefixes
        rf"(?:{heading_group})"        # one of our headings
        rf"[ \t]*:?[ \t]*\r?\n"        # optional colon, then newline
    )

    match = pattern.search(text)
    if match:
        return text[:match.start()].rstrip()

    return text


text = []
for filepath in md_filepaths:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Remove references:
    content = remove_references(content)
    text.append(content)


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
    #model_name="gaunernst/gemma-3-27b-it-int4-awq",
    model_name="gaunernst/gemma-3-27b-it-qat-autoawq",
    #model_name="cyankiwi/Olmo-3-32B-Think-AWQ-4bit",
    #model_name="allenai/Olmo-3-7B-Instruct",
    #model_name="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
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
    return_full_output=False,
    page_selection = True,
)


def filter_identify(text, outfile):
    data = []
    for i, paper in enumerate(text):
        data.append({'document_id': i, 'context' : paper})
    
    measurementlm.data = data
    data = measurementlm._identify()
    measurementlm.data = data
    data = measurementlm._measurements_filter()

    # Save intermediate results
    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def page_identify(infile, outfile):
    with open(infile, 'r') as f:
        data = json.load(f)
        
    measurementlm.data = data
    data = measurementlm._page_id()

    with open(outfile, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def measure(infile, outfile):
    with open(infile, 'r') as f:
        data = json.load(f)
    
    measurementlm.data = data
    data = measurementlm._measure_vllm()

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
        doc_metadata = md_info[document_id]
        dataset.append(
            doc_metadata | datapoint
        )

    with open(outfile, 'w') as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


#data = measurementlm.fit(text_chunks)

outfile1 = "data/12_10_25/pond_identified_mistral.json"
filter_identify(text, outfile1)

outfile2 = "data/12_10_25/pond_paged_mistral.json"
#page_identify(outfile1, outfile2)

outfile3 = "data/12_10_25/pond_measured_mistral.json"
measure(outfile1, outfile3)

outfile4 = "data/12_10_25/pond_mistral.json"
standardize_and_save(outfile3, outfile4)
