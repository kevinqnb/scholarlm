import os
import json
import math
import numpy as np
import random
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from scholarlm import JudgementLM
from scholarlm import JUDGE_INSTRUCTIONS

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


####################################################################################################


class ObservationSchema(BaseModel):
    name: str | None
    abbreviations: str | None
    location: str | None
    site: str | None
    state: str | None
    date: str | None
    ecosystem: str | None

fields = ObservationSchema.model_fields.keys()

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


####################################################################################################

ctxlm_params = {
    'do_sample': False,
    'max_new_tokens': 1,
}

llm = JudgementLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    sampling_params=ctxlm_params,
    nnsight_kwargs = {"torch_dtype": torch.bfloat16},
)

####################################################################################################

input_file = "data/experiments/2026_02_18/pond.json"
output_file = f"data/experiments/2026_02_18/pond_judged_llama.json"
attn_output_file = "data/experiments/2026_02_18/pond_judged_llama_attention_outputs.npz"

with open(input_file, "r") as f:
    data = json.load(f)

messages = []
message_ids = []
for entry in data:
    context = entry['context']
    attribute = entry.get('attribute')
    attribute_description = attribute_info_dict[attribute]['description']
    attribute_terms = entry.get('attribute_terms', [])
    entity_description = {k: v for k,v in entry.items() if k in fields}
    measurement_val = entry['value']
    measurement_id = entry['measurement_id']

    instructions = JUDGE_INSTRUCTIONS
    query = (
        f"attribute description: {attribute_description}\n"
        f"Terminology used for the attribute: {attribute_terms}\n"
        f"Entity description: {entity_description}\n"
        f"Extracted measurement: {measurement_val}\n\n"
        f"Is the extracted data point valid for the given entity and attribute?"
    )
    
    messages.append((instructions, context, query))
    message_ids.append(measurement_id)


responses = llm.predict(messages)
judged_data = []
attn_output_dict = {}
for i, response in enumerate(responses):
    measurement_id = str(message_ids[i])
    judged_data_point = data[i] | {
        'judgement': True if "true" in response['response'].strip().lower() else False,
        'judgement_confidence': math.exp(float(response['logprob'])),
        'judgement_model': 'Llama-3.1-8B-Instruct',
    }
    judged_data.append(judged_data_point)
    attn_output = response.get('attn_output', None)
    if attn_output is not None:
        attn_output_dict[measurement_id] = attn_output


with open(output_file, "w") as f:
    json.dump(judged_data, f, indent=4, ensure_ascii=False)

np.savez_compressed(attn_output_file, **attn_output_dict)

####################################################################################################