import os
import json
import numpy as np
import random
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()
from scholarlm import ContextLM
from scholarlm import JUDGE_INSTRUCTIONS

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


####################################################################################################

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

####################################################################################################

ctxlm_params = {
    'do_sample': False,
    'max_new_tokens': 1,
}

llm = ContextLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    sampling_params=ctxlm_params,
    nnsight_kwargs = {"torch_dtype": torch.bfloat16},
)

####################################################################################################

input_file = "data/01_14_26/ten_standardize3.json"
output_file = f"data/01_14_26/ten_judged3.json"
attn_output_file = "data/01_14_26/ten_judged3_attention_outputs.npz"

with open(input_file, "r") as f:
    data = json.load(f)

messages = []
message_ids = []
for entry in data:
    context = entry.get('context', None)
    name = entry.get('name', None)
    feature = MeasurementSchema.model_fields[entry['measurement']].description
    value = entry.get('value', None)
    units = entry.get('units', None)
    entity_names = entry.get('entity_names', [])
    feature_names = entry.get('measurement_names', [])
    measurement_id = entry.get('measurement_id', None)

    instructions = JUDGE_INSTRUCTIONS
    query = f"""Is the extracted data point valid for the given entity and feature?
Extracted Data Point:
    Entity Name: {name}
    Feature: {feature}
    Value: {value}
    Units: {units}
Note that the entity may be known by multiple names: {', '.join(entity_names)}.
Also, the feature may be referred to by different terms: {', '.join(feature_names)}.
    """
    messages.append((instructions, context, query))
    message_ids.append(measurement_id)

responses = llm.predict(messages)
judged_data = []
attn_output_dict = {}
for i, response in enumerate(responses):
    measurement_id = str(message_ids[i])
    judged_data_point = data[i] | {
        'judgement': response['response'], 'judgement_logprob': response['logprob']
    }
    judged_data.append(judged_data_point)
    attn_output = response.get('attn_output', None)
    if attn_output is not None:
        attn_output_dict[measurement_id] = attn_output


with open(output_file, "w") as f:
    json.dump(judged_data, f, indent=4, ensure_ascii=False)

np.savez_compressed(attn_output_file, **attn_output_dict)

####################################################################################################