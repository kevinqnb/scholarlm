import os
import json
import math
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field, field_validator, create_model
from enum import Enum
from typing import Callable
load_dotenv()

from scholarlm import ContextLM, ContextLM2, ContextLM3, jensen_shannon_divergence

text_directory = os.getenv("POND_TEXT_PATH")
filename = 'physical_and_chemical_limnological.json'
filepath = os.path.join(text_directory, filename)
with open(filepath, 'r', encoding='utf-8') as file:
    doc_chunks = json.load(file)


####################################################################################################
# Load model:

ctxlm_params = {
    "do_sample": False,
    "max_new_tokens": 20,
}

llm = ContextLM2(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    top_k=10,
    sampling_params=ctxlm_params,
    #return_full_output=True,
    nnsight_kwargs = {"torch_dtype": torch.bfloat16},
    cache_dir="data/test_cache"
)

instructions = (
    f"You are an expert in extracting precise numerical data from user provided, scientific text. "
    f"You will be queried with a description of an specific entity to be measured, along with the measurement type to report for. "
    f"Your task is to extract the corresponding value if it appears in the provided context. "
    f"* Respond 'None' if the the given entity or measurement feature do not appear in the context. "
    f"Respond 'None' if the context does not explicity provide data for the given feature and entity. "
    f"Respond 'None' if the data reported is not either a direct numerical measurement or a mean of numerical measurements. "
    f"Respond 'None' for if the data reported only contains values for parameter estimates or other statistical measures of fit. "
    f"Respond 'None' for ranges of values, inequalties, or other cases where there is not a clear choice for a single numerical value. "
    f"Respond with the extracted value only if the context explicity provides a direct numerical value measured for the given feature, with respect to the entity in question. "
    f"Copy the value exactly as it appears in the context. "
    f"Give the value only, and do not include any units of measurement, descriptors, or explanation in your response. "
    f"If the value is associated with uncertainty measures (e.g., ± values, confidence intervals), report only the central value without any uncertainty information. "
)

context = doc_chunks["6"]

entity = {"name": "BAJ", "date": "None", "location": "None", "ecosystem": "pond"}
measurement = "pH"
query = "Extract the value of " + f"{measurement} for the entity {entity}."

prompts = [
    (instructions, context, query)
]

import time
start_time = time.time()
responses = llm.predict(prompts)
end_time = time.time()

print(f"Time taken for {len(prompts)} prompts: {end_time - start_time} seconds")

print("Responses:")
for i, response in enumerate(responses):
    print(f"Prompt {i+1}:")
    print(response)
    print()


# Save responses to npz
#import numpy as np
#output_filepath = os.path.join("data", "probe.npz")
#output_dict = entity | responses[0]
#np.savez_compressed(output_filepath, **output_dict)


####################################################################################################
