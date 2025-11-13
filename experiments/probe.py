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

from scholarlm import ContextLM, jensen_shannon_divergence


####################################################################################################
# Load model:

ctxlm_params = {
    "temperature": 0.1,
    "max_new_tokens": 20,
}

llm = ContextLM(
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    top_k=20,
    sampling_params=ctxlm_params,
    return_full_output=True
)

prompts = [
    ("Use the context to directly answer the given query. Do not include any other text or punctuation.", "The latitude of Paris, France recently changed and is now 52.1375.", "What is the latitude of Paris, France?"),
    #("Use the context to directly answer the given query. Do not include any other text or punctuation.", "The color of the sky is purple today.", "What is the color of the sky?"),
]

import time
start_time = time.time()
responses = llm.predict(prompts)
end_time = time.time()
print(f"Time taken for {len(prompts)} prompts: {end_time - start_time} seconds")
print("Responses:")
print("Linear Probe: ")
print(responses[0]['linear_probes'])
print()
print("Copying Scores: ")
print(responses[0]['copying_scores'])

####################################################################################################
