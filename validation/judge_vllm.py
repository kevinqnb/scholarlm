from pydantic import BaseModel
import pandas as pd
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


import os
import json
import time
import random
from io import BytesIO
import pandas as pd
from PIL import Image
import asyncio
from openai import OpenAI, AsyncOpenAI
from openai import RateLimitError, APIError
from dotenv import load_dotenv
load_dotenv()

from scholarlm.utils import get_filenames_in_directory

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)


main_directory = os.getenv("POND_PATH")
pdf_directory = os.getenv("POND_PDF_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH")
image_directory = os.getenv("POND_IMAGE_PATH")
api_key = os.getenv("OPENAI_API_KEY")

# Directory
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)
registered_titles = [entry['title'] for entry in paper_info.values()]
registered_titles.sort()

filenames = get_filenames_in_directory(text_directory, ignore = [".DS_Store"])
filenames = [f.replace('.json', '') for f in filenames]
filenames.sort()

input_file = "data/pond_page_chunks_vllm_gemma.json"

with open(input_file, "r") as f:
    result_dict = json.load(f)


instructions = (
    f"You are an expert in discerning accuracy for data extraction tasks given to large language models. "
    f"You will be given context from a scientific research paper, along with a data point that "
    f"a language model has generated upon being prompted to extract relevant data. "
    f"Your task is to classify the extracted data point's relationship to the provided context, using the following categories:\n"
    f"hallucination: The extracted data point's 'value' feature does not explicity appear within the context.\n"
    f"disorientation: The extracted data point's 'value' feature explicity appears within the context, but is incorrectly attributed to the given entity or measurement type.\n"
    f"deviation: The extracted data point's 'value' feauture explicity appears within the context and is correctly attributed to the given entity and measurement type, but the given value is an aggregate statistic, range of values, inequality, non-numerical description, or a measurement for a collection of entities rather than a direct numerical measurement for a single entity.\n"
    f"valid: The extracted data point's 'value' feauture explicity appears within the context, is correctly attributed to the given entity and measurement type, and is a direct numerical measurement for a single entity.\n\n"
    f"Respond by choosing the category which best describes the data point's relation to the given context. "
    f"Only respond with one of the following labels: hallucination, disorientation, deviation, valid. Do not include any other text or explanation in your response."
)


messages = []
for i, entry in enumerate(result_dict):
    context = entry.get('context', None)
    datapoint = {
        "name": entry['name'],
        "location": entry['location'],
        "date": entry['date'],
        "ecosystem": entry['ecosystem'],
        "measurement": entry['measurement'],
        "value": entry['value'],
    }
    if entry.get('units', None) is not None:
        datapoint['units'] = entry['units']

    prompt = (
        f"## OCR Text:\n"
        f"{context}\n\n"
        f"## Extracted Data Point:\n"
        f"{json.dumps(datapoint)}\n\n"
        f"## Query:\n"
        f"Given the OCR text, which category best describes the extracted data point?"
    )

    message = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
            ],
        },
    ]

    messages.append(message)


model = "cyankiwi/Olmo-3-32B-Think-AWQ-4bit"
llm = LLM(model=model)
guided_decoding_params = GuidedDecodingParams(
    choice = ['hallucination', 'disorientation', 'deviation', 'valid'],
)
sampling_params = SamplingParams(
    temperature=0.0,
    guided_decoding=guided_decoding_params,
    seed=342
)

responses = llm.chat(messages = messages, sampling_params = sampling_params)
responses = [r.outputs[0].text for r in responses]

output_file = "data/pond_page_chunks_vllm_gemma_judged.json"
with open(output_file, "w") as f:
    output_data = []
    for entry, response in zip(result_dict, responses):
        entry['judgement'] = response
        output_data.append(entry)
    json.dump(output_data, f, indent=4)