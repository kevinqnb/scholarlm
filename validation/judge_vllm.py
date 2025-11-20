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

from scholarlm.utils import get_filenames_in_directory, encode_pil_image


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

input_file = "data/pond_results_10_papers_v1_vllm.json"

with open(input_file, "r") as f:
    result_dict = json.load(f)


instructions = (
    f"You are an expert in discerning accuracy for data extracted from research papers by large language models. "
    f"First you will be given a passage of OCR generated text. "
    f"You will then be given a data point which was extracted from the OCR text. "
    f"Your task is to classify the extracted data point's relationship to the provided OCR text, using the following categories:\n"
    f"hallucination: The extracted data point's 'value' feature does not explicity appear within the OCR text.\n"
    f"disorientation: The data point's 'value' feature appears to be derived from the OCR text, but is incorrectly attributed to the given entity or measurement type.\n"
    f"deviation: The data point's 'value' feauture is supported by the OCR text, but the given value is an aggregate statistic, range of values, inequality, non-numerical description, or a measurement for a collection of entities rather than a direct numerical measurement for a single entity.\n"
    f"valid: The data point is a direct measurement which is explicity supported by the context, and is made with respect to the correct entity and measurement type.\n\n"
    f"Respond by choosing the category which best describes the data point's relation to the given context. "
    f"Only respond with one of the following labels: hallucination, ocr_error, disorientation, deviation, valid. Do not include any other text or explanation in your response."
)


messages = []
for i, entry in enumerate(result_dict):
    paper_id = entry['paper_id']
    chunk_id = entry['chunk_id']
    img_filename = os.path.join(image_directory, filenames[paper_id], f"chunk_{chunk_id}.png")
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

    #img = Image.open(img_filename)
    #img_encoded = encode_pil_image(img)

    prompt = (
        f"## OCR Text:\n"
        f"{context}\n\n"
        f"## Extracted Data Point:\n"
        f"{json.dumps(datapoint)}\n\n"
        f"## Query:\n"
        f"Given the image and OCR text, which category best describes the extracted data point?"
    )

    message = [
        {"role": "system", "content": instructions},
        {
            "role": "user",
            "content": [
                #{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_encoded}"}},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    messages.append(message)



model = "gaunernst/gemma-3-27b-it-int4-awq"
llm = LLM(model=model)
guided_decoding_params = GuidedDecodingParams(
    choice = ['hallucination', 'ocr_error', 'disorientation', 'deviation', 'valid'],
)
sampling_params = SamplingParams(
    temperature = 0.0,
    guided_decoding=guided_decoding_params
)

responses = llm.chat(messages = messages, sampling_params = sampling_params)
responses = [r.outputs[0].text for r in responses]

output_file = "data/pond_results_10_papers_v1_vllm_validated_gemma.json"
with open(output_file, "w") as f:
    output_data = []
    for entry, response in zip(result_dict, responses):
        entry['judgement'] = response
        output_data.append(entry)
    json.dump(output_data, f, indent=4)