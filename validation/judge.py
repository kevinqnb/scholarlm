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
    f"First you will be given an image of a text passage, as well as OCR generated text for the image. "
    f"You will then be given a data point which was extracted from the OCR text. "
    f"Your task is to classify the extracted data point's relationship to the provided image and OCR text, using the following categories:\n"
    f"hallucination: The extracted data point's 'value' feature does not explicity appear within the OCR text.\n"
    f"ocr_error: The data point's 'value' feature appears to be derived from the OCR text, but is incorrectly attributed to the given entity or measurement type due to errors in the OCR representation of the text.\n"
    f"disorientation: The data point's 'value' feature appears to be derived from the OCR text, but is incorrectly attributed to the given entity or measurement type.\n"
    f"deviation: The data point's 'value' feauture is supported by the OCR text, but the given value is an aggregate statistic, range of values, inequality, non-numerical description, or a measurement for a collection of entities rather than a direct numerical measurement for a single entity.\n"
    f"valid: The data point is a direct measurement which is explicity supported by the context, and is made with respect to the correct entity and measurement type.\n\n"
    f"Respond by choosing the category which best describes the data point's relation to the given context. "
    f"Only respond with one of the following labels: hallucination, ocr_error, disorientation, deviation, valid. Do not include any other text or explanation in your response."
)

chats = []
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

    try:
        img = Image.open(img_filename)
        img_encoded = encode_pil_image(img)

        prompt = (
            f"## OCR Text:\n"
            f"{context}\n\n"
            f"## Extracted Data Point:\n"
            f"{json.dumps(datapoint)}\n\n"
            f"## Query:\n"
            f"Given the image and OCR text, which category best describes the extracted data point?"
        )

        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_encoded}"}},
                ],
            },
        ]

        chat = {
            "custom_id": f"validation_{i}",
            "messages": messages
        }

        chats.append(chat)

    except FileNotFoundError:
        prompt = (
            f"## OCR Text:\n"
            f"{context}\n\n"
            f"## Extracted Data Point:\n"
            f"{json.dumps(datapoint)}\n\n"
            f"## Query:\n"
            f"Image not available. Given only the OCR text, which category best describes the extracted data point?"
        )

        messages = [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            },
        ]

        chat = {
            "custom_id": f"validation_{i}",
            "messages": messages
        }

        chats.append(chat)



client = AsyncOpenAI()
MAX_CONCURRENT = 1 
MAX_RETRIES = 6
MODEL = "gpt-5-mini" 

'''
def run_single_chat(messages, model="gpt-5-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content
'''

async def run_single_chat(custom_id, messages, sem, max_retries=MAX_RETRIES):
    """
    Run a single chat completion with automatic retry on 429 errors.
    """
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=MODEL,
                    messages=messages
                )
                return custom_id, response.choices[0].message.content

            except RateLimitError:
                wait_time = (2 ** attempt) + random.random()
                print(f"[{custom_id}] Rate limit hit, retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)

            except APIError as e:
                wait_time = (2 ** attempt) + random.random()
                print(f"[{custom_id}] API error ({e}), retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"[{custom_id}] Failed after {max_retries} retries.")

async def run_all_chats(chats):
    """
    Run all chats in parallel with limited concurrency.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT)
    tasks = [
        run_single_chat(chat["custom_id"], chat["messages"], sem)
        for chat in chats
    ]
    results = await asyncio.gather(*tasks)
    return {cid: text for cid, text in results}

#responses = run_batch(chats, model="gpt-5-mini", poll_interval=60)

#responses = {
#    chat['custom_id']: run_single_chat(chat['messages'], model="gpt-5.1")
#    for chat in chats
#}
responses = asyncio.run(run_all_chats(chats))


# Save responses
for i, entry in enumerate(result_dict):
    cid = f"validation_{i}"
    entry['validation_label'] = responses.get(cid, None)

outfile = input_file.replace(".json", "_validated.json")
with open(outfile, "w") as f:
    json.dump(result_dict, f, indent=4)
