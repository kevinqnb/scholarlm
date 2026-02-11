import re
import os
import json
import time
from copy import deepcopy
import random
from io import BytesIO
import pandas as pd
from PIL import Image
import asyncio
from openai import OpenAI, AsyncOpenAI
from openai import RateLimitError, APIError
import backoff
from dotenv import load_dotenv
load_dotenv()

from scholarlm.utils import get_filenames_in_directory, encode_pil_image, process_pdf
from scholarlm.instruction_prompts import CLEAN_TABLE_INSTRUCTIONS

api_key = os.getenv("OPENAI_API_KEY")

####################################################################################################
# Load PDFs and set up filepaths

main_directory = "data/pond"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output")
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)


# Get all PDF files in the pdf directory
pdf_files = get_filenames_in_directory(pdf_directory, ignore = [".DS_Store", ".gitkeep"])
pdf_files.sort()

# Or specify specific files to process:
precomputed_pdf_files = [
    'physical_and_chemical_limnological.pdf',
    'physical-chemical_influences.pdf',
    'prairie_wetland.pdf',
    'net_heterotrophy.pdf',
    'habitat_characteristics.pdf',
    'biodiversity_of_constructed.pdf',
    'fish_production_in_lakes.pdf',
    'long-term_stability.pdf',
    'diversity_of_macroinvertebrates.pdf',
    'impact_of_macrophytes.pdf',
    #"bacterioplankton.pdf",
    "conservation_of_pond.pdf",
    "distinct_optical.pdf",
    "fish_assemblages.pdf",
    "lake_morphometry.pdf",
    "natural_variability.pdf",
    "productivity_and_depth.pdf",
    "relationships_of_fish.pdf",
    "sediment_characteristics.pdf",
    "vegetation-environmental.pdf"
]

pdf_files = [pf for pf in pdf_files if pf not in precomputed_pdf_files]

pdf_filepaths = []
for f in pdf_files:
    filepath = os.path.join(pdf_directory, f)
    pdf_filepaths.append(filepath)

# Process PDFs to extract images as base64 strings for embedding in prompts
pdf_images = []
for filepath in pdf_filepaths:
    # Reduce image size to 1536 to save tokens and reduce rate limiting
    b64_images = process_pdf(filepath, target_longest_dim = 1536)
    pdf_images.append(b64_images)


text_files = [pf.replace('.pdf', '.txt') for pf in pdf_files]

text_filepaths = []
for f in text_files:
    filepath = os.path.join(ocr_directory, f)
    text_filepaths.append(filepath)

text = []
for filepath in text_filepaths:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        text.append(content)


####################################################################################################
# Create prompts:

instructions = CLEAN_TABLE_INSTRUCTIONS
chats = []
chat_ids = []
for paper_idx, paper_text in enumerate(text):
    paper_images = pdf_images[paper_idx]
    pages = re.findall(r'<page number="(\d+)">', paper_text)
    pages = [int(p) for p in pages]

    for page_idx in pages:
        # Some PDFs may have sparse/offset page indices; skip if we don't have an image.
        if page_idx < 0 or page_idx >= len(paper_images):
            continue
        page_image = paper_images[page_idx]
        page_start = paper_text.find(f'<page number="{page_idx}">') + len(f'<page number="{page_idx}">')
        page_end = paper_text.find(f'</page>', page_start)
        page_text = paper_text[page_start: page_end].strip()

        # Find any tables within the page text:
        tables = re.findall(r'<table number="(\d+)">', page_text)
        tables = [int(t) for t in tables]

        for table_idx in tables:
            table_start = page_text.find(f'<table number="{table_idx}">')
            table_end = page_text.find(f'</table>', table_start) + len(f'</table>')
            table_text = page_text[table_start: table_end].strip()

            prompt = (
                f"## HTML Table:\n"
                f"{table_text}\n\n"
                f"## Query:\n"
                f"Please follow the instructions to reformat the HTML table, which corresponds to table {table_idx + 1} in the document."
            )
            messages = [
                {"role": "system", "content": instructions},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{page_image}", "detail": "high"}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            cid = f"paper_{paper_idx};page_{page_idx};table_{table_idx}"
            chat = {
                "custom_id": cid,
                "messages": messages
            }
            chats.append(chat)
            chat_ids.append(cid)


####################################################################################################
# Run chats sequentially with delay and backoff for rate limits

client = OpenAI(api_key=api_key)

@backoff.on_exception(backoff.expo, (RateLimitError, APIError), max_time=60, max_tries=6)
def completions_with_delay_and_backoff(delay: float, **kwargs):
    time.sleep(delay)
    return client.chat.completions.create(**kwargs)

rate_limit_per_minute = 30 # Limit to 30 requests per minute (~10 papers)
delay = 60.0 / rate_limit_per_minute

responses = {}
for i, chat in enumerate(chats):
    print(f"Running chat {i+1}/{len(chats)} with custom_id: {chat['custom_id']}")
    try:
        response = completions_with_delay_and_backoff(
            delay=delay,
            model="gpt-5.2",
            messages=chat["messages"],
            max_completion_tokens=8192,
            temperature=0.1,
            seed=342
        )
        responses[chat["custom_id"]] = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error processing chat with custom_id {chat['custom_id']}: {e}")
        print("Chat idx:", i)
        print()
        responses[chat["custom_id"]] = None  # Or some error message


####################################################################################################
# Save results:

updated_text = deepcopy(text)
for cid, response_text in responses.items():
    if not response_text:
        continue

    paper_idx, page_idx, table_idx = re.findall(r'paper_(\d+);page_(\d+);table_(\d+)', cid)[0]
    paper_idx = int(paper_idx)
    table_idx = int(table_idx)

    paper_text = updated_text[paper_idx]
    table_start = paper_text.find(f'<table number="{table_idx}">')
    if table_start == -1:
        continue
    table_end = paper_text.find(f'</table>', table_start)
    if table_end == -1:
        continue
    table_end += len(f'</table>')

    new_table_text = response_text.strip()
    updated_text[paper_idx] = paper_text[:table_start] + new_table_text + paper_text[table_end:]


# Save updated texts
for paper_idx, paper_text in enumerate(updated_text):
    filename = text_files[paper_idx]
    out_filepath = os.path.join(ocr_directory, filename)
    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.write(paper_text)


