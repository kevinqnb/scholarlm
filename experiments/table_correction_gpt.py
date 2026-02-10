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
    'impact_of_macrophytes.pdf'
    "bacterioplankton.pdf",
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

client = AsyncOpenAI()

async def run_single_chat(model, custom_id, messages, sem, max_retries):
    """
    Run a single chat completion with automatic retry on 429 errors.
    """
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=12288
                )
                return custom_id, response.choices[0].message.content

            except RateLimitError as e:
                wait_time = (2 ** attempt) + random.random()
                print(f"[{custom_id}] Rate limit hit (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)

            except APIError as e:
                wait_time = (2 ** attempt) + random.random()
                print(f"[{custom_id}] API error (attempt {attempt+1}/{max_retries}): {e}")
                print(f"  Retrying in {wait_time:.2f}s...") 
                await asyncio.sleep(wait_time)
            
            except Exception as e:
                print(f"[{custom_id}] Unexpected error: {type(e).__name__}: {e}")
                raise

        raise RuntimeError(f"[{custom_id}] Failed after {max_retries} retries.")

async def run_all_chats(chats, model="gpt-4o", max_concurrent=1, max_retries=10):
    """
    Run all chats in parallel with limited concurrency.
    """
    sem = asyncio.Semaphore(max_concurrent)
    
    async def run_with_delay(chat, delay):
        await asyncio.sleep(delay)
        return await run_single_chat(model, chat["custom_id"], chat["messages"], sem, max_retries=max_retries)
    
    # Stagger the initial requests by 5 seconds each to avoid rate limiting
    tasks = [
        run_with_delay(chat, idx * 5.0)
        for idx, chat in enumerate(chats)
    ]
    results = await asyncio.gather(*tasks)
    return {cid: text for cid, text in results}


####################################################################################################
# Run batches with size limit and parallel processing

print(f"Running {len(chats)} chat completions with up to 1 concurrent requests...")

responses = asyncio.run(run_all_chats(
    chats, 
    model="gpt-5.2",
    max_concurrent=5,
    max_retries=5
))

updated_text = deepcopy(text)
for cid, response_text in responses.items():
    paper_idx, page_idx, table_idx = re.findall(r'paper_(\d+);page_(\d+);table_(\d+)', cid)[0]
    paper_idx = int(paper_idx)
    page_idx = int(page_idx)
    table_idx = int(table_idx)

    # Locate the table in the text and replace with validated response
    paper_text = updated_text[paper_idx]
    table_start = paper_text.find(f'<table number="{table_idx}">')
    table_end = paper_text.find(f'</table>', table_start) + len(f'</table>')

    # Replace the table text with the validated response
    new_table_text = response_text.strip()

    # Update the full paper text
    updated_paper_text = (
        paper_text[:table_start] +
        new_table_text +
        paper_text[table_end:]
    )
    updated_text[paper_idx] = updated_paper_text


# Save updated texts
for paper_idx, paper_text in enumerate(updated_text):
    filename = text_files[paper_idx]
    out_filepath = os.path.join(ocr_directory, filename)
    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.write(paper_text)


