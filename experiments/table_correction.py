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


####################################################################################################
# Open text and pdf files 

main_directory = os.getenv("POND_PATH")
pdf_directory = os.getenv("POND_PDF_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH")
text_directory2 = os.getenv("POND_TEXT_PATH2")
image_directory = os.getenv("POND_IMAGE_PATH")
api_key = os.getenv("OPENAI_API_KEY")

# Directory
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

pdf_files = get_filenames_in_directory(pdf_directory, ignore = [".DS_Store"])
pdf_files.sort()
pdf_files = [
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
]

pdf_files = ['physical-chemical_influences.pdf']

pdf_filepaths = []
for f in pdf_files:
    filepath = os.path.join(pdf_directory, f)
    pdf_filepaths.append(filepath)

pdf_images = []
for filepath in pdf_filepaths:
    # Reduce image size to 1536 to save tokens and reduce rate limiting
    b64_images = process_pdf(filepath, target_longest_dim = 1536)
    pdf_images.append(b64_images)


text_files = get_filenames_in_directory(text_directory, ignore = [".DS_Store"])
text_files.sort()
text_files = [
    'physical_and_chemical_limnological.txt',
    'physical-chemical_influences.txt',
    'prairie_wetland.txt',
    'net_heterotrophy.txt',
    'habitat_characteristics.txt',
    'biodiversity_of_constructed.txt',
    'fish_production_in_lakes.txt',
    'long-term_stability.txt',
    'diversity_of_macroinvertebrates.txt',
    'impact_of_macrophytes.txt'
]

text_files = ['physical-chemical_influences.txt']

text_filepaths = []
for f in text_files:
    filepath = os.path.join(text_directory, f)
    text_filepaths.append(filepath)

text = []
for filepath in text_filepaths:
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        text.append(content)


####################################################################################################
# Create prompts:

instructions = """
You are an expert data engineer specializing in cleaning unstructured HTML tables for Python/Pandas processing. You will be provided with an HTML table from a research paper, along with an image of the page it was extracted from. Your job is to use the context to improve the structure and content of the table so that it is in accurate, clean, LLM readable, html format.

Formatting Instructions:
1. Standardize to 'long' format: If the table is 'wide' (e.g., it lists different categories side-by-side with repeating columns), you must 'melt' or unpivot the table by creating new rows for each category, while keeping unique, non-repeating column headers. You should not melt the table if columns are non-repeating or non-hierarchical, only do so if it is necessary for machine readability.
2. You must create a single index column so that rows are machine identifiable. The very first column of your output table must be named 'index'. This column must contain unique identifiers for each row. If the rows are hierarchical (e.g., Category -> Sub-category) or if you unpivoted the data, you must combine the identifying columns into a Python-tuple format. For example, if the indentifying columns are 'column A' and 'column B', each row should be identified by a tuple: '('column A' value, 'column B' value)'. Otherwise if the identifiers are simple and have no hierarchy, the index should be a single non-tupled value. Importantly, when choosing identifying attributes, you should give highest priority to columns which use descriptive names, even if they must be combined with other attribute columns to uniquely identify the row. If no such columns exist, you may use numerical columns to uniquely identify rows.
3. If there are multi-level column headers, flatten them in the same way by grouping the headers in a tuple format. For example, if the headers are 'Year' and 'Measurement', the combined header should be ('Year', 'Measurement').
4. Your job is mainly to modify structure without interfering on the data. However, if you notice any inaccuracies or inconsistencies in the given HTML table, you must correct them.
5. If a single cell contains a main value along with a separate range or interval of numbers, you must split these into separate columns. For example, if data is reported in a 'mean (minimum, maximum)' format, you should create three separate columns for the mean, minimum, and maximum. Similarly, if data is reported in a 'value ± uncertainty' format, you should create one column for the main value and another column for the uncertainty. Use the context to make sure that the new columns are named according to the statistic they represent. For example, in a single feature broken into mean, minimum, and maximum features you may use names such as 'feature_1_mean', 'feature_1_min', 'feature_1_max'. If there is no clear indication of what the statistic is, use generic names like 'feature_1_val_1', 'feature_1_val_2', etc.
6. The response table must be in HTML format, and wrapped inside <table number="i"></table> tags. Make sure to keep the table number attribute exactly as it appears in the given HTML.
7. At the very beginning of the table HTML, include <caption></caption> tags and use these to briefly describe the table and the measurements included within it. Use available table captioning on the pdf page to help, but make sure to include any additional information which might be relevant to understand the new formatting. 
8. Provide only the raw HTML for the full table, do not stop early (even if it is repetitive) and do not include any additional text or explanations.
"""

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
    model="gpt-5.1",
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
    out_filepath = os.path.join(text_directory2, filename)
    with open(out_filepath, 'w', encoding='utf-8') as f:
        f.write(paper_text)


