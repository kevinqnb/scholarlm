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
    if i < 1000:
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
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_encoded}"}},
                        {"type": "text", "text": prompt},
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



client = OpenAI()


def estimate_chat_size(chat, model="gpt-4o"):
    """
    Estimate the size in bytes of a single chat when serialized to JSONL.
    """
    line = {
        "custom_id": chat["custom_id"],
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": chat["messages"]
        }
    }
    return len((json.dumps(line) + "\n").encode("utf-8"))


def split_chats_by_size(chats, max_size_mb=150, model="gpt-4o"):
    """
    Split chats into chunks, each under max_size_mb.
    Returns a list of chat chunks.
    """
    max_bytes = max_size_mb * 1024 * 1024
    chunks = []
    current_chunk = []
    current_size = 0
    
    for chat in chats:
        chat_size = estimate_chat_size(chat, model=model)
        
        # If a single chat exceeds max_size, put it in its own chunk
        if chat_size >= max_bytes:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_size = 0
            chunks.append([chat])
            print(f"Warning: Single chat {chat['custom_id']} is {chat_size / (1024*1024):.2f} MB")
            continue
        
        # If adding this chat would exceed limit, start new chunk
        if current_size + chat_size > max_bytes:
            chunks.append(current_chunk)
            current_chunk = [chat]
            current_size = chat_size
        else:
            current_chunk.append(chat)
            current_size += chat_size
    
    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks


def create_batch_buffer(chats, model="gpt-4o"):
    """
    Build the JSONL batch input in memory.
    """
    buf = BytesIO()
    for chat in chats:
        line = {
            "custom_id": chat["custom_id"],
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": chat["messages"]
            }
        }
        buf.write((json.dumps(line) + "\n").encode("utf-8"))
    buf.seek(0)
    return buf


def run_single_batch(chats, batch_num, model="gpt-4o", poll_interval=60):
    """
    Creates a single batch job, waits for completion, and returns parsed responses.
    """
    print(f"\n[Batch {batch_num}] Building batch input with {len(chats)} chats...")
    batch_input = create_batch_buffer(chats, model=model)
    size_mb = batch_input.getbuffer().nbytes / (1024 * 1024)
    print(f"[Batch {batch_num}] Size: {size_mb:.2f} MB")

    print(f"[Batch {batch_num}] Uploading batch input file...")
    upload = client.files.create(
        file=(f"batch_{batch_num}.jsonl", batch_input),
        purpose="batch"
    )

    print(f"[Batch {batch_num}] Creating batch job...")
    batch = client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    print(f"[Batch {batch_num}] Batch created: {batch.id}")
    print(f"[Batch {batch_num}] Waiting for completion...")

    # Poll until done
    while True:
        status = client.batches.retrieve(batch.id)
        print(f"[Batch {batch_num}] Status: {status.status}")

        if status.status in ("completed", "failed", "expired", "cancelled"):
            break
        time.sleep(poll_interval)

    if status.status != "completed":
        raise RuntimeError(f"[Batch {batch_num}] Batch failed with status: {status.status}")

    print(f"[Batch {batch_num}] Batch completed. Downloading results...")

    # Download output JSONL file
    #result_bytes = client.files.content(status.output_file_id)
    #text = result_bytes.decode("utf-8")
    content = client.files.content(status.output_file_id)
    text = content.read().decode("utf-8")

    print(f"[Batch {batch_num}] Parsing results...")

    responses = {}
    for line in text.splitlines():
        rec = json.loads(line)
        cid = rec["custom_id"]
        body = rec["response"]["body"]
        out_text = body["choices"][0]["message"]["content"]
        responses[cid] = out_text

    print(f"[Batch {batch_num}] Done. Collected {len(responses)} responses.")
    return responses


async def run_batch_async(chats, batch_num, model="gpt-4o", poll_interval=60):
    """
    Async wrapper for running a single batch.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, 
        run_single_batch, 
        chats, 
        batch_num, 
        model, 
        poll_interval
    )


async def run_batches_parallel(chat_chunks, model="gpt-4o", poll_interval=60, max_concurrent=5):
    """
    Run multiple batch jobs in parallel with a concurrency limit.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def run_with_semaphore(chats, batch_num):
        async with semaphore:
            return await run_batch_async(chats, batch_num, model, poll_interval)
    
    tasks = [
        run_with_semaphore(chunk, i+1) 
        for i, chunk in enumerate(chat_chunks)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Merge all responses
    all_responses = {}
    for response_dict in results:
        all_responses.update(response_dict)
    
    return all_responses


def run_batches(chats, model="gpt-4o", max_size_mb=150, poll_interval=60, max_concurrent=5):
    """
    Main entry point: splits chats into manageable chunks and runs them in parallel.
    
    Args:
        chats: List of chat dictionaries
        model: OpenAI model to use
        max_size_mb: Maximum size per batch in MB
        poll_interval: How often to poll for batch status (seconds)
        max_concurrent: Maximum number of concurrent batch jobs
    
    Returns:
        Dictionary mapping custom_id to response text
    """
    print(f"\n{'='*60}")
    print(f"Starting batch processing with {len(chats)} total chats")
    print(f"Max batch size: {max_size_mb} MB")
    print(f"Max concurrent batches: {max_concurrent}")
    print(f"{'='*60}\n")
    
    # Split into chunks
    chat_chunks = split_chats_by_size(chats, max_size_mb=max_size_mb, model=model)
    print(f"\nSplit into {len(chat_chunks)} batches:")
    for i, chunk in enumerate(chat_chunks):
        print(f"  Batch {i+1}: {len(chunk)} chats")
    
    # Run in parallel
    responses = asyncio.run(
        run_batches_parallel(chat_chunks, model=model, poll_interval=poll_interval, max_concurrent=max_concurrent)
    )
    
    print(f"\n{'='*60}")
    print(f"All batches completed! Total responses: {len(responses)}")
    print(f"{'='*60}\n")
    
    return responses


# Run batches with size limit and parallel processing
responses = run_batches(
    chats, 
    model="gpt-5-mini",  # Fixed typo: gpt-5-mini -> gpt-4o-mini
    max_size_mb=150, 
    poll_interval=300,
    max_concurrent=3  # Adjust based on your rate limits
)

# Save responses
for i, entry in enumerate(result_dict):
    cid = f"validation_{i}"
    entry['validation_label'] = responses.get(cid, None)

outfile = input_file.replace(".json", "_validated.json")
with open(outfile, "w") as f:
    json.dump(result_dict, f, indent=4)
