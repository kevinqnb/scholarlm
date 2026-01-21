import os
import json
import time
from pydantic import BaseModel, Field
import random
import asyncio
from openai import OpenAI, AsyncOpenAI
from openai import RateLimitError, APIError
from dotenv import load_dotenv
load_dotenv()

from scholarlm import JUDGE_INSTRUCTIONS

####################################################################################################

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

input_file = "data/01_14_26/ten_judged3.json"

with open(input_file, "r") as f:
    data = json.load(f)


chats = []
chat_ids = []
for i, entry in enumerate(data):
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
    prompt = (
        f"## Context:\n{context}\n\n## Query:\n{query}"
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
    cid = str(i)
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
        run_with_delay(chat, idx * 1.0)
        for idx, chat in enumerate(chats)
    ]
    results = await asyncio.gather(*tasks)
    return {cid: text for cid, text in results}


####################################################################################################
# Run batches with size limit and parallel processing

responses = asyncio.run(run_all_chats(
    chats, 
    model="gpt-5.2",
    max_concurrent=500,
    max_retries=7
))

data_validated = []
for cid, response_text in responses.items():
    idx = int(cid)
    entry = data[idx]
    entry_validated = entry | {
        'validation': response_text
    }
    data_validated.append(entry_validated)

output_file = f"data/01_14_26/ten_validated3.json"

with open(output_file, "w") as f:
    json.dump(data_validated, f, indent=4, ensure_ascii=False)


