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

    Returns:
        (custom_id, result_dict)
        result_dict contains:
          - validation: model text (expected 'true'/'false')
          - confidence: probability of the (first) token the model actually produced
            for 'true'/'false' (best-effort; None if logprobs unavailable)
          - model: model name
    """
    async with sem:
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    # This task should only output a boolean.
                    max_completion_tokens=8,
                    temperature=0,
                    # Ask for token logprobs.
                    logprobs=True,
                    # Request enough alternatives so 'true'/'false' show up even if tokenization differs.
                    top_logprobs=5,
                )

                text = (response.choices[0].message.content or "").strip()

                confidence = None
                try:
                    import math

                    lp = getattr(response.choices[0], "logprobs", None)
                    content = getattr(lp, "content", None) if lp is not None else None

                    if content:
                        first = content[0]

                        # 1) Prefer the logprob of the actually generated token.
                        first_token = (getattr(first, "token", None) or "").strip().lower()
                        first_logprob = getattr(first, "logprob", None)

                        if first_logprob is not None and first_token in {"true", "false"}:
                            confidence = float(math.exp(first_logprob))
                        else:
                            # 2) Fallback: look for returned label in the top_logprobs list.
                            top = getattr(first, "top_logprobs", None) or []
                            label = text.strip().lower()

                            for t in top:
                                tok = (getattr(t, "token", None) or "").strip().lower()
                                if tok == label:
                                    lprob = getattr(t, "logprob", None)
                                    if lprob is not None:
                                        confidence = float(math.exp(lprob))
                                        break

                except Exception:
                    confidence = None

                return custom_id, {
                    "validation": text,
                    "confidence": confidence,
                    "model": model,
                }

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
    return {cid: payload for cid, payload in results}


####################################################################################################
# Run batches with size limit and parallel processing

responses = asyncio.run(run_all_chats(
    chats,
    model="gpt-5.2",
    max_concurrent=500,
    max_retries=7
))

data_validated = []
for cid, result in responses.items():
    idx = int(cid)
    entry = data[idx]
    entry_validated = entry | {
        'validation': result.get('validation'),
        'validation_confidence': result.get('confidence'),
        'validation_model': result.get('model'),
    }
    data_validated.append(entry_validated)

output_file = f"data/01_20_26/ten_validated_gpt.json"

with open(output_file, "w") as f:
    json.dump(data_validated, f, indent=4, ensure_ascii=False)


