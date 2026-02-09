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

class ObservationSchema(BaseModel):
    name: str | None
    abbreviations: str | None
    location: str | None
    site: str | None
    state: str | None
    date: str | None
    ecosystem: str | None

fields = ObservationSchema.model_fields.keys()

feature_info_dict = {
    "latitude": {
        "description": "Geographic latitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.",
        "units": ["degrees", "radians"]
    },
    "longitude": {
        "description": "Geographic longitude of the ecosystem location, expressed in a standard geographic coordinate system (e.g., WGS84). This should refer to the centroid or stated reference point of the ecosystem, not a bounding box or region.",
        "units": ["degrees", "radians"]
    },
    "surface_area": {
        "description": "Surface area of the water body itself (not the watershed or catchment area). This should represent the horizontal area of open water or the stated ecosystem boundary at the time of measurement or description.",
        "units": ["km^2", "mi^2", "ha", "m^2", "acres"]
    },
    "max_depth": {
        "description": "Maximum water depth of the ecosystem, defined as the deepest point of the water body at the time of measurement or as reported in the source. This is not the mean or average depth.",
        "units": ["m", "km", "ft"]
    },
    "vegetation_cover": {
        "description": "Fraction or percentage of the ecosystem surface area covered by aquatic macrophytes or other aquatic vegetation. This should refer to areal coverage, not biomass or volume.",
        "units": ["percent", "fraction"]
    },
    "ph": {
        "description": "pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH.",
        "units": []
    },
    "tn": {
        "description": "Total nitrogen concentration in the water column, including both dissolved and particulate forms and all major species (e.g., nitrate, nitrite, ammonium, organic nitrogen), as explicitly reported in the source.",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "tp": {
        "description": "Total phosphorus concentration in the water column, including both dissolved and particulate forms, as explicitly reported in the source (i.e., not just soluble reactive phosphorus or orthophosphate).",
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]
    },
    "chla": {
        "description": "Chlorophyll-a concentration in the water column, used as a proxy for phytoplankton biomass. This should refer to extracted or in situ chlorophyll-a measurements, not total chlorophyll or other pigments unless explicitly labeled as chlorophyll-a.",
        "units": ["µg/L", "mg/L", "mg/m^3"]
    },
}

####################################################################################################

input_file = "data/01_28_26/ten_standardize.json"
output_file = f"data/01_28_26/ten_judged_gpt.json"

with open(input_file, "r") as f:
    data = json.load(f)

chats = []
chat_ids = []
for i, entry in enumerate(data):
    context = entry['context']
    feature = entry.get('feature')
    feature_description = feature_info_dict[feature]['description']
    feature_terms = entry.get('feature_terms', [])
    entity_description = {k: v for k,v in entry.items() if k in fields}
    measurement_val = entry['value']

    instructions = JUDGE_INSTRUCTIONS
    query = (
        f"Feature description: {feature_description}\n"
        f"Terminology used for the feature: {feature_terms}\n"
        f"Entity description: {entity_description}\n"
        f"Extracted measurement: {measurement_val}\n\n"
        f"Is the extracted data point valid for the given entity and feature?"
    )
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
                    "judgement": text,
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
        'judgement': result.get('judgement'),
        'judgement_confidence': result.get('confidence'),
        'judgement_model': result.get('model'),
    }
    data_validated.append(entry_validated)

with open(output_file, "w") as f:
    json.dump(data_validated, f, indent=4, ensure_ascii=False)


