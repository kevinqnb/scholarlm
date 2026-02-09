import os
import json
import random
import asyncio
from typing import Any

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from scholarlm import JUDGE_INSTRUCTIONS

# NOTE: Requires the Anthropic SDK:
#   pip install anthropic
# and an API key in env var:
#   export ANTHROPIC_API_KEY=...
#
# This script mirrors `experiments/validate_gpt.py` but uses Claude.
#
# Token probabilities:
# - Anthropic's public API does not expose next-token logprobs.
# - This script outputs only the boolean `validation`.


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
output_file = "data/01_28_26/ten_judged_claude.json"


def build_chats(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chats: list[dict[str, Any]] = []

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

        prompt = f"## Context:\n{context}\n\n## Query:\n{query}"

        chats.append(
            {
                "custom_id": str(i),
                "system": instructions,
                "user": prompt,
            }
        )

    return chats


####################################################################################################

async def run_single_chat(
    client: Any,
    model: str,
    custom_id: str,
    system: str,
    user: str,
    sem: asyncio.Semaphore,
    max_retries: int,
) -> tuple[str, str]:
    """Run a single Claude request with retry."""

    async with sem:
        for attempt in range(max_retries):
            try:
                # Anthropic SDK is sync; run in worker thread.
                def _call():
                    # With Messages API: system prompt is separate.
                    return client.messages.create(
                        model=model,
                        system=system,
                        messages=[
                            {
                                "role": "user",
                                "content": user,
                            }
                        ],
                        max_tokens=8,
                        temperature=0.0,
                    )

                resp = await asyncio.to_thread(_call)

                # Text extraction
                text = ""
                try:
                    # Common: resp.content is a list of blocks like {type:'text', text:'...'}
                    blocks = getattr(resp, "content", None) or []
                    for b in blocks:
                        if getattr(b, "type", None) == "text":
                            text += getattr(b, "text", "")
                    text = text.strip()
                except Exception:
                    text = str(resp).strip()

                return custom_id, text

            except Exception as e:
                wait_time = (2**attempt) + random.random()
                print(f"[{custom_id}] Claude error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                print(f"  Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"[{custom_id}] Failed after {max_retries} retries.")


async def run_all_chats(
    chats: list[dict[str, Any]],
    model: str,
    max_concurrent: int,
    max_retries: int,
) -> dict[str, str]:
    """Run all chats in parallel with limited concurrency."""

    import anthropic  # type: ignore

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    sem = asyncio.Semaphore(max_concurrent)

    async def run_with_delay(chat: dict[str, Any], delay: float):
        await asyncio.sleep(delay)
        return await run_single_chat(
            client=client,
            model=model,
            custom_id=chat["custom_id"],
            system=chat["system"],
            user=chat["user"],
            sem=sem,
            max_retries=max_retries,
        )

    tasks = [run_with_delay(chat, idx * 1.0) for idx, chat in enumerate(chats)]
    results = await asyncio.gather(*tasks)
    return {cid: text for cid, text in results}


####################################################################################################

if __name__ == "__main__":
    with open(input_file, "r") as f:
        data = json.load(f)

    chats = build_chats(data)

    # "Latest" Claude model name changes; set via env var so you can swap easily.
    model = "claude-haiku-4-5"

    responses = asyncio.run(
        run_all_chats(
            chats,
            model=model,
            max_concurrent=2,
            max_retries=5,
        )
    )

    data_validated: list[dict[str, Any]] = []
    for cid, response_text in responses.items():
        idx = int(cid)
        entry = data[idx]
        entry_validated = entry | {
            "judgement": response_text,
            "judgement_model": model,
        }
        data_validated.append(entry_validated)

    with open(output_file, "w") as f:
        json.dump(data_validated, f, indent=4, ensure_ascii=False)
