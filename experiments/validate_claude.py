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

class MeasurementSchema(BaseModel):
    latitude: float | None = Field(
        description="latitude",
        json_schema_extra={"units": ["degrees", "radians"]},
    )
    longitude: float | None = Field(
        description="longitude",
        json_schema_extra={"units": ["degrees", "radians"]},
    )
    surface_area: float | None = Field(
        description="surface area",
        json_schema_extra={"units": ["km^2", "mi^2", "ha", "m^2", "acres"]},
    )
    max_depth: float | None = Field(
        description="maximum depth",
        json_schema_extra={"units": ["m", "km", "ft"]},
    )
    vegetation_cover: float | None = Field(
        description="aquatic macrophyte percent coverage",
        json_schema_extra={"units": ["percent", "fraction"]},
    )
    ph: float | None = Field(
        description="pH level",
        json_schema_extra={"units": None},
    )
    tn: float | None = Field(
        description="total nitrogen concentration",
        json_schema_extra={"units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]},
    )
    tp: float | None = Field(
        description="total phosphorus concentration",
        json_schema_extra={"units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"]},
    )
    chla: float | None = Field(
        description="chlorophyll-a concentration",
        json_schema_extra={"units": ["µg/L", "mg/L", "mg/m^3"]},
    )


####################################################################################################

input_file = "data/01_14_26/ten_judged3.json"
output_file = "data/01_20_26/ten_validated_claude.json"


def build_chats(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chats: list[dict[str, Any]] = []

    for i, entry in enumerate(data):
        context = entry.get("context", None)
        name = entry.get("name", None)
        feature = MeasurementSchema.model_fields[entry["measurement"]].description
        value = entry.get("value", None)
        units = entry.get("units", None)
        entity_names = entry.get("entity_names", [])
        feature_names = entry.get("measurement_names", [])

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
            "validation": response_text,
            "validation_model": model,
        }
        data_validated.append(entry_validated)

    with open(output_file, "w") as f:
        json.dump(data_validated, f, indent=4, ensure_ascii=False)
