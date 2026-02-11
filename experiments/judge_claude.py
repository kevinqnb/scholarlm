import os
import json
import random
import asyncio
import time
from typing import Any

from pydantic import BaseModel
from dotenv import load_dotenv

from scholarlm import JUDGE_INSTRUCTIONS

load_dotenv()

# NOTE: Requires the Anthropic SDK:
#   pip install anthropic
# and an API key in env var:
#   export ANTHROPIC_API_KEY=...
#
# This script mirrors `experiments/judge_gpt.py` / `experiments/judge_gemini.py` but uses Claude.
#
# Token probabilities:
# - Anthropic's public API does not expose next-token logprobs.
# - We set `judgement_confidence=None` for parity with the other judge scripts.


####################################################################################################

def normalize_bool_text(text: str | None) -> bool | None:
    """Strictly parse a model response into a boolean.

    Accepts only true/false (case-insensitive) with optional surrounding whitespace.
    Returns None if the response is not exactly parseable.
    """

    if text is None:
        return None
    t = text.strip().lower()
    if "true" in t:
        return True
    if "false" in t:
        return False
    return None


def build_user_prompt_from_entry(
    *,
    context: str,
    feature_description: str,
    feature_terms: list[Any],
    entity_description: dict[str, Any],
    measurement_val: Any,
) -> str:
    """Create the shared user prompt for judging.

    This must stay stable across providers to make results comparable.
    """

    query = (
        f"Feature description: {feature_description}\n"
        f"Terminology used for the feature: {feature_terms}\n"
        f"Entity description: {entity_description}\n"
        f"Extracted measurement: {measurement_val}\n\n"
        f"Is the extracted data point valid for the given entity and feature?"
    )
    return f"## Context:\n{context}\n\n## Query:\n{query}"


class AsyncRateLimiter:
    """A simple global rate limiter (token-less) that paces calls to ~rpm.

    Enforces an *average* request rate across all concurrent tasks.
    """

    def __init__(self, requests_per_minute: float):
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be > 0")
        self._interval = 60.0 / float(requests_per_minute)
        self._lock = asyncio.Lock()
        self._next_time = 0.0

    async def wait(self):
        async with self._lock:
            now = time.monotonic()
            if self._next_time <= now:
                self._next_time = now
            delay = self._next_time - now
            self._next_time += self._interval
        if delay > 0:
            await asyncio.sleep(delay)


async def run_single_chat(
    client: Any,
    *,
    model: str,
    custom_id: str,
    system: str,
    user: str,
    sem: asyncio.Semaphore,
    limiter: AsyncRateLimiter,
    max_retries: int,
) -> tuple[str, dict[str, Any]]:
    """Run a single Claude request with bounded concurrency + retries."""

    last_exc: Exception | None = None

    for attempt in range(max_retries):
        try:
            async with sem:
                await limiter.wait()

                # Anthropic SDK is sync; run in a worker thread.
                def _call():
                    return client.messages.create(
                        model=model,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                        # Mirror the other judge scripts: output should be only true/false.
                        max_tokens=5,
                        temperature=0.0,
                    )

                resp = await asyncio.to_thread(_call)

            # Text extraction
            text = ""
            blocks = getattr(resp, "content", None) or []
            for b in blocks:
                if getattr(b, "type", None) == "text":
                    text += getattr(b, "text", "")

            raw = text.strip()
            valid = normalize_bool_text(raw)

            return custom_id, {
                "judgement": valid,
                "confidence": None,
                "model": model,
                "raw_text": raw,
            }

        except Exception as e:
            last_exc = e
            # Exponential backoff with jitter.
            await asyncio.sleep(min(30.0, (2**attempt) + random.random()))

    raise last_exc if last_exc is not None else RuntimeError("Claude request failed")


async def run_all_chats(
    chats: list[dict[str, Any]],
    *,
    model: str,
    max_concurrent: int,
    requests_per_minute: float = 60,
    max_retries: int = 6,
) -> dict[str, dict[str, Any]]:
    """Run all chats concurrently with global pacing + bounded concurrency."""

    import anthropic  # type: ignore

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    client = anthropic.Anthropic(api_key=api_key)

    sem = asyncio.Semaphore(max_concurrent)
    limiter = AsyncRateLimiter(requests_per_minute=requests_per_minute)

    results = await asyncio.gather(
        *(
            run_single_chat(
                client=client,
                model=model,
                custom_id=chat["custom_id"],
                system=chat["system"],
                user=chat["user"],
                sem=sem,
                limiter=limiter,
                max_retries=max_retries,
            )
            for chat in chats
        )
    )

    return {cid: payload for cid, payload in results}


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

input_file = "data/experiments/2026_02_11/new_ten.json"
output_file = "data/experiments/2026_02_11/new_ten_judged_claude.json"


def build_chats(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chats: list[dict[str, Any]] = []

    for i, entry in enumerate(data):
        context = entry["context"]
        feature = entry.get("feature")
        feature_description = feature_info_dict[feature]["description"]
        feature_terms = entry.get("feature_terms", [])
        entity_description = {k: v for k, v in entry.items() if k in fields}
        measurement_val = entry["value"]

        # Keep the same system prompt as other scripts (contains hard constraint
        # to answer with exactly 'true' or 'false').
        system = JUDGE_INSTRUCTIONS

        user = build_user_prompt_from_entry(
            context=context,
            feature_description=feature_description,
            feature_terms=feature_terms,
            entity_description=entity_description,
            measurement_val=measurement_val,
        )

        chats.append(
            {
                "custom_id": str(i),
                "system": system,
                "user": user,
            }
        )

    return chats


####################################################################################################


if __name__ == "__main__":
    with open(input_file, "r") as f:
        data = json.load(f)

    chats = build_chats(data)

    model = "claude-opus-4-6"

    responses = asyncio.run(
        run_all_chats(
            chats,
            model=model,
            max_concurrent=3,
            requests_per_minute=40,
            max_retries=6,
        )
    )

    data_validated: list[dict[str, Any]] = []
    for cid, result in responses.items():
        idx = int(cid)
        entry = data[idx]
        entry_validated = entry | {
            "judgement": result.get("judgement"),
            "judgement_confidence": result.get("confidence"),
            "judgement_model": result.get("model"),
            # Keep for debugging prompt compliance; can be dropped later.
            "judgement_raw_text": result.get("raw_text"),
        }
        data_validated.append(entry_validated)

    with open(output_file, "w") as f:
        json.dump(data_validated, f, indent=4, ensure_ascii=False)
