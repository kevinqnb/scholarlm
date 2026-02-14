import json
import time
import asyncio
import backoff

from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any

from openai import AsyncOpenAI
from openai import RateLimitError, APIError

from scholarlm import JUDGE_INSTRUCTIONS

load_dotenv()

client = AsyncOpenAI()

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


def _extract_p_true_from_first_token(response) -> float | None:
    """Compute P('true') from the *first generated token* logprobs (best-effort)."""
    try:
        import math

        choice0 = response.choices[0]
        lp = getattr(choice0, "logprobs", None)
        content = getattr(lp, "content", None) if lp is not None else None
        if not content:
            return None

        first = content[0]
        first_token = (getattr(first, "token", None) or "").strip().lower()
        first_logprob = getattr(first, "logprob", None)

        # If the model actually emitted 'true' as the first token, use that.
        if first_token == "true" and first_logprob is not None:
            return float(math.exp(first_logprob))

        # Otherwise, look for 'true' in the alternatives.
        top = getattr(first, "top_logprobs", None) or []
        for alt in top:
            tok = (getattr(alt, "token", None) or "").strip().lower()
            if tok == "true":
                alt_lp = getattr(alt, "logprob", None)
                if alt_lp is not None:
                    return float(math.exp(alt_lp))

        return None

    except Exception:
        return None


class AsyncRateLimiter:
    """A simple global rate limiter (token-less) that paces calls to ~rpm.

    This enforces an *average* request rate across all concurrent tasks.
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
            # schedule next slot
            self._next_time += self._interval
        if delay > 0:
            await asyncio.sleep(delay)


async def _create_completion(model: str, messages):
    """Call the API expecting a bare true/false.

    IMPORTANT: Do not modify `messages` here; assume the prompt already contains
    the necessary instructions to respond with exactly 'true' or 'false'.
    """

    return await client.chat.completions.create(
        model=model,
        messages=messages,
        max_completion_tokens=5,
        temperature=0,
        logprobs=True,
        top_logprobs=5,
    )


async def run_all_chats(
    chats,
    model: str = "gpt-4o",
    max_concurrent: int = 25,
    requests_per_minute: float = 50,
    max_time: int = 60,
    max_tries: int = 6,
):
    """Run all chats concurrently with simple global pacing + automatic backoff."""

    sem = asyncio.Semaphore(max_concurrent)
    limiter = AsyncRateLimiter(requests_per_minute=requests_per_minute)

    async def _run_one(chat):
        custom_id = chat["custom_id"]
        messages = chat["messages"]

        @backoff.on_exception(
            backoff.expo,
            (RateLimitError, APIError),
            max_time=max_time,
            max_tries=max_tries,
            jitter=backoff.full_jitter,
        )
        async def _call_with_backoff():
            async with sem:
                await limiter.wait()
                return await _create_completion(model=model, messages=messages)

        response = await _call_with_backoff()
        raw = (response.choices[0].message.content or "").strip()

        valid = normalize_bool_text(raw)
        p_valid_true = _extract_p_true_from_first_token(response)

        return custom_id, {
            "judgement": valid,
            "confidence": p_valid_true,
            "model": model,
            "raw_text": raw,
        }

    results = await asyncio.gather(*(_run_one(chat) for chat in chats))
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
# Build the chats for all entries.

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

def build_chats(data):
    # Sort by document_id then context to improve temporal locality for any
    # provider-side prefix reuse / caching.
    # Carry original indices so we can write results in the original input order.
    data_with_idx = list(enumerate(data))
    data_with_idx.sort(
        key=lambda it: (
            str(it[1].get("document_id", "")),
            str(it[1].get("context", "")),
        )
    )

    chats = []

    for _i_sorted, (orig_idx, entry) in enumerate(data_with_idx):
        context = entry["context"]
        feature = entry.get("feature")
        feature_description = feature_info_dict[feature]["description"]
        feature_terms = entry.get("feature_terms", [])
        entity_description = {k: v for k, v in entry.items() if k in fields}
        measurement_val = entry["value"]

        system = JUDGE_INSTRUCTIONS
        user = build_user_prompt_from_entry(
            context=context,
            feature_description=feature_description,
            feature_terms=feature_terms,
            entity_description=entity_description,
            measurement_val=measurement_val,
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]

        chats.append({"custom_id": str(orig_idx), "messages": messages})

    return chats


####################################################################################################
# Run the script.

input_file = "data/experiments/2026_02_11/pond.json"
output_file = "data/experiments/2026_02_11/pond_judged_gpt.json"

if __name__ == "__main__":
    with open(input_file, "r") as f:
        data = json.load(f)

    chats = build_chats(data)

    responses = asyncio.run(
        run_all_chats(
            chats,
            model="gpt-5.2",
            max_concurrent=30,
            requests_per_minute=60,
            max_time=60,
            max_tries=6,
        )
    )

    # Reconstruct output in the original `data` order.
    data_validated = []
    for i in range(len(data)):
        result = responses.get(str(i), {})
        entry = data[i]
        entry_validated = entry | {
            "judgement": result.get("judgement"),
            "judgement_confidence": result.get("confidence"),
            "judgement_model": result.get("model"),
            "judgement_raw_text": result.get("raw_text"),
        }
        data_validated.append(entry_validated)

    with open(output_file, "w") as f:
        json.dump(data_validated, f, indent=4, ensure_ascii=False)


