import os
import json
import random
import asyncio
import time
import math
from typing import Any

from pydantic import BaseModel
from dotenv import load_dotenv

from scholarlm import JUDGE_INSTRUCTIONS

load_dotenv()

# NOTE: Requires the Google Gen AI SDK:
#   pip install google-genai
# and an API key in env var:
#   export GEMINI_API_KEY=...
#
# Token probabilities:
# - Gemini does not expose OpenAI-style logprobs uniformly across models/endpoints.
# - Some SDK responses include per-step candidate metadata (token + probability).
# - We compute P(true) from any available first-step candidate probabilities;
#   otherwise `confidence=None`.


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


def _extract_p_true_from_gemini_response(response: Any) -> float | None:
    """Best-effort P('true') from Gemini response metadata."""
    try:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return None

        cand0 = candidates[0]

        for attr in ("logprobs_result", "logprobs", "token_logprobs", "tokenMetadata"):
            lp_obj = getattr(cand0, attr, None)
            if not lp_obj:
                continue

            for steps_attr in ("top_candidates", "topCandidates", "steps", "tokens"):
                steps = getattr(lp_obj, steps_attr, None)
                if not steps:
                    continue

                step0 = steps[0] if isinstance(steps, list) else None
                if not step0:
                    continue

                top = step0
                if not isinstance(top, list):
                    top = (
                        getattr(step0, "candidates", None)
                        or getattr(step0, "top_candidates", None)
                        or getattr(step0, "topCandidates", None)
                    )

                if not top or not isinstance(top, list):
                    continue

                for alt in top:
                    tok = (
                        getattr(alt, "token", None)
                        or getattr(alt, "text", None)
                        or ""
                    ).strip().lower()
                    if tok != "true":
                        continue

                    p = getattr(alt, "probability", None)
                    if p is not None:
                        return float(p)

                    lp = getattr(alt, "log_probability", None)
                    if lp is None:
                        lp = getattr(alt, "logprob", None)
                    if lp is not None:
                        return float(math.exp(float(lp)))

        token_meta = getattr(cand0, "token_metadata", None)
        if token_meta and isinstance(token_meta, list) and token_meta:
            step0 = token_meta[0]
            top = (
                getattr(step0, "top_candidates", None)
                or getattr(step0, "topCandidates", None)
                or getattr(step0, "candidates", None)
            )
            if top and isinstance(top, list):
                for alt in top:
                    tok = (
                        getattr(alt, "token", None)
                        or getattr(alt, "text", None)
                        or ""
                    ).strip().lower()
                    if tok == "true":
                        p = getattr(alt, "probability", None)
                        if p is not None:
                            return float(p)
                        lp = getattr(alt, "log_probability", None) or getattr(alt, "logprob", None)
                        if lp is not None:
                            return float(math.exp(float(lp)))

        return None
    except Exception:
        return None


class AsyncRateLimiter:
    """A simple global rate limiter (token-less) that paces calls to ~rpm."""

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


async def _create_gemini_response(
    *,
    client: Any,
    model: str,
    system: str,
    user: str,
):
    """Call Gemini expecting a bare true/false."""

    # Lazy import so this file can still be imported without the SDK installed.
    from google import genai  # type: ignore

    # NOTE: In the `google-genai` SDK, `generate_content` returns a
    # `GenerateContentResponse` (non-awaitable) in many versions/configs.
    # Keep the outer function async for compatibility with our concurrency
    # plumbing, but do not `await` the SDK call.
    return client.models.generate_content(
        model=model,
        contents=[
            {"role": "user", "parts": [{"text": user}]},
        ],
        config={
            "system_instruction": system,
            "temperature": 0,
            "max_output_tokens": 5,
            "response_mime_type": "text/plain",
            "candidate_count": 1,
        },
    )


async def run_all_chats(
    chats: list[dict[str, Any]],
    *,
    model: str,
    max_concurrent: int = 20,
    requests_per_minute: float = 60,
    max_retries: int = 6,
) -> dict[str, dict[str, Any]]:
    """Run Gemini chats concurrently with global pacing + simple retries."""

    from google import genai  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

    sem = asyncio.Semaphore(max_concurrent)
    limiter = AsyncRateLimiter(requests_per_minute=requests_per_minute)

    async def _run_one(chat: dict[str, Any]):
        custom_id = chat["custom_id"]
        system = chat.get("system", "")
        user = chat.get("user", "")

        last_exc: Exception | None = None
        for attempt in range(max_retries):
            try:
                async with sem:
                    await limiter.wait()
                    response = await _create_gemini_response(
                        client=client,
                        model=model,
                        system=system,
                        user=user,
                    )

                raw = (getattr(response, "text", None) or "").strip()
                valid = normalize_bool_text(raw)
                p_true = _extract_p_true_from_gemini_response(response)

                return custom_id, {
                    "judgement": valid,
                    "confidence": p_true,
                    "model": model,
                    "raw_text": raw,
                }
            except Exception as e:
                last_exc = e
                await asyncio.sleep(min(30.0, (2**attempt) + random.random()))

        raise last_exc if last_exc is not None else RuntimeError("Gemini request failed")

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


def build_chats(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chats: list[dict[str, Any]] = []

    for i, entry in enumerate(data):
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

        chats.append({"custom_id": str(i), "system": system, "user": user})

    return chats


####################################################################################################
# Run the script.

input_file = "data/experiments/2026_02_11/new_ten.json"
output_file = "data/experiments/2026_02_11/new_ten_judged_gemini.json"

if __name__ == "__main__":
    with open(input_file, "r") as f:
        data = json.load(f)

    chats = build_chats(data)

    model = "gemini-3-flash-preview"

    responses = asyncio.run(
        run_all_chats(
            chats,
            model=model,
            max_concurrent=50,
            requests_per_minute=100,
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
            "judgement_raw_text": result.get("raw_text"),
        }
        data_validated.append(entry_validated)

    with open(output_file, "w") as f:
        json.dump(data_validated, f, indent=4, ensure_ascii=False)
