import os
import json
import random
import asyncio
import time
from typing import Any

from pydantic import BaseModel
from dotenv import load_dotenv
from scholarlm.utils import get_filenames_in_directory

from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS_TEXT, JUDGE_INSTRUCTIONS_TABLE

load_dotenv()

# NOTE: Requires the Anthropic SDK:
#   pip install anthropic
# and an API key in env var:
#   export ANTHROPIC_API_KEY=...
#
# This script mirrors `experiments/judge_gpt.py` / `experiments/judge_gemini.py` but uses Claude.
#
# Prompt caching:
# - Anthropic supports prompt caching by marking *stable* prompt segments with
#   `cache_control` on content blocks.
# - We mark the (1) system instructions and (2) per-entry context block as cacheable.
# - Variable query parts remain uncached.
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


def build_user_prompt_blocks_from_entry(
    *,
    document: str,
    source: str,
    attribute_description: str,
    attribute_terms: list[Any],
    entity_type_description: str,
    entity_description: dict[str, Any],
    page_number: int | None,
    table_number: int | None,
    measurement_val: Any = None,
    row_index: Any = None,
    column_index: Any = None,
    units: Any = None,
) -> tuple[str, str]:
    """Return (cached_context_text, query_text) for judging.

    We split the prompt so the large, reused context can be explicitly cached.
    """

    # Stable cached prefix for this entry (full document for entity validation).
    cached_context = f"## Document:\n{document}\n\n"

    units_str = units if units is not None else "not reported"

    # NOTE: Do not change entity_description formatting (keep Python dict repr).
    entity_section = (
        f"Target entity type: {entity_type_description}\n"
        f"Extracted entity: {entity_description}"
    )

    attribute_section = (
        f"Target attribute: {attribute_description}\n"
        f"Attribute terminology: {attribute_terms}"
    )

    location_parts = []
    if page_number is not None:
        location_parts.append(f"Page number: {page_number}")
    if source == "table" and table_number is not None:
        location_parts.append(f"Table number: {table_number}")
    location_section = "\n".join(location_parts)

    if source == "table":
        value_section = (
            f"Extracted row index: {row_index}\n"
            f"Extracted column index: {column_index}\n"
            f"Extracted units: {units_str}"
        )
        closing = (
            "Is the extracted (entity, attribute, row index, column index) tuple fully valid — "
            "meaning the entity is correctly identified and together the row index and column index "
            "correctly locate the value for that (entity, target attribute) pair in the specified table?"
        )
    else:
        value_section = (
            f"Extracted value: {measurement_val}\n"
            f"Extracted units: {units_str}"
        )
        closing = (
            "Is the extracted (entity, attribute, value) triplet fully valid — "
            "meaning the entity is correctly identified and the extracted value "
            "correctly corresponds to the target attribute for that entity, as evidenced by the document?"
        )

    sections = [entity_section, attribute_section]
    if location_section:
        sections.append(location_section)
    sections.append(value_section)
    sections.append(closing)

    query_text = f"## Query:\n" + "\n\n".join(sections)

    return cached_context, query_text


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
    cached_context: str,
    query_text: str,
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
                        # Mark stable instructions as cacheable.
                        system=[
                            {
                                "type": "text",
                                "text": system,
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    # Mark the reused context as cacheable.
                                    {
                                        "type": "text",
                                        "text": cached_context,
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                    # Keep variable parts uncached.
                                    {"type": "text", "text": query_text},
                                ],
                            }
                        ],
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
                cached_context=chat["cached_context"],
                query_text=chat["query_text"],
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

attribute_info_dict = {
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

main_directory = "data/pond"
ocr_directory = os.path.join(main_directory, "ocr_output_cleaned_openai")

input_file = "data/experiments/2026_02_25/pond_openai.json"
output_file = "data/experiments/2026_02_25/pond_openai_judged_claude.json"

ENTITY_TYPE_DESCRIPTION = (
    "A distinct aquatic ecosystem observation — a specific pond, lake, wetland, or "
    "similar water body — potentially further identified by treatment site, treatment "
    "state, or date of measurement."
)


def build_chats(data: list[dict[str, Any]], documents: list[str]) -> list[dict[str, Any]]:
    # Sort by document_id to increase cache locality (Claude caches expire).
    # We carry the original index so outputs can be re-mapped to the original order.
    data_with_idx = list(enumerate(data))
    data_with_idx.sort(key=lambda it: str(it[1].get("document_id", "")))

    chats: list[dict[str, Any]] = []

    for i_sorted, (orig_idx, entry) in enumerate(data_with_idx):
        document = documents[entry["document_id"]]
        attribute = entry.get("attribute")
        attribute_description = attribute_info_dict[attribute]["description"]
        attribute_terms = entry.get("attribute_terms", [])
        entity_description = {k: v for k, v in entry.items() if k in fields}
        page_number = entry.get("page_number")
        table_number = entry.get("table_number")
        source = entry.get("source", "text")
        units = entry.get("units")

        if source == "table":
            system = JUDGE_INSTRUCTIONS_TABLE
            row_index = entry.get("row_index")
            column_index = entry.get("column_index")
            measurement_val = None
        else:
            system = JUDGE_INSTRUCTIONS_TEXT
            measurement_val = entry["value"]
            row_index = column_index = None

        cached_context, query_text = build_user_prompt_blocks_from_entry(
            document=document,
            source=source,
            attribute_description=attribute_description,
            attribute_terms=attribute_terms,
            entity_type_description=ENTITY_TYPE_DESCRIPTION,
            entity_description=entity_description,
            page_number=page_number,
            table_number=table_number,
            measurement_val=measurement_val,
            row_index=row_index,
            column_index=column_index,
            units=units,
        )

        chats.append(
            {
                # Use the original index as the custom_id so we can map results
                # back to the original input order.
                "custom_id": str(orig_idx),
                "system": system,
                "cached_context": cached_context,
                "query_text": query_text,
            }
        )

    return chats


####################################################################################################


if __name__ == "__main__":
    with open(input_file, "r") as f:
        data = json.load(f)

    # Load full documents in the same sorted order used during extraction.
    text_files = get_filenames_in_directory(ocr_directory, ignore=[".DS_Store", ".gitkeep"])
    text_files.sort()
    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(ocr_directory, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())

    chats = build_chats(data, documents)

    model = "claude-haiku-4-5"

    responses = asyncio.run(
        run_all_chats(
            chats,
            model=model,
            max_concurrent=3,
            requests_per_minute=25,
            max_retries=6,
        )
    )

    # Reconstruct output in the original `data` order.
    data_validated: list[dict[str, Any]] = []
    for i in range(len(data)):
        result = responses.get(str(i), {})
        entry = data[i]
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
