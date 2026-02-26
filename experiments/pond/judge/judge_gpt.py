import os
import json
import time
import asyncio
import backoff

from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Any

from openai import AsyncOpenAI
from openai import RateLimitError, APIError

from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS_TEXT, JUDGE_INSTRUCTIONS_TABLE
from scholarlm.utils import get_filenames_in_directory

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


async def _create_completion(model: str, messages, use_logprobs: bool = True):
    """Call the API expecting a bare true/false.

    IMPORTANT: Do not modify `messages` here; assume the prompt already contains
    the necessary instructions to respond with exactly 'true' or 'false'.
    """

    kwargs: dict[str, Any] = dict(
        model=model,
        messages=messages,
        max_completion_tokens=2048,
        #temperature=0,
    )

    if use_logprobs:
        kwargs["logprobs"] = True
        kwargs["top_logprobs"] = 5

    return await client.chat.completions.create(**kwargs)


async def run_all_chats(
    chats,
    model: str = "gpt-5-mini",
    max_concurrent: int = 25,
    requests_per_minute: float = 50,
    max_time: int = 60,
    max_tries: int = 6,
    use_logprobs: bool = True,
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
                return await _create_completion(
                    model=model, messages=messages, use_logprobs=use_logprobs
                )

        response = await _call_with_backoff()
        raw = (response.choices[0].message.content or "").strip()

        valid = normalize_bool_text(raw)
        p_valid_true = (
            _extract_p_true_from_first_token(response) if use_logprobs else None
        )

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
# Build the chats for all entries.

main_directory = "data/pond"
ocr_directory = os.path.join(main_directory, "ocr_output_cleaned_openai")

input_file = "data/experiments/2026_02_25/pond_openai.json"
output_file = "data/experiments/2026_02_25/pond_openai_judged_gpt.json"

ENTITY_TYPE_DESCRIPTION = (
    "A distinct aquatic ecosystem observation — a specific pond, lake, wetland, or "
    "similar water body — potentially further identified by treatment site, treatment "
    "state, or date of measurement."
)


def build_user_prompt_from_entry(
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
) -> str:
    """Create the shared user prompt for judging.

    This must stay stable across providers to make results comparable.
    """

    units_str = units if units is not None else "not reported"

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

    query = "\n\n".join(sections)
    return f"## Document:\n{document}\n\n## Query:\n{query}"


def build_chats(data, documents: list[str]):
    # Sort by document_id to improve temporal locality for any provider-side
    # prefix reuse / caching.
    # Carry original indices so we can write results in the original input order.
    data_with_idx = list(enumerate(data))
    data_with_idx.sort(key=lambda it: str(it[1].get("document_id", "")))

    chats = []

    for _i_sorted, (orig_idx, entry) in enumerate(data_with_idx):
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

        user = build_user_prompt_from_entry(
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

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]

        chats.append({"custom_id": str(orig_idx), "messages": messages})

    return chats


####################################################################################################
# Run the script.

if __name__ == "__main__":
    with open(input_file, "r") as f:
        data = json.load(f)

    #data = data[:100]  # limit to 100 for testing; remove or increase as needed

    # Load full documents in the same sorted order used during extraction.
    text_files = get_filenames_in_directory(ocr_directory, ignore=[".DS_Store", ".gitkeep"])
    text_files.sort()
    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(ocr_directory, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())

    chats = build_chats(data, documents)

    USE_LOGPROBS = False  # Set to False to disable logprobs computation

    responses = asyncio.run(
        run_all_chats(
            chats,
            model="gpt-5-mini",
            max_concurrent=30,
            requests_per_minute=60,
            max_time=60,
            max_tries=6,
            use_logprobs=USE_LOGPROBS,
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


