import os
import json
import random
import asyncio
from typing import Any, Optional

from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from scholarlm import JUDGE_INSTRUCTIONS

# NOTE: Requires the Google Gen AI SDK:
#   pip install google-genai
# and an API key in env var:
#   export GEMINI_API_KEY=...
#
# This script mirrors `experiments/validate_gpt.py` but uses Gemini.
#
# Token probabilities:
# - Gemini's public API does not currently expose full per-token logprobs in the
#   same way OpenAI does.
# - However, Gemini *can* return top candidate response tokens for constrained
#   decoding. In the SDK this is typically surfaced via candidate/token metadata
#   when enabled.
# - The implementation below attempts to read those fields if present and will
#   otherwise set `confidence=None`.


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

# Match your GPT script defaults; edit as needed.
input_file = "data/01_28_26/ten_standardize.json"
output_file = "data/01_28_26/ten_judged_gemini.json"


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

        # Gemini expects a single "contents" list (role + parts). We'll keep a
        # similar mental model: system instructions + user content.
        chats.append(
            {
                "custom_id": str(i),
                "system": instructions,
                "user": prompt,
            }
        )

    return chats


####################################################################################################

# Gemini integration
# We import inside a function so the file can be imported without the dependency.

def _extract_text_and_confidence(resp: Any) -> tuple[str, Optional[float]]:
    """Best-effort extraction of (text, confidence) from Gemini SDK response.

    Confidence matches the GPT script's intent:
      - confidence = P(first_generated_token == produced_label)
        for produced_label in {'true','false'}.

    If token probability metadata isn't available from Gemini, returns None.
    """

    # Text extraction (SDK response shape varies across versions).
    text: str = ""

    # 1) Prefer the canonical structure: resp.candidates[0].content.parts[].text
    try:
        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            c0 = candidates[0]
            content = getattr(c0, "content", None)
            parts = getattr(content, "parts", None) if content is not None else None
            if parts:
                chunk_list: list[str] = []
                for p in parts:
                    t = getattr(p, "text", None)
                    if isinstance(t, str) and t:
                        chunk_list.append(t)
                text = "".join(chunk_list).strip()
    except Exception:
        pass

    # 1b) Some SDK versions store text directly on content (rare but seen in the wild)
    if not text:
        try:
            candidates = getattr(resp, "candidates", None) or []
            if candidates:
                c0 = candidates[0]
                content = getattr(c0, "content", None)
                direct_text = getattr(content, "text", None) if content is not None else None
                if isinstance(direct_text, str) and direct_text.strip():
                    text = direct_text.strip()
        except Exception:
            pass

    # 2) Fallback: resp.text convenience accessor
    if not text:
        try:
            text = (getattr(resp, "text", None) or "").strip()
        except Exception:
            text = ""

    # NOTE: Do NOT fall back to str(resp). The SDK's __repr__ can be fairly
    # verbose and includes metadata rather than the model text (and can look like
    # a valid "response" even when no text was returned).

    # Confidence extraction (best-effort).
    confidence: Optional[float] = None

    try:
        import math

        candidates = getattr(resp, "candidates", None) or []
        if candidates:
            c0 = candidates[0]
            produced_label = (text or "").strip().lower()

            # Option A: logprobs_result-style object (version/endpoint dependent)
            lpr = getattr(c0, "logprobs_result", None)
            if lpr is not None:
                tokens = getattr(lpr, "tokens", None)
                if tokens:
                    t0 = tokens[0]

                    # Prefer the logprob of the actually generated token if present.
                    gen_tok = (getattr(t0, "token", None) or getattr(t0, "text", None) or "").strip().lower()
                    gen_lp = getattr(t0, "logprob", None)
                    if gen_lp is not None and gen_tok in {"true", "false"}:
                        confidence = float(math.exp(gen_lp))
                    else:
                        # Fallback: look up the produced label in the top candidates.
                        top = getattr(t0, "top_candidates", None) or getattr(t0, "top_logprobs", None) or []
                        for cand in top:
                            tok = (getattr(cand, "token", None) or getattr(cand, "text", None) or "").strip().lower()
                            if tok == produced_label:
                                lp = getattr(cand, "logprob", None)
                                if lp is not None:
                                    confidence = float(math.exp(lp))
                                break

            # Option B: direct per-token logprobs for generated tokens
            if confidence is None:
                token_logprobs = getattr(c0, "token_logprobs", None)
                if token_logprobs and isinstance(token_logprobs, list) and len(token_logprobs) > 0:
                    # Assume token_logprobs[0] corresponds to the first generated token.
                    confidence = float(math.exp(token_logprobs[0]))

    except Exception:
        confidence = None

    return text, confidence


async def run_single_chat(
    client: Any,
    model: str,
    custom_id: str,
    system: str,
    user: str,
    sem: asyncio.Semaphore,
    max_retries: int,
) -> tuple[str, dict[str, Any]]:
    """Run a single Gemini request with retry. Returns (custom_id, result_dict)."""

    async with sem:
        for attempt in range(max_retries):
            try:
                # google-genai Client is sync; run in worker thread.
                def _call():
                    # For strict boolean outputs, set temperature=0 and restrict output tokens.
                    # Also strongly instruct: "Respond with true/false only" already in JUDGE_INSTRUCTIONS.
                    return client.models.generate_content(
                        model=model,
                        # Use structured contents to reduce SDK-version differences.
                        contents=[
                            {
                                "role": "user",
                                "parts": [{"text": user}],
                            }
                        ],
                        config={
                            "system_instruction": system,
                            "temperature": 0.0,
                            # Avoid MAX_TOKENS truncation; still tiny.
                            "max_output_tokens": 100,
                            # If supported by your SDK/version, you can try enabling logprobs-like metadata here.
                            # "response_logprobs": True,
                            # "logprobs": True,
                        },
                    )

                resp = await asyncio.to_thread(_call)
                text, confidence = _extract_text_and_confidence(resp)

                if not (text or "").strip():
                    # Minimal debug; helpful when Gemini returns Candidate(content=Content())
                    try:
                        cands = getattr(resp, "candidates", None) or []
                        c0 = cands[0] if cands else None
                        content = getattr(c0, "content", None) if c0 is not None else None
                        parts = getattr(content, "parts", None) if content is not None else None
                        finish_reason = getattr(c0, "finish_reason", None) if c0 is not None else None
                        print(
                            f"[{custom_id}] Empty Gemini text. "
                            f"finish_reason={finish_reason} "
                            f"content_type={type(content).__name__ if content is not None else None} "
                            f"parts_len={len(parts) if isinstance(parts, list) else None}"
                        )
                    except Exception:
                        print(f"[{custom_id}] Empty Gemini text. Response type={type(resp).__name__}")

                return custom_id, {
                    "judgement": text,
                    "confidence": confidence,
                    "model": model,
                }

            except Exception as e:
                # Rate limit / transient errors can manifest as generic exceptions depending on SDK.
                wait_time = (2**attempt) + random.random()
                print(f"[{custom_id}] Gemini error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                print(f"  Retrying in {wait_time:.2f}s...")
                await asyncio.sleep(wait_time)

        raise RuntimeError(f"[{custom_id}] Failed after {max_retries} retries.")


async def run_all_chats(
    chats: list[dict[str, Any]],
    model: str,
    max_concurrent: int,
    max_retries: int,
) -> dict[str, dict[str, Any]]:
    """Run all chats in parallel with limited concurrency."""

    from google import genai  # type: ignore

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    client = genai.Client(api_key=api_key)

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
    return {cid: payload for cid, payload in results}


####################################################################################################

if __name__ == "__main__":
    with open(input_file, "r") as f:
        data = json.load(f)

    chats = build_chats(data)

    # Suggested Gemini 3 model name; update to the exact deployed name in your project.
    # Examples sometimes look like: "gemini-2.0-flash" / "gemini-2.0-pro" / etc.
    #model = "gemini-3-pro-preview"
    model = "gemini-3-flash-preview"

    responses = asyncio.run(
        run_all_chats(
            chats,
            model=model,
            max_concurrent=20,
            max_retries=5,
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
        }
        data_validated.append(entry_validated)

    with open(output_file, "w") as f:
        json.dump(data_validated, f, indent=4, ensure_ascii=False)
