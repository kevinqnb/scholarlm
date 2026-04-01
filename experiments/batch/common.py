"""Shared data, prompt builders, and utilities for batch judging.

All provider modules import from here so the prompt stays identical across
OpenAI, Anthropic, and Gemini — making results directly comparable.

The key public function ``prepare_chat_entries`` accepts an optional
``dataset_config`` argument (a ``scholarlm.config.DatasetConfig`` instance).
When omitted it falls back to the hardcoded pond schema and attribute catalogue
below, preserving backward compatibility for all existing pond judge scripts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel

from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS_TABLE, JUDGE_INSTRUCTIONS_TEXT
from scholarlm.utils import get_filenames_in_directory

load_dotenv()

# ─── Pond-specific defaults (kept for backward compatibility) ─────────────────
#
# New code should pass a DatasetConfig to prepare_chat_entries() instead of
# relying on these module-level constants.


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
        "description": (
            "Geographic latitude of the ecosystem location, expressed in a standard geographic "
            "coordinate system (e.g., WGS84). This should refer to the centroid or stated reference "
            "point of the ecosystem, not a bounding box or region."
        ),
        "units": ["degrees", "radians"],
    },
    "longitude": {
        "description": (
            "Geographic longitude of the ecosystem location, expressed in a standard geographic "
            "coordinate system (e.g., WGS84). This should refer to the centroid or stated reference "
            "point of the ecosystem, not a bounding box or region."
        ),
        "units": ["degrees", "radians"],
    },
    "surface_area": {
        "description": (
            "Surface area of the water body itself (not the watershed or catchment area). This should "
            "represent the horizontal area of open water or the stated ecosystem boundary at the time "
            "of measurement or description."
        ),
        "units": ["km^2", "mi^2", "ha", "m^2", "acres"],
    },
    "max_depth": {
        "description": (
            "Maximum water depth of the ecosystem, defined as the deepest point of the water body at "
            "the time of measurement or as reported in the source. This is not the mean or average depth."
        ),
        "units": ["m", "km", "ft"],
    },
    "vegetation_cover": {
        "description": (
            "Fraction or percentage of the ecosystem surface area covered by aquatic macrophytes or "
            "other aquatic vegetation. This should refer to areal coverage, not biomass or volume."
        ),
        "units": ["percent", "fraction"],
    },
    "ph": {
        "description": (
            "pH of the water, i.e., the negative logarithm of the hydrogen ion activity. This is a "
            "dimensionless quantity and should refer to a measured water pH value, not soil or sediment pH."
        ),
        "units": [],
    },
    "tn": {
        "description": (
            "Total nitrogen concentration in the water column, including both dissolved and particulate "
            "forms and all major species (e.g., nitrate, nitrite, ammonium, organic nitrogen), as "
            "explicitly reported in the source."
        ),
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"],
    },
    "tp": {
        "description": (
            "Total phosphorus concentration in the water column, including both dissolved and particulate "
            "forms, as explicitly reported in the source (i.e., not just soluble reactive phosphorus or "
            "orthophosphate)."
        ),
        "units": ["µg/L", "mg/L", "μmol/L", "ppm", "ppb"],
    },
    "chla": {
        "description": (
            "Chlorophyll-a concentration in the water column, used as a proxy for phytoplankton biomass. "
            "This should refer to extracted or in situ chlorophyll-a measurements, not total chlorophyll "
            "or other pigments unless explicitly labeled as chlorophyll-a."
        ),
        "units": ["µg/L", "mg/L", "mg/m^3"],
    },
}

ENTITY_TYPE_DESCRIPTION = (
    "A distinct aquatic ecosystem observation — a specific pond, lake, wetland, or "
    "similar water body — potentially further identified by treatment site, treatment "
    "state, or date of measurement."
)


# ─── Prompt builder ───────────────────────────────────────────────────────────


def build_user_prompt(
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
    """Build the shared user prompt string (identical across all providers)."""
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


# ─── Data preparation ─────────────────────────────────────────────────────────


def prepare_chat_entries(
    data: list[dict],
    documents: list[str],
    dataset_config: Any = None,
) -> list[dict[str, Any]]:
    """Convert raw extraction data to provider-agnostic chat entries.

    Entries are sorted by document_id for cache locality. Each entry contains:
        custom_id     – str(original index in data), used to map results back
        system        – system prompt string
        user          – user prompt string
        user_document – document prefix used for Anthropic prompt caching

    Args:
        data: List of extraction records from a ``final.json`` file.
        documents: List of OCR text strings indexed by ``document_id``.
        dataset_config: Optional ``DatasetConfig`` instance.  When provided,
            entity fields, attribute descriptions, and the entity type
            description are sourced from the config.  When ``None``, the
            module-level pond defaults are used (backward-compatible behaviour).

    Returns:
        List of chat entry dicts ready for any provider batch module.
    """
    # Resolve schema fields, attribute catalogue, and entity description from
    # the dataset config when provided, or fall back to pond defaults.
    if dataset_config is not None:
        _fields = dataset_config.entity_schema.model_fields.keys()
        _attr_dict = dataset_config.attribute_info_dict
        _entity_type_desc = dataset_config.entity_type_description
    else:
        _fields = fields
        _attr_dict = attribute_info_dict
        _entity_type_desc = ENTITY_TYPE_DESCRIPTION

    data_with_idx = list(enumerate(data))
    data_with_idx.sort(key=lambda it: str(it[1].get("document_id", "")))

    entries: list[dict[str, Any]] = []
    for _i_sorted, (orig_idx, entry) in enumerate(data_with_idx):
        document = documents[entry["document_id"]]
        attribute = entry.get("attribute")
        attribute_description = _attr_dict[attribute]["description"]
        attribute_terms = entry.get("attribute_terms", [])
        entity_description = {k: v for k, v in entry.items() if k in _fields}
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

        user = build_user_prompt(
            document=document,
            source=source,
            attribute_description=attribute_description,
            attribute_terms=attribute_terms,
            entity_type_description=_entity_type_desc,
            entity_description=entity_description,
            page_number=page_number,
            table_number=table_number,
            measurement_val=measurement_val,
            row_index=row_index,
            column_index=column_index,
            units=units,
        )

        user_document = f"## Document:\n{document}\n\n"
        entries.append({
            "custom_id": str(orig_idx),
            "system": system,
            "user": user,
            # Cached prefix for Anthropic — the document text shared across many requests.
            # OpenAI and Gemini ignore this field and use the combined "user" string.
            "user_document": user_document,
        })

    return entries


# ─── I/O helpers ──────────────────────────────────────────────────────────────


def load_data(input_file: str) -> list[dict]:
    with open(input_file, "r") as f:
        return json.load(f)


def load_documents(ocr_directory: str) -> list[str]:
    text_files = get_filenames_in_directory(
        ocr_directory, ignore=[".DS_Store", ".gitkeep"]
    )
    text_files.sort()
    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(ocr_directory, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())
    return documents


def merge_results(data: list[dict], results: dict[str, dict]) -> list[dict]:
    """Merge batch results back into data in the original input order."""
    output = []
    for i, entry in enumerate(data):
        result = results.get(str(i), {})
        output.append(
            entry
            | {
                "judgement": result.get("judgement"),
                "judgement_prob": result.get("prob"),
                "judgement_model": result.get("model"),
                "judgement_raw_text": result.get("raw_text"),
            }
        )
    return output


def normalize_bool_text(text: str | None) -> bool | None:
    """Parse a model response into a boolean, or None if not parseable."""
    if text is None:
        return None
    t = text.strip().lower()
    if "true" in t:
        return True
    if "false" in t:
        return False
    return None


def chunk_by_size(requests: list[dict], max_bytes: int) -> list[list[dict]]:
    """Split requests into chunks so each chunk's JSONL serialization fits within max_bytes.

    Size per record is estimated as len(json.dumps(r)) + 1 (the newline).
    This is exact for JSONL-based providers (OpenAI, Gemini) and a close
    approximation for JSON-array-based providers (Anthropic).
    """
    if max_bytes <= 0:
        raise ValueError("max_bytes must be > 0")

    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_bytes = 0

    for req in requests:
        req_bytes = len(json.dumps(req, ensure_ascii=False).encode("utf-8")) + 1
        if current and current_bytes + req_bytes > max_bytes:
            chunks.append(current)
            current = []
            current_bytes = 0
        current.append(req)
        current_bytes += req_bytes

    if current:
        chunks.append(current)

    return chunks


def write_jsonl(records: list[dict], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
