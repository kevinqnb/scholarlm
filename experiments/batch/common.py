"""Shared data, prompt builders, and utilities for batch judging.

All provider modules import from here so the prompt stays identical across
OpenAI, Anthropic, and Gemini — making results directly comparable.

The key public functions are:

    ``build_judge_query``   — builds the ## QUERY content (entity + attribute +
                              value sections) without the document prefix.
    ``build_user_prompt``   — wraps build_judge_query with the ## CONTEXT header
                              to produce the full user message for chat-API runners.
    ``prepare_chat_entries`` — converts raw extraction records to provider-agnostic
                              chat entries sorted by document_id for cache locality.
                              Each entry includes ``user_query`` and ``document_id``
                              for use by the interpretability judge runner.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scholarlm.config import DatasetConfig
from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS_TABLE, JUDGE_INSTRUCTIONS_TEXT
from scholarlm.utils import get_filenames_in_directory

load_dotenv()


# ─── Prompt builder ───────────────────────────────────────────────────────────


def build_judge_query(
    *,
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
    """Build the ## QUERY content without the document prefix.

    Returns the joined sections string that appears after ``## QUERY:`` in the
    full prompt.  Both ``build_user_prompt`` and the interpretability judge
    runner (which passes document and query separately to JudgementLM) call
    this function so the query content is identical across all judge runners.
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

    return "\n\n".join(sections)


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
    """Build the full user prompt string (## CONTEXT prefix + ## QUERY content).

    Identical across all chat-API providers (OpenAI, Anthropic, Gemini, vLLM).
    """
    query = build_judge_query(
        source=source,
        attribute_description=attribute_description,
        attribute_terms=attribute_terms,
        entity_type_description=entity_type_description,
        entity_description=entity_description,
        page_number=page_number,
        table_number=table_number,
        measurement_val=measurement_val,
        row_index=row_index,
        column_index=column_index,
        units=units,
    )
    return f"## CONTEXT:\n{document}\n\n## QUERY:\n{query}"


# ─── Data preparation ─────────────────────────────────────────────────────────


def prepare_chat_entries(
    data: list[dict],
    documents: list[str],
    dataset_config: DatasetConfig,
) -> list[dict[str, Any]]:
    """Convert raw extraction data to provider-agnostic chat entries.

    Entries are sorted by document_id for cache locality. Each entry contains:
        custom_id     – str(original index in data), used to map results back
        document_id   – int index into documents list (for the interp judge runner)
        system        – system prompt string (JUDGE_INSTRUCTIONS_TEXT or _TABLE)
        user          – full user prompt string (## CONTEXT + ## QUERY)
        user_query    – the ## QUERY content only (for JudgementLM interp runner)
        user_document – ## CONTEXT prefix only (used for Anthropic prompt caching)

    Args:
        data: List of extraction records from a ``final.json`` file.
        documents: List of OCR text strings indexed by ``document_id``.
        dataset_config: ``DatasetConfig`` instance supplying entity schema,
            attribute catalogue, and entity type description.

    Returns:
        List of chat entry dicts ready for any provider batch module or the
        interpretability judge runner.
    """
    _fields = dataset_config.entity_schema.model_fields.keys()
    _attr_dict = dataset_config.attribute_info_dict
    _entity_type_desc = dataset_config.entity_type_description

    data_with_idx = list(enumerate(data))
    data_with_idx.sort(key=lambda it: str(it[1].get("document_id", "")))

    entries: list[dict[str, Any]] = []
    for _i_sorted, (orig_idx, entry) in enumerate(data_with_idx):
        document_id = entry["document_id"]
        document = documents[document_id]
        attribute = entry.get("attribute")
        try:
            attribute_description = _attr_dict[attribute]["description"]
        except KeyError:
            print(f"Attribute '{attribute}' not found in dataset_config.attribute_info_dict")
            continue

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

        query = build_judge_query(
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

        user = f"## CONTEXT:\n{document}\n\n## QUERY:\n{query}"
        # Cached prefix for Anthropic — the document text shared across many
        # requests for the same paper.  OpenAI and Gemini ignore this field.
        user_document = f"## CONTEXT:\n{document}\n\n"

        entries.append({
            "custom_id": str(orig_idx),
            "document_id": document_id,
            "system": system,
            "user": user,
            "user_query": query,
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


def load_documents_for_dataset(dataset_config: DatasetConfig, ocr_directory: str) -> list[str]:
    """Load OCR documents applying the same paper_filter and paper_subset as extraction.

    ``document_id`` values in ``final.json`` are indices into the list produced
    by this function.  Using a plain directory listing (without filtering) causes
    index misalignment whenever a subset config (e.g. ``pond_ten``) is used.

    Args:
        dataset_config: Dataset configuration supplying metadata_file, paper_filter,
            and paper_subset.
        ocr_directory: Directory containing ``.txt`` OCR files.

    Returns:
        List of document strings in the same order as during extraction.
    """
    with open(dataset_config.metadata_file) as f:
        paper_info: dict = json.load(f)

    text_files = get_filenames_in_directory(ocr_directory, ignore=[".DS_Store", ".gitkeep"])
    text_files.sort()

    if dataset_config.paper_filter is not None:
        registered_ids = {k for k, v in paper_info.items() if dataset_config.paper_filter(v)}
        text_files = [f for f in text_files if f.replace(".txt", "") in registered_ids]

    if dataset_config.paper_subset is not None:
        subset_set = set(dataset_config.paper_subset)
        text_files = [f for f in text_files if f.replace(".txt", "") in subset_set]

    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(ocr_directory, fname), "r", encoding="utf-8") as fh:
            documents.append(fh.read())
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
