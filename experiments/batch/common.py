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
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scholarlm.config import DatasetConfig
from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS
from scholarlm.utils import get_filenames_in_directory

load_dotenv()

# Debug: set to None to print all, or a small number to limit output
_DEBUG_PAPERS_LIMIT: int | None = 3


# ─── Page extraction ──────────────────────────────────────────────────────────

_PAGE_BLOCK_RE = re.compile(r'<page number="(\d+)">.*?</page>', re.DOTALL)


def _extract_page_text(document: str, page_numbers: list[int]) -> str:
    """Extract specific pages from an OCR document.

    OCR documents use ``<page number="N">...</page>`` blocks.  This returns the
    blocks for the requested page numbers concatenated with a blank line between
    them, preserving the original ``<page number="N">`` tags so the judge can
    still orient itself within the document.

    Falls back to the full document string if no ``<page number>`` tags are found
    or if none of the requested pages exist.

    Args:
        document: Full raw OCR document text.
        page_numbers: Page numbers to extract (0-indexed, matching the
            ``<page number="N">`` tags in the OCR document).

    Returns:
        Concatenated text of the requested page blocks, or the full document
        as a fallback.
    """
    if not page_numbers:
        return document
    target = {pn for pn in page_numbers if pn is not None}
    if not target:
        return document

    parts = [m.group(0) for m in _PAGE_BLOCK_RE.finditer(document)
             if int(m.group(1)) in target]
    return "\n\n".join(parts) if parts else document


# ─── Prompt builder ───────────────────────────────────────────────────────────


def build_judge_query(
    *,
    attribute_description: str,
    attribute_terms: list[Any],
    entity_type_description: str,
    entity_description: dict[str, Any],
    measurement_val: Any,
    units: Any = None,
    event_description: dict[str, Any] | None = None,
) -> str:
    """Build the ## QUERY content for the judge prompt.

    Returns the joined sections string that appears after ``## QUERY:`` in the
    full prompt.  Both ``build_user_prompt`` and the interpretability judge
    runner call this function so the query content is identical across all
    judge runners.

    Args:
        attribute_description: Full attribute description string.
        attribute_terms: List of terminology strings for the attribute.
        entity_type_description: One-sentence entity type description.
        entity_description: Dict of entity field values (entity fields only;
            event fields are passed separately via ``event_description``).
        measurement_val: The extracted scalar value.
        units: Extracted units string, or ``None`` if not reported.
        event_description: Dict of measurement event field values extracted
            from ``measurement_event_schema`` fields.  ``None`` or empty dict
            omits the event section entirely (datasets with no event schema).

    Returns:
        Multi-section query string ready to follow ``## QUERY:\\n``.
    """
    units_str = units if units is not None else "not reported"

    entity_display = {k: v for k, v in entity_description.items() if v is not None}
    entity_section = (
        f"Target entity type: {entity_type_description}\n"
        f"Extracted entity: {entity_display}"
    )

    attribute_section = f"Target attribute: {attribute_description}"
    if attribute_terms:
        attribute_section += f"\nAttribute terminology: {attribute_terms}"

    value_section = (
        f"Extracted value: {measurement_val}\n"
        f"Extracted units: {units_str}"
    )

    closing = "(true or false) Is this extraction correct?"
    sections = [entity_section, attribute_section]

    if event_description:
        event_display = {k: v for k, v in event_description.items() if v is not None}
        if event_display:
            sections.append(f"Measurement event information: {event_display}")
    
    sections += [value_section, closing]
    return "\n\n".join(sections)


# ─── Data preparation ─────────────────────────────────────────────────────────


def prepare_chat_entries(
    data: list[dict],
    documents: dict[str, str],
    dataset_config: DatasetConfig,
) -> list[dict[str, Any]]:
    """Convert raw extraction data to provider-agnostic chat entries.

    Entries are sorted by document_id for cache locality. Each entry contains:
        custom_id     – str(original index in data), used to map results back
        document_id   – string paper code identifying the source document
        system        – system prompt string (JUDGE_INSTRUCTIONS)
        user          – full user prompt string (## CONTEXT page text + ## QUERY)
        user_query    – the ## QUERY content only (for JudgementLM interp runner)
        user_document – ## CONTEXT prefix only (used for Anthropic prompt caching)
        page_text     – extracted page(s) text (for JudgementLM interp runner)

    Provenance fields (``source``, ``page_number``, ``table_number``) are stored
    as lists in ``final.json`` because deduplication merges multiple source
    occurrences.  This function unwraps them to the appropriate scalar or list
    types before building prompts.

    Args:
        data: List of extraction records from a ``final.json`` file.
        documents: Dict mapping paper_code strings to raw OCR text, as returned
            by ``load_documents_for_dataset``.
        dataset_config: ``DatasetConfig`` instance supplying entity schema,
            attribute catalogue, entity type description, and optional
            measurement event schema.

    Returns:
        List of chat entry dicts ready for any provider batch module or the
        interpretability judge runner.
    """
    _filter: set[str] = set(dataset_config.judge_filter_fields or [])
    _entity_fields: list[str] = [
        k for k in dataset_config.entity_schema.model_fields.keys()
        if k not in _filter
    ]
    _event_fields: list[str] = (
        [k for k in dataset_config.measurement_event_schema.model_fields.keys()
         if k not in _filter]
        if dataset_config.measurement_event_schema is not None
        else []
    )
    _attr_dict = dataset_config.attribute_info_dict
    _entity_type_desc = dataset_config.entity_type_description

    data_with_idx = list(enumerate(data))
    data_with_idx.sort(key=lambda it: str(it[1].get("document_id", "")))

    entries: list[dict[str, Any]] = []
    papers_printed: int = 0
    for _i_sorted, (orig_idx, entry) in enumerate(data_with_idx):
        document_id = str(entry["document_id"])
        document = documents.get(document_id)
        if document is None:
            print(f"Warning: document_id '{document_id}' not found in documents, skipping")
            continue
        attribute = entry.get("attribute")
        _judge_attr_map = dataset_config.judge_attribute_map or {}
        judge_attribute = _judge_attr_map.get(attribute, attribute)
        try:
            attribute_description = _attr_dict[judge_attribute]["description"]
        except KeyError:
            print(f"Attribute '{attribute}' not found in dataset_config.attribute_info_dict")
            continue

        attribute_terms = entry.get("attribute_terms", [])
        entity_description = {k: entry.get(k) for k in _entity_fields}
        event_description = (
            {k: entry.get(k) for k in _event_fields}
            if _event_fields else None
        )
        units = entry.get("units")
        measurement_val = entry.get("value")

        # Provenance fields are stored as lists in final.json — one element per
        # source occurrence (a value may appear in both prose and a table).
        source_raw = entry.get("source", "text")
        source_list: list[str] = source_raw if isinstance(source_raw, list) else [source_raw]

        # page_number / table_number: keep all unique non-None values for
        # page extraction and table identification.
        pn_raw = entry.get("page_number")
        page_numbers: list[int] = (
            [pn for pn in pn_raw if pn is not None]
            if isinstance(pn_raw, list)
            else ([pn_raw] if pn_raw is not None else [])
        )
        tn_raw = entry.get("table_number")
        table_numbers: list[int | None] = (
            [tn for tn in tn_raw if tn is not None]
            if isinstance(tn_raw, list)
            else ([tn_raw] if tn_raw is not None else [])
        )

        # Extract only the relevant page(s) from the raw OCR document.
        page_text = _extract_page_text(document, page_numbers)

        system = JUDGE_INSTRUCTIONS

        query = build_judge_query(
            attribute_description=attribute_description,
            attribute_terms=attribute_terms,
            entity_type_description=_entity_type_desc,
            entity_description=entity_description,
            measurement_val=measurement_val,
            units=units,
            event_description=event_description,
        )

        user = f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
        # Cached prefix for Anthropic — the page text shared across requests
        # for the same source page.  OpenAI and Gemini ignore this field.
        user_document = f"## CONTEXT:\n{page_text}\n\n"

        if _DEBUG_PAPERS_LIMIT is None or papers_printed < _DEBUG_PAPERS_LIMIT:
            print(f"DEBUG: User message for document_id={document_id}, orig_idx={orig_idx}:\n{user}\n")
            print()
            papers_printed += 1

        entries.append({
            "custom_id": str(orig_idx),
            "document_id": document_id,
            "system": system,
            "user": user,
            "user_query": query,
            "user_document": user_document,
            "page_text": page_text,
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


def load_documents_for_dataset(dataset_config: DatasetConfig, ocr_directory: str) -> dict[str, str]:
    """Load OCR documents applying the same paper_filter and paper_subset as extraction.

    Returns a dict keyed by paper_code so that ``document_id`` values (which are
    string paper codes) can be looked up directly without any index mapping.

    Args:
        dataset_config: Dataset configuration supplying metadata_file, paper_filter,
            and paper_subset.
        ocr_directory: Directory containing ``.txt`` OCR files.

    Returns:
        Dict mapping paper_code strings to document text.
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

    documents: dict[str, str] = {}
    for fname in text_files:
        paper_code = fname.replace(".txt", "")
        with open(os.path.join(ocr_directory, fname), "r", encoding="utf-8") as fh:
            documents[paper_code] = fh.read()
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
