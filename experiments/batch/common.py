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
from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS_UNIFIED, JUDGE_INSTRUCTIONS_UNIFIED_TABLE
from scholarlm.utils import get_filenames_in_directory

load_dotenv()


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
        page_numbers: Page numbers to extract (0-based integer, matching
            ``page_number`` values stored in ``final.json``).

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
    sources: list[str],
    attribute_description: str,
    attribute_terms: list[Any],
    entity_type_description: str,
    entity_description: dict[str, Any],
    table_numbers: list[int | None],
    measurement_val: Any,
    units: Any = None,
    row_indices: list[str] | None = None,
    column_indices: list[str] | None = None,
) -> str:
    """Build the ## QUERY content for the unified judge prompt.

    Returns the joined sections string that appears after ``## QUERY:`` in the
    full prompt.  Both ``build_user_prompt`` and the interpretability judge
    runner call this function so the query content is identical across all
    judge runners.

    Measurement event fields, when present in the dataset, are folded into
    ``entity_description`` by ``prepare_chat_entries`` before this function is
    called, so the query structure is always the same regardless of whether the
    dataset has an event schema.

    Args:
        sources: Full list of source types for this extraction (e.g.
            ``["text"]``, ``["table"]``, or ``["text", "table"]``).  All
            source types are reflected in the query's Source line.
        attribute_description: Full attribute description string.
        attribute_terms: List of terminology strings for the attribute.
        entity_type_description: One-sentence entity type description.
        entity_description: Dict of entity field values (may include event
            fields merged in by ``prepare_chat_entries``).
        table_numbers: List of table numbers (may contain ``None``).  Unique
            non-``None`` values are shown when any source is ``"table"``.
        measurement_val: The extracted scalar value.
        units: Extracted units string, or ``None`` if not reported.
        row_indices: Non-``None`` row names from table-sourced occurrences,
            pre-filtered by ``prepare_chat_entries``.  Appended to the query
            only when non-empty (requires ``JUDGE_INSTRUCTIONS_UNIFIED_TABLE``).
        column_indices: Non-``None`` column names from table-sourced
            occurrences.  Same conditions as ``row_indices``.

    Returns:
        Multi-section query string ready to follow ``## QUERY:\\n``.
    """
    units_str = units if units is not None else "not reported"

    entity_display = {k: v for k, v in entity_description.items() if v is not None}
    entity_section = (
        f"Target entity type: {entity_type_description}\n"
        f"Extracted entity: {entity_display}"
    )

    attribute_section = (
        f"Target attribute: {attribute_description}\n"
        f"Attribute terminology: {attribute_terms}"
    )

    value_section = (
        f"Extracted value: {measurement_val}\n"
        f"Extracted units: {units_str}"
    )

    # Source location — commented out to keep prompts consistent between extraction
    # and synthetic probe runs (probe records have no source/page provenance).
    # unique_src = set(sources)
    # source_parts: list[str] = []
    # if "text" in unique_src:
    #     source_parts.append("prose text")
    # if "table" in unique_src:
    #     unique_tables = sorted({tn for tn in table_numbers if tn is not None})
    #     source_parts.append(
    #         ", ".join(f"Table {tn}" for tn in unique_tables) if unique_tables else "table"
    #     )
    # source_section = "Source: " + " and ".join(source_parts) if source_parts else "Source: not reported"
    # if row_indices and column_indices:
    #     row_label = "Row names" if len(row_indices) > 1 else "Row name"
    #     col_label = "Column names" if len(column_indices) > 1 else "Column name"
    #     source_section += f"\n{row_label}: {', '.join(row_indices)}\n{col_label}: {', '.join(column_indices)}"

    closing = "Is this extraction correct? (true or false)"
    sections = [entity_section, attribute_section, value_section, closing]
    return "\n\n".join(sections)


# ─── Data preparation ─────────────────────────────────────────────────────────


def prepare_chat_entries(
    data: list[dict],
    documents: dict[str, str],
    dataset_config: DatasetConfig,
    include_row_col: bool = False,
) -> list[dict[str, Any]]:
    """Convert raw extraction data to provider-agnostic chat entries.

    Entries are sorted by document_id for cache locality. Each entry contains:
        custom_id     – str(original index in data), used to map results back
        document_id   – string paper code identifying the source document
        system        – system prompt string (JUDGE_INSTRUCTIONS_UNIFIED)
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
        include_row_col: If ``True``, append ``Row name`` / ``Column name``
            fields to the query for table-sourced extractions and switch the
            system prompt to ``JUDGE_INSTRUCTIONS_UNIFIED_TABLE``.  Defaults
            to ``False`` to preserve the standard prompt for all other runners.

    Returns:
        List of chat entry dicts ready for any provider batch module or the
        interpretability judge runner.
    """
    _all_entity_fields = dataset_config.entity_schema.model_fields.keys()
    _judge_entity_fields = (
        set(dataset_config.judge_entity_fields)
        if dataset_config.judge_entity_fields is not None
        else set(_all_entity_fields)
    )
    _attr_dict = dataset_config.attribute_info_dict
    _entity_type_desc = dataset_config.entity_type_description

    data_with_idx = list(enumerate(data))
    data_with_idx.sort(key=lambda it: str(it[1].get("document_id", "")))

    entries: list[dict[str, Any]] = []
    for _i_sorted, (orig_idx, entry) in enumerate(data_with_idx):
        document_id = str(entry["document_id"])
        document = documents.get(document_id)
        if document is None:
            print(f"Warning: document_id '{document_id}' not found in documents, skipping")
            continue
        attribute = entry.get("attribute")
        try:
            attribute_description = _attr_dict[attribute]["description"]
        except KeyError:
            print(f"Attribute '{attribute}' not found in dataset_config.attribute_info_dict")
            continue

        attribute_terms = entry.get("attribute_terms", [])
        # Fold event fields into the entity description so the query structure
        # is always identical regardless of whether the dataset has an event schema.
        # Entity fields are filtered to judge_entity_fields when defined, to avoid
        # passing noisy or irrelevant fields (e.g. coordinates) to the judge.
        entity_description = {
            k: v for k, v in entry.items()
            if k in _judge_entity_fields #or k in _event_fields
        }
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

        system = JUDGE_INSTRUCTIONS_UNIFIED_TABLE if include_row_col else JUDGE_INSTRUCTIONS_UNIFIED

        # For row/col mode: zip source_list with the index lists so that we
        # only pick up row/column names from table-sourced occurrences.
        if include_row_col:
            ri_raw = entry.get("row_index") or []
            ci_raw = entry.get("column_index") or []
            ri_list = ri_raw if isinstance(ri_raw, list) else [ri_raw]
            ci_list = ci_raw if isinstance(ci_raw, list) else [ci_raw]
            row_indices = [ri for src, ri in zip(source_list, ri_list) if src == "table" and ri is not None]
            col_indices = [ci for src, ci in zip(source_list, ci_list) if src == "table" and ci is not None]
        else:
            row_indices, col_indices = [], []

        query = build_judge_query(
            sources=source_list,
            attribute_description=attribute_description,
            attribute_terms=attribute_terms,
            entity_type_description=_entity_type_desc,
            entity_description=entity_description,
            table_numbers=table_numbers,
            measurement_val=measurement_val,
            units=units,
            row_indices=row_indices or None,
            column_indices=col_indices or None,
        )

        user = f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
        # Cached prefix for Anthropic — the page text shared across requests
        # for the same source page.  OpenAI and Gemini ignore this field.
        user_document = f"## CONTEXT:\n{page_text}\n\n"

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
