"""Shared prompt builders and document-loading utilities for judge runners.

All judge runner scripts (run_judge_interp.py, run_judge_local.py,
validation.py) import from here so prompts are identical across backends.

Public API:

    ``build_judge_query``         — builds the ## QUERY content (entity +
                                    attribute + value sections) without the
                                    document prefix.
    ``prepare_chat_entries``      — converts raw extraction records to
                                    provider-agnostic chat entries sorted by
                                    document_id for cache locality.  Each entry
                                    includes ``user_query`` and ``document_id``
                                    for use by the interpretability judge runner.
    ``load_documents_for_dataset`` — loads OCR text files applying the same
                                    paper_filter and paper_subset as extraction.
    ``extract_page_text``         — extracts specific page blocks from an OCR
                                    document string.
"""
from __future__ import annotations

import json
import re
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


def extract_page_text(document: str, page_numbers: list[int]) -> str:
    """Extract specific pages from an OCR document.

    OCR documents use ``<page number="N">...</page>`` blocks.  Returns the
    blocks for the requested page numbers concatenated with a blank line,
    preserving the original tags so the judge can orient itself.

    Falls back to the full document if no ``<page number>`` tags are found or
    none of the requested pages exist.

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
    full prompt.  All judge runners call this function so the query content is
    identical across backends.

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
        user          – full user prompt string (## CONTEXT full paper + ## QUERY)
        user_query    – the ## QUERY content only (for JudgementLM interp runner)
        user_document – ## CONTEXT prefix only (for prompt caching)
        context_text  – the ## CONTEXT body: the full OCR document text, or the
                        row's ``context_override`` verbatim when it carries one
                        (consumed directly by the NNsight-based runners:
                        interp judge, jacobian-lens, attribution)

    The judge always sees the full paper: ``## CONTEXT`` is the entire OCR
    document, never a page slice. Provenance fields (``page_number`` etc.) on
    the extraction record are not consulted here — narrowing the judge's
    context to the extracted page(s) was retired in favour of always judging
    against the full paper.

    Context override: when a row carries a non-null ``context_override`` field,
    its ``## CONTEXT`` is that text verbatim instead of the full OCR document
    (the synthetic-probe augmentation pipeline sets this on rows whose context
    was edited alongside the measurement — see ``probe_augment.py``). A row
    with no such field, or ``context_override: null``, gets the full OCR
    document. Because the override travels *on* the row, it cannot be "for" the
    wrong row or the wrong file by construction — it either is that row's own
    field or it doesn't exist.

    Args:
        data: List of extraction records from a ``final.json`` file.
        documents: Dict mapping paper_code strings to raw OCR text, as returned
            by ``load_documents_for_dataset``.
        dataset_config: ``DatasetConfig`` instance supplying entity schema,
            attribute catalogue, entity type description, and optional
            measurement event schema.

    Returns:
        List of chat entry dicts ready for any judge runner.
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
        try:
            attribute_description = _attr_dict[attribute]["description"]
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

        override_text = entry.get("context_override")
        if override_text is not None:
            # Context was edited by the augmentation pipeline; use it verbatim.
            context_text = override_text
        else:
            # The judge always sees the whole paper, never a page slice.
            context_text = document

        system = dataset_config.judge_instructions or JUDGE_INSTRUCTIONS

        query = build_judge_query(
            attribute_description=attribute_description,
            attribute_terms=attribute_terms,
            entity_type_description=_entity_type_desc,
            entity_description=entity_description,
            measurement_val=measurement_val,
            units=units,
            event_description=event_description,
        )

        user = f"## CONTEXT:\n{context_text}\n\n## QUERY:\n{query}"
        user_document = f"## CONTEXT:\n{context_text}\n\n"

        if _DEBUG_PAPERS_LIMIT is None or papers_printed < _DEBUG_PAPERS_LIMIT:
            # The context is a whole paper now; print head+tail, not the lot.
            if len(user) > 2400:
                preview = f"{user[:1200]}\n...[{len(user) - 2400} chars elided]...\n{user[-1200:]}"
            else:
                preview = user
            print(f"DEBUG: User message for document_id={document_id}, orig_idx={orig_idx}:\n{preview}\n")
            papers_printed += 1

        entries.append({
            "custom_id": str(orig_idx),
            "document_id": document_id,
            "system": system,
            "user": user,
            "user_query": query,
            "user_document": user_document,
            "context_text": context_text,
        })

    return entries


# ─── Document loading ─────────────────────────────────────────────────────────


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
        with open(f"{ocr_directory}/{fname}", "r", encoding="utf-8") as fh:
            documents[paper_code] = fh.read()
    return documents
