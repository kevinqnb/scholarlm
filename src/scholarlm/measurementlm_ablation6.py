"""MeasurementLM Ablation 6: Full-Document Pair Provenance with Direct List Response

Ablation goal: understand what happens when the model is asked to directly return
a list of (page, table) provenance locations for each (entity, attribute) pair
using the full document context, rather than iterating page-by-page and asking a
binary has_data question for each one.

Changes from the baseline MeasurementLM:

1. Both `_entity_provenance()` and `_attribute_provenance()` are replaced by a
   single `_pair_provenance_full_context()` method that operates over all
   (entity, attribute) combinations per document. For each pair it gives the model
   the full document context and asks it to return a JSON list of provenance items,
   each with 'explanation', 'page_number', and 'table_number' fields. Page numbers
   are identified via <page number="x"> tags and table numbers via <table number="x">
   tags in the document text.

2. Two new Pydantic response models are defined in this module:
     - ProvenanceItem  : a single provenance location (page_number, table_number)
     - ProvenanceListResponse : the list wrapper returned by the model

3. `fit()` is updated to call `_pair_provenance_full_context()` instead of the two
   separate provenance methods, then adapts its output into the (entity_prov,
   attr_prov) format expected by the unchanged extraction methods via the module-level
   helper `_adapt_pair_prov()`. Event resolution is run after provenance adaptation
   and passed through to value extraction.

Unchanged from baseline: `_extract_entities()`, `_detect_attributes()`,
`_extract_values_from_text()`, `_extract_values_from_tables()`,
`_standardize()`, `_deduplicate()`, `save()`.
"""

from pydantic import BaseModel
from .measurementlm import MeasurementLM, response_validator
from .instruction_prompts import FULL_CONTEXT_PROVENANCE_INSTRUCTIONS


# -----------------------------------------------------------------------
# Response models for full-document provenance
# -----------------------------------------------------------------------

class ProvenanceItem(BaseModel):
    """A single provenance location for an (entity, attribute) pair."""
    explanation: str
    page_number: int
    table_number: int | None = None


class ProvenanceListResponse(BaseModel):
    """All provenance locations for an (entity, attribute) pair in one document."""
    items: list[ProvenanceItem]


# -----------------------------------------------------------------------
# Shared adaptation helper (also used by the experiment script)
# -----------------------------------------------------------------------

def _adapt_pair_prov(pair_prov, entity_data, doc_attributes):
    """
    Adapts pair_prov (keyed by (doc_id, entity_id, attr_name)) into the
    (extended_entity_data, entity_prov, attr_prov) format expected by the
    baseline extraction methods.

    Creates a unique pair_id = f"{entity_id}|{attr_name}" for each
    (entity, attribute) combination so that entity_prov holds a distinct
    page-set per pair. Because entity_prov[(doc_id, pair_id)] is a subset
    of attr_prov[(doc_id, attr_name)] by construction, the intersection
    inside the extraction methods reduces exactly to the pair's own pages.

    Args:
        pair_prov : dict mapping (doc_id, entity_id, attr_name)
                    -> list[{"page": int, "table": int|None}]
        entity_data   : list of entity records (with 'document_id', 'entity_id')
        doc_attributes: {doc_id: {attr_name: terms}} mapping

    Returns:
        extended_entity_data : entity records with entity_id replaced by pair_id
        entity_prov          : dict (doc_id, pair_id) -> list[{page, table}]
        attr_prov            : dict (doc_id, attr_name) -> list[{page, table}]
    """
    unique_entities = {}
    for record in entity_data:
        key = (record["document_id"], record["entity_id"])
        if key not in unique_entities:
            unique_entities[key] = record

    extended_entity_data = []
    entity_prov = {}
    attr_prov = {}

    for (doc_id, entity_id), record in unique_entities.items():
        for attr_name in doc_attributes.get(doc_id, {}):
            pair_id = f"{entity_id}|{attr_name}"
            entries = pair_prov.get((doc_id, entity_id, attr_name), [])

            extended_entity_data.append(record | {"entity_id": pair_id})
            entity_prov[(doc_id, pair_id)] = entries

            attr_key = (doc_id, attr_name)
            for entry in entries:
                attr_prov.setdefault(attr_key, []).append(entry)

    return extended_entity_data, entity_prov, attr_prov


class MeasurementLMAblation6(MeasurementLM):
    """
    Ablation 6: both provenance steps are replaced by a single full-document
    query per (entity, attribute) pair that returns a list of provenance locations.
    """

    # -----------------------------------------------------------------------
    # Steps 3 + 4 combined: full-document (entity, attribute) pair provenance
    # -----------------------------------------------------------------------

    def _pair_provenance_full_context(self, entity_data, doc_attributes):
        """
        For each (entity, attribute) pair in each document, query the full
        document context to directly obtain a list of provenance locations.

        CHANGED: replaces both _entity_provenance() and _attribute_provenance()
        from the baseline. Instead of asking a per-page binary question, the
        model receives the entire document and returns a list of
        (page_number, table_number) items identifying where the (entity,
        attribute) pair has data. Page numbers are read from <page number="x">
        tags; table numbers from <table number="x"> tags (null if prose).

        Args:
            entity_data   : list of entity records from _extract_entities()
            doc_attributes: {doc_id: {attr_name: terms}} from _detect_attributes()

        Returns:
            dict mapping (doc_id, entity_id, attr_name)
                -> list[{"page": int, "table": int|None}]
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        unique_entities = {}
        for record in entity_data:
            key = (record["document_id"], record["entity_id"])
            if key not in unique_entities:
                unique_entities[key] = record

        messages = []
        message_ids = []  # (doc_id, entity_id, attr_name)

        for (doc_id, entity_id), record in unique_entities.items():
            context = record["context"]
            entity_description = {k: v for k, v in record.items() if k in entity_fields}

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                attr_description = self.attribute_info_dict[attr_name].get("description", "")

                # CHANGED: single full-document query per (entity, attribute) pair
                query = (
                    f"Entity description: {entity_description}\n"
                    f"Attribute: {attr_name}\n"
                    f"Attribute description: {attr_description}\n"
                    f"Terminology used for the attribute: {terms}\n\n"
                    f"List all locations in this document where a direct numerical "
                    f"measurement for this entity and attribute appears. "
                    f"For each location, provide the page number (from the nearest "
                    f"preceding <page number=\"x\"> tag) and, if the data is in a "
                    f"table, the table number (from the enclosing <table number=\"x\"> "
                    f"tag). Set table_number to null if the data is in prose text.\n\n"
                )
                # CHANGED: uses FULL_CONTEXT_PROVENANCE_INSTRUCTIONS and full context
                prompt = (
                    f"## INSTRUCTIONS:\n{FULL_CONTEXT_PROVENANCE_INSTRUCTIONS}\n\n"
                    f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
                )
                messages.append([{"role": "user", "content": prompt}])
                message_ids.append((doc_id, entity_id, attr_name))

        if not messages:
            return {}

        # CHANGED: response schema is ProvenanceListResponse (list of items)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "provenance_list_response",
                "schema": ProvenanceListResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(ProvenanceListResponse, r),
        )

        pair_prov = {}
        for msg_idx, resp in enumerate(response_texts):
            doc_id, entity_id, attr_name = message_ids[msg_idx]
            try:
                result = response_validator(ProvenanceListResponse, resp)
            except Exception as e:
                print(f"Validation error in full-context provenance response: {e}")
                print(f"Response text: {resp[:500]}")
                continue

            for item in result["items"]:
                key = (doc_id, entity_id, attr_name)
                pair_prov.setdefault(key, []).append({
                    "page": item["page_number"],
                    "table": item.get("table_number"),
                })

        return pair_prov

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def fit(
        self,
        documents: list[str],
        processed_pdf_dirs: list[str] | None = None,
    ) -> list[dict]:
        """
        Runs the ablation 6 pipeline on the provided documents.

        CHANGED: Steps 3 and 4 (entity provenance, attribute provenance) are
        replaced by a single _pair_provenance_full_context() call. Its output
        is adapted via _adapt_pair_prov() before being passed to the unchanged
        extraction methods. Event resolution is run after provenance adaptation
        and passed to value extraction steps.
        """
        if self.clean_tables:
            if processed_pdf_dirs is None:
                raise ValueError(
                    "processed_pdf_dirs is required when clean_tables=True. "
                    "Run 'python experiments/process_pdfs.py' first."
                )
            documents = self._clean_tables(documents, processed_pdf_dirs)

        self.data = []
        for i, doc in enumerate(documents):
            self.data.append({"document_id": i, "context": doc})
        doc_data = list(self.data)

        # Step 1: Entity extraction (unchanged)
        entity_data = self._extract_entities()

        # Step 2: Document-level attribute detection (unchanged)
        self.data = doc_data
        doc_attributes = self._detect_attributes()

        # Steps 3 + 4 (CHANGED): full-document pair provenance
        pair_prov = self._pair_provenance_full_context(entity_data, doc_attributes)

        # Adapt pair_prov into the (entity_prov, attr_prov) interface that the
        # unchanged extraction methods expect.
        extended_entity_data, entity_prov, attr_prov = _adapt_pair_prov(
            pair_prov, entity_data, doc_attributes
        )

        # Step 4.5: Event resolution (optional, unchanged)
        if self.measurement_event_schema is not None:
            event_resolution = self._resolve_events(
                extended_entity_data, doc_attributes, entity_prov, attr_prov
            )
        else:
            event_resolution = None

        # Step 5: Extract values from text (unchanged)
        text_values = self._extract_values_from_text(
            extended_entity_data, doc_attributes, entity_prov, attr_prov, event_resolution
        )

        # Step 6: Extract values from tables (unchanged)
        table_values = self._extract_values_from_tables(
            extended_entity_data, doc_attributes, entity_prov, attr_prov, event_resolution
        )

        self.data = text_values + table_values

        # Step 7: Standardize (unchanged)
        self.data = self._standardize()

        # Step 8: Deduplicate (unchanged)
        self.data = self._deduplicate(self.data)

        return self.data
