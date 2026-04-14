"""MeasurementLM Ablation 1: Combined Entity-Attribute Extraction

Ablation goal: understand what happens when entity identification and attribute
detection are combined into a single extraction step rather than run independently.

Changes from the baseline MeasurementLM:

1. The two-step baseline (entity extraction → attribute detection) is replaced by
   a single step that extracts (entity, attribute) pairs directly. Only pairs for
   which the model believes a direct numerical measurement exists are emitted.
   This reuses `_extract_entities()` from the parent; the difference is that the
   caller-supplied `entity_identification_schema` must include two reserved fields:
     - attribute (str)             : exact attribute name from attribute_info_dict
     - attribute_terms (list[str]) : terminology used in the document

2. Separate entity provenance and attribute provenance are replaced by a single
   `_entity_attribute_provenance()` that checks per page whether BOTH the entity
   AND the attribute have a measurement together, using the new
   ENTITY_ATTRIBUTE_PROVENANCE_INSTRUCTIONS prompt.

3. `fit()` is reduced from 8 steps to 6 steps by eliminating the standalone
   attribute detection and merging the two provenance steps into one.  Before
   calling the (unchanged) extraction methods, `fit()` adapts the pair provenance
   into the `(entity_prov, attr_prov, doc_attributes)` interface those methods
   expect.

Unchanged from baseline: `_extract_values_from_text()`,
`_extract_values_from_tables()`, `_standardize()`, `_deduplicate()`, `save()`.
"""

from .measurementlm import (
    MeasurementLM,
    ProvenanceResponse,
    response_validator,
)
from .instruction_prompts import (
    ENTITY_ATTRIBUTE_PROVENANCE_INSTRUCTIONS,  # NEW: combined provenance prompt
)


# Fields that carry attribute identity inside the combined schema.
# The entity_identification_schema in ablation 1 must define these fields.
_ATTRIBUTE_FIELDS = frozenset({"attribute", "attribute_terms"})


class MeasurementLMAblation1(MeasurementLM):
    """
    Ablation 1: entity and attribute detection combined into one extraction step.

    The entity_identification_schema passed to __init__ must include:
      - attribute (str)           : one of the keys in attribute_info_dict
      - attribute_terms (list[str]): terminology used in the document
    in addition to the usual entity-identifying fields.
    """

    # -----------------------------------------------------------------------
    # Step 1 + 3 combined: Extract (entity, attribute) pairs
    # -----------------------------------------------------------------------

    def _extract_entity_attribute_pairs(self):
        """
        Extract (entity, attribute) pairs in a single LLM pass.

        CHANGED: replaces the separate _extract_entities() + _detect_attributes()
        steps of the baseline.  The parent's _extract_entities() implementation
        is reused here unchanged — the only difference is that the caller supplies
        an entity_identification_schema that also contains 'attribute' and
        'attribute_terms' fields, and a prompt that instructs the model to emit
        one item per (entity, attribute) pair rather than one item per entity.
        """
        # Delegate to the parent's extraction logic; the schema and prompt
        # supplied by the caller already encode the combined structure.
        return self._extract_entities()

    # -----------------------------------------------------------------------
    # Step 2 + 4 combined: (Entity, attribute) pair provenance
    # -----------------------------------------------------------------------

    def _entity_attribute_provenance(self, pair_data):
        """
        For each unique (document, entity-attribute pair), determine which pages
        contain data for that pair.

        CHANGED: replaces both _entity_provenance() and _attribute_provenance()
        from the baseline.  Uses ENTITY_ATTRIBUTE_PROVENANCE_INSTRUCTIONS, which
        requires evidence for BOTH the entity and the attribute on the same page.

        Args:
            pair_data: list of pair records from _extract_entity_attribute_pairs()

        Returns:
            dict mapping (doc_id, entity_id) -> list[{"page": int, "table": int|None}]
        """
        # Determine which schema fields are entity-describing (not attribute fields)
        entity_fields = [
            f for f in self.entity_identification_schema.model_fields.keys()
            if f not in _ATTRIBUTE_FIELDS
        ]

        # Collect unique (doc_id, entity_id) pairs
        unique_pairs = {}
        for record in pair_data:
            key = (record["document_id"], record["entity_id"])
            if key not in unique_pairs:
                unique_pairs[key] = record

        messages = []
        message_ids = []  # (doc_id, entity_id, page_number)

        for (doc_id, entity_id), record in unique_pairs.items():
            context = record["context"]
            entity_description = {k: v for k, v in record.items() if k in entity_fields}
            # CHANGED: pull attribute info from the pair record directly
            attr_name = record["attribute"]

            try:
                attr_description = self.attribute_info_dict[attr_name].get("description", "")
            except KeyError:
                print(f"Warning: attribute '{attr_name}' not found in attribute_info_dict. Skipping this entity.")
                continue

            attr_terms = record.get("attribute_terms", [])
            pages = self._get_page_numbers(context)

            for p in pages:
                page_text = self._get_page_text(context, p)
                if not page_text:
                    continue

                # CHANGED: query asks about the (entity, attribute) pair together
                query = (
                    f"Entity description: {entity_description}\n"
                    f"Attribute: {attr_name}\n"
                    f"Attribute description: {attr_description}\n"
                    f"Terminology used for the attribute: {attr_terms}\n\n"
                    f"Does this page contain directly reported numerical measurements "
                    f"for the described attribute AND entity? If yes, indicate whether "
                    f"the data appears in a table or in prose text.\n\n"
                )
                # CHANGED: uses ENTITY_ATTRIBUTE_PROVENANCE_INSTRUCTIONS
                prompt = (
                    f"## INSTRUCTIONS:\n{ENTITY_ATTRIBUTE_PROVENANCE_INSTRUCTIONS}\n\n"
                    f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                )
                messages.append([{"role": "user", "content": prompt}])
                message_ids.append((doc_id, entity_id, p))

        if not messages:
            return {}

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "provenance_response",
                "schema": ProvenanceResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=1,
            validator=lambda r: response_validator(ProvenanceResponse, r),
        )

        provenance = {}
        for msg_idx, resp in enumerate(response_texts):
            doc_id, entity_id, page_number = message_ids[msg_idx]
            try:
                result = response_validator(ProvenanceResponse, resp)
            except Exception:
                print("Validation error in entity-attribute provenance response.")
                continue

            if result.get("has_data"):
                key = (doc_id, entity_id)
                if result.get("in_table"):
                    page_text = self._get_page_text(
                        unique_pairs[(doc_id, entity_id)]["context"],
                        page_number,
                    )
                    for t in self._get_table_numbers_on_page(page_text):
                        provenance.setdefault(key, []).append(
                            {"page": page_number, "table": t}
                        )
                else:
                    provenance.setdefault(key, []).append(
                        {"page": page_number, "table": None}
                    )

        return provenance

    # -----------------------------------------------------------------------
    # Full pipeline (simplified: 6 steps instead of 8)
    # -----------------------------------------------------------------------

    def fit(
        self,
        documents: list[str],
        processed_pdf_dirs: list[str] | None = None,
    ) -> list[dict]:
        """
        Runs the ablation 1 pipeline on the provided documents.

        CHANGED: 6-step pipeline instead of 8 steps.
          Step 1: Extract (entity, attribute) pairs [was: steps 1 + 3]
          Step 2: Combined pair provenance          [was: steps 2 + 4]
          Step 3: Extract values from text          [unchanged]
          Step 4: Extract values from tables        [unchanged]
          Step 5: Standardize
          Step 6: Deduplicate

        After pair provenance, pair_prov is adapted into the (entity_prov,
        attr_prov, doc_attributes) format expected by the unchanged extraction
        methods before calling them.
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

        # Step 1: Extract (entity, attribute) pairs
        pair_data = self._extract_entity_attribute_pairs()

        # Step 2: Combined (entity, attribute) pair provenance
        pair_prov = self._entity_attribute_provenance(pair_data)

        # Adapt pair_prov to the (entity_prov, attr_prov, doc_attributes) interface
        # expected by the unchanged _extract_values_from_text / _extract_values_from_tables.
        #
        # entity_prov: reuse pair_prov as-is — it is already keyed by (doc_id, entity_id).
        #
        # attr_prov: for each (doc_id, attr_name), union the pair_prov page entries
        # across all entity-attribute pairs in that doc that share the same attr_name.
        # This ensures the (entity_prov ∩ attr_prov) intersection inside the extraction
        # methods reduces to just the entity's own pair_prov pages.
        #
        # doc_attributes: {doc_id: {attr_name: terms}} derived from pair_data,
        # replacing the separate _detect_attributes() result from the baseline.
        entity_prov = pair_prov
        attr_prov = {}
        doc_attributes = {}
        for record in pair_data:
            doc_id = record["document_id"]
            entity_id = record["entity_id"]
            attr_name = record["attribute"]
            terms = record.get("attribute_terms", [])

            doc_attributes.setdefault(doc_id, {})[attr_name] = terms

            attr_key = (doc_id, attr_name)
            for entry in pair_prov.get((doc_id, entity_id), []):
                attr_prov.setdefault(attr_key, []).append(entry)

        # Step 3: Extract values from text (identical to baseline)
        text_values = self._extract_values_from_text(
            pair_data, doc_attributes, entity_prov, attr_prov
        )

        # Step 4: Extract values from tables (identical to baseline)
        table_values = self._extract_values_from_tables(
            pair_data, doc_attributes, entity_prov, attr_prov
        )

        # Combine
        self.data = text_values + table_values

        # Step 5: Standardize
        self.data = self._standardize()

        # Step 6: Deduplicate
        self.data = self._deduplicate(self.data)

        return self.data
