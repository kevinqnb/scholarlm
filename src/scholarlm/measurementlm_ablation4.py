"""MeasurementLM Ablation 4: No Standardization

Ablation goal: understand the contribution of the standardization step by
removing it from the pipeline and recording extracted values directly.

Changes from the baseline MeasurementLM:

1. `fit()`: the `_standardize()` call is removed. Extracted values pass directly
   from the combination of text and table results into deduplication without
   any LLM-based value cleanup.

Unchanged from baseline: all extraction steps, provenance steps,
`_standardize()` (the method still exists but is not called), and
`_deduplicate()`.
"""

from .measurementlm import MeasurementLM


class MeasurementLMAblation4(MeasurementLM):
    """
    Ablation 4: standardization step removed from the pipeline.
    Extracted values are passed directly to deduplication.
    """

    def fit(self, documents: list[str]):
        """
        Runs the ablation 4 pipeline on the provided documents.

        CHANGED: `_standardize()` is not called; values extracted by the model
        are recorded as-is into the output dataset.
        """
        self.data = []
        for i, doc in enumerate(documents):
            self.data.append({"document_id": i, "context": doc})
        doc_data = list(self.data)

        # Step 1: Entity extraction
        entity_data = self._extract_entities()

        # Step 2: Entity provenance
        entity_prov = self._entity_provenance(entity_data)

        # Step 3: Document-level attribute detection
        self.data = doc_data
        doc_attributes = self._detect_attributes()

        # Step 4: Attribute provenance
        attr_prov = self._attribute_provenance(doc_attributes)

        # Step 5: Extract values from text
        text_values = self._extract_values_from_text(
            entity_data, doc_attributes, entity_prov, attr_prov
        )

        # Step 6: Extract values from tables
        table_values = self._extract_values_from_tables(
            entity_data, doc_attributes, entity_prov, attr_prov
        )

        # Combine text and table extractions
        self.data = text_values + table_values

        # CHANGED: _standardize() is intentionally omitted here

        # Step 7: Deduplicate
        self.data = self._deduplicate(self.data)

        return self.data
