"""MeasurementLM Ablation 6: Direct Triple Extraction (No Pipeline Structure)

Ablation goal: understand what happens when we remove the extraction pipeline
entirely and simply ask the model to extract a list of (entity, attribute, value)
triples directly from each document in a single pass.

Changes from the baseline MeasurementLM:

1. All intermediate pipeline steps (attribute detection, entity/attribute provenance,
   per-page and per-table value extraction) are eliminated. A single LLM call per
   document extracts all (entity, attribute, value, units) triples at once.

2. The extraction schema is built dynamically by extending entity_identification_schema
   with three additional fields: 'attribute' (str), 'value' (str | None), and
   'units' (str | None). The model is asked to return a list of these extended items.

3. The query combines self.entity_identification_prompt with the full attribute list
   from attribute_info_dict, instructing the model to identify entities and
   simultaneously extract measured values for each attribute.

4. After extraction, the standard _standardize() and _deduplicate() steps are applied
   unchanged. No provenance fields (page_number, table_number, etc.) are produced.

Unchanged from baseline: _standardize(), _deduplicate(), save().
"""

from pydantic import create_model
from .measurementlm import MeasurementLM, response_validator
from .instruction_prompts import DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS


class MeasurementLMAblation6(MeasurementLM):
    """
    Ablation 6: the multi-step extraction pipeline is replaced by a single
    LLM call per document that extracts all (entity, attribute, value) triples
    directly.
    """

    # -----------------------------------------------------------------------
    # Single extraction step: extract all triples directly
    # -----------------------------------------------------------------------

    def _extract_triples(self):
        """
        Extract all (entity, attribute, value, units) triples from each document
        in a single LLM call per document.

        CHANGED: replaces all of steps 1-6 of the baseline pipeline. The model
        receives the full document along with entity identification instructions
        and the complete attribute list, and returns a list of items each
        containing entity fields plus 'attribute', 'value', and 'units'.

        Returns a list of records suitable for _standardize() and _deduplicate().
        """
        # Dynamically extend entity_identification_schema with attribute/value/units
        ExtendedTripleSchema = create_model(
            "EntityAttributeTriple",
            __base__=self.entity_identification_schema,
            attribute=(str, ...),
            value=(str | None, None),
            units=(str | None, None),
        )
        TripleList = create_model(
            "TripleList",
            items=(list[ExtendedTripleSchema], ...),
        )
        triple_list_json = TripleList.model_json_schema()

        attribute_list_text = self._format_attribute_list()

        messages = []
        for datapoint in self.data:
            context = datapoint['context']
            query = (
                f"Entity identification instructions:\n{self.entity_identification_prompt}\n\n"
                f"Attributes to extract:\n{attribute_list_text}\n\n"
                f"Extract all (entity, attribute, value) triples for the entities and "
                f"attributes described above. Return one item per (entity, attribute) pair "
                f"where a direct numerical measurement exists in the document."
            )
            prompt = (
                f"## Instructions:\n{DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS}\n\n"
                f"## Context:\n{context}\n\n## Query:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "triple_list",
                "schema": triple_list_json,
            },
        }
        response_texts = self._call_batch(messages, response_format=response_format)

        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        triple_data = []
        for i, r in enumerate(response_texts):
            try:
                resp_validated = response_validator(TripleList, r)
            except Exception as e:
                print(f"Validation error in triple extraction response: {e}")
                print(f"Response text: {r[:500]}")
                resp_validated = {'items': []}

            for j, item in enumerate(resp_validated['items']):
                if item.get('value') is None:
                    continue
                entity_id = f"doc_{i}_entity_{j}"
                triple_data.append(
                    self.data[i] | item | {
                        'entity_id': entity_id,
                        'attribute_terms': [],
                    }
                )

        return triple_data

    # -----------------------------------------------------------------------
    # Full pipeline (simplified: 3 steps instead of 8)
    # -----------------------------------------------------------------------

    def fit(
        self,
        documents: list[str],
        processed_pdf_dirs: list[str] | None = None,
    ) -> list[dict]:
        """
        Runs the ablation 6 pipeline on the provided documents.

        CHANGED: all intermediate extraction steps replaced by a single
        _extract_triples() call.
          Step 1: Extract all (entity, attribute, value) triples [was: steps 1-6]
          Step 2: Standardize                                    [unchanged]
          Step 3: Deduplicate                                    [unchanged]
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
            self.data.append({'document_id': i, 'context': doc})

        # Step 1: Extract all (entity, attribute, value) triples directly
        self.data = self._extract_triples()

        # Step 2: Standardize
        self.data = self._standardize()

        # Step 3: Deduplicate
        self.data = self._deduplicate(self.data)

        return self.data
