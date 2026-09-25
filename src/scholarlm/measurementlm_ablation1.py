"""MeasurementLM Ablation 1: Direct Extraction (No Pipeline Structure)

Ablation goal: understand what happens when we remove the extraction pipeline
entirely and simply ask the model to extract a list of measurement records
directly from each document in a single pass.

Changes from the baseline MeasurementLM:

1. All intermediate pipeline steps (attribute detection, entity/attribute provenance,
   per-page and per-table value extraction, event resolution) are eliminated. A single
   LLM call per document extracts all records at once.

2. The extraction schema and prompt are defined at the dataset level
   (direct_extraction_schema and direct_extraction_prompt on DatasetConfig). The schema
   is a flat Pydantic model combining entity fields, measurement event fields, and
   'attribute', 'value', 'units', and the qualifier/shape fields ('qualifiers',
   'point_value', 'lower', 'upper', 'list_values', 'tolerance',
   'standard_deviation' -- the same shape MeasurementLM._parse_quantities()
   produces via a separate step; here it's asked for directly in the one
   extraction call). The prompt describes all of them in one block.

3. fit() returns straight from _extract_triples() -- the standard _standardize()
   and _parse_quantities()/_deduplicate() steps are not called; this ablation's
   direct_extraction_schema asks the model for the qualifier/shape fields
   directly instead (see point 2), by design (unlike every other ablation,
   which does call _parse_quantities() -- see MeasurementLM.fit()). No
   provenance fields (page_number, table_number, etc.) are produced.

Unchanged from baseline: save().

direct_extraction_instructions defaults to DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS
but can be overridden -- run_ablation.py passes
DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS along with a dataset's
*_no_qualifiers schema/prompt when params.include_qualifiers is explicitly
false, to test whether a later parsing step (analysis/postprocessing.py) can
recover the qualifier/shape fields as well as asking the model for them
directly (see point 2/3 above).
"""

from functools import partial
from pydantic import create_model
from .measurementlm import MeasurementLM, response_validator
from .instruction_prompts import DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS


class MeasurementLMAblation1(MeasurementLM):
    """
    Ablation 1: the multi-step extraction pipeline is replaced by a single
    LLM call per document that extracts all measurement records directly.

    Requires direct_extraction_schema and direct_extraction_prompt, which are
    passed explicitly to this class rather than to the base MeasurementLM.
    """

    def __init__(
        self,
        *args,
        max_concurrent: int = 1,
        direct_extraction_schema=None,
        direct_extraction_prompt=None,
        direct_extraction_instructions: str = DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS,
        **kwargs,
    ):
        super().__init__(*args, max_concurrent=max_concurrent, **kwargs)
        self.direct_extraction_schema = direct_extraction_schema
        self.direct_extraction_prompt = direct_extraction_prompt
        self.direct_extraction_instructions = direct_extraction_instructions

    # -----------------------------------------------------------------------
    # Single extraction step: extract all records directly
    # -----------------------------------------------------------------------

    def _extract_triples(self):
        """
        Extract all measurement records from each document in a single LLM call.

        Uses self.direct_extraction_schema (a flat Pydantic model combining entity,
        event, attribute, value, and units fields) and self.direct_extraction_prompt
        (a dataset-specific block describing entities, events, and attributes).

        Returns a list of records suitable for _standardize() and _deduplicate().
        """
        if self.direct_extraction_schema is None or self.direct_extraction_prompt is None:
            raise ValueError(
                "direct_extraction_schema and direct_extraction_prompt must be set "
                "for MeasurementLMAblation1. Define them in the dataset config."
            )

        DirectExtractionList = create_model(
            "DirectExtractionList",
            items=(list[self.direct_extraction_schema], ...),
        )
        direct_extraction_list_json = DirectExtractionList.model_json_schema()

        messages = []
        for datapoint in self.data:
            context = datapoint['context']
            query = "Extract all measurement records from this document as described in the instructions."
            prompt = (
                f"## INSTRUCTIONS:\n{self.direct_extraction_instructions}\n\n"
                f"## DATASET SPECIFIC INSTRUCTIONS:\n{self.direct_extraction_prompt}\n\n"
                f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "direct_extraction_list",
                "schema": direct_extraction_list_json,
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_tokens=32768,
            max_retries=4,
            max_concurrent=1,
            validator=partial(response_validator, DirectExtractionList),
            timeout=600.0,
        )

        triple_data = []
        for i, r in enumerate(response_texts):
            try:
                resp_validated = response_validator(DirectExtractionList, r)
            except Exception as e:
                print(f"Validation error in direct extraction response: {e}")
                print(f"Response text: {r}")
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
    # Full pipeline (simplified: single extraction step)
    # -----------------------------------------------------------------------

    def fit(
        self,
        documents: list[str],
    ) -> list[dict]:
        """
        Runs the ablation 1 pipeline on the provided documents.

        Replaces the multi-step baseline pipeline with a single _extract_triples()
        call. Standardize and deduplicate steps are intentionally skipped.
        """
        self.data = []
        for i, doc in enumerate(documents):
            self.data.append({'document_id': i, 'context': doc})

        # Step 1: Extract all measurement records directly
        self.data = self._extract_triples()

        return self.data
