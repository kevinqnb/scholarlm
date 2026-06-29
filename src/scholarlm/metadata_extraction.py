"""MetadataLM: Single-pass document metadata extraction.

Extracts a fixed set of bibliographic or administrative metadata fields from
each document in one LLM call.  Unlike MeasurementLMAblation1, which extracts
a *list* of (entity, attribute, value) records, this class extracts exactly one
JSON object per document whose shape is defined entirely by the dataset config.

The response format is the metadata schema itself (not wrapped in a list), so
the model is forced to return a single object.  A warning is printed for any
document where all returned fields are null.
"""
from __future__ import annotations

import time
from functools import partial

from .measurementlm import MeasurementLM, response_validator
from .instruction_prompts import METADATA_EXTRACTION_INSTRUCTIONS


class MetadataLM(MeasurementLM):
    """
    Single-pass metadata extraction: one LLM call per document, one record out.

    Requires metadata_extraction_schema and metadata_extraction_prompt to be set,
    typically via DatasetConfig.
    """

    def __init__(
        self,
        *args,
        max_concurrent: int = 4,
        extract_max_tokens: int = 4096,
        max_input_chars: int = 150_000,
        metadata_extraction_schema=None,
        metadata_extraction_prompt=None,
        **kwargs,
    ):
        super().__init__(*args, max_concurrent=max_concurrent, **kwargs)
        self.metadata_extraction_schema = metadata_extraction_schema
        self.metadata_extraction_prompt = metadata_extraction_prompt
        self.extract_max_tokens = extract_max_tokens
        self.max_input_chars = max_input_chars

    # -----------------------------------------------------------------------
    # Core extraction step
    # -----------------------------------------------------------------------

    def _extract_metadata(self) -> list[dict]:
        if self.metadata_extraction_schema is None or self.metadata_extraction_prompt is None:
            raise ValueError(
                "metadata_extraction_schema and metadata_extraction_prompt must be set "
                "for MetadataLM. Define them in the dataset config."
            )

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "metadata_extraction",
                "schema": self.metadata_extraction_schema.model_json_schema(),
            },
        }

        messages = []
        n_truncated = 0
        for datapoint in self.data:
            context = datapoint["context"]
            if len(context) > self.max_input_chars:
                context = context[: self.max_input_chars]
                n_truncated += 1
            prompt = (
                f"## INSTRUCTIONS:\n{METADATA_EXTRACTION_INSTRUCTIONS}\n\n"
                f"## DATASET SPECIFIC INSTRUCTIONS:\n{self.metadata_extraction_prompt}\n\n"
                f"## CONTEXT:\n{context}\n\n"
                f"## QUERY:\nExtract the metadata fields from this document."
            )
            messages.append([{"role": "user", "content": prompt}])
        if n_truncated:
            print(
                f"  Note: {n_truncated}/{len(self.data)} documents truncated to "
                f"{self.max_input_chars:,} chars (~{self.max_input_chars // 4:,} tokens)."
            )

        print(
            f"Extracting metadata from {len(messages)} documents "
            f"(max_concurrent={self.max_concurrent}, max_tokens={self.extract_max_tokens})..."
        )
        t0 = time.perf_counter()
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_tokens=self.extract_max_tokens,
            max_retries=4,
            max_concurrent=self.max_concurrent,
            validator=partial(response_validator, self.metadata_extraction_schema),
            timeout=600.0,
        )
        elapsed = time.perf_counter() - t0
        print(f"  Total: {elapsed:.1f}s  avg: {elapsed / max(len(messages), 1):.1f}s/doc\n")

        null_fields = set(self.metadata_extraction_schema.model_fields)
        results = []
        for i, r in enumerate(response_texts):
            try:
                record = response_validator(self.metadata_extraction_schema, r)
            except Exception as e:
                doc_id = self.data[i].get("document_id", i)
                print(f"Warning: validation error for document {doc_id}: {e}")
                record = {field: None for field in self.metadata_extraction_schema.model_fields}

            if all(record.get(f) is None for f in null_fields):
                doc_id = self.data[i].get("document_id", i)
                print(f"Warning: all fields null for document {doc_id} — model returned empty metadata.")

            results.append(self.data[i] | record)

        return results

    # -----------------------------------------------------------------------
    # Pipeline entry point
    # -----------------------------------------------------------------------

    def fit(self, documents: list[str]) -> list[dict]:
        """Run metadata extraction on the provided documents.

        Returns one dict per document, merging the document's internal
        metadata (document_id, context) with the extracted fields.
        """
        self.data = [{"document_id": i, "context": doc} for i, doc in enumerate(documents)]
        self.data = self._extract_metadata()
        return self.data
