"""NuExtract-2.0-8B baseline: single-shot, image-based structured extraction.

Unlike MeasurementLM (which reads OCR'd `<page>/<table>` tagged text) and unlike
Ablation 1 (which is also single-shot but text-only), this baseline sends the
rendered page images of a document directly to a vision-language model in one
call. It reuses `MeasurementLM`'s OpenAI-compatible async client, retry, and
concurrency machinery unchanged — only prompt/message construction differs.

Requires `direct_extraction_schema` and `direct_extraction_prompt` (the same
flat Pydantic model and dataset-specific instructions Ablation 1 uses,
describing entities, measurement events, and attributes in one block) and
pre-rendered page images produced by `experiments/process_pdfs.py`
(`{processed_pdf_dir}/{page_index}.b64`). The dataset's existing, iterated-on
`direct_extraction_prompt` is reused verbatim rather than writing a new,
untested prompt per dataset — only a NuExtract-specific typed JSON template is
appended, to describe the target schema in NuExtract's own convention.
"""

import json
from functools import partial
from pathlib import Path

from .measurementlm import MeasurementLM, response_validator


def _build_template(direct_extraction_schema, attribute_info_dict: dict) -> dict:
    """Convert a flat direct-extraction schema into a NuExtract JSON template.

    Every field is typed as `verbatim-string` (copy exactly from the document)
    except `attribute`, which is expressed as an enum of the known attribute
    names so NuExtract is constrained to the same attribute vocabulary as the
    rest of the pipeline.
    """
    attribute_names = sorted(attribute_info_dict.keys())
    item_template: dict = {}
    for field_name in direct_extraction_schema.model_fields:
        if field_name == "attribute":
            item_template[field_name] = attribute_names
        else:
            item_template[field_name] = "verbatim-string"
    return {"items": [item_template]}


class MeasurementLMNuExtract(MeasurementLM):
    """Single-shot image-based extraction baseline using NuExtract-2.0-8B."""

    def __init__(
        self,
        *args,
        direct_extraction_schema=None,
        direct_extraction_prompt: str | None = None,
        max_concurrent: int = 2,
        max_images_per_document: int = 32,
        **kwargs,
    ):
        super().__init__(*args, max_concurrent=max_concurrent, **kwargs)
        if direct_extraction_schema is None or direct_extraction_prompt is None:
            raise ValueError(
                "direct_extraction_schema and direct_extraction_prompt must both be set "
                "for MeasurementLMNuExtract. Use the same values defined in the dataset "
                "config for Ablation 1 — do not write a new prompt here."
            )
        self.direct_extraction_schema = direct_extraction_schema
        self.direct_extraction_prompt = direct_extraction_prompt
        self.max_images_per_document = max_images_per_document

    # -----------------------------------------------------------------------
    # Single extraction step: extract all records directly from page images
    # -----------------------------------------------------------------------

    def _extract_records(self, processed_pdf_dirs: list[str]) -> list[dict]:
        """Extract all measurement records from each document's page images.

        Args:
            processed_pdf_dirs: Paths to pre-processed image directories, one
                per document (in the same order as `self.data`), each
                containing `{page_index}.b64` files from `process_pdfs.py`.

        Returns:
            List of records suitable for `_deduplicate()`.
        """
        from pydantic import create_model

        template = _build_template(self.direct_extraction_schema, self.attribute_info_dict)
        # Reuse the dataset's existing, iterated-on direct-extraction prompt verbatim
        # (the same one Ablation 1 uses) rather than writing new, untested instructions
        # per dataset. Only the NuExtract-specific typed template is appended.
        text_block = (
            f"{self.direct_extraction_prompt}\n\n"
            f"TEMPLATE:\n{json.dumps(template, indent=2)}\n\n"
            "Return ONLY a JSON object matching the template above, with one item per "
            "distinct measurement found."
        )

        messages: list[list[dict]] = []
        for doc_dir in processed_pdf_dirs:
            doc_path = Path(doc_dir)
            if not doc_path.exists():
                raise FileNotFoundError(
                    f"Processed PDF directory not found: {doc_dir}\n"
                    f"Run 'python experiments/process_pdfs.py' first."
                )
            page_files = sorted(doc_path.glob("*.b64"), key=lambda p: int(p.stem))
            if len(page_files) > self.max_images_per_document:
                print(
                    f"WARNING: {doc_dir} has {len(page_files)} pages, exceeding "
                    f"max_images_per_document={self.max_images_per_document}. "
                    f"Only the first {self.max_images_per_document} pages will be sent."
                )
                page_files = page_files[: self.max_images_per_document]
            images_b64 = [p.read_text().strip() for p in page_files]

            content: list[dict] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                }
                for img in images_b64
            ]
            content.append({"type": "text", "text": text_block})
            messages.append([{"role": "user", "content": content}])

        DirectExtractionList = create_model(
            "DirectExtractionList",
            items=(list[self.direct_extraction_schema], ...),
        )
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "direct_extraction_list",
                "schema": DirectExtractionList.model_json_schema(),
            },
        }

        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_tokens=8192,
            max_retries=4,
            validator=partial(response_validator, DirectExtractionList),
            timeout=600.0,
        )

        records: list[dict] = []
        for i, r in enumerate(response_texts):
            try:
                resp_validated = response_validator(DirectExtractionList, r)
            except Exception as e:
                print(f"Validation error in NuExtract response: {e}")
                print(f"Response text: {r}")
                resp_validated = {"items": []}

            for j, item in enumerate(resp_validated["items"]):
                if item.get("value") is None:
                    continue
                entity_id = f"doc_{i}_entity_{j}"
                records.append(
                    self.data[i] | item | {
                        "entity_id": entity_id,
                        "attribute_terms": [],
                    }
                )

        return records

    # -----------------------------------------------------------------------
    # Full pipeline (single extraction step + programmatic dedup)
    # -----------------------------------------------------------------------

    def fit(self, processed_pdf_dirs: list[str]) -> list[dict]:
        """Run the NuExtract baseline on the given documents' page images.

        Deliberately skips `_standardize()` (an LLM normalization pass) so the
        comparison isn't blended with an extra MeasurementLM-specific LLM
        call; `_deduplicate()` is purely programmatic and is applied for a
        fair, non-conflating cleanup of near-duplicate mentions.
        """
        self.data = [{"document_id": i} for i in range(len(processed_pdf_dirs))]
        self.data = self._extract_records(processed_pdf_dirs)
        self.data = self._deduplicate(self.data)
        return self.data
