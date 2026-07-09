"""NuExtract-2.0-8B baseline: single-shot, image-based structured extraction.

Unlike MeasurementLM (which reads OCR'd `<page>/<table>` tagged text) and unlike
Ablation 1 (which is also single-shot but text-only), this baseline sends the
rendered page images of a document directly to a vision-language model in one
call. It reuses `MeasurementLM`'s OpenAI-compatible async client, retry, and
concurrency machinery unchanged — only message/request construction differs.

NuExtract has its own fixed calling convention (see its `chat_template.jinja`
on HuggingFace): the JSON extraction schema must be passed as a separate
`extra_body.chat_template_kwargs.template` field, not embedded as prose in the
message content. Its template also special-cases message content: if a `text`
content block is present whose text isn't the literal string `"<image>"`, the
template treats the whole request as text-only and emits *no* image
placeholder tokens at all, even when images are attached — silently breaking
every request. There is also no field in this protocol for freeform
instructions alongside the template (only `template` and optional few-shot
`examples`), so — unlike Ablation 1 — this baseline cannot reuse a dataset's
`direct_extraction_prompt`; it is inherently template-only, matching how
NuMind's own examples call the model.

To offset the lack of attribute descriptions, callers may pass `examples`
(from `DatasetConfig.nuextract_examples`) — small, synthetic input/output
pairs that teach the model the dataset's attribute vocabulary and field
conventions the way NuExtract expects: by demonstration, not description.
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
        examples: list[dict] | None = None,
        max_concurrent: int = 2,
        max_images_per_document: int = 45,
        use_extra_body: bool = True,
        **kwargs,
    ):
        super().__init__(*args, max_concurrent=max_concurrent, use_extra_body=use_extra_body, **kwargs)
        if direct_extraction_schema is None:
            raise ValueError(
                "direct_extraction_schema must be set for MeasurementLMNuExtract. "
                "Use the same schema defined in the dataset config for Ablation 1."
            )
        self.direct_extraction_schema = direct_extraction_schema
        self.examples = examples
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
        template_json = json.dumps(template, indent=2)

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

            # NuExtract's chat template only accepts image content blocks here
            # (plus an optional literal "<image>" text sentinel) — any other
            # text block makes it silently drop all image placeholders. The
            # schema itself is sent out-of-band via extra_body below.
            content: list[dict] = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img}"},
                }
                for img in images_b64
            ]
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
        chat_template_kwargs = {"template": template_json}
        if self.examples:
            chat_template_kwargs["examples"] = self.examples
        extra_body = {"chat_template_kwargs": chat_template_kwargs}

        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_tokens=8192,
            max_retries=4,
            validator=partial(response_validator, DirectExtractionList),
            timeout=600.0,
            extra_body=extra_body,
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
