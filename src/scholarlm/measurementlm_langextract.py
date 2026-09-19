"""LangExtract baseline: chunked, schema-grounded extraction via Google's
langextract library (https://github.com/google/langextract).

langextract handles its own chunking, per-chunk few-shot prompting, and
character-level grounding of each extraction back onto the source text. This
module is a translation layer: it builds langextract's call shape
(`prompt_description`, `examples`) from this repo's dataset-config resources,
and converts its output back into the flat measurement-record schema every
other baseline here emits.

To stay faithful to what langextract itself measures, this baseline reuses
Ablation 1's schema/prompt and the NuExtract baseline's example passages
rather than authoring new ones, and skips `_standardize`/`_deduplicate` and
any out-of-vocabulary filtering -- those are this repo's own post-processing,
not part of langextract's method, and applying them would measure something
other than langextract itself. langextract owns its own HTTP client and
chunking; `MeasurementLM`'s `_acall`/`_call_batch` machinery is unused, only
inherited for constructor bookkeeping. The one real addition is an explicit
`output_schema` (`_build_output_schema`) that enum-constrains `attribute` at
generation time when `use_schema_constraints` is set -- a supported
langextract extension point, not a bypass of it.

The shared direct-extraction prompt text prescribes an `{"items": [...]}`
output envelope that conflicts with langextract's own required
`{"extractions": [...]}` shape, so that portion is stripped before use (see
`_prompt_description`). OCR page tags are left in the input text so page
numbers can be recovered from extraction offsets after the fact.

Known issue: at high concurrency, a run can hang on a single stuck chunk and
crash with nothing saved (`fit()` has no try/except around `lx.extract()`).
Not reliably reproduced or root-caused; lower concurrency avoids it, and
retrying is the current mitigation.
"""

from __future__ import annotations

import json
import re

from .instruction_prompts import DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS
from .measurementlm import MeasurementLM

# Canonical OCR page-tag regex (independently duplicated per-module elsewhere
# in this repo -- see page_attribution.py, chandra_format.py,
# measurementlm_chatextract.py -- rather than a shared import).
_PAGE_RE = re.compile(r'<page number="(\d+)">(.*?)</page>', re.DOTALL)

# The one sentence in DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS that prescribes the
# direct-extraction-mode ("items" list) output shape -- dropped for langextract,
# which imposes its own extraction_class/extraction_text/attributes envelope.
_JSON_ENVELOPE_LINE = (
    '- Structure your response as a JSON object with an "items" list, where '
    'each item contains the entity fields, event fields, and "attribute", '
    '"value", and "units" fields as specified in the dataset-specific instructions.'
)

_EXTRACTION_CLASS = "measurement"

# Every dataset config's direct_extraction_prompt currently ends with this
# heading, followed by nothing but a restatement of the "items" JSON envelope
# -- see module docstring. Truncated rather than patched line-by-line since
# its exact wording differs per dataset.
_OUTPUT_FORMAT_HEADING = "Output format requirements:"


def _prompt_description(direct_extraction_prompt: str) -> str:
    """Build langextract's `prompt_description`: the shared extraction
    guidelines plus the dataset's own instructions, each with its own
    conflicting `{"items": [...]}` output-format text stripped out.
    """
    if _JSON_ENVELOPE_LINE not in DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS:
        raise AssertionError(
            "Expected JSON-envelope line not found in "
            "DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS -- the source text changed; "
            "update _JSON_ENVELOPE_LINE (and re-check it still conflicts with "
            "langextract's own output contract) before reusing this blindly."
        )
    guidelines = DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS.replace(_JSON_ENVELOPE_LINE, "").rstrip()

    if _OUTPUT_FORMAT_HEADING not in direct_extraction_prompt:
        raise AssertionError(
            f"Expected {_OUTPUT_FORMAT_HEADING!r} heading not found in "
            "direct_extraction_prompt -- every dataset config's direct "
            "extraction prompt currently ends with this section; if the "
            "dataset config's shape changed, re-check what (if anything) "
            "still needs stripping before reusing this blindly."
        )
    dataset_prompt = direct_extraction_prompt.split(_OUTPUT_FORMAT_HEADING)[0].rstrip()

    return (
        f"{guidelines}\n\n"
        f'Emit every extraction with extraction_class="{_EXTRACTION_CLASS}". '
        "extraction_text must be the exact verbatim numeric value substring "
        "from the document (the same string as the \"value\" attribute below); "
        "put every other field -- the entity fields, event fields, \"attribute\", "
        "\"value\", and \"units\" -- into the extraction's attributes.\n\n"
        f"{dataset_prompt}"
    )


def _build_examples(nuextract_examples: list[dict] | None) -> list:
    """Translate `DatasetConfig.nuextract_examples` into langextract few-shot
    `ExampleData`, using each item's `value` (an exact substring of its
    passage) as the grounded `extraction_text`.
    """
    import langextract as lx

    examples = []
    for example in nuextract_examples or []:
        items = json.loads(example["output"])["items"]
        extractions = []
        for item in items:
            value = item.get("value")
            if not value:
                raise ValueError(
                    f"nuextract_examples item has no non-empty 'value' to ground "
                    f"extraction_text on: {item!r}"
                )
            extractions.append(
                lx.data.Extraction(
                    extraction_class=_EXTRACTION_CLASS,
                    extraction_text=value,
                    attributes={k: v for k, v in item.items() if v is not None},
                )
            )
        examples.append(lx.data.ExampleData(text=example["input"], extractions=extractions))
    return examples


def _attribute_object_schema(direct_extraction_schema, attribute_info_dict: dict) -> dict:
    """JSON Schema for one extraction's attributes: every entity/event field
    as a nullable string, except `attribute`, enum-constrained to
    `attribute_info_dict`'s known vocabulary -- a real generation-time
    constraint, unlike langextract's own example-inferred schema, which only
    types each attribute by Python type, never by its specific values.
    """
    known_attributes = sorted(attribute_info_dict.keys())
    properties: dict[str, dict] = {}
    for name in direct_extraction_schema.model_fields:
        if name == "attribute":
            properties[name] = {"type": "string", "enum": known_attributes}
        else:
            properties[name] = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _build_output_schema(direct_extraction_schema, attribute_info_dict: dict) -> dict:
    """Full langextract output-schema for the single `_EXTRACTION_CLASS`,
    passed to `lx.extract(output_schema=...)` to enum-constrain `attribute`
    at generation time. Mirrors the shape langextract's own schema builder
    produces, using its own key-name constants rather than hardcoding them.
    """
    import langextract as lx

    attributes_key = f"{_EXTRACTION_CLASS}{lx.data.ATTRIBUTE_SUFFIX}"
    item_schema = {
        "type": "object",
        "properties": {
            _EXTRACTION_CLASS: {"type": "string"},
            attributes_key: {
                "anyOf": [
                    _attribute_object_schema(direct_extraction_schema, attribute_info_dict),
                    {"type": "null"},
                ]
            },
        },
        "required": [_EXTRACTION_CLASS, attributes_key],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            lx.data.EXTRACTIONS_KEY: {"type": "array", "items": item_schema},
        },
        "required": [lx.data.EXTRACTIONS_KEY],
        "additionalProperties": False,
    }


def _page_for_offset(context: str, char_pos: int | None) -> int | None:
    """Which `<page number="N">` block (in `context`, the exact string handed
    to langextract) contains character offset `char_pos`. None if ungrounded
    or the offset falls in untagged text.
    """
    if char_pos is None:
        return None
    for m in _PAGE_RE.finditer(context):
        if m.start() <= char_pos < m.end():
            return int(m.group(1))
    return None


class MeasurementLMLangExtract(MeasurementLM):
    """Baseline extraction via Google's langextract, served through a local
    vLLM OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        *args,
        direct_extraction_schema=None,
        direct_extraction_prompt: str | None = None,
        nuextract_examples: list[dict] | None = None,
        max_char_buffer: int,
        extraction_passes: int,
        max_workers: int,
        batch_length: int,
        use_schema_constraints: bool,
        fence_output: bool,
        max_concurrent: int = 32,
        **kwargs,
    ):
        # langextract reads full document text directly; never run the
        # image-based table cleaning pass (mirrors the other text baselines).
        kwargs.setdefault("clean_tables", False)
        super().__init__(*args, max_concurrent=max_concurrent, **kwargs)
        if direct_extraction_schema is None or direct_extraction_prompt is None:
            raise ValueError(
                "direct_extraction_schema and direct_extraction_prompt must be set "
                "for MeasurementLMLangExtract. Use the same values defined in the "
                "dataset config for Ablation 1's direct extraction mode."
            )
        self.direct_extraction_schema = direct_extraction_schema
        self.direct_extraction_prompt = direct_extraction_prompt
        self.nuextract_examples = nuextract_examples
        self.max_char_buffer = max_char_buffer
        self.extraction_passes = extraction_passes
        self.max_workers = max_workers
        self.batch_length = batch_length
        self.use_schema_constraints = use_schema_constraints
        self.fence_output = fence_output

    def _entity_field_names(self) -> list[str]:
        """Schema fields carried through unchanged (everything except the
        measurement fields, which are handled explicitly in `fit`)."""
        return [
            name for name in self.direct_extraction_schema.model_fields
            if name not in ("attribute", "value", "units")
        ]

    def _record_from_extraction(self, doc_idx: int, item_idx: int, extraction, context: str) -> dict:
        """One langextract `Extraction` -> one flat measurement record,
        unfiltered -- an out-of-vocabulary `attribute` is kept as-is, not
        dropped (see module docstring).
        """
        attrs = extraction.attributes or {}
        record = {field: attrs.get(field) for field in self._entity_field_names()}
        record["attribute"] = attrs.get("attribute")
        record["value"] = attrs.get("value")
        record["units"] = attrs.get("units")

        char_interval = extraction.char_interval
        page_number = (
            _page_for_offset(context, char_interval.start_pos)
            if char_interval is not None else None
        )

        entity_id = f"doc_{doc_idx}_measurement_{item_idx}"
        return {"document_id": doc_idx} | record | {
            "entity_id": entity_id, "attribute_terms": [], "page_number": page_number,
        }

    def fit(self, documents: list[str]) -> list[dict]:
        """Run langextract over each document and translate its output into
        the standard flat schema, one record per extraction, unmodified (see
        module docstring for what's deliberately skipped and why).

        Note: langextract requires `fence_output=False` whenever
        `output_schema` is passed (i.e. whenever `use_schema_constraints` is
        set); it raises its own error otherwise.
        """
        import langextract as lx
        from langextract.factory import ModelConfig

        examples = _build_examples(self.nuextract_examples)
        prompt_description = _prompt_description(self.direct_extraction_prompt)
        config = ModelConfig(
            model_id=self.model_name,
            provider="openai",
            # `lx.extract`'s `config=` path (unlike its `model_id=` path) never
            # reads `language_model_params` -- max_output_tokens/top_p must go
            # here, as provider constructor kwargs, or vLLM gets no completion
            # budget at all and a chunk can generate unbounded.
            provider_kwargs={
                "api_key": self.client.api_key,
                "base_url": str(self.client.base_url),
                "max_output_tokens": self.sampling_params.get("max_tokens"),
                "top_p": self.sampling_params.get("top_p"),
            },
        )
        output_schema = (
            _build_output_schema(self.direct_extraction_schema, self.attribute_info_dict)
            if self.use_schema_constraints else None
        )

        records: list[dict] = []
        n_ungrounded = 0

        for doc_idx, context in enumerate(documents):
            annotated = lx.extract(
                text_or_documents=context,
                prompt_description=prompt_description,
                examples=examples,
                config=config,
                max_char_buffer=self.max_char_buffer,
                extraction_passes=self.extraction_passes,
                max_workers=self.max_workers,
                batch_length=self.batch_length,
                use_schema_constraints=self.use_schema_constraints,
                output_schema=output_schema,
                fence_output=self.fence_output,
                temperature=self.sampling_params.get("temperature"),
                show_progress=False,
            )

            for item_idx, extraction in enumerate(annotated.extractions or []):
                if extraction.char_interval is None:
                    n_ungrounded += 1
                records.append(self._record_from_extraction(doc_idx, item_idx, extraction, context))

        if n_ungrounded:
            print(f"LangExtract: {n_ungrounded} record(s) had no character-grounded span (page_number=None).")

        self.data = records
        return self.data
