"""LangExtract baseline: chunked, schema-grounded extraction via Google's
langextract library (https://github.com/google/langextract).

langextract does the heavy lifting itself -- chunking a document, prompting per
chunk with a few-shot template, and aligning each extracted span back onto the
source text (character-level grounding). This module is a translation layer: it
turns a dataset config into langextract's own call shape (`prompt_description`,
`examples`) and turns its output (a list of `Extraction`s per document) back into
the flat measurement-record schema every other baseline in this repo emits.

Deliberately reuses existing dataset-config resources rather than authoring new
ones for this baseline specifically:

* `direct_extraction_schema` / `direct_extraction_prompt` (Ablation 1's flat
  entity+event+attribute+value+units schema, see `measurementlm.py`'s
  `_extract_triples`) supply the attribute vocabulary and field descriptions.
  langextract's `attributes` dict is built with exactly this schema's field
  names, so `_entity_field_names` below is dataset-generic rather than
  hardcoded.
* `nuextract_examples` (the NuExtract baseline's synthetic, already-audited
  passages -- see e.g. `experiments/dataset-configs/pond.py`) become
  langextract's few-shot `ExampleData`. Each item's `value` field is documented
  there as an exact substring of its passage, so it doubles as the grounded
  `extraction_text` langextract's alignment step needs -- no new example
  authoring, and no risk of leaking real eval-paper text into a baseline's
  headline number.

Two prompt fragments prescribe the direct-extraction-mode output shape
(`{"items": [...]}`), which conflicts with langextract's own required
`{"extractions": [...]}` envelope -- confirmed the hard way in the first smoke
run: every chunk came back rejected by langextract's resolver
("Content must contain an 'extractions' key") because the model dutifully
followed these instructions instead. Both are stripped programmatically
(asserting if the expected text can't be found, rather than silently keeping
a contradictory instruction) instead of reused verbatim:

* `DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS` (`instruction_prompts.py`) -- its
  last line ("Structure your response as a JSON object with an 'items'
  list...").
* Each dataset's own `direct_extraction_prompt` -- every one of them ends
  with an "Output format requirements:" section spelling out the same
  `{"items": [...]}` envelope in more detail (confirmed identical in shape
  across pond/nfix/supermat/measeval); everything from that heading onward is
  pure output-shape boilerplate, no substantive entity/attribute content, so
  it's truncated there rather than patched line-by-line.

Full OCR-tagged text (`<page number="N">...</page>`, same as every other
baseline's input) is passed straight through, unstripped -- langextract chunks
it internally regardless, and the "direct extraction" mode this baseline mirrors
(`_extract_triples`) already feeds the model tag-included text with no ill
effect. Keeping the tags means page numbers can be recovered by scanning the
same original string langextract saw for the `<page>` block containing a given
character offset -- no separate stripped-text representation to keep in sync.

langextract itself owns the OpenAI-compatible HTTP client for this baseline
(via `langextract.factory.ModelConfig(provider="openai", ...)`, pointed at a
local vLLM server's `base_url`); `MeasurementLM`'s own `_acall`/`_call_batch`
machinery is unused here, only inherited for `_deduplicate` and the shared
constructor bookkeeping (model name, attribute vocabulary, sampling params).
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
    """Build langextract's `prompt_description` from the shared extraction
    guidelines (minus the conflicting JSON-envelope line) plus the dataset's
    own entity/attribute/event instructions (minus its own conflicting
    "Output format requirements" section).
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
    `ExampleData`. Each nuextract example item's `value` field is documented at
    its point of definition as an exact substring of the example passage, so it
    is reused directly as the grounded `extraction_text`.
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
        """One langextract `Extraction` -> one flat measurement record.
        Caller has already checked `attribute` is in vocabulary.
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
        """Run langextract over each document, then deduplicate into the
        standard flat schema (like NuExtract/ChatExtract, this skips
        `_standardize` -- an extra MeasurementLM-specific LLM pass -- for a fair,
        non-conflating baseline comparison).
        """
        import langextract as lx
        from langextract.factory import ModelConfig

        examples = _build_examples(self.nuextract_examples)
        prompt_description = _prompt_description(self.direct_extraction_prompt)
        config = ModelConfig(
            model_id=self.model_name,
            provider="openai",
            provider_kwargs={"api_key": self.client.api_key, "base_url": str(self.client.base_url)},
        )

        records: list[dict] = []
        oov_attribute_counts: dict[str | None, int] = {}
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
                fence_output=self.fence_output,
                temperature=self.sampling_params.get("temperature"),
                show_progress=False,
            )

            for item_idx, extraction in enumerate(annotated.extractions or []):
                attrs = extraction.attributes or {}
                attribute = attrs.get("attribute")
                if attribute not in self.attribute_info_dict:
                    oov_attribute_counts[attribute] = oov_attribute_counts.get(attribute, 0) + 1
                    continue
                if extraction.char_interval is None:
                    n_ungrounded += 1
                records.append(self._record_from_extraction(doc_idx, item_idx, extraction, context))

        if oov_attribute_counts:
            n_oov = sum(oov_attribute_counts.values())
            by_name = ", ".join(f"{name!r}: {count}" for name, count in oov_attribute_counts.items())
            print(f"LangExtract: dropped {n_oov} record(s) with an out-of-vocabulary attribute ({by_name}).")
        if n_ungrounded:
            print(f"LangExtract: {n_ungrounded} record(s) had no character-grounded span (page_number=None).")

        self.data = self._deduplicate(records)
        return self.data
