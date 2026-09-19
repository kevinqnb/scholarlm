"""NuExtract3 baseline: single-shot, full-document-text structured extraction.

NuExtract3 (`numind/NuExtract3`, a 4B Qwen3.5-based vision-language model)
takes a `template` chat-template kwarg (a JSON schema of fields to extract)
plus a dedicated `instructions` kwarg for freeform extraction guidance, and
supports multiple images/messages in one call. This adapter uses that to send
one call per document -- covering every attribute at once -- rather than
dispatching per page or per attribute.

To compare fairly against the other baselines (Ablation 1, GLiNER2,
ChatExtract), this adapter reads full OCR'd document text, not page images,
even though NuExtract3 also supports image input. It also skips
`_deduplicate()`, since deduplication is the actual MeasurementLM pipeline's
own contribution, not something a single-shot baseline should be credited
with.

Full-document text pushes prompts much larger than a single page image, so
context-length overflow is a first-class risk here -- size the model config's
`max_model_len` accordingly. Docs that overflow are tracked in
`self.context_length_exceeded_docs` rather than silently dropped; response
validation failures are tracked in `self.validation_failures`.

Determinism: `sampling_params["seed"]` is forwarded per-call via
`extra_body["seed"]` (the base `_acall` doesn't forward `seed` itself), and
`_extract_records` pins `max_concurrent=1`, since vLLM continuous batching can
flip outputs across concurrent requests even at temperature 0. Still run a
two-run same-seed diff before trusting numbers.

`max_tokens` is a constructor parameter (falling back to
`sampling_params["max_tokens"]`, then a generous default) rather than
hardcoded, since a full paper's measurement count varies output length a lot
by document -- raise it via the runner's `params` if a run needs more.

Two build decisions worth flagging:

- **Units enforcement is flat, not per-attribute.** NuExtract's `template`
  can't express "this attribute never takes a unit, that one takes one of
  several" -- it has no per-attribute conditional typing. This adapter uses
  one flat `Literal` enum per field (`attribute` over the known attributes,
  `units` over the union of all declared units), which constrains guided
  decoding but doesn't tie a specific unit to a specific attribute; the
  dataset's own prompt text carries that association. A value still
  out-of-vocabulary after a client-side retry is kept, not dropped -- the
  same enforce-no-drop policy Ablation 1 and LangExtract use.
- **`attribute_info_dict` is the units source of truth**, not the dataset
  prompt text or `unit_conversion_table` -- those can disagree (see
  `pond.py`); reconciling them is a dataset-config-level fix, out of scope
  here.
"""

import json
from typing import Literal

from pydantic import create_model

from .instruction_prompts import DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS
from .measurementlm import ContextLengthExceededError, MeasurementLM, response_validator


def _units_vocabulary(attribute_info_dict: dict) -> list[str]:
    """Union of every attribute's declared units, in first-seen order (see
    module docstring on why `attribute_info_dict` is the source of truth).

    Can be empty -- some datasets (e.g. measeval) have no closed unit
    vocabulary at all; callers fall back to unconstrained `units` typing
    rather than treating that as a config error.
    """
    seen: list[str] = []
    for info in attribute_info_dict.values():
        for unit in info.get("units", []):
            if unit not in seen:
                seen.append(unit)
    return seen


def _build_response_schema(direct_extraction_schema, attribute_names: list[str], unit_names: list[str]):
    """Pydantic model mirroring `direct_extraction_schema`, with `attribute`
    and `units` narrowed from plain `str` to `Literal` enums -- this is what
    actually constrains vLLM's guided decoding; the NuExtract `template`
    string is prompt text only, never a decoding constraint on its own.

    Used only to build `response_format`, never to parse a response: parsing
    uses the plain (non-Literal) `direct_extraction_schema` instead, so one
    out-of-vocabulary item fails validation on its own rather than discarding
    every other item in the same response.
    """
    fields: dict = {}
    for name, finfo in direct_extraction_schema.model_fields.items():
        if name == "attribute":
            annotation = Literal[tuple(attribute_names)]
        elif name == "units" and unit_names:
            annotation = Literal[tuple(unit_names)] | None
        else:
            annotation = finfo.annotation
        default = ... if finfo.is_required() else finfo.default
        fields[name] = (annotation, default)
    return create_model(f"{direct_extraction_schema.__name__}Literal", **fields)


def _build_nuextract_template(direct_extraction_schema, attribute_names: list[str], unit_names: list[str]) -> dict:
    """Convert `direct_extraction_schema` into a NuExtract JSON template.
    Every field is `"verbatim-string"` except `attribute` and `units`, which
    become NuExtract enum lists -- the prompt-facing counterpart of
    `_build_response_schema`'s Literal typing.
    """
    item_template: dict = {}
    for field_name in direct_extraction_schema.model_fields:
        if field_name == "attribute":
            item_template[field_name] = list(attribute_names)
        elif field_name == "units" and unit_names:
            item_template[field_name] = list(unit_names)
        else:
            item_template[field_name] = "verbatim-string"
    return {"items": [item_template]}


def _build_example_messages(examples: list[dict] | None) -> list[dict]:
    """Convert `DatasetConfig.nuextract_examples` into NuExtract3's
    `developer`-message few-shot format: a content list whose items are all
    input except the last, which is the expected output.
    """
    return [
        {
            "role": "developer",
            "content": [
                {"type": "text", "text": example["input"]},
                {"type": "text", "text": example["output"]},
            ],
        }
        for example in (examples or [])
    ]


class MeasurementLMNuExtract3(MeasurementLM):
    """Single-shot, full-document-text extraction baseline using NuExtract3."""

    def __init__(
        self,
        *args,
        direct_extraction_schema=None,
        direct_extraction_prompt=None,
        examples: list[dict] | None = None,
        max_tokens: int | None = None,
        **kwargs,
    ):
        if direct_extraction_schema is None or direct_extraction_prompt is None:
            raise ValueError(
                "direct_extraction_schema and direct_extraction_prompt must be set "
                "for MeasurementLMNuExtract3. Use the same values defined in the "
                "dataset config for Ablation 1."
            )
        if kwargs.get("clean_tables", False):
            raise ValueError(
                "MeasurementLMNuExtract3 does not support clean_tables=True: it has "
                "no table-cleaning step and takes no processed_pdf_dirs. Clean text "
                "upstream (or use MeasurementLMAblation1) if table cleaning is needed."
            )
        kwargs["clean_tables"] = False
        if not kwargs.get("use_extra_body", True):
            # _acall (measurementlm.py) gates its *entire* extra_body handling --
            # including the caller-supplied extra_body this adapter passes for
            # `template`/`instructions`/`seed` -- behind `self.use_extra_body`.
            # With it False, `template` never reaches the request: the chat
            # template's `mode` then defaults to 'content' (markdown/OCR mode),
            # not 'structured', while `response_format` still forces valid JSON
            # out of that wrong mode -- a parseable response built from a
            # markdown-conversion prompt, with no exception anywhere.
            raise ValueError(
                "MeasurementLMNuExtract3 requires use_extra_body=True: its "
                "chat_template_kwargs (template/instructions/seed) are carried "
                "entirely through extra_body."
            )
        super().__init__(
            *args,
            extraction_mode="direct",
            direct_extraction_schema=direct_extraction_schema,
            direct_extraction_prompt=direct_extraction_prompt,
            **kwargs,
        )
        self.examples = examples
        self.max_tokens = max_tokens
        self.validation_failures = 0

    # -----------------------------------------------------------------------
    # Single extraction step: extract all records directly from document text
    # -----------------------------------------------------------------------

    def _extract_records(self) -> list[dict]:
        """Extract all measurement records from each document's full text, one
        call per document covering every attribute (see module docstring).

        Returns:
            List of measurement records, one per extracted item.
        """
        attribute_names = list(self.attribute_info_dict.keys())
        unit_names = _units_vocabulary(self.attribute_info_dict)
        known_attributes = set(attribute_names)
        # None (not an empty set) when there's no closed unit vocabulary at all
        # (measeval) -- distinguishes "skip this check" from "nothing is valid".
        known_units = set(unit_names) if unit_names else None

        # Strict schema (Literal attribute/units) for the decoding constraint only.
        ResponseItem = _build_response_schema(self.direct_extraction_schema, attribute_names, unit_names)
        StrictDirectExtractionList = create_model("StrictDirectExtractionList", items=(list[ResponseItem], ...))
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "direct_extraction_list",
                "schema": StrictDirectExtractionList.model_json_schema(),
            },
        }

        # Lenient schema (plain str) for parsing whatever text actually comes
        # back -- see _build_response_schema's docstring for why these differ.
        DirectExtractionList = create_model(
            "DirectExtractionList", items=(list[self.direct_extraction_schema], ...)
        )

        template_json = json.dumps(
            _build_nuextract_template(self.direct_extraction_schema, attribute_names, unit_names),
            indent=2,
        )
        instructions = f"{DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS}\n\n{self.direct_extraction_prompt}"
        example_messages = _build_example_messages(self.examples)

        extra_body = {"chat_template_kwargs": {"template": template_json, "instructions": instructions}}
        if "seed" in self.sampling_params:
            extra_body["seed"] = self.sampling_params["seed"]

        messages: list[list[dict]] = []
        for datapoint in self.data:
            user_message = {
                "role": "user",
                "content": [{"type": "text", "text": datapoint["context"]}],
            }
            messages.append([*example_messages, user_message])

        def _validate(r):
            parsed = response_validator(DirectExtractionList, r)
            for item in parsed["items"]:
                if item.get("value") is None:
                    continue
                if item.get("attribute") not in known_attributes:
                    raise ValueError(f"attribute {item.get('attribute')!r} not in attribute_info_dict")
                if known_units is not None and item.get("units") is not None and item.get("units") not in known_units:
                    raise ValueError(f"units {item.get('units')!r} not in attribute_info_dict's unit vocabulary")
            return parsed

        # max_tokens: an explicit constructor value wins; otherwise fall back to
        # sampling_params (matching _acall's own resolution order) before the
        # code default -- so a model-config YAML's own max_tokens isn't silently
        # shadowed by this call always passing something.
        max_tokens = self.max_tokens if self.max_tokens is not None else self.sampling_params.get("max_tokens", 32768)

        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_tokens=max_tokens,
            max_retries=2,
            # Pinned to 1, not exposed as a constructor knob: one call per
            # document already means high per-request cost (full-paper prompt,
            # large output), and vLLM continuous batching flips outputs across
            # concurrent requests even at temperature 0 (see module docstring,
            # issue #3) -- concurrency, not just a missing seed, was named as
            # part of that determinism failure. Matches the same literal in
            # MeasurementLM._extract_triples and MeasurementLMAblation1.
            max_concurrent=1,
            validator=_validate,
            timeout=600.0,
            extra_body=extra_body,
        )

        records: list[dict] = []
        out_of_vocab_count = 0
        for doc_idx, r in enumerate(response_texts):
            if isinstance(r, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(doc_idx)
                continue
            try:
                resp_validated = response_validator(DirectExtractionList, r)
            except Exception as e:
                self.validation_failures += 1
                print(f"Validation error in NuExtract3 response (doc {doc_idx}): {e}")
                print(f"Response text: {r}")
                resp_validated = {"items": []}

            for item_idx, item in enumerate(resp_validated["items"]):
                if item.get("value") is None:
                    continue
                if item.get("attribute") not in known_attributes:
                    out_of_vocab_count += 1
                    print(
                        f"NuExtract3 record with out-of-vocabulary attribute "
                        f"{item.get('attribute')!r} (doc {doc_idx}, item {item_idx}); "
                        f"still not in attribute_info_dict after retries -- kept, not dropped "
                        f"(enforce-no-drop: matches Ablation 1 and LangExtract's policy)."
                    )
                elif known_units is not None and item.get("units") is not None and item.get("units") not in known_units:
                    out_of_vocab_count += 1
                    print(
                        f"NuExtract3 record with out-of-vocabulary units "
                        f"{item.get('units')!r} (doc {doc_idx}, item {item_idx}); "
                        f"still not in attribute_info_dict's unit vocabulary after retries -- kept, "
                        f"not dropped (enforce-no-drop: matches Ablation 1 and LangExtract's policy)."
                    )
                entity_id = f"doc_{doc_idx}_entity_{item_idx}"
                records.append(
                    self.data[doc_idx] | item | {
                        "entity_id": entity_id,
                        "attribute_terms": [],
                    }
                )

        if out_of_vocab_count:
            print(
                f"NuExtract3 extraction: {out_of_vocab_count} record(s) had an "
                f"out-of-vocabulary attribute or units after retries; kept (not dropped)."
            )

        return records

    # -----------------------------------------------------------------------
    # Full pipeline (single extraction step, nothing else)
    # -----------------------------------------------------------------------

    def fit(self, documents: list[str]) -> list[dict]:
        """Run the NuExtract3 baseline on the given documents' full OCR text.

        Skips both `_standardize()` (an LLM normalization pass) and
        `_deduplicate()`: both are part of the actual MeasurementLM pipeline's
        own contribution, not something a single-shot baseline should be
        credited with. This adapter takes raw text only -- no page images, no
        table-cleaning step.

        Args:
            documents: OCR text strings, one per document.
        """
        self.context_length_exceeded_docs = set()
        self.data = [{"document_id": i, "context": doc} for i, doc in enumerate(documents)]
        self.data = self._extract_records()
        return self.data
