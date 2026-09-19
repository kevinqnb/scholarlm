"""NuExtract3 baseline: single-shot, full-document-text structured extraction.

NuExtract3 (`numind/NuExtract3`, a 4B Qwen3.5-based vision-language model) is a
successor to NuExtract-2.0-8B with three capabilities the old adapter
(`measurementlm_nuextract.py`) couldn't use:

1. A dedicated `instructions` chat-template kwarg, alongside the `template`
   kwarg, for freeform extraction guidance (see NuExtract3's model card,
   "Structured extraction" section). NuExtract-2.0 had no such field, which is
   why the old adapter folded each attribute's description into its `value`
   field's JSON key instead (`_value_field_key`). Here, `instructions` carries
   `DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS` (the same generic direct-extraction
   guardrails `MeasurementLM._extract_triples`/`MeasurementLMAblation1` use --
   omit uncertainty bounds, use the central value, no model-fit statistics,
   etc.) followed by the dataset's own `direct_extraction_prompt`.
2. A fixed chat-template bug: NuExtract-2.0's template only ever emitted one
   image placeholder per message, forcing the old adapter to dispatch one
   request per (page, attribute). NuExtract3's template
   (`chat_template.jinja`, confirmed by inspecting the actual downloaded
   model files under `$HF_CACHE/models--numind--NuExtract3`) loops over every
   image and every user message via `render_content(..., do_vision_count=True)`
   -- multi-image, multi-message input works in one call. Combined with (3)
   below, this baseline now sends one call per *document*, not one per
   (page, attribute).
3. Per the build decision to compare fairly against the other baselines
   (Ablation 1, GLiNER2, ChatExtract), this adapter reads full OCR'd document
   TEXT (like `MeasurementLMAblation1`), not rendered page images -- despite
   NuExtract3 also supporting image and mixed text+image input.

Issues carried over from `ISSUE-nuextract-baseline.md` (scoped to the old
adapter) and how each is handled here:

- **#1 dedup no-op**: dissolves by removal, not by fixing the key. This
  adapter runs no `_deduplicate()` step at all -- deduplication of
  near-duplicate mentions is part of the actual MeasurementLM pipeline's own
  contribution (see `notes/scholarlm`), and applying it to a baseline would
  credit the baseline with a capability it doesn't have, blurring the
  comparison. `entity_id` is therefore a plain per-item loop index
  (`doc_{i}_entity_{j}`), the same as `MeasurementLM._extract_triples` and
  `MeasurementLMAblation1` -- unique by construction, never merged.
- **#2 silent exception swallowing**: a `ContextLengthExceededError` is
  isolated to its own `isinstance` check per document (mirroring
  `MeasurementLM._extract_triples`) and counted in
  `self.context_length_exceeded_docs`, never reaching the generic
  `except Exception` block below it. That block (a validation-failure
  fallback to `{"items": []}`) is the same idiom used everywhere else in
  `measurementlm.py`; what was missing before was any visible count -- this
  adapter tracks `self.validation_failures` so a dropped call is no longer
  indistinguishable from a genuine negative. Full-document text pushes prompts
  far larger than a single page image (pond papers already reach ~47k tokens
  under the full-paper judge), so context-length overflow is a first-class
  risk here, not an edge case -- the model config's `max_model_len` needs
  sizing accordingly.
- **#3 no determinism guarantee**: two contributing causes, both addressed.
  `MeasurementLM._acall` never forwards a `seed` -- rather than editing that
  shared method (which every other runner also goes through), this adapter
  forwards `sampling_params["seed"]`, when present, via its own per-call
  `extra_body["seed"]` (`_acall`'s generic `extra_body` merge, see its
  docstring, passes it straight through to the vLLM request body, the same
  mechanism `run_judge_local.py` already uses for the judge path). Separately,
  the issue doc also names concurrency itself ("vLLM continuous batching
  causes flips ... `max_concurrent=16` plus retry rounds vary batch
  composition run to run") -- `_extract_records` pins `max_concurrent=1` at
  the `_call_batch` call site (not a constructor knob), matching the literal
  both `MeasurementLM._extract_triples` and `MeasurementLMAblation1` already
  use for this same one-call-per-document shape. Still run the two-run
  same-seed diff before trusting numbers.
- **#4 hardcoded page cap**: dissolves. There is no per-page dispatch anymore,
  so there is nothing to cap -- the full document is one call. A document that
  doesn't fit the model's context window is issue #2's context-length path,
  not a silent truncation. Relatedly, `max_tokens` (the *output* cap) is a
  constructor parameter here, not a hardcoded literal -- unlike a single page
  image, a full paper's measurement count (and therefore its output JSON
  length) varies a lot by document. When left unset, it falls back to
  `sampling_params["max_tokens"]` (so a model-config YAML's own value isn't
  silently shadowed) and then to 32768, matching
  `MeasurementLMAblation1`'s own single-call-per-document precedent. That
  32768 default was checked, not just matched by analogy: the worst-case
  document across every real dataset's ground truth (nfix, ~121
  measurements/paper) needs on the order of 10k output tokens at this
  schema's per-item verbosity -- see the build session for the calculation.
  Ground truth is a lower bound on what the model may emit, so treat this as
  a checked starting point, not a hard guarantee; plumb it through the
  runner's `params` if a specific run needs raising it (same fix issue #4
  itself prescribes for `max_images_per_document`).
- **#5 few-shot examples biased toward emitting nothing**: dissolves. The old
  adapter filtered each dataset's `nuextract_examples` down to one attribute
  per call, so any attribute absent from the examples saw only empty-output
  demonstrations. One call now covers every attribute, so
  `dataset_config.nuextract_examples` is reused unmodified (just rewrapped
  into NuExtract3's `developer`-message example format) -- no per-attribute
  filtering, no all-empty bias.
- **#6 missing seed-vs-repo-config check**: belongs in the runner
  (`experiments/run_baseline_nuextract3.py`, not yet written), same as
  `run_extraction.py`'s existing check.
- **#7 no experiment config yet**: also a runner-level follow-up.

Two build decisions worth flagging explicitly (both discussed and confirmed
during the build session, not inferred):

- **Units enforcement is flat, not per-attribute.** NuExtract's `template`
  field is a single flat item shape -- it cannot express "ph never has a unit
  but tn takes one of seven" (no discriminated-union/oneOf construct in its
  type DSL). So even though `response_format` (the actual vLLM-enforced
  decoding constraint, independent of the template text) *could* use a
  Pydantic discriminated union to tie units to attribute, the template shown
  to the model couldn't express the same tie -- template and constraint would
  disagree. This adapter instead uses one flat `Literal` per field:
  `attribute` over `attribute_info_dict`'s keys, `units` over the union of
  every attribute's declared units (or `None`). This closes the old adapter's
  actual hole (attribute was plain `str` in NuExtract-2.0's `response_format`,
  so guided decoding never constrained it -- see that module's docstring) but
  does not stop a `ph` item from picking a `tn` unit; the dataset's
  `direct_extraction_prompt`, passed via `instructions`, is what tells the
  model which units belong to which attribute. Both `attribute` and `units`
  get the same client-side retry backstop (in `_extract_records`'s
  `_validate`, `max_retries=2`): if guided decoding doesn't hold for this new
  model/backend combination -- unverified, same reason the base pipeline
  keeps its own vocabulary check even with a plain-`str` schema -- an
  out-of-vocabulary value is retried. **Enforce-no-drop**: unlike the old
  NuExtract-2.0 adapter's issue #1 discussion might suggest, a value still
  out-of-vocabulary after retries is kept, not dropped (`out_of_vocab_count`
  in `_extract_records` tracks it for visibility) -- Ablation 1 has no
  vocabulary check at all and LangExtract's `output_schema` path
  (`use_schema_constraints=True`) never drops either, so this adapter's
  post-hoc filtering used to be a third, uniquely strict policy for the same
  underlying failure mode. Aligned across all three baselines/points of
  comparison as of the build session that added this note.
- **`attribute_info_dict` is the units source of truth.** `pond.py` currently
  has three disagreeing unit vocabularies for the same attributes:
  `attribute_info_dict` (e.g. 7 entries for `tn`/`tp`/`chla`), the
  `direct_extraction_prompt` text (3), and `unit_conversion_table` (3, a
  different 3). `attribute_info_dict` was chosen because it's the structured
  value already passed to this class and to `MeasurementLM` generally --
  reconciling the other two is a separate, dataset-config-level fix, not part
  of this adapter.
"""

import json
from typing import Literal

from pydantic import create_model

from .instruction_prompts import DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS
from .measurementlm import ContextLengthExceededError, MeasurementLM, response_validator


def _units_vocabulary(attribute_info_dict: dict) -> list[str]:
    """Union of every attribute's declared units, in first-seen order.

    `attribute_info_dict` is the units source of truth for this adapter (see
    module docstring) -- not `direct_extraction_prompt` text or
    `unit_conversion_table`, which currently disagree with it in `pond.py`.

    Can be empty: measeval's sole attribute ("measurement") declares no units
    at all -- by design, not omission, since its units are free text rather
    than a closed vocabulary (the same reason `collect_attribute_terms=False`
    exists for it elsewhere -- see `MeasurementLM.__init__`'s docstring). An
    empty result means "this dataset has no closed unit vocabulary"; callers
    fall back to unconstrained `units` typing rather than treating it as a
    config error.
    """
    seen: list[str] = []
    for info in attribute_info_dict.values():
        for unit in info.get("units", []):
            if unit not in seen:
                seen.append(unit)
    return seen


def _build_response_schema(direct_extraction_schema, attribute_names: list[str], unit_names: list[str]):
    """Pydantic model mirroring `direct_extraction_schema`, with `attribute`
    and `units` narrowed from plain `str` to `Literal` enums.

    This is what actually constrains vLLM's guided decoding (`response_format`)
    -- the NuExtract `template` string is prompt text only, never a decoding
    constraint on its own (see module docstring on the old NuExtract-2.0
    adapter's identical off-vocabulary hole).

    Used only to build `response_format`, never to parse a response: parsing
    uses the plain (non-Literal) `direct_extraction_schema` instead, so that
    one out-of-vocabulary item -- decoding-time enforcement notwithstanding --
    fails validation on its own rather than raising a `ValidationError` that
    discards every other item in the same document's response. The
    post-`_call_batch` loop in `_extract_records` retries such an item
    (`_validate`, `max_retries=2`) but keeps it either way (enforce-no-drop --
    see module docstring), matching Ablation 1's `_extract_triples`, which
    never checks vocabulary at all.

    `units` keeps its original (non-Literal) typing when `unit_names` is
    empty -- some datasets (measeval) have no closed unit vocabulary at all
    (see `_units_vocabulary`).
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
    become NuExtract enum lists (JSON arrays of allowed strings) -- the
    prompt-facing counterpart of `_build_response_schema`'s Literal typing.
    `units` stays `"verbatim-string"` when `unit_names` is empty (no closed
    vocabulary to enumerate -- see `_units_vocabulary`).
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
    input except the last, which is the expected output (model card,
    "In-context examples for extraction").

    Reused unmodified -- unlike the old NuExtract-2.0 adapter, there is no
    per-attribute filtering to do, since one call now covers every attribute.

    Offline-rendering `chat_template.jinja` against this exact message shape
    (developer examples followed by the final user message, per the model
    card's own documented ordering) confirmed `mode` correctly flips to
    `structured` and `template`/`instructions` render as expected, but that
    the template's `【examples_end】` marker never appears: its emission is
    gated on `loop.last` over the *whole* `messages` list, which is the final
    `user` message, not the last `developer` one -- so the closing tag branch
    is never reached for either message role. This reproduces NuMind's own
    documented example ordering, so it isn't something to work around by
    reordering messages; `【document_start】` immediately follows and still
    marks an unambiguous boundary.
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
        """Extract all measurement records from each document's full text.

        One call per document: the `template` + `instructions` chat-template
        kwargs describe every attribute at once (see module docstring), so
        there is no per-attribute or per-page dispatch left to do.

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

        Deliberately skips both `_standardize()` (an LLM normalization pass)
        and `_deduplicate()`: both are part of the actual MeasurementLM
        pipeline's own contribution, not something a single-shot baseline
        should be credited with (see module docstring, issue #1). This
        adapter takes raw text only -- no page images, no table-cleaning step.

        Args:
            documents: OCR text strings, one per document.
        """
        self.context_length_exceeded_docs = set()
        self.data = [{"document_id": i, "context": doc} for i, doc in enumerate(documents)]
        self.data = self._extract_records()
        return self.data
