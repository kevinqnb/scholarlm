import asyncio
import contextlib
import json
import time
from typing import Any, Callable
from pydantic import BaseModel
import numpy as np
import pandas as pd
import math
import re
from io import StringIO
from openai import AsyncOpenAI, BadRequestError, OpenAI
from .instruction_prompts import (
    DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS,
    ENTITY_PROVENANCE_INSTRUCTIONS,
    ATTRIBUTE_PROVENANCE_INSTRUCTIONS,
    MEASUREMENT_EVENT_INSTRUCTIONS,
    EXTRACT_TEXT_VALUE_INSTRUCTIONS,
    EXTRACT_TABLE_VALUE_INSTRUCTIONS,
    STANDARDIZE_MEASUREMENTS_INSTRUCTIONS,
    STANDARDIZE_MEASUREMENTS_VALUE_ONLY_INSTRUCTIONS,
    PARSE_QUANTITY_INSTRUCTIONS,
    PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS,
    DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS,
)

# Orthogonal tags describing the shape of a parsed quantity (see
# PARSE_QUANTITY_INSTRUCTIONS and _parse_quantities()). Any combination is valid;
# this is not a closed enum of mutually exclusive labels -- a census of the raw
# MeasEval `mods` annotations (data/measeval/) found real quantities tagged with
# more than one of these simultaneously (e.g. IsApproximate + IsRange, 35
# occurrences), which a single mutually-exclusive qualifier field cannot represent.
QUALIFIER_TAGS = (
    "IsCount", "IsApproximate", "IsList", "IsRange",
    "IsMean", "IsMedian", "HasTolerance", "HasSD",
)


class ContextLengthExceededError(Exception):
    """Raised by _acall when the model server reports the prompt exceeded the
    model's context window. Distinguishes this specific, deterministic failure
    from every other exception _acall swallows into an empty-string result --
    see notes/scholarlm/builds/2026-08-20-per-document-isolation-01.md.
    """


# Confirmed against a real vLLM OpenAI-compatible error body (see the build note
# above): the response's `type`/`code` fields are generic ('BadRequestError', 400)
# and do not distinguish a context-length failure from any other 400 -- only the
# free-text message does.
_CONTEXT_LENGTH_MESSAGE_RE = re.compile(r"maximum context length", re.IGNORECASE)


def response_validator(response_structure, response):
    # Strip any leading prose or markdown fences before the JSON object/array.
    # Some frontier models prepend text like "Here is the JSON:" even when
    # response_format is set; raw_decode stops at the first complete top-level value.
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch in ('{', '['):
            try:
                obj, _ = decoder.raw_decode(response, i)
                response = json.dumps(obj)
                break
            except json.JSONDecodeError:
                continue
    pyd = response_structure.model_validate_json(response)
    out_dict = pyd.model_dump()
    return out_dict


class ListResponse(BaseModel):
    items: list[str]


class AttributeDetectionItem(BaseModel):
    """Detection result for a single attribute."""
    attribute_name: str
    explanation: str
    detected: bool
    terms: list[str]


class BatchAttributeDetectionResponse(BaseModel):
    """Batched detection results for all attributes."""
    items: list[AttributeDetectionItem]


class ProvenanceResponse(BaseModel):
    """Structured response for provenance detection on a single page."""
    explanation: str
    has_data: bool
    in_table: bool


class TextValueExtractionResponse(BaseModel):
    """Response for extracting a value from prose text."""
    explanation: str
    has_value: bool
    value: str | None = None
    units: str | None = None


class TableValueExtractionResponse(BaseModel):
    """Response for extracting a value from a table."""
    explanation: str
    has_value: bool
    row_index: str | None = None
    column_index: str | None = None
    units: str | None = None


class StandardizeResponse(BaseModel):
    """Response for standardizing an extracted measurement's units. Value
    standardization is a separate step (see ParseQuantityResponse) -- this
    response never touches value."""
    explanation: str
    units: str | None = None


class ParseQuantityResponse(BaseModel):
    """Response for parsing an extracted measurement value into its structured
    shape: qualifier tags plus whichever of point_value/lower/upper/list_values/
    tolerance/standard_deviation the reported value actually has.

    point_value/lower/upper/tolerance/standard_deviation accept a bare float
    in addition to str: a str-only schema forces vLLM's guided decoding to
    open a quote even when the model wants to emit an unquoted number, which
    causes gpt-oss-120b to stall mid-token and emit a malformed value on a
    large fraction of records (see notes/scholarlm/experiments/
    2026-09-24-gptoss120b-parsequantity-schema-diag-{01,02}.md). list_values
    stays list[str]-only -- untested, and it's a list shape, not a bare
    scalar."""
    explanation: str
    qualifiers: list[str]
    point_value: str | float | None = None
    lower: str | float | None = None
    upper: str | float | None = None
    list_values: list[str] | None = None
    tolerance: str | float | None = None
    standard_deviation: str | float | None = None


def check_quantity_consistency(parsed: dict) -> list[str]:
    """Non-fatal structural cross-checks between a parsed quantity's qualifier
    tags and its populated fields. Returns human-readable warnings for a caller
    to log -- never raises and never drops or alters data. A model tagging a
    quantity IsRange without populating lower/upper (or vice versa) is a sign the
    parse went wrong, but per-project policy we log and keep the record rather
    than discarding it or failing the run.
    """
    tags = set(parsed["qualifiers"])
    warnings = []
    has_range = parsed["lower"] is not None or parsed["upper"] is not None
    if "IsRange" in tags and not has_range:
        warnings.append("IsRange tagged but lower/upper both null")
    if has_range and "IsRange" not in tags:
        warnings.append("lower/upper populated but IsRange not tagged")
    if "IsList" in tags and not parsed["list_values"]:
        warnings.append("IsList tagged but list_values empty/null")
    if parsed["list_values"] and "IsList" not in tags:
        warnings.append("list_values populated but IsList not tagged")
    if "HasTolerance" in tags and parsed["tolerance"] is None:
        warnings.append("HasTolerance tagged but tolerance null")
    if "HasSD" in tags and parsed["standard_deviation"] is None:
        warnings.append("HasSD tagged but standard_deviation null")
    if {"IsMean", "IsMedian", "IsCount"} & tags and parsed["point_value"] is None and not has_range:
        warnings.append("central-tendency tag (IsMean/IsMedian/IsCount) but point_value null")
    return warnings


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            if math.isnan(obj):
                return None
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class BatchLLMBase:
    """Shared plumbing for a class that dispatches batched chat-completion
    requests against a vLLM/OpenAI-compatible endpoint: client setup, the
    async single-call helper with context-length isolation, batched dispatch
    with per-item retry, and the OCR page/table tag helpers used to slice
    ``<page number="N">``/``<table number="M">``-tagged text.

    Used by both ``MeasurementLM`` (and its subclasses) and ``TableCleaner``
    -- the two classes that need to make batched, retried LLM calls over
    OCR text, but otherwise have no pipeline logic in common.
    """

    def __init__(
        self,
        model_name: str,
        sampling_params: dict[str, any] = {},
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 32,
        use_extra_body: bool = True,
    ):
        self.model_name = model_name
        if sampling_params is None:
            self.sampling_params = {
                "temperature" : 0.90,
                "top_p" : 0.95,
                "top_k" : 64,
                "repetition_penalty" : 1.0,
                "max_tokens" : 2048,
                "enable_thinking": False
            }
        else:
            self.sampling_params = sampling_params

        self.max_concurrent = max_concurrent
        self.use_extra_body = use_extra_body
        self.max_prompt_tokens: int = 0
        self.token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "successful_calls": 0,
            "failed_calls": 0,
        }
        self.step_seconds: dict[str, float] = {}
        self.context_length_exceeded_docs: set[int] = set()
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=2400.0)

    # -----------------------------------------------------------------------
    # Core API call helpers
    # -----------------------------------------------------------------------

    async def _acall(
        self,
        messages: list[dict],
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        timeout: float = 600.0,
        extra_body: dict | None = None,
    ) -> str:
        """Single async API call to the vLLM OpenAI-compatible endpoint."""
        # Frontier models may require 'max_completion_tokens' (OpenAI o-series / gpt-5+)
        # instead of 'max_tokens'. Detect the right key from sampling_params so the
        # caller's explicit max_tokens value is forwarded under the correct parameter name.
        _TOKEN_KEYS = ("max_completion_tokens", "max_tokens")
        token_param = next((k for k in _TOKEN_KEYS if k in self.sampling_params), "max_tokens")
        token_value = max_tokens if max_tokens is not None else self.sampling_params.get(token_param, 2048)
        # Some frontier models (e.g. gpt-5-mini) reject temperature/top_p entirely.
        # Only include them when explicitly provided or present in sampling_params.
        effective_temp = temperature if temperature is not None else self.sampling_params.get("temperature")
        effective_top_p = self.sampling_params.get("top_p")
        kwargs: dict = {
            "model": self.model_name,
            "messages": messages,
            token_param: token_value,
        }
        if effective_temp is not None:
            kwargs["temperature"] = effective_temp
        if effective_top_p is not None:
            kwargs["top_p"] = effective_top_p
        if response_format is not None:
            kwargs["response_format"] = response_format
        if self.use_extra_body:
            extra = {}
            if "top_k" in self.sampling_params:
                extra["top_k"] = self.sampling_params["top_k"]
            if "repetition_penalty" in self.sampling_params:
                extra["repetition_penalty"] = self.sampling_params["repetition_penalty"]
            if "seed" in self.sampling_params:
                # Chat Completions has no top-level `seed` field for vLLM's
                # server; it must ride in extra_body like top_k/repetition_penalty.
                # Previously never forwarded here at all -- a configured seed
                # was silently a no-op for every caller of this base _acall
                # (extraction v1, ablations 1-6, NuExtract-v1, ChatExtract).
                # MeasurementLMv2 and NuExtract3 noticed and worked around it
                # locally (_seed_extra_body/extra_body["seed"]); fixing it here
                # covers everyone else instead of requiring the same patch
                # per subclass. Does not by itself restore run-to-run
                # determinism -- MeasurementLMv2 already forwarded seed this
                # way and still diverged (see 2026-09-21-qualifiers-seeddet-01);
                # the residual cause is temperature-amplified vLLM float noise
                # compounding across a chained multi-step pipeline, not a
                # missing seed.
                extra["seed"] = self.sampling_params["seed"]
            chat_template_kwargs = {}
            if "enable_thinking" in self.sampling_params:
                # Disable thinking by default for extraction tasks
                chat_template_kwargs["enable_thinking"] = self.sampling_params["enable_thinking"]
            if "reasoning_effort" in self.sampling_params:
                # gpt-oss's harmony chat template kwarg (low/medium/high). See
                # experiments/model-configs/extraction/gpt-oss-120b.yaml's own
                # comment for why this is pinned rather than left at the
                # template's implicit default.
                chat_template_kwargs["reasoning_effort"] = self.sampling_params["reasoning_effort"]
            if chat_template_kwargs:
                extra["chat_template_kwargs"] = chat_template_kwargs
            if extra_body:
                for key, value in extra_body.items():
                    if key == "chat_template_kwargs" and isinstance(value, dict) and isinstance(extra.get(key), dict):
                        extra[key] = {**extra[key], **value}
                    else:
                        extra[key] = value
            if extra:
                kwargs["extra_body"] = extra
        try:
            response = await self.async_client.chat.completions.create(
                **kwargs, timeout=timeout
            )
            self.token_usage["successful_calls"] += 1
            if response.usage is not None:
                if response.usage.prompt_tokens:
                    self.token_usage["prompt_tokens"] += response.usage.prompt_tokens
                    if response.usage.prompt_tokens > self.max_prompt_tokens:
                        self.max_prompt_tokens = response.usage.prompt_tokens
                if response.usage.completion_tokens:
                    self.token_usage["completion_tokens"] += response.usage.completion_tokens
            return response.choices[0].message.content
        except BadRequestError as e:
            message = e.body.get("message", "") if isinstance(e.body, dict) else str(e)
            if _CONTEXT_LENGTH_MESSAGE_RE.search(message):
                self.token_usage["failed_calls"] += 1
                raise ContextLengthExceededError(message) from e
            print(f"API call failed: {e}")
            self.token_usage["failed_calls"] += 1
            return ""
        except Exception as e:
            print(f"API call failed: {e}")
            self.token_usage["failed_calls"] += 1
            return ""

    def _call_batch(
        self,
        message_sets: list[list[dict]],
        response_format: dict | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = 2,
        validator: Callable[[str], Any] | None = None,
        timeout: float = 600.0,
        max_concurrent: int | None = None,
        extra_body: dict | None = None,
    ) -> list[str | ContextLengthExceededError]:
        """Dispatch all message sets concurrently; return response texts in order.

        If max_retries > 0, any response that is empty or causes validator to raise
        is retried up to max_retries times with exponential backoff between rounds.
        validator is called only to detect failure — its return value is ignored.
        max_concurrent overrides self.max_concurrent for this call only, allowing
        per-step concurrency tuning without changing the instance default.
        extra_body is forwarded unchanged to every request in this batch (merged
        under self.use_extra_body, same as the sampling-params-derived fields).

        A ContextLengthExceededError from an individual call is isolated to that
        call's slot in the returned list (never retried, never propagated into
        asyncio.gather) rather than failing the whole batch. Every other exception
        still propagates and crashes loudly.
        """
        async def _run():
            sem = asyncio.Semaphore(max_concurrent if max_concurrent is not None else self.max_concurrent)

            async def _limited(msgs):
                async with sem:
                    try:
                        return await self._acall(msgs, response_format, temperature, max_tokens, timeout, extra_body)
                    except ContextLengthExceededError as e:
                        return e

            results = list(await asyncio.gather(*[_limited(msgs) for msgs in message_sets]))

            for attempt in range(max_retries):
                failed = []
                for i, resp in enumerate(results):
                    if isinstance(resp, ContextLengthExceededError):
                        continue
                    if not resp:
                        failed.append(i)
                        continue
                    if validator is not None:
                        try:
                            validator(resp)
                        except Exception:
                            failed.append(i)

                if not failed:
                    break

                print(f"Retrying {len(failed)} failed responses (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(2 ** attempt)
                retried = await asyncio.gather(*[_limited(message_sets[i]) for i in failed])
                for local_i, global_i in enumerate(failed):
                    results[global_i] = retried[local_i]

            return results

        return asyncio.run(_run())

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _get_page_numbers(self, context: str) -> list[int]:
        """Return sorted list of page numbers found in *context*."""
        return [int(p) for p in re.findall(r'<page number="(\d+)">', context)]

    def _get_page_text(self, context: str, page_number: int) -> str:
        """Extract the text content of a single page from *context*."""
        tag = f'<page number="{page_number}">'
        start = context.find(tag)
        if start == -1:
            return ""
        start += len(tag)
        end = context.find('</page>', start)
        if end == -1:
            return ""
        return context[start:end].strip()

    def _get_table_numbers_on_page(self, page_text: str) -> list[int]:
        """Return sorted list of table numbers found in *page_text*."""
        return sorted(int(t) for t in re.findall(r'<table number="(\d+)">', page_text))


class MeasurementLM(BatchLLMBase):
    """
    A language model class designed for organized collection of measurements from scientific text.

    Args:
        model_name (str): The name or path of the pre-trained language model from the huggingface
            collection.
        entity_identification_prompt (str): The prompt template for entity identification.
        entity_identification_schema (BaseModel): The pydantic schema for entity identification.
        attribute_info_dict (dict[str, any]): A dictionary containing information about the
            attributes to be measured. Each key is an attribute name, and each value is a dict
            with at least a 'description' key and optionally a 'units' key.
        sampling_params (dict[str, any]): A dictionary of sampling parameters for text generation.
        collect_attribute_terms (bool): Whether to request and forward per-attribute
            terminology ("terms") during attribute detection. Meaningful only for
            datasets with a real closed attribute vocabulary (pond/nfix/supermat),
            where "terms" means synonyms/abbreviations for a named attribute. For
            datasets whose attribute space collapses to a single abstract bucket
            (e.g. measeval's "measurement"), there is no textual referent for
            "terminology used to refer to this attribute", and models asked for it
            tend to dump unrelated numeric values instead. Set to False to discard
            whatever the model returns for `terms` and omit the "Terminology used
            for the attribute: ..." line from downstream prompts, without changing
            the detection prompt/schema itself (so this has no effect on other
            datasets). Defaults to True for backward compatibility.
        extraction_mode ("pipeline" | "direct"): "pipeline" (default) runs the full
            seven-step pipeline unchanged. "direct" replaces entity/attribute
            detection, provenance, and value extraction with a single per-document
            LLM call (see `_extract_triples`), still followed by `_standardize` and
            `_deduplicate`. Requires `direct_extraction_schema` and
            `direct_extraction_prompt` when set to "direct".
        direct_extraction_schema (BaseModel | None): Flat pydantic schema combining
            entity, event, attribute, value, and units fields, used only when
            `extraction_mode="direct"`.
        direct_extraction_prompt (str | None): Dataset-specific prompt describing
            entities, events, and attributes for the single direct-extraction call;
            used only when `extraction_mode="direct"`.
        parse_quantities_context ("full" | "value_only"): Controls what
            `_parse_quantities()` shows the model. "full" (default) is
            unchanged: source-text context, entity description, attribute
            description, units, and the extracted value. "value_only" gives
            it nothing but the extracted value string -- see
            PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS and
            2026-09-22-pond-parsequantities-valueonly-01.
        standardize_context ("full" | "value_only"): Controls what
            `_standardize()` shows the model. "full" (default) is unchanged:
            source-text context, entity description, attribute description,
            attribute terminology, available units, and the extracted
            measurement/units. "value_only" gives it only the available
            units list and the extracted measurement/units -- no source
            text, entity description, attribute description, or
            terminology. The standardization decision rule itself (best-
            matching notational variant, else unchanged, else null) is the
            same in both modes -- see STANDARDIZE_MEASUREMENTS_VALUE_ONLY_INSTRUCTIONS.
    """
    def __init__(
        self,
        model_name: str,
        entity_identification_prompt: str,
        entity_identification_schema: BaseModel,
        attribute_info_dict: dict[str, any],
        sampling_params: dict[str, any] = {},
        api_base: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        max_concurrent: int = 32,
        measurement_event_schema: BaseModel | None = None,
        measurement_event_prompt: str | None = None,
        use_extra_body: bool = True,
        collect_attribute_terms: bool = True,
        extraction_mode: str = "pipeline",
        direct_extraction_schema: BaseModel | None = None,
        direct_extraction_prompt: str | None = None,
        parse_quantities_context: str = "full",
        standardize_context: str = "full",
    ):
        super().__init__(
            model_name=model_name,
            sampling_params=sampling_params,
            api_base=api_base,
            api_key=api_key,
            max_concurrent=max_concurrent,
            use_extra_body=use_extra_body,
        )
        self.entity_identification_prompt = entity_identification_prompt
        self.entity_identification_schema = entity_identification_schema
        self.attribute_info_dict = attribute_info_dict
        self.measurement_event_schema = measurement_event_schema
        self.measurement_event_prompt = measurement_event_prompt
        self.collect_attribute_terms = collect_attribute_terms
        if extraction_mode not in ("pipeline", "direct"):
            raise ValueError(
                f"extraction_mode must be 'pipeline' or 'direct', got {extraction_mode!r}."
            )
        self.extraction_mode = extraction_mode
        self.direct_extraction_schema = direct_extraction_schema
        self.direct_extraction_prompt = direct_extraction_prompt
        if parse_quantities_context not in ("full", "value_only"):
            raise ValueError(
                f"parse_quantities_context must be 'full' or 'value_only', got {parse_quantities_context!r}."
            )
        self.parse_quantities_context = parse_quantities_context
        if standardize_context not in ("full", "value_only"):
            raise ValueError(
                f"standardize_context must be 'full' or 'value_only', got {standardize_context!r}."
            )
        self.standardize_context = standardize_context

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _auto_provenance_for_single_page(
        self, context: str, pages: list[int]
    ) -> list[dict] | None:
        """Provenance entries for a single-page document, without a per-page query.

        The per-page provenance loop exists to identify *which* page (of several)
        carries an entity's/attribute's data. On a single-page document there is no
        "which page" to resolve -- the one page is the only candidate, and the
        surrounding pipeline (entity identification requiring a measurement to be
        present, or measeval's document-level "measurement" gate) already
        established that data exists somewhere in the document. A "does this page
        have data" round-trip only adds a chance for the model to answer "no" and
        silently drop an entity that's known to be there. So every representation
        the page could take -- prose, and each table on it -- is recorded directly.

        Returns:
            A list of provenance entries (possibly empty, if the page has no text)
            for a single-page document, or ``None`` for a multi-page/empty one,
            signalling the caller should fall back to the per-page query loop.
        """
        if len(pages) != 1:
            return None
        page_number = pages[0]
        page_text = self._get_page_text(context, page_number)
        if not page_text:
            return []
        entries = [{'page': page_number, 'table': None}]
        entries += [
            {'page': page_number, 'table': t}
            for t in self._get_table_numbers_on_page(page_text)
        ]
        return entries

    # -----------------------------------------------------------------------
    # Step 1: Document Level entity extraction
    # -----------------------------------------------------------------------

    def _extract_entities(self, max_tokens: int = 8192):
        """
        Extracts entities from documents in two passes:
        1. Full-context extraction using the entity identification prompt and schema.
        2. Per-table enrichment that finds new entities or fills in missing fields
           on existing entities.

        Reads from self.data (one record per document) and returns one record per
        (document, entity) with entity schema fields merged in.
        """
        from pydantic import create_model

        IdentificationList = create_model(
            "IdentificationList",
            items=(list[self.entity_identification_schema], ...),
        )
        identification_list_json = IdentificationList.model_json_schema()

        # --- Pass 1: Full-context entity identification ---
        messages = []
        for i, datapoint in enumerate(self.data):
            instructions = self.entity_identification_prompt
            context = datapoint['context']
            query = (
                "Scan the full context and identify all distinct entities of the described type. "
                "Return one item per entity, populating all fields using only information "
                "explicitly stated in the text. Do not infer or fabricate any field values."
            )
            prompt = (
                f"## INSTRUCTIONS:\n{instructions}\n\n## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "identification_list",
                "schema": identification_list_json,
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=4,
            max_tokens=max_tokens,
            max_concurrent=2,
            timeout=300,
            validator=lambda r: response_validator(IdentificationList, r),
        )

        # Build per-document entity lists
        doc_entities: dict[int, list[dict]] = {}
        for i, r in enumerate(response_texts):
            if isinstance(r, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(i)
                doc_entities[i] = []
                continue
            try:
                resp_validated = response_validator(IdentificationList, r)
            except Exception as e:
                print(f"Validation error in identification response: {e}")
                print(f"Response text: {r}")
                resp_validated = {'items': []}

            doc_entities[i] = list(resp_validated['items'])

        # --- Build output: one record per (document, entity) ---
        entity_data = []
        for i, datapoint in enumerate(self.data):
            for j, entity in enumerate(doc_entities.get(i, [])):
                entity_id = f"doc_{i}_entity_{j}"
                entity_data.append(datapoint | entity | {'entity_id': entity_id})

        return entity_data
    

    # -----------------------------------------------------------------------
    # Step 1b: Entity provenance
    # -----------------------------------------------------------------------

    def _entity_provenance(self, entity_data):
        """
        For each unique (document, entity), determine which pages contain
        data for that entity.

        Args:
            entity_data: list of entity records from _extract_entities()

        Returns:
            dict mapping (doc_id, entity_id) -> list[{"page": int, "table": int|None}]
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        # Collect unique (doc_id, entity_id) pairs with their context and description
        unique_entities = {}
        for record in entity_data:
            key = (record['document_id'], record['entity_id'])
            if key not in unique_entities:
                unique_entities[key] = record

        messages = []
        message_ids = []  # (doc_id, entity_id, page_number)
        provenance = {}

        for (doc_id, entity_id), record in unique_entities.items():
            context = record['context']
            entity_description = {k: v for k, v in record.items() if k in entity_fields}
            pages = self._get_page_numbers(context)

            auto = self._auto_provenance_for_single_page(context, pages)
            if auto is not None:
                if auto:
                    provenance[(doc_id, entity_id)] = auto
                continue

            for p in pages:
                page_text = self._get_page_text(context, p)
                if not page_text:
                    continue

                query = (
                    f"Entity description: {entity_description}\n\n"
                    f"Does this page contain directly reported numerical measurements "
                    f"for the described entity? If yes, indicate whether the data "
                    f"appears in a table or in prose text.\n\n"
                )
                prompt = (
                    f"## INSTRUCTIONS:\n{ENTITY_PROVENANCE_INSTRUCTIONS}\n\n"
                    f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                )
                messages.append([{"role": "user", "content": prompt}])
                message_ids.append((doc_id, entity_id, p))

        if not messages:
            return provenance

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "provenance_response",
                "schema": ProvenanceResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=2048,
            max_concurrent=32,
            timeout=120,
            validator=lambda r: response_validator(ProvenanceResponse, r),
        )

        for msg_idx, resp in enumerate(response_texts):
            doc_id, entity_id, page_number = message_ids[msg_idx]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(doc_id)
                continue
            try:
                result = response_validator(ProvenanceResponse, resp)
            except Exception as e:
                print(f"Validation error in entity provenance response: {e}")
                print(f"Response text: {resp}")
                continue

            if result.get('has_data'):
                key = (doc_id, entity_id)
                if result.get('in_table'):
                    page_text = self._get_page_text(
                        unique_entities[(doc_id, entity_id)]['context'],
                        page_number,
                    )
                    for t in self._get_table_numbers_on_page(page_text):
                        provenance.setdefault(key, []).append({
                            'page': page_number,
                            'table': t,
                        })
                else:
                    provenance.setdefault(key, []).append({
                        'page': page_number,
                        'table': None,
                    })

        return provenance


    # -----------------------------------------------------------------------
    # Step 2: Document-level attribute detection
    # -----------------------------------------------------------------------

    def _format_attribute_list(self, attr_names=None):
        """Format attributes as a numbered list for inclusion in prompts.

        Args:
            attr_names: Subset of attribute names to include. If None, includes all.
        """
        if attr_names is None:
            attr_names = list(self.attribute_info_dict.keys())
        lines = []
        for idx, attr_name in enumerate(attr_names, 1):
            desc = self.attribute_info_dict[attr_name].get('description', '')
            lines.append(f"{idx}. {attr_name}: {desc}")
        return "\n".join(lines)

    def _detect_attributes(self):
        """
        Document-level attribute detection in two phases:
        A. Batched full-context detection — one prompt per document evaluating
           all attributes at once, with inline term identification.
        B. Batched per-table fallback — one prompt per (document, table) for
           attributes not yet detected in Phase A.

        Reads from self.data (one record per document) and returns a dict
        mapping document index to detected attributes with their terms:
            {doc_idx: {attr_name: [term, ...], ...}, ...}
        """
        attr_names = list(self.attribute_info_dict.keys())
        attribute_list_text = self._format_attribute_list()

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "batch_attribute_detection",
                "schema": BatchAttributeDetectionResponse.model_json_schema(),
            },
        }

        # --- Phase A: Batched full-context detection (one prompt per document) ---
        messages = []
        message_ids = []  # doc_idx
        for i, datapoint in enumerate(self.data):
            context = datapoint['context']
            query = (
                f"Attributes to evaluate:\n{attribute_list_text}\n\n"
                f"For each attribute listed above, determine whether the document "
                f"contains any direct numerical measurements for that attribute. "
                f"Return one item per attribute using the exact attribute name.\n\n"
            )
            prompt = (
                f"## INSTRUCTIONS:\n{DETECT_ATTRIBUTES_BATCH_INSTRUCTIONS}\n\n"
                f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])
            message_ids.append(i)

        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=4,
            max_tokens=4096,
            max_concurrent=2,
            timeout=300,
            validator=lambda r: response_validator(BatchAttributeDetectionResponse, r),
        )

        # Build detection results and attribute terms per document
        detection_results: dict[int, dict[str, bool]] = {}
        attribute_terms: dict[int, dict[str, list[str]]] = {}

        for msg_idx, resp in enumerate(response_texts):
            doc_idx = message_ids[msg_idx]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(doc_idx)
                detection_results[doc_idx] = {a: False for a in attr_names}
                continue
            try:
                batch = response_validator(BatchAttributeDetectionResponse, resp)
            except Exception as e:
                print(f"Validation error in batched attribute detection response: {e}")
                print(f"Response text: {resp}")
                detection_results[doc_idx] = {a: False for a in attr_names}
                continue

            responded_attrs = {}
            for item in batch['items']:
                responded_attrs[item['attribute_name']] = item

            detection_results[doc_idx] = {}
            attribute_terms[doc_idx] = {}
            for attr_name in attr_names:
                item = responded_attrs.get(attr_name)
                if item and item.get('detected', False):
                    detection_results[doc_idx][attr_name] = True
                    attribute_terms[doc_idx][attr_name] = (
                        item.get('terms', []) if self.collect_attribute_terms else []
                    )
                else:
                    detection_results[doc_idx][attr_name] = False

        # --- Build output: {doc_idx: {attr_name: terms}} for detected attrs ---
        doc_attributes: dict[int, dict[str, list[str]]] = {}
        for doc_idx in range(len(self.data)):
            detected = {}
            for attr_name in attr_names:
                if detection_results.get(doc_idx, {}).get(attr_name, False):
                    detected[attr_name] = attribute_terms.get(doc_idx, {}).get(attr_name, [])
            if detected:
                doc_attributes[doc_idx] = detected

        return doc_attributes


    # -----------------------------------------------------------------------
    # Step 2b: Attribute provenance
    # -----------------------------------------------------------------------

    def _attribute_provenance(self, doc_attributes):
        """
        For each (document, detected attribute), determine which pages
        contain data for that attribute.

        Args:
            doc_attributes: dict from _detect_attributes()
                {doc_idx: {attr_name: [terms]}}

        Returns:
            dict mapping (doc_id, attr_name) -> list[{"page": int, "table": int|None}]
        """
        messages = []
        message_ids = []  # (doc_id, attr_name, page_number)
        provenance = {}

        for doc_idx, attrs in doc_attributes.items():
            # doc_idx may be int or str depending on caller
            doc_idx_int = int(doc_idx)
            context = self.data[doc_idx_int]['context']
            pages = self._get_page_numbers(context)
            auto = self._auto_provenance_for_single_page(context, pages)

            for attr_name, terms in attrs.items():
                if auto is not None:
                    if auto:
                        provenance[(doc_idx_int, attr_name)] = auto
                    continue

                attr_description = self.attribute_info_dict[attr_name].get('description', '')

                for p in pages:
                    page_text = self._get_page_text(context, p)
                    if not page_text:
                        continue

                    terms_line = (
                        f"Terminology used for the attribute: {terms}\n\n"
                        if self.collect_attribute_terms else ""
                    )
                    query = (
                        f"Attribute: {attr_name}\n"
                        f"Attribute description: {attr_description}\n"
                        f"{terms_line}"
                        f"Does this page contain directly reported numerical measurements "
                        f"for the described attribute? If yes, indicate whether the data "
                        f"appears in a table or in prose text.\n\n"
                    )
                    prompt = (
                        f"## INSTRUCTIONS:\n{ATTRIBUTE_PROVENANCE_INSTRUCTIONS}\n\n"
                        f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((doc_idx_int, attr_name, p))

        if not messages:
            return provenance

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "provenance_response",
                "schema": ProvenanceResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=2048,
            max_concurrent=32,
            timeout=120,
            validator=lambda r: response_validator(ProvenanceResponse, r),
        )

        for msg_idx, resp in enumerate(response_texts):
            doc_id, attr_name, page_number = message_ids[msg_idx]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(doc_id)
                continue
            try:
                result = response_validator(ProvenanceResponse, resp)
            except Exception as e:
                print(f"Validation error in attribute provenance response: {e}")
                print(f"Response text: {resp}")
                continue

            if result.get('has_data'):
                key = (doc_id, attr_name)
                if result.get('in_table'):
                    page_text = self._get_page_text(
                        self.data[doc_id]['context'],
                        page_number,
                    )
                    for t in self._get_table_numbers_on_page(page_text):
                        provenance.setdefault(key, []).append({
                            'page': page_number,
                            'table': t,
                        })
                else:
                    provenance.setdefault(key, []).append({
                        'page': page_number,
                        'table': None,
                    })

        return provenance


    # -----------------------------------------------------------------------
    # Step 2c: Measurement event resolution (optional)
    # -----------------------------------------------------------------------

    def _resolve_events(self, entity_data, doc_attributes, entity_prov, attr_prov):
        """
        For each (entity, attribute, page) intersection, enumerate the distinct
        measurement events present on that page.

        Only runs when ``self.measurement_event_schema`` is set.  Returns an empty
        dict otherwise.

        Args:
            entity_data: list of entity records from _extract_entities().
            doc_attributes: dict from _detect_attributes().
            entity_prov: dict from _entity_provenance().
            attr_prov: dict from _attribute_provenance().

        Returns:
            dict mapping (doc_id, entity_id, attr_name, page_number) ->
            list[event_dict].  An empty list means no events were found on
            that page (caller falls back to a single all-None default event).
        """
        if self.measurement_event_schema is None:
            return {}

        from pydantic import create_model

        EventList = create_model(
            "EventList",
            items=(list[self.measurement_event_schema], ...),
        )
        event_list_json = EventList.model_json_schema()
        entity_fields = list(self.entity_identification_schema.model_fields.keys())

        unique_entities = {}
        for record in entity_data:
            key = (record['document_id'], record['entity_id'])
            if key not in unique_entities:
                unique_entities[key] = record

        messages = []
        message_ids = []  # (doc_id, entity_id, attr_name, page_number)

        for (doc_id, entity_id), record in unique_entities.items():
            context = record['context']
            entity_description = {k: v for k, v in record.items() if k in entity_fields}

            for attr_name, _terms in doc_attributes.get(doc_id, {}).items():
                attr_description = self.attribute_info_dict[attr_name].get('description', '')

                # Collect all pages where both entity and attribute have provenance
                e_pages = {entry['page'] for entry in entity_prov.get((doc_id, entity_id), [])}
                a_pages = {entry['page'] for entry in attr_prov.get((doc_id, attr_name), [])}
                intersecting_pages = sorted(e_pages & a_pages)

                for p in intersecting_pages:
                    page_text = self._get_page_text(context, p)
                    if not page_text:
                        continue

                    query = (
                        f"Entity description: {entity_description}\n"
                        f"Attribute: {attr_name}\n"
                        f"Attribute description: {attr_description}\n"
                        f"Enumerate all distinct measurement events for the given entity and attribute.\n\n"
                    )
                    prompt = (
                        f"## INSTRUCTIONS:\n{MEASUREMENT_EVENT_INSTRUCTIONS}\n\n"
                        f"## EVENT DETAILS:\n{self.measurement_event_prompt}\n\n"
                        f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((doc_id, entity_id, attr_name, p))

        if not messages:
            return {}

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "event_list",
                "schema": event_list_json,
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=8192,
            max_concurrent=8,
            timeout=300,
            validator=lambda r: response_validator(EventList, r),
        )

        event_resolution = {}
        for msg_idx, resp in enumerate(response_texts):
            key = message_ids[msg_idx]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(key[0])
                event_resolution[key] = []
                continue
            try:
                result = response_validator(EventList, resp)
            except Exception as e:
                print(f"Validation error in event resolution response: {e}")
                print(f"Response text: {resp}")
                event_resolution[key] = []
                continue
            event_resolution[key] = result['items']

        return event_resolution


    # -----------------------------------------------------------------------
    # Step 3: Extract values from text (per-page)
    # -----------------------------------------------------------------------

    def _extract_values_from_text(self, entity_data, doc_attributes, entity_prov, attr_prov, event_resolution=None):
        """
        Extracts measurement values from prose text using provenance intersection.

        For each (entity, attribute) pair per document, finds pages where BOTH
        have provenance with table=None, and only prompts for those pages.

        Args:
            entity_data: list of entity records from _extract_entities()
            doc_attributes: dict from _detect_attributes()
            entity_prov: dict from _entity_provenance()
            attr_prov: dict from _attribute_provenance()

        Returns records with 'value', 'units', 'page_number', and 'source'='text'.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, page_number)

        for record in entity_data:
            doc_id = record['document_id']
            entity_id = record['entity_id']
            context = record['context']

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                # Provenance intersection: find pages where both entity and attribute
                # have provenance with table=None (prose text)
                e_pages = {
                    entry['page'] for entry in entity_prov.get((doc_id, entity_id), [])
                    if entry['table'] is None
                }
                a_pages = {
                    entry['page'] for entry in attr_prov.get((doc_id, attr_name), [])
                    if entry['table'] is None
                }
                intersecting_pages = sorted(e_pages & a_pages)

                if not intersecting_pages:
                    continue

                attr_description = self.attribute_info_dict[attr_name]['description']
                entity_description = {k: v for k, v in record.items() if k in entity_fields}
                unit_options = self.attribute_info_dict[attr_name].get('units', [])

                pair_record = record | {
                    'attribute': attr_name,
                    'attribute_terms': terms,
                }

                units_guidance = ""
                if unit_options:
                    units_guidance = (
                        f"Preferred unit options: {unit_options}. "
                        f"Strongly prioritize choosing the best option from this list. "
                        f"If none of the options fit, specify the unit exactly as it appears in the text.\n"
                    )

                for p in intersecting_pages:
                    page_text = self._get_page_text(context, p)
                    if not page_text:
                        continue

                    # Determine measurement events for this (entity, attribute, page)
                    if event_resolution is not None:
                        events = event_resolution.get((doc_id, entity_id, attr_name, p), [])
                        if not events:
                            events = [{f: None for f in self.measurement_event_schema.model_fields}]
                    else:
                        events = [None]

                    for event in events:
                        event_record = pair_record | (event if event is not None else {})
                        event_context = ""
                        if event and any(v is not None for v in event.values()):
                            event_context = f"Measurement event context: {event}\n"
                        terms_line = (
                            f"Terminology used for the attribute: {terms}\n"
                            if self.collect_attribute_terms else ""
                        )

                        query = (
                            f"Entity description: {entity_description}\n"
                            f"Attribute description: {attr_description}\n"
                            f"{terms_line}"
                            f"{event_context}"
                            f"{units_guidance}\n"
                            f"Does this page contain a measured value for the given entity, attribute, and event? "
                            f"If yes, extract the value and its units.\n\n"
                        )
                        prompt = (
                            f"## INSTRUCTIONS:\n{EXTRACT_TEXT_VALUE_INSTRUCTIONS}\n\n"
                            f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                        )
                        messages.append([{"role": "user", "content": prompt}])
                        message_ids.append((event_record, p))

        if not messages:
            return []

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "text_value_extraction",
                "schema": TextValueExtractionResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=2048,
            max_concurrent=32,
            timeout=120,
            validator=lambda r: response_validator(TextValueExtractionResponse, r),
        )

        text_values = []
        for msg_idx, resp in enumerate(response_texts):
            pair_record, page_number = message_ids[msg_idx]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(pair_record['document_id'])
                continue
            try:
                result = response_validator(TextValueExtractionResponse, resp)
            except Exception as e:
                print(f"Validation error in text value extraction response: {e}")
                print(f"Response text: {resp}")
                continue

            if result.get('has_value') and result.get('value') is not None:
                page_text = self._get_page_text(pair_record['context'], page_number)

                text_values.append(
                    pair_record | {
                        'context': page_text,
                        'value': result['value'],
                        'units': result.get('units'),
                        'page_number': page_number,
                        'source': 'text',
                    }
                )

        return text_values


    # -----------------------------------------------------------------------
    # Step 4: Extract values from tables
    # -----------------------------------------------------------------------

    def _extract_values_from_tables(self, entity_data, doc_attributes, entity_prov, attr_prov, event_resolution=None):
        """
        Extracts measurement values from HTML tables using provenance intersection.

        For each (entity, attribute) pair per document, finds table numbers where
        BOTH have provenance, and only prompts for those tables.

        Args:
            entity_data: list of entity records from _extract_entities()
            doc_attributes: dict from _detect_attributes()
            entity_prov: dict from _entity_provenance()
            attr_prov: dict from _attribute_provenance()

        Returns records with 'value', 'units', 'table_number', 'row_index',
        'column_index', and 'source'='table'.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_ids = []  # (record_dict, table_number)
        table_cache = {}  # (doc_id, table_number) -> (table_text, row_names, column_names)

        def _get_table(context, t, doc_id):
            """Parse and cache a table, returning (table_text, row_names, column_names) or None."""
            cache_key = (doc_id, t)
            if cache_key in table_cache:
                return table_cache[cache_key]
            tag = f'<table number="{t}">'
            table_tag_start = context.find(tag)
            if table_tag_start == -1:
                return None
            table_content_start = table_tag_start + len(tag)
            table_end = context.find('</table>', table_content_start)
            table_text = context[table_tag_start:table_end + len('</table>')].strip()
            if not table_text:
                return None
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]
                row_names = table_df.loc[:, "index"].to_list() if 'index' in table_df.columns else []
                row_names = [str(name) for name in row_names]
                column_names = [str(name) for name in table_df.columns.tolist()]
                column_names = [name for name in column_names if name != 'index']
            except:
                print(f"Error parsing table {t} in doc {doc_id}.")
                return None
            table_cache[cache_key] = (table_text, row_names, column_names)
            return table_cache[cache_key]

        for record in entity_data:
            doc_id = record['document_id']
            entity_id = record['entity_id']
            context = record['context']

            for attr_name, terms in doc_attributes.get(doc_id, {}).items():
                # Provenance intersection: find table numbers where both entity
                # and attribute have provenance
                entity_prov_entries = entity_prov.get((doc_id, entity_id), [])
                e_tables = {
                    entry['table'] for entry in entity_prov_entries
                    if entry['table'] is not None
                }
                a_tables = {
                    entry['table'] for entry in attr_prov.get((doc_id, attr_name), [])
                    if entry['table'] is not None
                }
                intersecting_tables = sorted(e_tables & a_tables)

                # Map table number -> page number for provenance attribution.
                table_to_page = {
                    entry['table']: entry['page']
                    for entry in entity_prov_entries
                    if entry['table'] is not None
                }

                if not intersecting_tables:
                    continue

                attr_description = self.attribute_info_dict[attr_name]['description']
                entity_description = {k: v for k, v in record.items() if k in entity_fields}
                unit_options = self.attribute_info_dict[attr_name].get('units', [])

                pair_record = record | {
                    'attribute': attr_name,
                    'attribute_terms': terms,
                }

                units_guidance = ""
                if unit_options:
                    units_guidance = (
                        f"Preferred unit options: {unit_options}. "
                        f"Strongly prioritize choosing the best option from this list. "
                        f"If none of the options fit, specify the unit exactly as it appears in the text.\n"
                    )

                for t in intersecting_tables:
                    parsed = _get_table(context, t, doc_id)
                    if parsed is None:
                        continue
                    table_text, row_names, column_names = parsed
                    table_page_number = table_to_page.get(t)

                    # Determine measurement events for this (entity, attribute, page)
                    if event_resolution is not None:
                        events = event_resolution.get(
                            (doc_id, entity_id, attr_name, table_page_number), []
                        )
                        if not events:
                            events = [{f: None for f in self.measurement_event_schema.model_fields}]
                    else:
                        events = [None]

                    for event in events:
                        event_record = pair_record | (event if event is not None else {})
                        event_context = ""
                        if event and any(v is not None for v in event.values()):
                            event_context = f"Measurement event context: {event}\n"
                        terms_line = (
                            f"Terminology used for the attribute: {terms}\n"
                            if self.collect_attribute_terms else ""
                        )

                        query = (
                            f"Entity description: {entity_description}\n"
                            f"Attribute description: {attr_description}\n"
                            f"{terms_line}"
                            f"{event_context}"
                            f"{units_guidance}"
                            f"Row names in the table: {row_names}\n"
                            f"Column names in the table: {column_names}\n\n"
                            f"Does this table contain a measured value for the given entity, attribute, and event? "
                            f"If yes, provide the corresponding row_index and column_index names, and the units.\n\n"
                        )
                        prompt = (
                            f"## INSTRUCTIONS:\n{EXTRACT_TABLE_VALUE_INSTRUCTIONS}\n\n"
                            f"## CONTEXT:\n{table_text}\n\n## QUERY:\n{query}"
                        )
                        messages.append([{"role": "user", "content": prompt}])
                        message_ids.append((event_record, t, table_page_number))

        if not messages:
            return []

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "table_value_extraction",
                "schema": TableValueExtractionResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=2048,
            max_concurrent=32,
            timeout=120,
            validator=lambda r: response_validator(TableValueExtractionResponse, r),
        )

        table_values = []
        for msg_idx, resp in enumerate(response_texts):
            pair_record, table_number, page_number = message_ids[msg_idx]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(pair_record['document_id'])
                continue
            try:
                result = response_validator(TableValueExtractionResponse, resp)
            except Exception as e:
                print(f"Validation error in table value extraction response: {e}")
                print(f"Response text: {resp}")
                continue

            if not result.get('has_value'):
                continue
            row_index = result.get('row_index')
            column_index = result.get('column_index')
            if row_index is None or column_index is None:
                continue

            # Extract cell value from the table using pandas
            doc_id = pair_record['document_id']
            parsed = table_cache.get((doc_id, table_number))
            if parsed is None:
                continue
            table_text, row_names, column_names = parsed
            try:
                table_dfs = pd.read_html(StringIO(table_text))
                table_df = table_dfs[0]

                # Ensure row and column indices are strings for matching
                table_df.columns = [str(c) for c in table_df.columns]
                if 'index' in table_df.columns:
                    table_df['index'] = table_df['index'].astype(str)

                matched_rows = table_df.loc[table_df["index"] == row_index][column_index]
                if len(matched_rows) == 0:
                    print("No matching row found in table extraction.")
                    val = None
                elif len(matched_rows) == 1:
                    val = matched_rows.item()
                else:
                    print("Multiple matching rows found in table extraction, taking the first match.")
                    val = matched_rows.iloc[0]
            except:
                print(f"Error extracting value from table {table_number} in doc {doc_id}.")
                val = None

            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                table_values.append(
                    pair_record | {
                        'context': table_text,
                        'value': val,
                        'units': result.get('units'),
                        'page_number': page_number,
                        'table_number': table_number,
                        'row_index': row_index,
                        'column_index': column_index,
                        'source': 'table',
                    }
                )

        return table_values


    # -----------------------------------------------------------------------
    # Direct extraction (extraction_mode="direct"): single call per document
    # -----------------------------------------------------------------------

    def _extract_triples(self):
        """
        Extract all measurement records from each document in a single LLM call.

        Uses self.direct_extraction_schema (a flat Pydantic model combining entity,
        event, attribute, value, and units fields) and self.direct_extraction_prompt
        (a dataset-specific block describing entities, events, and attributes). The
        schema's ``attribute`` field is unconstrained free text (not a Literal/enum),
        so a response can name an attribute outside ``self.attribute_info_dict`` —
        this is retried (via the validator below, same as any other malformed
        response) and, if still present after retries are exhausted, dropped with a
        loud printed count rather than reaching `_standardize()` (which indexes
        `attribute_info_dict` directly and would otherwise raise `KeyError`).

        Returns a list of records suitable for _standardize() and _deduplicate().
        """
        if self.direct_extraction_schema is None or self.direct_extraction_prompt is None:
            raise ValueError(
                "direct_extraction_schema and direct_extraction_prompt must be set "
                "when extraction_mode='direct'. Define them in the dataset config."
            )

        from pydantic import create_model

        DirectExtractionList = create_model(
            "DirectExtractionList",
            items=(list[self.direct_extraction_schema], ...),
        )
        direct_extraction_list_json = DirectExtractionList.model_json_schema()
        known_attributes = set(self.attribute_info_dict.keys())

        def _validate_direct_extraction(r):
            parsed = response_validator(DirectExtractionList, r)
            for item in parsed['items']:
                if item.get('value') is not None and item.get('attribute') not in known_attributes:
                    raise ValueError(
                        f"attribute {item.get('attribute')!r} not in attribute_info_dict"
                    )
            return parsed

        messages = []
        for datapoint in self.data:
            context = datapoint['context']
            query = "Extract all measurement records from this document as described in the instructions."
            prompt = (
                f"## INSTRUCTIONS:\n{DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS}\n\n"
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
            validator=_validate_direct_extraction,
            timeout=600.0,
        )

        triple_data = []
        dropped_count = 0
        for i, r in enumerate(response_texts):
            if isinstance(r, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(i)
                continue
            try:
                resp_validated = response_validator(DirectExtractionList, r)
            except Exception as e:
                print(f"Validation error in direct extraction response: {e}")
                print(f"Response text: {r}")
                resp_validated = {'items': []}

            for j, item in enumerate(resp_validated['items']):
                if item.get('value') is None:
                    continue
                if item.get('attribute') not in known_attributes:
                    dropped_count += 1
                    print(
                        f"Dropping direct-extraction record with out-of-vocabulary "
                        f"attribute {item.get('attribute')!r} (doc {i}, item {j}); "
                        f"still not in attribute_info_dict after retries."
                    )
                    continue
                entity_id = f"doc_{i}_entity_{j}"
                triple_data.append(
                    self.data[i] | item | {
                        'entity_id': entity_id,
                        'attribute_terms': [],
                    }
                )

        if dropped_count:
            print(
                f"Direct extraction: dropped {dropped_count} record(s) with an "
                f"out-of-vocabulary attribute after retries."
            )

        return triple_data


    # -----------------------------------------------------------------------
    # Step 5: Standardize and deduplicate
    # -----------------------------------------------------------------------

    def _standardize(self):
        """
        LLM-based unit cleanup: maps extracted units onto the attribute's
        preferred unit list where they're a notational variant. Value is left
        untouched here -- parsing its shape (range/list/mean/tolerance/etc.) is
        a separate step, _parse_quantities().

        In the default ``standardize_context="full"`` mode, this is grounded
        against the source text the value was extracted from (entity/attribute
        description, attribute terminology, page/table context). In
        ``"value_only"`` mode, the model sees nothing but the available units
        list and the extracted measurement/units -- see
        STANDARDIZE_MEASUREMENTS_VALUE_ONLY_INSTRUCTIONS. The standardization
        decision rule (best-matching notational variant, else unchanged, else
        null) is identical in both modes.

        Reads from self.data and returns the standardized list.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_data_ids = []
        for i, datapoint in enumerate(self.data):
            attribute = datapoint.get('attribute')
            unit_options = self.attribute_info_dict[attribute].get('units', [])
            measurement_val = datapoint['value']
            measurement_units = datapoint.get('units')

            if self.standardize_context == "value_only":
                query = (
                    f"Available units for the attribute: {unit_options}\n\n"
                    f"Extracted measurement: {measurement_val}\n"
                    f"Extracted units: {measurement_units}\n"
                    f"Standardize the units for the extracted data point. "
                )
                prompt = (
                    f"## INSTRUCTIONS:\n{STANDARDIZE_MEASUREMENTS_VALUE_ONLY_INSTRUCTIONS}\n\n## QUERY:\n{query}"
                )
            else:
                context = datapoint['context']
                attr_description = self.attribute_info_dict[attribute]['description']
                attr_terms = datapoint.get('attribute_terms', [])
                entity_description = {k: v for k, v in datapoint.items() if k in entity_fields}

                terms_line = (
                    f"Terminology used for the attribute: {attr_terms}\n"
                    if self.collect_attribute_terms else ""
                )
                query = (
                    f"Entity description: {entity_description}\n"
                    f"Attribute description: {attr_description}\n"
                    f"{terms_line}"
                    f"Available units for the attribute: {unit_options}\n\n"
                    f"Extracted measurement: {measurement_val}\n"
                    f"Extracted units: {measurement_units}\n"
                    f"Standardize the units for the extracted data point. "
                )
                prompt = (
                    f"## INSTRUCTIONS:\n{STANDARDIZE_MEASUREMENTS_INSTRUCTIONS}\n\n"
                    f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
                )
            messages.append([{"role": "user", "content": prompt}])
            message_data_ids.append(i)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "standardize_response",
                "schema": StandardizeResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_tokens=1024,
            max_retries=2,
            max_concurrent=32,
            timeout=120,
            validator=lambda r: response_validator(StandardizeResponse, r),
        )

        standardized_data = [dict(datapoint) for datapoint in self.data]
        for i, resp in enumerate(response_texts):
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(
                    standardized_data[message_data_ids[i]]['document_id']
                )
                continue
            try:
                result = response_validator(StandardizeResponse, resp)
                standardized_data[message_data_ids[i]]['units'] = result['units']
            except Exception as e:
                print(f"Validation error in standardize response (keeping original units): {e}")
                print(f"Response text: {resp}")
                # fallback: leave units unchanged (standardized_data was initialised
                # as a copy of self.data, so the original is already in place)

        return standardized_data


    def _parse_quantities(self):
        """
        LLM-based quantity parsing: decomposes each extracted (already
        unit-standardized) value into qualifier tags plus whichever of
        point_value/lower/upper/list_values/tolerance/standard_deviation the
        reported value has.

        In the default ``parse_quantities_context="full"`` mode, this is
        grounded against the source text the value was extracted from
        (entity/attribute description, units, page/table context). In
        ``"value_only"`` mode, the model sees nothing but the extracted value
        string itself -- see PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS.

        Non-fatal by design: a response that fails validation, or whose
        qualifier tags don't match its populated fields, is kept with a
        logged warning rather than dropped or raised -- see
        check_quantity_consistency().

        Reads from self.data and returns the parsed list.
        """
        entity_fields = list(self.entity_identification_schema.model_fields.keys())
        messages = []
        message_data_ids = []
        for i, datapoint in enumerate(self.data):
            measurement_val = datapoint['value']

            if self.parse_quantities_context == "value_only":
                query = (
                    f"Extracted value: {measurement_val}\n"
                    f"Parse this extracted value into its structured components. "
                )
                prompt = (
                    f"## INSTRUCTIONS:\n{PARSE_QUANTITY_VALUE_ONLY_INSTRUCTIONS}\n\n## QUERY:\n{query}"
                )
            else:
                context = datapoint['context']
                attribute = datapoint.get('attribute')
                attr_description = self.attribute_info_dict[attribute]['description']
                entity_description = {k: v for k, v in datapoint.items() if k in entity_fields}
                measurement_units = datapoint.get('units')

                query = (
                    f"Entity description: {entity_description}\n"
                    f"Attribute description: {attr_description}\n\n"
                    f"Extracted value: {measurement_val}\n"
                    f"Extracted units: {measurement_units}\n"
                    f"Parse this extracted value into its structured components. "
                )
                prompt = (
                    f"## INSTRUCTIONS:\n{PARSE_QUANTITY_INSTRUCTIONS}\n\n"
                    f"## CONTEXT:\n{context}\n\n## QUERY:\n{query}"
                )
            messages.append([{"role": "user", "content": prompt}])
            message_data_ids.append(i)

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "parse_quantity_response",
                "schema": ParseQuantityResponse.model_json_schema(),
            },
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_tokens=1024,
            max_retries=2,
            max_concurrent=32,
            timeout=120,
            validator=lambda r: response_validator(ParseQuantityResponse, r),
        )

        parsed_data = [dict(datapoint) for datapoint in self.data]
        quantity_fields = (
            "qualifiers", "point_value", "lower", "upper",
            "list_values", "tolerance", "standard_deviation",
        )
        consistency_warnings = []
        for i, resp in enumerate(response_texts):
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(
                    parsed_data[message_data_ids[i]]['document_id']
                )
                continue
            try:
                result = response_validator(ParseQuantityResponse, resp)
            except Exception as e:
                print(f"Validation error in parse-quantity response (leaving quantity fields unset): {e}")
                print(f"Response text: {resp}")
                continue

            for field in quantity_fields:
                parsed_data[message_data_ids[i]][field] = result[field]
            warnings = check_quantity_consistency(result)
            consistency_warnings.extend(
                f"doc {parsed_data[message_data_ids[i]]['document_id']}: {w}" for w in warnings
            )

        if consistency_warnings:
            print(
                f"Quantity parsing: {len(consistency_warnings)} qualifier/field "
                f"consistency warning(s) -- logged only, no records dropped:"
            )
            for w in consistency_warnings:
                print(f"  {w}")

        return parsed_data


    def _deduplicate(self, data):
        """
        Equality-based deduplication using entity_id, with provenance aggregation.

        Groups records by (entity_id, attribute), then within each group
        compares values: np.isclose for numerics, case-insensitive string
        match otherwise. Keeps first occurrence and aggregates provenance
        fields (page_number, table_number, row_index, column_index, source,
        context) from duplicates into aligned lists.

        Args:
            data: list of measurement records

        Returns:
            list[dict]: Deduplicated records with aggregated provenance.
        """
        def _norm(v):
            if v is None:
                return None
            return str(v).strip().lower()

        def _values_equal(a, b):
            """Compare two values: np.isclose for numerics, case-insensitive otherwise."""
            try:
                fa, fb = float(a), float(b)
                return np.isclose(fa, fb)
            except (ValueError, TypeError):
                return _norm(a) == _norm(b)

        # Provenance fields to aggregate into aligned lists
        _PROV_FIELDS = ('page_number', 'table_number', 'row_index', 'column_index', 'source', 'context')

        def _extract_provenance(record):
            """Extract a provenance tuple from a record, using None for missing fields."""
            return {field: record.get(field) for field in _PROV_FIELDS}

        event_field_names = (
            list(self.measurement_event_schema.model_fields.keys())
            if self.measurement_event_schema is not None
            else []
        )

        groups: dict[tuple, list[int]] = {}
        for idx, record in enumerate(data):
            event_key = tuple(record.get(f) for f in event_field_names)
            key = (record.get('entity_id'), record.get('attribute')) + event_key
            groups.setdefault(key, []).append(idx)

        deduplicated: list[dict] = []

        for key, indices in groups.items():
            # Each entry: (value, units, index into deduplicated list)
            kept_values: list[tuple] = []
            for idx in indices:
                record = data[idx]
                val = record.get('value')
                units = _norm(record.get('units'))
                prov = _extract_provenance(record)

                is_dup = False
                for kept_val, kept_units, dedup_idx in kept_values:
                    if _values_equal(val, kept_val) and units == kept_units:
                        # Aggregate provenance: append all fields together to stay aligned
                        kept_record = deduplicated[dedup_idx]
                        for field in _PROV_FIELDS:
                            kept_record[field].append(prov[field])
                        is_dup = True
                        break

                if not is_dup:
                    # Create new record with provenance fields as single-element lists
                    new_record = {
                        k: v for k, v in record.items() if k not in _PROV_FIELDS
                    }
                    for field in _PROV_FIELDS:
                        new_record[field] = [prov[field]]
                    dedup_idx = len(deduplicated)
                    deduplicated.append(new_record)
                    kept_values.append((val, units, dedup_idx))

        return deduplicated


    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    @contextlib.contextmanager
    def _timed_step(self, name: str):
        """Records elapsed wall-clock time for one fit() step into
        self.step_seconds[name]. Shared by the base pipeline and by any
        ablation subclass's own fit() override, so step names stay comparable
        across ablations that keep a step vs. merge/replace it."""
        t0 = time.time()
        try:
            yield
        finally:
            self.step_seconds[name] = round(time.time() - t0, 1)

    def fit(
        self,
        documents: list[str],
    ) -> list[dict]:
        """
        Runs measurement extraction on the provided documents.

        If ``extraction_mode="direct"`` (set at construction), runs a single
        ``_extract_triples()`` call per document followed by ``_standardize()``
        and ``_deduplicate()``, instead of the full pipeline below.

        Table cleaning is not part of this class -- it's a separate,
        explicit step (``TableCleaner``, ``experiments/run_table_cleaning.py``).
        If a dataset's OCR text needs it, run that first and pass its output
        directory as ``documents``' source (``params.ocr_dir`` at the
        experiment-config layer).

        Per-step wall-clock time is recorded into ``self.step_seconds`` (pipeline
        mode only -- direct mode is a single call, nothing to break down) under
        keys ``entities``, ``entity_prov``, ``attributes``, ``attribute_prov``,
        ``events``, ``values_text``, ``values_tables``, ``final`` (standardize +
        parse + deduplicate combined). ``values_text``/``values_tables`` are kept
        separate (rather than one combined ``values`` key) because ablation 5
        changes table extraction only and ablation 4 changes both but
        differently -- a combined number would blend a changed step with an
        unchanged one. Summing them reproduces run_extraction.py's single
        ``values`` checkpoint, so the two remain comparable at that level.

        Args:
            documents: OCR text strings, one per document.
        Returns:
            Measurement records extracted from the documents.
        """
        # Reset per-batch state: a second fit() call on the same instance must not
        # carry forward document indices from the previous batch (those indices are
        # positions within *this* batch's documents list, not stable document ids).
        self.context_length_exceeded_docs = set()
        self.step_seconds = {}

        self.data = []
        for i, doc in enumerate(documents):
            self.data.append({'document_id': i, 'context': doc})

        if self.extraction_mode == "direct":
            self.data = self._extract_triples()
            self.data = self._standardize()
            self.data = self._parse_quantities()
            self.data = self._deduplicate(self.data)
            return self.data

        doc_data = list(self.data)

        # Step 1: Entity extraction
        with self._timed_step("entities"):
            entity_data = self._extract_entities()

        # Step 2: Entity provenance
        with self._timed_step("entity_prov"):
            entity_prov = self._entity_provenance(entity_data)

        # Step 3: Document-level attribute detection
        self.data = doc_data
        with self._timed_step("attributes"):
            doc_attributes = self._detect_attributes()

        # Step 4: Attribute provenance
        with self._timed_step("attribute_prov"):
            attr_prov = self._attribute_provenance(doc_attributes)

        # Step 4.5: Measurement event resolution (optional)
        with self._timed_step("events"):
            if self.measurement_event_schema is not None:
                event_resolution = self._resolve_events(
                    entity_data, doc_attributes, entity_prov, attr_prov
                )
            else:
                event_resolution = None

        # Steps 5+6: Extract values from text and tables (provenance intersection)
        with self._timed_step("values_text"):
            text_values = self._extract_values_from_text(
                entity_data, doc_attributes, entity_prov, attr_prov, event_resolution
            )
        with self._timed_step("values_tables"):
            table_values = self._extract_values_from_tables(
                entity_data, doc_attributes, entity_prov, attr_prov, event_resolution
            )

        # Combine text and table extractions
        self.data = text_values + table_values

        # Steps 7+7.5+8: Standardize, parse quantities, deduplicate
        with self._timed_step("final"):
            self.data = self._standardize()
            self.data = self._parse_quantities()
            self.data = self._deduplicate(self.data)

        return self.data


    def save(self, filepath: str):
        """
        Saves the measurement data to a JSON file.

        Args:
            filepath (str): The path to the file where the data will be saved.
        """
        with open(filepath, 'w') as f:
            json.dump(self.data, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
