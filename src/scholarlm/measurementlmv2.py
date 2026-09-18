"""MeasurementLMv2: quantity-first measurement extraction.

Flips MeasurementLM's extraction order. Instead of identifying entities and
events before looking for a value, v2 collects every quantity reported for
each (page, attribute) pair first, deduplicates them (a code-level task once
the fields are structured), and only then asks the model to attribute each
deduplicated quantity to an entity and measurement event using the full paper
as context. See notes/scholarlm/builds/ for the build note that introduced
this class.

Subclasses MeasurementLM to reuse its API-call plumbing (_acall, _call_batch,
response_validator) and page/table text helpers unchanged; only the pipeline
steps and fit() are new. Table cleaning is not one of them: v2 assumes
documents are already cleaned by a separate process (construct with
clean_tables=False) and fit() takes only the OCR text.
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, create_model

from .instruction_prompts import (
    QUANTITY_COLLECTION_INSTRUCTIONS,
    STANDARDIZE_QUANTITY_INSTRUCTIONS,
    CONTEXTUALIZE_QUANTITY_INSTRUCTIONS,
)
from .measurementlm import ContextLengthExceededError, MeasurementLM, response_validator


# -----------------------------------------------------------------------
# Quantity shape: type/quantifier/value/CI invariants
# -----------------------------------------------------------------------

QuantityType = Literal["point", "range", "inequality"]
Quantifier = Literal["<", ">", "<=", ">="]

_INEQUALITY_QUANTIFIERS = {"<", ">", "<=", ">="}
_NUMBER = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_SCALAR_RE = re.compile(rf"^\s*{_NUMBER}\s*$")
_RANGE_RE = re.compile(rf"^\s*\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)\s*$")


def validate_quantity_shape(item: dict) -> None:
    """Raise ValueError if a quantity's type/quantifier/value/CI fields are
    internally inconsistent.

    This is the one place these genuinely-new fields are checked structurally
    rather than trusted from the prompt: a model can return `type: "range"`
    with `value: "3.5"` (dropping the range) or misplace a CI bound, and none
    of that would otherwise raise. Called from the collection-step batch
    validator (so a violation triggers a retry) and again after
    standardization (so reformatting can't silently break the shape).
    """
    qtype = item.get("type")
    quantifier = item.get("quantifier")
    value = item.get("value")

    if qtype == "range":
        if quantifier is not None:
            raise ValueError(f"range quantity must not have a quantifier, got {quantifier!r}")
        m = _RANGE_RE.match(value or "")
        if not m:
            raise ValueError(f"range value must be formatted '(lower, upper)', got {value!r}")
        lower, upper = float(m.group(1)), float(m.group(2))
        if lower > upper:
            raise ValueError(f"range lower bound {lower} exceeds upper bound {upper}")
    elif qtype == "inequality":
        if quantifier not in _INEQUALITY_QUANTIFIERS:
            raise ValueError(
                f"inequality quantity requires quantifier in {_INEQUALITY_QUANTIFIERS}, got {quantifier!r}"
            )
        if not _SCALAR_RE.match(value or ""):
            raise ValueError(f"inequality value must be a bare number, got {value!r}")
    elif qtype == "point":
        if quantifier is not None:
            raise ValueError(f"point quantity must not have a quantifier, got {quantifier!r}")
        if not _SCALAR_RE.match(value or ""):
            raise ValueError(f"point value must be a bare number, got {value!r}")
    else:
        raise ValueError(f"unknown quantity type {qtype!r}")

    ci_lower, ci_upper, ci = item.get("ci_lower"), item.get("ci_upper"), item.get("ci")
    if (ci_lower is None) != (ci_upper is None):
        raise ValueError("ci_lower and ci_upper must both be set or both be None")
    if ci is not None and (ci_lower is not None or ci_upper is not None):
        raise ValueError("ci is mutually exclusive with ci_lower/ci_upper")
    for label, val in (("ci_lower", ci_lower), ("ci_upper", ci_upper), ("ci", ci)):
        if val is not None and not _SCALAR_RE.match(val):
            raise ValueError(f"{label} must be a bare number, got {val!r}")


# -----------------------------------------------------------------------
# Response schemas
# -----------------------------------------------------------------------


class QuantityItem(BaseModel):
    """A single extracted quantity, pre-standardization and pre-attribution."""
    value: str
    units: str | None = None
    type: QuantityType
    quantifier: Quantifier | None = None
    ci_lower: str | None = None
    ci_upper: str | None = None
    ci: str | None = None


class QuantityListResponse(BaseModel):
    items: list[QuantityItem]


class StandardizeQuantityResponse(BaseModel):
    explanation: str
    value: str
    units: str | None = None
    ci_lower: str | None = None
    ci_upper: str | None = None
    ci: str | None = None


def _merge_field_schemas(entity_schema: type[BaseModel], event_schema: type[BaseModel] | None, model_name: str):
    """Combine an entity schema and (optional) event schema into one flat model.

    Used for the contextualization step's response schema: given a quantity,
    the model fills in entity fields and event fields together in one item.
    Field-name collisions between the two schemas fail loud rather than
    silently letting one schema's field shadow the other's.
    """
    event_fields = set(event_schema.model_fields) if event_schema is not None else set()
    overlap = set(entity_schema.model_fields) & event_fields
    if overlap:
        raise ValueError(
            f"entity and event schemas share field name(s) {overlap}; rename to disambiguate"
        )
    schemas = (entity_schema,) if event_schema is None else (entity_schema, event_schema)
    fields = {}
    for schema in schemas:
        for name, info in schema.model_fields.items():
            fields[name] = (info.annotation, ... if info.is_required() else info.default)
    return create_model(model_name, **fields)


# -----------------------------------------------------------------------
# Dedup key
# -----------------------------------------------------------------------


def _norm_units(v: str | None) -> str | None:
    return None if v is None else str(v).strip().lower()


def _norm_scalar(v: str | None):
    if v is None:
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v).strip().lower()


def _quantity_dedup_key(q: dict) -> tuple:
    """Equality key for deduplicating quantities within (document, attribute).

    Two quantities on different pages of the same document, for the same
    attribute, with the same type/quantifier/value/units/CI collapse into one
    entry here — this is a deliberate consequence of dedup running before any
    entity is known (see build note): if the paper genuinely reports this same
    value for two distinct entities/events, the contextualization step is
    responsible for re-expanding it into multiple final records.
    """
    value = q["value"]
    if q["type"] == "range":
        m = _RANGE_RE.match(value)
        value_key = (_norm_scalar(m.group(1)), _norm_scalar(m.group(2))) if m else value
    else:
        value_key = _norm_scalar(value)
    return (
        q["document_id"], q["attribute"], q["type"], q.get("quantifier"),
        value_key, _norm_units(q.get("units")),
        _norm_scalar(q.get("ci_lower")), _norm_scalar(q.get("ci_upper")), _norm_scalar(q.get("ci")),
    )


# -----------------------------------------------------------------------
# MeasurementLMv2
# -----------------------------------------------------------------------


class MeasurementLMv2(MeasurementLM):
    """Quantity-first measurement extraction.

    fit() runs: collect quantities per (page, attribute) -> standardize units
    and numeric formatting -> deduplicate (code, within document+attribute) ->
    attribute each deduplicated quantity to entity/event using the full paper.
    Constructor is inherited unchanged from MeasurementLM.
    """

    def _seed_extra_body(self) -> dict | None:
        """extra_body carrying sampling_params['seed'], if set.

        MeasurementLM._acall never forwards 'seed' from sampling_params (only
        temperature/top_p/top_k/repetition_penalty/chat-template kwargs) — v2
        threads it through explicitly here, on every _call_batch call, so a
        configured seed actually reaches the request rather than being a
        no-op key sitting in the model config.
        """
        seed = self.sampling_params.get("seed")
        return {"seed": seed} if seed is not None else None

    # -----------------------------------------------------------------------
    # Step 1: Quantity collection, per (page, attribute)
    # -----------------------------------------------------------------------

    def _collect_quantities(self, max_tokens: int = 2048) -> list[dict]:
        """One call per (document, page, attribute); returns a flat list of
        quantity records with 'document_id', 'page_number', 'attribute', and
        the QuantityItem fields. Page text itself is not stored on the
        record -- later steps that need it (standardization) re-fetch it from
        self.data by (document_id, page_number).
        """
        messages = []
        message_ids = []  # (doc_id, page_number, attr_name)

        for doc_id, datapoint in enumerate(self.data):
            context = datapoint["context"]
            for page_number in self._get_page_numbers(context):
                page_text = self._get_page_text(context, page_number)
                if not page_text:
                    continue
                for attr_name, attr_info in self.attribute_info_dict.items():
                    unit_options = attr_info.get("units", [])
                    units_guidance = ""
                    if unit_options:
                        units_guidance = (
                            f"Preferred unit options: {unit_options}. "
                            f"Strongly prioritize choosing the best option from this list. "
                            f"If none of the options fit, specify the unit exactly as it appears in the text.\n"
                        )
                    query = (
                        f"Attribute: {attr_name}\n"
                        f"Attribute description: {attr_info.get('description', '')}\n"
                        f"{units_guidance}\n"
                        f"Extract every distinct quantity reported on this page for the given attribute.\n\n"
                    )
                    prompt = (
                        f"## INSTRUCTIONS:\n{QUANTITY_COLLECTION_INSTRUCTIONS}\n\n"
                        f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
                    )
                    messages.append([{"role": "user", "content": prompt}])
                    message_ids.append((doc_id, page_number, attr_name))

        if not messages:
            return []

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "quantity_list", "schema": QuantityListResponse.model_json_schema()},
        }

        def _validate(r):
            parsed = response_validator(QuantityListResponse, r)
            for item in parsed["items"]:
                validate_quantity_shape(item)
            return parsed

        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=max_tokens,
            max_concurrent=32,
            timeout=120,
            validator=_validate,
            extra_body=self._seed_extra_body(),
        )

        quantities = []
        for msg_idx, resp in enumerate(response_texts):
            doc_id, page_number, attr_name = message_ids[msg_idx]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(doc_id)
                continue
            try:
                parsed = _validate(resp)
            except Exception as e:
                print(
                    f"Validation error in quantity collection response "
                    f"(doc {doc_id}, page {page_number}, attribute {attr_name!r}): {e}"
                )
                print(f"Response text: {resp}")
                continue
            for item in parsed["items"]:
                quantities.append({
                    "document_id": doc_id,
                    "page_number": page_number,
                    "attribute": attr_name,
                    **item,
                })

        return quantities

    # -----------------------------------------------------------------------
    # Step 1.5: Standardize units and numeric formatting
    # -----------------------------------------------------------------------

    def _standardize_quantities(self, quantities: list[dict], max_tokens: int = 1024) -> list[dict]:
        if not quantities:
            return []

        messages = []
        for q in quantities:
            attr_info = self.attribute_info_dict[q["attribute"]]
            query = (
                f"Attribute description: {attr_info.get('description', '')}\n"
                f"Available units for the attribute: {attr_info.get('units', [])}\n\n"
                f"Quantity type: {q['type']}\n"
                f"Quantifier: {q.get('quantifier')}\n"
                f"Extracted value: {q['value']}\n"
                f"Extracted units: {q.get('units')}\n"
                f"Extracted CI lower/upper: {q.get('ci_lower')} / {q.get('ci_upper')}\n"
                f"Extracted CI (±): {q.get('ci')}\n\n"
                f"Standardize the units and numeric formatting for this quantity.\n"
            )
            page_text = self._get_page_text(self.data[q["document_id"]]["context"], q["page_number"])
            prompt = (
                f"## INSTRUCTIONS:\n{STANDARDIZE_QUANTITY_INSTRUCTIONS}\n\n"
                f"## CONTEXT:\n{page_text}\n\n## QUERY:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])

        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "standardize_quantity", "schema": StandardizeQuantityResponse.model_json_schema()},
        }
        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=max_tokens,
            max_concurrent=32,
            timeout=120,
            validator=lambda r: response_validator(StandardizeQuantityResponse, r),
            extra_body=self._seed_extra_body(),
        )

        standardized = [dict(q) for q in quantities]
        for i, resp in enumerate(response_texts):
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(standardized[i]["document_id"])
                continue
            try:
                result = response_validator(StandardizeQuantityResponse, resp)
                candidate = standardized[i] | {
                    "value": result["value"],
                    "units": result.get("units"),
                    "ci_lower": result.get("ci_lower"),
                    "ci_upper": result.get("ci_upper"),
                    "ci": result.get("ci"),
                }
                validate_quantity_shape(candidate)
            except Exception as e:
                print(f"Validation error in quantity standardization (keeping original value/units): {e}")
                print(f"Response text: {resp}")
                continue
            standardized[i] = candidate

        return standardized

    # -----------------------------------------------------------------------
    # Step 2: Deduplicate quantities (code, no LLM call)
    # -----------------------------------------------------------------------

    def _deduplicate_quantities(self, quantities: list[dict]) -> list[dict]:
        """Groups by (document_id, attribute, type, quantifier, value, units,
        CI), aggregating 'page_number' into a list so every surviving
        quantity carries the full list of pages it was found on (see
        _quantity_dedup_key's docstring for why this granularity).
        """
        _PROV_FIELDS = ("page_number",)
        index_by_key: dict[tuple, int] = {}
        deduplicated: list[dict] = []

        for q in quantities:
            key = _quantity_dedup_key(q)
            if key in index_by_key:
                kept = deduplicated[index_by_key[key]]
                for f in _PROV_FIELDS:
                    kept[f].append(q[f])
            else:
                new_record = {k: v for k, v in q.items() if k not in _PROV_FIELDS}
                for f in _PROV_FIELDS:
                    new_record[f] = [q[f]]
                index_by_key[key] = len(deduplicated)
                deduplicated.append(new_record)

        return deduplicated

    # -----------------------------------------------------------------------
    # Step 3: Contextualize — attribute each quantity to entity(s) and event(s)
    # -----------------------------------------------------------------------

    def _contextualize_quantities(self, quantities: list[dict], max_tokens: int = 8192) -> list[dict]:
        if not quantities:
            return []

        EntityEventItem = _merge_field_schemas(
            self.entity_identification_schema, self.measurement_event_schema, "EntityEventItem",
        )
        EntityEventList = create_model("EntityEventList", items=(list[EntityEventItem], ...))
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "entity_event_list", "schema": EntityEventList.model_json_schema()},
        }

        event_reference = (
            f"\n\nMeasurement event field reference:\n{self.measurement_event_prompt}"
            if self.measurement_event_schema is not None else ""
        )

        messages = []
        for q in quantities:
            doc_context = self.data[q["document_id"]]["context"]
            query = (
                f"Attribute: {q['attribute']}\n"
                f"Attribute description: {self.attribute_info_dict[q['attribute']].get('description', '')}\n"
                f"Quantity type: {q['type']}\n"
                f"Quantifier: {q.get('quantifier')}\n"
                f"Value: {q['value']}\n"
                f"Units: {q.get('units')}\n"
                f"CI lower/upper: {q.get('ci_lower')} / {q.get('ci_upper')}\n"
                f"CI (±): {q.get('ci')}\n"
                f"Found on page(s): {q['page_number']}\n\n"
                f"Entity field reference:\n{self.entity_identification_prompt}"
                f"{event_reference}\n\n"
                f"Identify every distinct (entity, event) this quantity describes.\n"
            )
            prompt = (
                f"## INSTRUCTIONS:\n{CONTEXTUALIZE_QUANTITY_INSTRUCTIONS}\n\n"
                f"## CONTEXT:\n{doc_context}\n\n## QUERY:\n{query}"
            )
            messages.append([{"role": "user", "content": prompt}])

        response_texts = self._call_batch(
            messages,
            response_format=response_format,
            max_retries=2,
            max_tokens=max_tokens,
            max_concurrent=2,
            timeout=300,
            validator=lambda r: response_validator(EntityEventList, r),
            extra_body=self._seed_extra_body(),
        )

        records = []
        dropped = 0
        for i, resp in enumerate(response_texts):
            q = quantities[i]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(q["document_id"])
                continue
            try:
                result = response_validator(EntityEventList, resp)
            except Exception as e:
                print(f"Validation error in contextualization response: {e}")
                print(f"Response text: {resp}")
                continue
            if not result["items"]:
                dropped += 1
                continue
            for item in result["items"]:
                records.append(q | item)

        if dropped:
            print(
                f"Contextualization: {dropped} quantity/quantities could not be "
                "attributed to any entity and were dropped."
            )

        return records

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def fit(self, documents: list[str]) -> list[dict]:
        """Runs the quantity-first extraction pipeline on the provided documents.

        Unlike MeasurementLM.fit(), table cleaning is not run here: v2 assumes
        it has already been done as a separate process (see
        experiments/run_table_cleaning.py), so ``documents`` should already be
        cleaned OCR text. Construct with ``clean_tables=False``.

        Args:
            documents: OCR text strings, one per document, already cleaned.
        Returns:
            Measurement records extracted from the documents.
        """
        self.context_length_exceeded_docs = set()
        if self.measurement_event_schema is not None and self.measurement_event_prompt is None:
            raise ValueError("measurement_event_prompt is required when measurement_event_schema is set.")
        if self.clean_tables:
            raise ValueError(
                "MeasurementLMv2 does not run table cleaning itself -- it assumes "
                "documents are already cleaned by a separate process. Construct "
                "with clean_tables=False and pass already-cleaned OCR text to fit()."
            )

        self.data = [{"document_id": i, "context": doc} for i, doc in enumerate(documents)]

        quantities = self._collect_quantities()
        quantities = self._standardize_quantities(quantities)
        quantities = self._deduplicate_quantities(quantities)
        self.data = self._contextualize_quantities(quantities)
        return self.data
