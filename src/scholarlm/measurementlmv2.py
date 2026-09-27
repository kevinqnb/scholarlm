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
steps and fit() are new. Table cleaning is not one of them: it's a fully
separate step (TableCleaner, experiments/run_table_cleaning.py) -- v2 assumes
documents are already cleaned by that process, and fit() takes only the OCR
text.
"""
from __future__ import annotations

from pydantic import BaseModel, create_model

from .instruction_prompts import (
    QUANTITY_COLLECTION_INSTRUCTIONS,
    STANDARDIZE_QUANTITY_INSTRUCTIONS,
    CONTEXTUALIZE_QUANTITIES_INSTRUCTIONS,
)
from .measurementlm import (
    ContextLengthExceededError,
    MeasurementLM,
    check_quantity_consistency,
    response_validator,
)


# -----------------------------------------------------------------------
# Response schemas
#
# Quantity shape (qualifiers/point_value/lower/upper/list_values/tolerance/
# standard_deviation) matches MeasurementLM's ParseQuantityResponse -- see
# check_quantity_consistency() there for the qualifier/field cross-checks,
# reused unchanged here. Unlike the pre-reconciliation type/quantifier/value
# scheme this replaces, there's no packed format (e.g. "(lower, upper)") left
# to validate structurally: every field is already its own scalar (str or,
# since the 2026-09-24 gpt-oss-120b stall fix, float), so inconsistency
# checking is purely the qualifier-vs-populated-field logging
# check_quantity_consistency() does -- non-fatal, per-project policy (see
# MeasurementLM._parse_quantities()'s docstring): a mismatched parse is kept
# and logged, never dropped or retried.
# -----------------------------------------------------------------------


class QuantityItem(BaseModel):
    """A single extracted quantity, pre-standardization and pre-attribution.

    point_value/lower/upper/tolerance/standard_deviation are str | float |
    None, matching MeasurementLM's ParseQuantityResponse -- same gpt-oss-120b
    guided-decoding stall, same fix (see notes/scholarlm/experiments/
    2026-09-24-gptoss120b-parsequantity-schema-diag-{01,02}.md)."""
    value: str
    units: str | None = None
    qualifiers: list[str]
    point_value: str | float | None = None
    lower: str | float | None = None
    upper: str | float | None = None
    list_values: list[str] | None = None
    tolerance: str | float | None = None
    standard_deviation: str | float | None = None
    table_number: int | None = None


class QuantityListResponse(BaseModel):
    items: list[QuantityItem]


class StandardizeQuantityResponse(BaseModel):
    explanation: str
    units: str | None = None


def _model_field_kwargs(schema: type[BaseModel]) -> dict[str, tuple]:
    """(annotation, default-or-Ellipsis) kwargs for create_model, keyed by field name."""
    return {
        name: (info.annotation, ... if info.is_required() else info.default)
        for name, info in schema.model_fields.items()
    }


def _merge_field_schemas(entity_schema: type[BaseModel], event_schema: type[BaseModel] | None, model_name: str):
    """Combine two schemas into one flat model (the second argument may be None).

    Used both for the contextualization step's entity+event response schema,
    and again to add the quantity-echo fields on top of that merged schema.
    Field-name collisions fail loud rather than silently letting one schema's
    field shadow the other's.
    """
    event_fields = set(event_schema.model_fields) if event_schema is not None else set()
    overlap = set(entity_schema.model_fields) & event_fields
    if overlap:
        raise ValueError(
            f"schemas share field name(s) {overlap}; rename to disambiguate"
        )
    fields = _model_field_kwargs(entity_schema)
    if event_schema is not None:
        fields.update(_model_field_kwargs(event_schema))
    return create_model(model_name, **fields)


# -----------------------------------------------------------------------
# Dedup key
# -----------------------------------------------------------------------


def _norm_units(v: str | None) -> str | None:
    return None if v is None else str(v).strip().lower()


def _norm_scalar(v: str | float | None):
    if v is None:
        return None
    try:
        return round(float(v), 6)
    except (TypeError, ValueError):
        return str(v).strip().lower()


def _quantity_value_key(q: dict):
    """Normalized value key, shared by the dedup key and the (looser)
    contextualization match key below. value is now always a raw string (no
    packed range format to parse out), so this is just a numeric-aware
    normalization of it.
    """
    return _norm_scalar(q["value"])


def _quantity_dedup_key(q: dict) -> tuple:
    """Equality key for deduplicating quantities within (document, attribute).

    Two quantities on different pages of the same document, for the same
    attribute, with the same qualifiers/value/units/shape fields collapse
    into one entry here — this is a deliberate consequence of dedup running
    before any entity is known (see build note): if the paper genuinely
    reports this same value for two distinct entities/events, the
    contextualization step is responsible for re-expanding it into multiple
    final records.
    """
    list_values = q.get("list_values")
    return (
        q["document_id"], q["attribute"], tuple(sorted(q["qualifiers"])),
        _quantity_value_key(q), _norm_units(q.get("units")),
        _norm_scalar(q.get("point_value")), _norm_scalar(q.get("lower")), _norm_scalar(q.get("upper")),
        tuple(_norm_scalar(v) for v in list_values) if list_values else None,
        _norm_scalar(q.get("tolerance")), _norm_scalar(q.get("standard_deviation")),
    )


def _quantity_match_key(q: dict) -> tuple:
    """Looser identity key for matching a contextualization response's
    echoed quantity back to its source: (document_id, attribute, value,
    units) only -- qualifiers/shape fields excluded.

    A 2026-09-19 smoke test (gemma-3-27b, pond, pre-qualifier-rework type/
    quantifier/CI scheme) showed the model reliably echoes value/units/
    attribute verbatim, but not the shape fields: it added a spurious
    inequality quantifier to a plain point quantity (apparently re-derived
    from a symbol near the value in the source table, rather than copied),
    and echoed an absent field as the literal string "None" (a symptom of
    our own prompt text rendering `None` as that word -- see the
    query-building code below, fixed the same day). The same caution applies
    to the current qualifiers/point_value/lower/upper/list_values/tolerance/
    standard_deviation fields: the model's echoed versions are never
    trusted, only the matched original quantity's own fields reach the
    output record. When more than one quantity in a group shares this
    reduced key (e.g. a point value that coincidentally equals a separately-
    reported range bound), the full _quantity_dedup_key is used to
    disambiguate among just those candidates.
    """
    return (q["document_id"], q["attribute"], _quantity_value_key(q), _norm_units(q.get("units")))


def _describe_quantity(q: dict, attribute_info_dict: dict) -> str:
    """Render one quantity for the contextualization prompt.

    A field that is None is omitted entirely rather than interpolated (an
    f-string on None renders the literal text "None") -- see
    _quantity_match_key's docstring for the bug that caused: a model told to
    copy fields back "exactly as given" copied that literal word instead of
    emitting JSON null for an absent field.
    """
    lines = [
        f"- Attribute: {q['attribute']} -- {attribute_info_dict[q['attribute']].get('description', '')}",
        f"  Value: {q['value']}",
    ]
    if q.get("qualifiers"):
        lines.append(f"  Qualifiers: {q['qualifiers']}")
    if q.get("units") is not None:
        lines.append(f"  Units: {q['units']}")
    if q.get("point_value") is not None:
        lines.append(f"  Point value: {q['point_value']}")
    if q.get("lower") is not None or q.get("upper") is not None:
        lines.append(f"  Lower/upper: {q.get('lower')} / {q.get('upper')}")
    if q.get("list_values"):
        lines.append(f"  List values: {q['list_values']}")
    if q.get("tolerance") is not None:
        lines.append(f"  Tolerance: {q['tolerance']}")
    if q.get("standard_deviation") is not None:
        lines.append(f"  Standard deviation: {q['standard_deviation']}")
    lines.append(f"  Found on page(s): {q['page_number']}")
    return "\n".join(lines) + "\n"


# -----------------------------------------------------------------------
# MeasurementLMv2
# -----------------------------------------------------------------------


class MeasurementLMv2(MeasurementLM):
    """Quantity-first measurement extraction.

    fit() runs: collect quantities per (page, attribute) -> standardize units
    -> deduplicate (code, within document+attribute) -> attribute each
    deduplicated quantity to entity/event using the full paper.
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
            return response_validator(QuantityListResponse, r)

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
        consistency_warnings = []
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
                consistency_warnings.extend(
                    f"doc {doc_id} page {page_number} attribute {attr_name!r}: {w}"
                    for w in check_quantity_consistency(item)
                )

        if consistency_warnings:
            print(
                f"Quantity collection: {len(consistency_warnings)} qualifier/field "
                f"consistency warning(s) -- logged only, no records dropped:"
            )
            for w in consistency_warnings:
                print(f"  {w}")

        return quantities

    # -----------------------------------------------------------------------
    # Step 1.5: Standardize units
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
                f"Extracted value: {q['value']}\n"
                f"Extracted units: {q.get('units')}\n\n"
                f"Standardize the units for this quantity.\n"
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
                standardized[i]["units"] = result["units"]
            except Exception as e:
                print(f"Validation error in quantity standardization (keeping original units): {e}")
                print(f"Response text: {resp}")

        return standardized

    # -----------------------------------------------------------------------
    # Step 2: Deduplicate quantities (code, no LLM call)
    # -----------------------------------------------------------------------

    def _deduplicate_quantities(self, quantities: list[dict]) -> list[dict]:
        """Groups by (document_id, attribute, qualifiers, value, units, and
        the other shape fields), aggregating 'page_number' and 'table_number'
        into index-aligned lists so every surviving quantity carries every
        (page, table) location it was found on (see _quantity_dedup_key's
        docstring for why dedup ignores these fields, and
        _group_quantities_for_contextualization for how they're used
        downstream).
        """
        _PROV_FIELDS = ("page_number", "table_number")
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

    def _group_quantities_for_contextualization(
        self, quantities: list[dict]
    ) -> dict[tuple[int, int | None], list[dict]]:
        """Buckets deduplicated quantities into (document_id, table_number) groups.

        table_number is None for a document's prose group. A quantity whose
        aggregated table_number list (see _deduplicate_quantities) contains
        any non-null value is assigned to the first such table -- a table is a
        more specific source location than prose, so a quantity seen in both
        is grouped with the table. Dict insertion order determines message
        order, which the caller relies on to zip results back to groups.
        """
        groups: dict[tuple[int, int | None], list[dict]] = {}
        for q in quantities:
            table_number = next((t for t in q["table_number"] if t is not None), None)
            groups.setdefault((q["document_id"], table_number), []).append(q)
        return groups

    def _contextualize_quantities(self, quantities: list[dict], max_tokens: int = 8192) -> list[dict]:
        if not quantities:
            return []

        EntityEventItem = _merge_field_schemas(
            self.entity_identification_schema, self.measurement_event_schema, "EntityEventItem",
        )
        # The response must echo each quantity's identifying fields back
        # (QuantityEcho) so we can match an item to its source quantity by
        # content rather than by an index the model could misapply -- see
        # notes/scholarlm/builds/ for the build note that introduced this.
        # table_number is provenance, not identity (see _quantity_dedup_key),
        # so it's excluded here even though it's a QuantityItem field.
        quantity_identity_fields = _model_field_kwargs(QuantityItem)
        quantity_identity_fields.pop("table_number")
        QuantityEcho = create_model("QuantityEcho", attribute=(str, ...), **quantity_identity_fields)
        AttributedQuantityItem = _merge_field_schemas(QuantityEcho, EntityEventItem, "AttributedQuantityItem")
        AttributedQuantityList = create_model("AttributedQuantityList", items=(list[AttributedQuantityItem], ...))
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "attributed_quantity_list", "schema": AttributedQuantityList.model_json_schema()},
        }
        echo_fields = set(QuantityEcho.model_fields)

        event_reference = (
            f"\n\nMeasurement event field reference:\n{self.measurement_event_prompt}"
            if self.measurement_event_schema is not None else ""
        )

        groups = self._group_quantities_for_contextualization(quantities)
        group_keys = list(groups.keys())

        messages = []
        for doc_id, table_number in group_keys:
            group = groups[(doc_id, table_number)]
            doc_context = self.data[doc_id]["context"]
            scope_line = (
                f"The quantities below were all extracted from Table {table_number} of this paper."
                if table_number is not None
                else "The quantities below were all extracted from prose text (not from a table)."
            )
            quantity_lines = [_describe_quantity(q, self.attribute_info_dict) for q in group]
            query = (
                f"{scope_line}\n\n"
                f"Entity field reference:\n{self.entity_identification_prompt}"
                f"{event_reference}\n\n"
                f"Quantities:\n\n" + "\n".join(quantity_lines) +
                "\nFor each quantity above, copy back its fields and identify every "
                "distinct (entity, event) it describes.\n"
            )
            prompt = (
                f"## INSTRUCTIONS:\n{CONTEXTUALIZE_QUANTITIES_INSTRUCTIONS}\n\n"
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
            validator=lambda r: response_validator(AttributedQuantityList, r),
            extra_body=self._seed_extra_body(),
        )

        records = []
        dropped = 0
        for msg_idx, resp in enumerate(response_texts):
            doc_id, table_number = group_keys[msg_idx]
            group = groups[(doc_id, table_number)]
            if isinstance(resp, ContextLengthExceededError):
                self.context_length_exceeded_docs.add(doc_id)
                continue
            try:
                result = response_validator(AttributedQuantityList, resp)
            except Exception as e:
                print(f"Validation error in contextualization response (doc {doc_id}, table {table_number}): {e}")
                print(f"Response text: {resp}")
                continue

            candidates_by_key: dict[tuple, list[dict]] = {}
            for q in group:
                candidates_by_key.setdefault(_quantity_match_key(q), []).append(q)

            matched_ids = set()
            for item in result["items"]:
                echo = {k: v for k, v in item.items() if k in echo_fields}
                echo["document_id"] = doc_id
                candidates = candidates_by_key.get(_quantity_match_key(echo), [])
                if len(candidates) == 1:
                    matched_q = candidates[0]
                elif len(candidates) > 1:
                    # Rare: several quantities in this group share (attribute,
                    # value, units) -- disambiguate via the full identity key.
                    matched_q = {_quantity_dedup_key(c): c for c in candidates}.get(_quantity_dedup_key(echo))
                else:
                    matched_q = None
                if matched_q is None:
                    print(
                        f"Contextualization: doc {doc_id} table {table_number} returned an "
                        f"item whose copied-back quantity fields ({echo}) don't match any "
                        "quantity sent in this batch; dropping."
                    )
                    continue
                matched_ids.add(id(matched_q))
                entity_event_fields = {k: v for k, v in item.items() if k not in echo_fields}
                records.append(matched_q | entity_event_fields)

            dropped += len(group) - len(matched_ids)

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

        Table cleaning is not run here: it's a separate, explicit step
        (TableCleaner, see experiments/run_table_cleaning.py) -- run it first
        if needed, so ``documents`` is already cleaned OCR text.

        Args:
            documents: OCR text strings, one per document, already cleaned.
        Returns:
            Measurement records extracted from the documents.
        """
        self.context_length_exceeded_docs = set()
        if self.measurement_event_schema is not None and self.measurement_event_prompt is None:
            raise ValueError("measurement_event_prompt is required when measurement_event_schema is set.")

        self.data = [{"document_id": i, "context": doc} for i, doc in enumerate(documents)]

        quantities = self._collect_quantities()
        quantities = self._standardize_quantities(quantities)
        quantities = self._deduplicate_quantities(quantities)
        self.data = self._contextualize_quantities(quantities)
        return self.data
