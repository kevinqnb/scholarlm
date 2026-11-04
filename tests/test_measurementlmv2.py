"""Unit tests for MeasurementLMv2's quantity-first pipeline.

Rung 1: the code-level dedup key and the qualifier/field consistency-logging
integration in _collect_quantities, on hand-built tuples -- no network, no
LLM. Since the reconciliation of v2's quantity schema onto MeasurementLM's
qualifiers/point_value/lower/upper/list_values/tolerance/standard_deviation
shape (see tests/test_measurementlm.py for check_quantity_consistency's own
exhaustive cases, reused unchanged here rather than duplicated), there's no
packed value format (the old "(lower, upper)" range encoding) left to
validate structurally -- every field is already its own string.

Rung 2: fit() end-to-end with a stubbed _call_batch (same pattern as
tests/test_measurementlm.py), verifying the three-call sequence (collect ->
standardize -> contextualize), that page numbers are aggregated across
duplicate-quantity pages, and that a single deduplicated quantity can expand
into multiple final records via the contextualization step.
"""
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.measurementlm import response_validator
from scholarlm.measurementlmv2 import (
    MeasurementLMv2,
    QuantityItem,
    _merge_field_schemas,
    _quantity_dedup_key,
    check_quantity_consistency,
)


# ---------------------------------------------------------------------------
# Rung 1: fixtures shared by the dedup-key and collection-logging tests below
# ---------------------------------------------------------------------------


def _raw_quantity(**overrides):
    """A single not-yet-deduplicated quantity record, as _collect_quantities
    emits it: scalar page_number/table_number, not yet list-wrapped (see
    _quantity() further below for the post-dedup, list-wrapped shape used by
    the group/contextualize tests)."""
    base = {
        "document_id": 0, "page_number": 1, "table_number": None,
        "attribute": "depth", "qualifiers": [], "value": "3.2", "units": "m",
        "point_value": "3.2", "lower": None, "upper": None, "list_values": None,
        "tolerance": None, "standard_deviation": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Rung 1: dedup key
# ---------------------------------------------------------------------------


def test_dedup_key_matches_across_pages_for_equal_point_quantities():
    a = _raw_quantity()
    b = dict(a)  # same quantity, would appear on a different page
    assert _quantity_dedup_key(a) == _quantity_dedup_key(b)


def test_dedup_key_normalizes_numeric_and_unit_formatting():
    a = _raw_quantity(value="3.20", units="M", point_value="3.20")
    b = _raw_quantity(value="3.2", units="m", point_value="3.2")
    assert _quantity_dedup_key(a) == _quantity_dedup_key(b)


def test_dedup_key_normalizes_range_bounds():
    """lower/upper carry the normalized numeric identity of a range now --
    value is raw provenance text, not a packed "(lower, upper)" format to
    parse, so this holds value constant and varies only lower/upper's
    formatting (unlike the point-quantity test above, which varies value)."""
    a = _raw_quantity(qualifiers=["IsRange"], value="3-7", point_value=None,
                       lower="3.0", upper="7.0")
    b = _raw_quantity(qualifiers=["IsRange"], value="3-7", point_value=None,
                       lower="3", upper="7")
    assert _quantity_dedup_key(a) == _quantity_dedup_key(b)


def test_quantity_item_schema_matches_parse_quantity_response_scalar_shape():
    """QuantityItem's scalar quantity fields must be widened the same way as
    MeasurementLM's ParseQuantityResponse (same gpt-oss-120b guided-decoding
    stall, same fix -- see tests/test_measurementlm.py's own
    test_parse_quantity_response_schema_matches_validated_variant_d_shape)."""
    props = QuantityItem.model_json_schema()["properties"]
    for field in ("point_value", "lower", "upper", "tolerance", "standard_deviation"):
        any_of = props[field]["anyOf"]
        assert [t.get("type") for t in any_of] == ["string", "number", "null"], (
            f"{field}: expected str | float | None (in that order), got {any_of}"
        )
    assert props["list_values"]["anyOf"] == [{"type": "array", "items": {"type": "string"}}, {"type": "null"}]


def test_quantity_item_validates_bare_numbers_for_scalar_fields():
    payload = {
        "value": "60 ± 5", "units": None, "qualifiers": ["HasTolerance"],
        "point_value": 60.0, "lower": None, "upper": None, "list_values": None,
        "tolerance": 5.0, "standard_deviation": None,
    }
    result = response_validator(QuantityItem, json.dumps(payload))
    assert result["point_value"] == 60.0 and isinstance(result["point_value"], float)
    assert result["tolerance"] == 5.0 and isinstance(result["tolerance"], float)


def test_dedup_key_treats_str_and_float_scalars_as_equal():
    """2026-09-24-gptoss120b-parsequantity-schema-diag-{01,02}'s fix widened
    point_value/lower/upper/tolerance/standard_deviation to str | float |
    None: a quantity collected as the bare float 3.2 (the new, previously-
    stalling shape) must dedup-collapse with one collected as the string
    "3.2" (the old shape) -- _norm_scalar's float(v) round-trips either
    representation to the same normalized key."""
    a = _raw_quantity(point_value="3.2")
    b = _raw_quantity(point_value=3.2)
    assert _quantity_dedup_key(a) == _quantity_dedup_key(b)


def test_dedup_key_differs_across_documents_and_attributes():
    same_doc_diff_attr = _raw_quantity(document_id=0, attribute="depth")
    same_doc_diff_attr2 = _raw_quantity(document_id=0, attribute="tn")
    diff_doc = _raw_quantity(document_id=1, attribute="depth")
    keys = {
        _quantity_dedup_key(same_doc_diff_attr),
        _quantity_dedup_key(same_doc_diff_attr2),
        _quantity_dedup_key(diff_doc),
    }
    assert len(keys) == 3


def test_dedup_key_differs_when_qualifiers_differ():
    """qualifiers are now part of the dedup key: a plain point and an
    approximate point with the same value/units must not collapse into one
    quantity -- they're distinct claims about what the paper reported."""
    plain = _raw_quantity(qualifiers=[])
    approximate = _raw_quantity(qualifiers=["IsApproximate"])
    assert _quantity_dedup_key(plain) != _quantity_dedup_key(approximate)


def test_deduplicate_quantities_aggregates_page_numbers(monkeypatch):
    mlm = _make_mlm()
    page1 = _raw_quantity(page_number=1)
    page2 = dict(page1, page_number=2)
    distinct = dict(page1, page_number=3, value="5.0", point_value="5.0")

    deduped = mlm._deduplicate_quantities([page1, page2, distinct])

    assert len(deduped) == 2
    by_value = {r["value"]: r for r in deduped}
    assert by_value["3.2"]["page_number"] == [1, 2]
    assert by_value["5.0"]["page_number"] == [3]


def test_deduplicate_quantities_aggregates_table_numbers():
    mlm = _make_mlm()
    prose = _raw_quantity(page_number=1, table_number=None)
    same_value_in_table = dict(prose, page_number=2, table_number=3)

    deduped = mlm._deduplicate_quantities([prose, same_value_in_table])

    assert len(deduped) == 1
    assert deduped[0]["page_number"] == [1, 2]
    assert deduped[0]["table_number"] == [None, 3]


# ---------------------------------------------------------------------------
# Rung 1: _collect_quantities logs but keeps an inconsistent item
# ---------------------------------------------------------------------------


def test_collect_quantities_logs_but_keeps_inconsistent_item(monkeypatch, capsys):
    """A collected item whose qualifiers don't match its populated fields is
    kept in full and only logged -- the same non-fatal policy as
    MeasurementLM._parse_quantities(), via the shared check_quantity_consistency()."""
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        # Tagged IsRange but lower/upper both left null -- inconsistent, not fatal.
        item = (
            '{"value": "3-7", "units": "m", "qualifiers": ["IsRange"], '
            '"point_value": null, "lower": null, "upper": null, '
            '"list_values": null, "tolerance": null, "standard_deviation": null, '
            '"table_number": null}'
        )
        return [f'{{"items": [{item}]}}' for _ in message_sets]

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    quantities = mlm._collect_quantities(max_tokens=2048)

    assert len(quantities) == 2  # 2 pages in _DOC x 1 attribute
    assert quantities[0]["qualifiers"] == ["IsRange"]
    assert quantities[0]["lower"] is None and quantities[0]["upper"] is None
    out = capsys.readouterr().out
    assert "IsRange tagged but lower/upper both null" in out


# ---------------------------------------------------------------------------
# _merge_field_schemas
# ---------------------------------------------------------------------------


class _EntitySchema(BaseModel):
    name: str | None
    location: str | None


class _EventSchema(BaseModel):
    date: str | None


class _OverlappingEventSchema(BaseModel):
    name: str | None  # collides with _EntitySchema.name


def test_merge_field_schemas_combines_entity_and_event_fields():
    Merged = _merge_field_schemas(_EntitySchema, _EventSchema, "Merged")
    assert set(Merged.model_fields) == {"name", "location", "date"}


def test_merge_field_schemas_with_no_event_schema():
    Merged = _merge_field_schemas(_EntitySchema, None, "Merged")
    assert set(Merged.model_fields) == {"name", "location"}


def test_merge_field_schemas_raises_on_field_name_collision():
    with pytest.raises(ValueError):
        _merge_field_schemas(_EntitySchema, _OverlappingEventSchema, "Merged")


# ---------------------------------------------------------------------------
# Shared fixture: a post-dedup quantity
# ---------------------------------------------------------------------------


def _quantity(**overrides):
    """Post-dedup quantity shape: page_number/table_number are already
    list-wrapped (see _raw_quantity() above for the pre-dedup shape)."""
    base = {
        "document_id": 0, "attribute": "depth", "qualifiers": [],
        "value": "3.2", "units": "m", "point_value": "3.2", "lower": None,
        "upper": None, "list_values": None, "tolerance": None, "standard_deviation": None,
        "page_number": [1], "table_number": [None],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Rung 2: fit() end-to-end with a stubbed _call_batch
# ---------------------------------------------------------------------------


_ATTRIBUTE_INFO = {
    "depth": {"description": "Maximum depth", "units": ["m"]},
}


def _make_mlm(**overrides):
    kwargs = dict(
        model_name="test-model",
        entity_identification_prompt="Identify entities. Fields: name, location.",
        entity_identification_schema=_EntitySchema,
        attribute_info_dict=_ATTRIBUTE_INFO,
        api_base="http://localhost:0/v1",
        use_extra_body=False,
        measurement_event_schema=_EventSchema,
        measurement_event_prompt="EVENT FIELDS: - date: the date of measurement.",
    )
    kwargs.update(overrides)
    return MeasurementLMv2(**kwargs)


_DOC = (
    '<page number="1">Lake A maximum depth is 3.2 m.</page>'
    '<page number="2">As reported earlier, the site\'s maximum depth was 3.2 m.</page>'
)


def test_fit_collects_standardizes_dedupes_and_contextualizes(monkeypatch):
    mlm = _make_mlm()
    calls = []

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        name = response_format["json_schema"]["name"]
        calls.append((name, len(message_sets)))
        if name == "quantity_list":
            # One call per (page, attribute): 2 pages x 1 attribute = 2 calls.
            assert len(message_sets) == 2
            item = (
                '{"value": "3.2", "units": "m", "qualifiers": [], '
                '"point_value": "3.2", "lower": null, "upper": null, '
                '"list_values": null, "tolerance": null, "standard_deviation": null, '
                '"table_number": null}'
            )
            return [f'{{"items": [{item}]}}', f'{{"items": [{item}]}}']
        if name == "standardize_quantity":
            # One call per raw quantity, pre-dedup: still 2 (one per page).
            assert len(message_sets) == 2
            resp = '{"explanation": "ok", "units": "m"}'
            return [resp, resp]
        if name == "entity_event_list":
            # Both pages produced an identical quantity -> deduped to 1 quantity -> 1 call (no copy-back).
            assert len(message_sets) == 1
            return [
                '{"items": [{"name": "Lake A", "location": "WI", "date": "2020"}, '
                '{"name": "Lake B", "location": "WI", "date": "2021"}]}'
            ]
        raise AssertionError(f"unexpected _call_batch invocation: {name}")

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    records = mlm.fit([_DOC], collection_max_tokens=2048)

    assert [c[0] for c in calls] == ["quantity_list", "standardize_quantity", "entity_event_list"]
    assert len(records) == 2  # one deduplicated quantity -> two distinct measurements

    names = {r["name"] for r in records}
    assert names == {"Lake A", "Lake B"}
    for r in records:
        assert r["attribute"] == "depth"
        assert r["value"] == "3.2"
        assert r["units"] == "m"
        assert r["qualifiers"] == []
        assert r["point_value"] == "3.2"
        assert r["page_number"] == [1, 2]  # found on both pages, aggregated


def test_fit_raises_if_event_schema_set_without_event_prompt():
    mlm = _make_mlm(measurement_event_prompt=None)
    with pytest.raises(ValueError, match="measurement_event_prompt"):
        mlm.fit([_DOC], collection_max_tokens=2048)


# ---------------------------------------------------------------------------
# standardize_context: v2 honors the constructor argument MeasurementLM already has
# ---------------------------------------------------------------------------


def _capture_standardize_prompts(monkeypatch, mlm_cls, mlm, call):
    """Run ``call(mlm)`` with _call_batch stubbed; return the prompt strings it sent."""
    sent = []

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        sent.extend(m[0]["content"] for m in message_sets)
        return ['{"explanation": "ok", "units": "m"}'] * len(message_sets)

    monkeypatch.setattr(mlm_cls, "_call_batch", fake_call_batch)
    call(mlm)
    return sent


def test_value_only_standardization_prompt_matches_v1_byte_for_byte(monkeypatch):
    from scholarlm.measurementlm import MeasurementLM

    v1 = MeasurementLM(
        model_name="test-model", entity_identification_prompt="x", entity_identification_schema=_EntitySchema,
        attribute_info_dict=_ATTRIBUTE_INFO, api_base="http://localhost:0/v1", use_extra_body=False,
        standardize_context="value_only",
    )
    v1.data = [{"document_id": 0, "attribute": "depth", "value": "3.2", "units": "meters", "context": "PAGE TEXT"}]
    v1_prompts = _capture_standardize_prompts(monkeypatch, MeasurementLM, v1, lambda m: m._standardize())

    v2 = _make_mlm(standardize_context="value_only")
    v2.data = [{"document_id": 0, "context": _DOC}]
    quantity = {"document_id": 0, "page_number": 1, "attribute": "depth", "value": "3.2", "units": "meters"}
    v2_prompts = _capture_standardize_prompts(
        monkeypatch, MeasurementLMv2, v2, lambda m: m._standardize_quantities([quantity])
    )

    assert len(v1_prompts) == len(v2_prompts) == 1
    assert v2_prompts == v1_prompts
    assert "PAGE TEXT" not in v2_prompts[0] and "Lake A" not in v2_prompts[0]   # no page context


def test_full_standardization_prompt_still_carries_the_page_text(monkeypatch):
    v2 = _make_mlm(standardize_context="full")
    v2.data = [{"document_id": 0, "context": _DOC}]
    quantity = {"document_id": 0, "page_number": 1, "attribute": "depth", "value": "3.2", "units": "meters"}
    (prompt,) = _capture_standardize_prompts(
        monkeypatch, MeasurementLMv2, v2, lambda m: m._standardize_quantities([quantity])
    )
    assert "Lake A maximum depth is 3.2 m." in prompt


def test_standardize_context_rejects_unknown_value():
    with pytest.raises(ValueError):
        _make_mlm(standardize_context="page_only")


# ---------------------------------------------------------------------------
# _contextualize_quantities: one call per quantity, nothing echoed or matched
# ---------------------------------------------------------------------------


def _ctx_mlm(monkeypatch, responder):
    """MeasurementLMv2 whose _call_batch returns responder(message_sets)."""
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}, {"document_id": 1, "context": "<page number=\"1\">Other paper.</page>"}]
    mlm.context_length_exceeded_docs = set()
    seen = {}

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        seen["name"] = response_format["json_schema"]["name"]
        seen["messages"] = message_sets
        seen["kwargs"] = kwargs
        return responder(message_sets)

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)
    return mlm, seen


def _items(*pairs):
    return "{\"items\": [" + ", ".join(
        f'{{"name": "{n}", "location": "WI", "date": "{d}"}}' for n, d in pairs) + "]}"


def test_contextualize_makes_one_call_per_quantity_and_attaches_items_to_the_quantity_sent(monkeypatch):
    qs = [_quantity(value="1.5 m", point_value="1.5"), _quantity(value="7.4 ± 0.2", point_value="7.4"),
          _quantity(document_id=1, value="3.2")]
    mlm, seen = _ctx_mlm(monkeypatch, lambda ms: [_items(("Lake A", "2020")), _items(("Lake B", "2021")), _items(("Lake C", "2022"))])
    records = mlm._contextualize_quantities(qs)
    assert seen["name"] == "entity_event_list" and len(seen["messages"]) == 3
    # raw value strings (units text, +/-) survive untouched: nothing was echoed or matched
    assert [(r["value"], r["name"]) for r in records] == [("1.5 m", "Lake A"), ("7.4 ± 0.2", "Lake B"), ("3.2", "Lake C")]
    assert [r["document_id"] for r in records] == [0, 0, 1]
    assert mlm.contextualization_stats == {"sent": 3, "records": 3, "dropped": 0,
                                           "dropped_by_reason": {"context_length": 0, "invalid_response": 0, "no_entity": 0}}


def test_contextualize_each_call_gets_its_own_documents_full_text_and_only_its_quantity(monkeypatch):
    qs = [_quantity(value="1.5 m"), _quantity(document_id=1, value="3.2")]
    mlm, seen = _ctx_mlm(monkeypatch, lambda ms: [_items(("A", "2020")), _items(("B", "2021"))])
    mlm._contextualize_quantities(qs)
    p0, p1 = (m[0]["content"] for m in seen["messages"])
    assert "Lake A maximum depth is 3.2 m." in p0 and "Other paper." not in p0
    assert "Other paper." in p1 and "Lake A maximum depth" not in p1
    assert "Value: 1.5 m" in p0 and "Value: 3.2" in p1 and "Value: 1.5 m" not in p1
    assert "copy" not in p0.lower()                       # no copy-back instruction survives


def test_contextualize_expands_a_quantity_with_several_items_into_several_records(monkeypatch):
    mlm, _ = _ctx_mlm(monkeypatch, lambda ms: [_items(("Lake A", "2020"), ("Lake B", "2021"))])
    records = mlm._contextualize_quantities([_quantity(value="3.2")])
    assert [r["name"] for r in records] == ["Lake A", "Lake B"] and all(r["value"] == "3.2" for r in records)
    assert mlm.contextualization_stats["sent"] == 1 and mlm.contextualization_stats["records"] == 2


def test_contextualize_counts_and_reports_every_kind_of_drop(monkeypatch, capsys):
    from scholarlm.measurementlm import ContextLengthExceededError
    qs = [_quantity(value="1"), _quantity(value="2"), _quantity(value="3"), _quantity(value="4")]
    mlm, _ = _ctx_mlm(monkeypatch, lambda ms: [
        _items(("A", "2020")), "{\"items\": []}", "not json", ContextLengthExceededError("too long")])
    records = mlm._contextualize_quantities(qs)
    assert [r["value"] for r in records] == ["1"]
    assert mlm.contextualization_stats == {"sent": 4, "records": 1, "dropped": 3,
                                           "dropped_by_reason": {"context_length": 1, "invalid_response": 1, "no_entity": 1}}
    assert mlm.context_length_exceeded_docs == {0}
    out = capsys.readouterr().out
    assert "3 quantity/quantities could not be attributed to any entity and were dropped" in out
    assert "Validation error in contextualization response" in out


def test_contextualize_raises_if_entity_schema_collides_with_quantity_fields(monkeypatch):
    class _Bad(BaseModel):
        units: str | None = None                          # collides with the quantity's units
    mlm, _ = _ctx_mlm(monkeypatch, lambda ms: [])
    mlm.entity_identification_schema = _Bad
    mlm.measurement_event_schema = None
    with pytest.raises(ValueError, match="collide"):
        mlm._contextualize_quantities([_quantity()])


def test_contextualize_no_quantities_sends_nothing_and_records_zero_stats(monkeypatch):
    mlm, seen = _ctx_mlm(monkeypatch, lambda ms: pytest.fail("no call expected"))
    assert mlm._contextualize_quantities([]) == [] and seen == {}
    assert mlm.contextualization_stats["sent"] == 0


# ---------------------------------------------------------------------------
# collection max_tokens: explicit, validated, and actually sent
# ---------------------------------------------------------------------------


def test_collect_quantities_sends_the_given_max_tokens_to_every_call(monkeypatch):
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    seen = {}

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        seen["max_tokens"] = kwargs["max_tokens"]
        return ['{"items": []}'] * len(message_sets)

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)
    mlm._collect_quantities(max_tokens=8192)
    assert seen["max_tokens"] == 8192
    mlm._collect_quantities(max_tokens=2048)
    assert seen["max_tokens"] == 2048


@pytest.mark.parametrize("bad", [0, -1, 2048.0, "2048", None, True])
def test_collect_quantities_rejects_a_bad_max_tokens(bad):
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    with pytest.raises(ValueError, match="max_tokens"):
        mlm._collect_quantities(max_tokens=bad)


def test_collect_quantities_and_fit_have_no_default_cap():
    import inspect
    assert inspect.signature(MeasurementLMv2._collect_quantities).parameters["max_tokens"].default is inspect.Parameter.empty
    assert inspect.signature(MeasurementLMv2.fit).parameters["collection_max_tokens"].default is inspect.Parameter.empty
