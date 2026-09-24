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

    quantities = mlm._collect_quantities()

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
# _group_quantities_for_contextualization
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


def test_group_quantities_prefers_table_over_prose_and_keeps_docs_separate():
    mlm = _make_mlm()
    prose = _quantity(value="3.2", table_number=[None])
    in_table = _quantity(value="5.0", table_number=[2])
    mixed = _quantity(value="7.5", table_number=[None, 2])
    other_doc = _quantity(document_id=1, value="3.2", table_number=[None])

    groups = mlm._group_quantities_for_contextualization([prose, in_table, mixed, other_doc])

    assert set(groups.keys()) == {(0, None), (0, 2), (1, None)}
    assert groups[(0, None)] == [prose]
    assert groups[(0, 2)] == [in_table, mixed]  # mixed goes with the table, not prose
    assert groups[(1, None)] == [other_doc]


# ---------------------------------------------------------------------------
# _contextualize_quantities: matching by echoed fields, not position
# ---------------------------------------------------------------------------


def _attributed_item(value="3.2", name="Lake A", location="WI", date="2020"):
    return (
        f'{{"attribute": "depth", "value": "{value}", "units": "m", "qualifiers": [], '
        f'"point_value": "{value}", "lower": null, "upper": null, "list_values": null, '
        f'"tolerance": null, "standard_deviation": null, '
        f'"name": "{name}", "location": "{location}", "date": "{date}"}}'
    )


def test_contextualize_quantities_matches_by_echo_and_drops_mismatch_and_missing(monkeypatch):
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    mlm.context_length_exceeded_docs = set()
    q_a = _quantity(value="3.2")
    q_b = _quantity(value="5.0")  # never echoed back by the model below

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        assert response_format["json_schema"]["name"] == "attributed_quantity_list"
        assert len(message_sets) == 1  # both quantities are prose, same document -> 1 group
        return [
            "{\"items\": [" + _attributed_item(value="3.2", name="Lake A") + ", "
            # echoes a value that was never sent -- must be dropped, not matched by position
            + _attributed_item(value="9.9", name="Ghost Lake") + "]}"
        ]

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    records = mlm._contextualize_quantities([q_a, q_b])

    assert len(records) == 1
    assert records[0]["value"] == "3.2"
    assert records[0]["name"] == "Lake A"


def test_contextualize_quantities_batches_prose_and_table_separately(monkeypatch):
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    mlm.context_length_exceeded_docs = set()
    q_prose = _quantity(value="3.2", page_number=[1], table_number=[None])
    q_table = _quantity(value="5.0", page_number=[2], table_number=[2])

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        assert len(message_sets) == 2  # prose group and table-2 group, sent separately
        return [
            "{\"items\": [" + _attributed_item(value="3.2", name="Lake A") + "]}",
            "{\"items\": [" + _attributed_item(value="5.0", name="Lake B") + "]}",
        ]

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    records = mlm._contextualize_quantities([q_prose, q_table])

    by_name = {r["name"]: r for r in records}
    assert by_name["Lake A"]["value"] == "3.2"
    assert by_name["Lake B"]["value"] == "5.0"


def test_contextualize_quantities_matches_despite_echoed_shape_field_drift(monkeypatch):
    """Regression for the 2026-09-19 smoke test (gemma-3-27b, pond) under the
    pre-reconciliation type/quantifier/CI scheme: the model drifted echoed
    shape fields away from what was sent, and echoed an absent field as the
    literal string "None" instead of JSON null. Matching on (attribute,
    value, units) must still succeed, and the ORIGINAL qualifiers/shape
    fields -- not the model's drifted echo -- must reach the record.
    """
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    mlm.context_length_exceeded_docs = set()
    q = _quantity(value="26", units="km^2", qualifiers=[], point_value="26")

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        return [
            '{"items": [{"attribute": "depth", "value": "26", "units": "km^2", '
            '"qualifiers": ["IsRange"], "point_value": "None", "lower": "None", '
            '"upper": "None", "list_values": null, "tolerance": "None", '
            '"standard_deviation": "None", '
            '"name": "Lake A", "location": "WI", "date": "2020"}]}'
        ]

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    records = mlm._contextualize_quantities([q])

    assert len(records) == 1
    assert records[0]["qualifiers"] == []      # original, not the model's ["IsRange"]
    assert records[0]["point_value"] == "26"   # original, not the string "None"
    assert records[0]["lower"] is None         # original, not the string "None"
    assert records[0]["name"] == "Lake A"


def test_contextualize_quantities_disambiguates_reduced_key_collision(monkeypatch):
    """Two quantities share (attribute, value, units) but differ in shape --
    a point value that coincidentally equals a separately-reported range
    bound. Matching must fall back to the full identity key rather than
    attaching the response to whichever candidate comes first.
    """
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    mlm.context_length_exceeded_docs = set()
    q_point = _quantity(value="5", units="m", qualifiers=[], point_value="5", page_number=[1])
    q_bound = _quantity(value="5", units="m", qualifiers=["IsRange"], point_value=None,
                         lower=None, upper="5", page_number=[2])

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        return [
            '{"items": [{"attribute": "depth", "value": "5", "units": "m", '
            '"qualifiers": ["IsRange"], "point_value": null, "lower": null, '
            '"upper": "5", "list_values": null, "tolerance": null, '
            '"standard_deviation": null, '
            '"name": "Bound Lake", "location": "WI", "date": "2020"}]}'
        ]

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    records = mlm._contextualize_quantities([q_point, q_bound])

    assert len(records) == 1
    assert records[0]["page_number"] == [2]  # matched q_bound, not q_point
    assert records[0]["name"] == "Bound Lake"


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
        if name == "attributed_quantity_list":
            # Both pages produced an identical quantity -> deduped to 1 group (prose, doc 0).
            assert len(message_sets) == 1
            return [
                "{\"items\": [" + _attributed_item(value="3.2", name="Lake A", date="2020")
                + ", " + _attributed_item(value="3.2", name="Lake B", date="2021") + "]}"
            ]
        raise AssertionError(f"unexpected _call_batch invocation: {name}")

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    records = mlm.fit([_DOC])

    assert [c[0] for c in calls] == ["quantity_list", "standardize_quantity", "attributed_quantity_list"]
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
        mlm.fit([_DOC])
