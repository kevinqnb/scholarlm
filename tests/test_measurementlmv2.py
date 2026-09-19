"""Unit tests for MeasurementLMv2's quantity-first pipeline.

Rung 1: validate_quantity_shape (the type/quantifier/value/CI invariant
checker) and the code-level dedup key, on hand-built tuples -- no network,
no LLM. This is the highest-value test in this build: it's the one place a
genuinely new field (type/quantifier/CI) is checked structurally rather than
trusted from a prompt.

Rung 2: fit() end-to-end with a stubbed _call_batch (same pattern as
tests/test_measurementlm.py), verifying the three-call sequence (collect ->
standardize -> contextualize), that page numbers are aggregated across
duplicate-quantity pages, and that a single deduplicated quantity can expand
into multiple final records via the contextualization step.
"""
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.measurementlmv2 import (
    MeasurementLMv2,
    _merge_field_schemas,
    _quantity_dedup_key,
    validate_quantity_shape,
)


# ---------------------------------------------------------------------------
# Rung 1: validate_quantity_shape
# ---------------------------------------------------------------------------


def _item(**overrides):
    base = {
        "value": "12.3", "units": "m", "type": "point", "quantifier": None,
        "ci_lower": None, "ci_upper": None, "ci": None,
    }
    base.update(overrides)
    return base


@pytest.mark.parametrize("item", [
    _item(),
    _item(type="range", value="(3, 7)"),
    _item(type="range", value="(3.0, 7.5)"),
    _item(type="inequality", value="5", quantifier="<"),
    _item(type="inequality", value="5", quantifier=">="),
    _item(ci_lower="1.0", ci_upper="2.0"),
    _item(ci="0.5"),
])
def test_validate_quantity_shape_accepts_well_formed_items(item):
    validate_quantity_shape(item)  # must not raise


@pytest.mark.parametrize("item, reason", [
    (_item(type="range", value="3.5"), "range value not tuple-formatted"),
    (_item(type="range", value="(7, 3)"), "range bounds reversed"),
    (_item(type="range", value="(3, 7)", quantifier="<"), "range must not carry a quantifier"),
    (_item(type="inequality", value="5", quantifier=None), "inequality missing quantifier"),
    (_item(type="inequality", value="5", quantifier="=="), "inequality quantifier not in allowed set"),
    (_item(type="inequality", value="(3, 7)", quantifier="<"), "inequality value must be a bare number"),
    (_item(type="point", value="12.3", quantifier="<"), "point must not carry a quantifier"),
    (_item(type="point", value="(3, 7)"), "point value must be a bare number"),
    (_item(type="unknown_type"), "unrecognized type"),
    (_item(ci_lower="1.0", ci_upper=None), "ci_lower without ci_upper"),
    (_item(ci_lower=None, ci_upper="2.0"), "ci_upper without ci_lower"),
    (_item(ci="0.5", ci_lower="1.0", ci_upper="2.0"), "ci and ci_lower/ci_upper both set"),
    (_item(ci="not-a-number"), "ci not a bare number"),
])
def test_validate_quantity_shape_rejects_malformed_items(item, reason):
    with pytest.raises(ValueError):
        validate_quantity_shape(item)


# ---------------------------------------------------------------------------
# Rung 1: dedup key
# ---------------------------------------------------------------------------


def test_dedup_key_matches_across_pages_for_equal_point_quantities():
    a = {"document_id": 0, "attribute": "depth", "type": "point", "quantifier": None,
         "value": "3.2", "units": "m", "ci_lower": None, "ci_upper": None, "ci": None}
    b = dict(a)  # same quantity, would appear on a different page
    assert _quantity_dedup_key(a) == _quantity_dedup_key(b)


def test_dedup_key_normalizes_numeric_and_unit_formatting():
    a = {"document_id": 0, "attribute": "depth", "type": "point", "quantifier": None,
         "value": "3.20", "units": "M", "ci_lower": None, "ci_upper": None, "ci": None}
    b = {"document_id": 0, "attribute": "depth", "type": "point", "quantifier": None,
         "value": "3.2", "units": "m", "ci_lower": None, "ci_upper": None, "ci": None}
    assert _quantity_dedup_key(a) == _quantity_dedup_key(b)


def test_dedup_key_normalizes_range_bounds():
    a = {"document_id": 0, "attribute": "depth", "type": "range", "quantifier": None,
         "value": "(3.0, 7.0)", "units": "m", "ci_lower": None, "ci_upper": None, "ci": None}
    b = {"document_id": 0, "attribute": "depth", "type": "range", "quantifier": None,
         "value": "(3, 7)", "units": "m", "ci_lower": None, "ci_upper": None, "ci": None}
    assert _quantity_dedup_key(a) == _quantity_dedup_key(b)


def test_dedup_key_differs_across_documents_and_attributes():
    base = {"type": "point", "quantifier": None, "value": "3.2", "units": "m",
            "ci_lower": None, "ci_upper": None, "ci": None}
    same_doc_diff_attr = base | {"document_id": 0, "attribute": "depth"}
    same_doc_diff_attr2 = base | {"document_id": 0, "attribute": "tn"}
    diff_doc = base | {"document_id": 1, "attribute": "depth"}
    keys = {
        _quantity_dedup_key(same_doc_diff_attr),
        _quantity_dedup_key(same_doc_diff_attr2),
        _quantity_dedup_key(diff_doc),
    }
    assert len(keys) == 3


def test_deduplicate_quantities_aggregates_page_numbers(monkeypatch):
    mlm = _make_mlm()
    page1 = {"document_id": 0, "page_number": 1, "table_number": None, "attribute": "depth", "type": "point",
             "quantifier": None, "value": "3.2", "units": "m", "ci_lower": None, "ci_upper": None, "ci": None}
    page2 = dict(page1, page_number=2)
    distinct = dict(page1, page_number=3, value="5.0")

    deduped = mlm._deduplicate_quantities([page1, page2, distinct])

    assert len(deduped) == 2
    by_value = {r["value"]: r for r in deduped}
    assert by_value["3.2"]["page_number"] == [1, 2]
    assert by_value["5.0"]["page_number"] == [3]


def test_deduplicate_quantities_aggregates_table_numbers():
    mlm = _make_mlm()
    prose = {"document_id": 0, "page_number": 1, "table_number": None, "attribute": "depth", "type": "point",
              "quantifier": None, "value": "3.2", "units": "m", "ci_lower": None, "ci_upper": None, "ci": None}
    same_value_in_table = dict(prose, page_number=2, table_number=3)

    deduped = mlm._deduplicate_quantities([prose, same_value_in_table])

    assert len(deduped) == 1
    assert deduped[0]["page_number"] == [1, 2]
    assert deduped[0]["table_number"] == [None, 3]


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
    base = {
        "document_id": 0, "attribute": "depth", "type": "point", "quantifier": None,
        "value": "3.2", "units": "m", "ci_lower": None, "ci_upper": None, "ci": None,
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
        f'{{"attribute": "depth", "value": "{value}", "units": "m", "type": "point", '
        f'"quantifier": null, "ci_lower": null, "ci_upper": null, "ci": null, '
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


def test_contextualize_quantities_matches_despite_echoed_quantifier_and_ci_drift(monkeypatch):
    """Regression for the 2026-09-19 smoke test (gemma-3-27b, pond): the model
    added an inequality quantifier to a plain point quantity and echoed its
    absent CI fields as the literal string "None" instead of JSON null.
    Matching on (attribute, value, units) must still succeed, and the
    ORIGINAL type/quantifier/CI -- not the model's drifted ones -- must reach
    the record.
    """
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    mlm.context_length_exceeded_docs = set()
    q = _quantity(value="26", units="km^2")  # type point, quantifier None, ci None

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        return [
            '{"items": [{"attribute": "depth", "value": "26", "units": "km^2", '
            '"type": "point", "quantifier": "<=", "ci_lower": "None", "ci_upper": "None", '
            '"ci": "None", "name": "Lake A", "location": "WI", "date": "2020"}]}'
        ]

    monkeypatch.setattr(MeasurementLMv2, "_call_batch", fake_call_batch)

    records = mlm._contextualize_quantities([q])

    assert len(records) == 1
    assert records[0]["quantifier"] is None  # original, not the model's "<="
    assert records[0]["ci"] is None          # original, not the string "None"
    assert records[0]["name"] == "Lake A"


def test_contextualize_quantities_disambiguates_reduced_key_collision(monkeypatch):
    """Two quantities share (attribute, value, units) but differ in type --
    a point value that coincidentally equals a separately-reported
    inequality bound. Matching must fall back to the full identity key
    rather than attaching the response to whichever candidate comes first.
    """
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": _DOC}]
    mlm.context_length_exceeded_docs = set()
    q_point = _quantity(value="5", units="m", type="point", quantifier=None, page_number=[1])
    q_bound = _quantity(value="5", units="m", type="inequality", quantifier="<", page_number=[2])

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        return [
            '{"items": [{"attribute": "depth", "value": "5", "units": "m", '
            '"type": "inequality", "quantifier": "<", "ci_lower": null, "ci_upper": null, '
            '"ci": null, "name": "Bound Lake", "location": "WI", "date": "2020"}]}'
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
        clean_tables=False,
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
                '{"value": "3.2", "units": "m", "type": "point", '
                '"quantifier": null, "ci_lower": null, "ci_upper": null, "ci": null}'
            )
            return [f'{{"items": [{item}]}}', f'{{"items": [{item}]}}']
        if name == "standardize_quantity":
            # One call per raw quantity, pre-dedup: still 2 (one per page).
            assert len(message_sets) == 2
            resp = (
                '{"explanation": "ok", "value": "3.2", "units": "m", '
                '"ci_lower": null, "ci_upper": null, "ci": null}'
            )
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
        assert r["page_number"] == [1, 2]  # found on both pages, aggregated


def test_fit_raises_if_clean_tables_true():
    mlm = _make_mlm(clean_tables=True)
    with pytest.raises(ValueError, match="table cleaning"):
        mlm.fit([_DOC])


def test_fit_raises_if_event_schema_set_without_event_prompt():
    mlm = _make_mlm(measurement_event_prompt=None)
    with pytest.raises(ValueError, match="measurement_event_prompt"):
        mlm.fit([_DOC])
