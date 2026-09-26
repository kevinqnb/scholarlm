"""Unit tests for MeasurementLM Ablation 7 (no standardize/parse-quantities/
deduplicate).

Uses a hand-built fixture -- no network calls, no vLLM/frontier endpoint
required. Verifies:
  * `_standardize()` leaves every record's `units` (and everything else)
    untouched and never calls `_call_batch`.
  * `_parse_quantities()` writes every qualifier/shape field
    (`scholarlm.config.QUALIFIER_FIELD_NAMES`) as null on every record,
    leaves `value`/`units` untouched, and never calls `_call_batch`.
  * `_deduplicate()` is the identity function -- two records that the base
    `MeasurementLM._deduplicate()` would merge (same entity_id/attribute,
    equal value+units) survive as two separate rows instead.
  * `fit()` (inherited unchanged from the base class) still dispatches the
    original seven pipeline steps, and the "final" step's output reflects
    all three skips together.
"""
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.config import QUALIFIER_FIELD_NAMES
from scholarlm.measurementlm_ablation7 import MeasurementLMAblation7


class _EntitySchema(BaseModel):
    name: str | None
    location: str | None


_ATTRIBUTE_INFO = {
    "depth": {"description": "Maximum depth", "units": ["m"]},
}


def _make_mlm(**overrides):
    kwargs = dict(
        model_name="test-model",
        entity_identification_prompt="Identify entities.",
        entity_identification_schema=_EntitySchema,
        attribute_info_dict=_ATTRIBUTE_INFO,
        api_base="http://localhost:0/v1",
        use_extra_body=False,
    )
    kwargs.update(overrides)
    return MeasurementLMAblation7(**kwargs)


def _base_datapoint(document_id, value, units="m"):
    return {
        "document_id": document_id, "entity_id": f"doc_{document_id}_entity_0",
        "name": "Lake A", "location": "WI", "context": f"DOC{document_id} text",
        "attribute": "depth", "value": value, "units": units, "attribute_terms": [],
    }


def _forbid_call_batch(*args, **kwargs):
    raise AssertionError("ablation 7 must never call _call_batch from the cleanup stage")


def test_standardize_leaves_units_untouched_without_llm_call(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(mlm, "_call_batch", _forbid_call_batch)
    mlm.data = [_base_datapoint(0, "12.5", units="ft"), _base_datapoint(1, "3", units="m")]

    result = mlm._standardize()

    assert result == mlm.data
    assert result is not mlm.data  # plain copy, per base class's own convention
    assert [r["units"] for r in result] == ["ft", "m"]


def test_parse_quantities_nulls_every_qualifier_field_without_llm_call(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(mlm, "_call_batch", _forbid_call_batch)
    mlm.data = [_base_datapoint(0, "12.5", units="ft"), _base_datapoint(1, "3", units="m")]

    result = mlm._parse_quantities()

    assert len(result) == 2
    for original, parsed in zip(mlm.data, result):
        assert parsed["value"] == original["value"]
        assert parsed["units"] == original["units"]
        for field in QUALIFIER_FIELD_NAMES:
            assert parsed[field] is None


def test_deduplicate_is_identity_even_for_would_be_duplicates():
    mlm = _make_mlm()
    data = [
        _base_datapoint(0, "12.5", units="m"),
        _base_datapoint(0, "12.5", units="m"),  # same entity_id/attribute/value/units
    ]

    result = mlm._deduplicate(data)

    assert result is data
    assert len(result) == 2


def test_fit_dispatches_original_seven_steps_and_skips_cleanup(monkeypatch):
    mlm = _make_mlm()

    called = []

    def stub(name, ret):
        def _stub(*args, **kwargs):
            called.append(name)
            return ret
        return _stub

    raw_records = [
        _base_datapoint(0, "12.5", units="ft"),
        _base_datapoint(0, "12.5", units="ft"),  # a would-be duplicate
    ]

    monkeypatch.setattr(mlm, "_extract_entities", stub("_extract_entities", []))
    monkeypatch.setattr(mlm, "_entity_provenance", stub("_entity_provenance", {}))
    monkeypatch.setattr(mlm, "_detect_attributes", stub("_detect_attributes", {}))
    monkeypatch.setattr(mlm, "_attribute_provenance", stub("_attribute_provenance", {}))
    monkeypatch.setattr(mlm, "_extract_values_from_text", stub("_extract_values_from_text", raw_records))
    monkeypatch.setattr(mlm, "_extract_values_from_tables", stub("_extract_values_from_tables", []))
    monkeypatch.setattr(mlm, "_call_batch", _forbid_call_batch)

    result = mlm.fit(["doc text"])

    assert called == [
        "_extract_entities",
        "_entity_provenance",
        "_detect_attributes",
        "_attribute_provenance",
        "_extract_values_from_text",
        "_extract_values_from_tables",
    ]
    assert set(mlm.step_seconds) == {
        "entities", "entity_prov", "attributes", "attribute_prov", "events",
        "values_text", "values_tables", "final",
    }
    # No deduplication: both raw records survive.
    assert len(result) == 2
    # No standardization: units left exactly as extracted.
    assert all(r["units"] == "ft" for r in result)
    # No quantity parsing: every qualifier field null.
    for record in result:
        for field in QUALIFIER_FIELD_NAMES:
            assert record[field] is None
