"""Unit tests for MeasurementLM's extraction_mode branch (pipeline vs direct).

Uses a hand-built two-document fixture with a stubbed ``_call_batch`` — no
network calls, no vLLM/frontier endpoint required. Verifies:
  * extraction_mode="direct" runs fit() -> _extract_triples() -> _standardize()
    -> _deduplicate() and shapes records the way MeasurementLMAblation1's
    _extract_triples() does, with standardize/deduplicate applied on top
    (ablation1 itself skips both).
  * The default-constructed (extraction_mode="pipeline") instance dispatches
    fit() through the original seven-step sequence, never touching
    _extract_triples, so pipeline-mode behavior is provably unchanged.
  * _extract_triples()'s out-of-vocabulary-attribute retry/drop logic (added
    after a live smoke run surfaced the model emitting attribute names outside
    attribute_info_dict): a bad attribute triggers a retry, and a record still
    bad after retries are exhausted is dropped with a loud printed count
    rather than reaching _standardize() (which would KeyError). These tests
    exercise the real _call_batch retry loop via a stubbed _acall, not a
    stubbed _call_batch, since the retry loop itself is what's under test.
"""
import asyncio
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.measurementlm import MeasurementLM


class _EntitySchema(BaseModel):
    name: str | None
    location: str | None


class _DirectSchema(BaseModel):
    name: str | None
    location: str | None
    attribute: str
    value: str | None
    units: str | None


_ATTRIBUTE_INFO = {
    "depth": {"description": "Maximum depth", "units": ["m"]},
}


async def _fast_sleep(*args, **kwargs):
    """No-op stand-in for asyncio.sleep, to skip _call_batch's retry backoff in tests."""
    return None


def _make_mlm(**overrides):
    kwargs = dict(
        model_name="test-model",
        entity_identification_prompt="Identify entities.",
        entity_identification_schema=_EntitySchema,
        attribute_info_dict=_ATTRIBUTE_INFO,
        api_base="http://localhost:0/v1",
        clean_tables=False,
        use_extra_body=False,
    )
    kwargs.update(overrides)
    return MeasurementLM(**kwargs)


def test_direct_mode_fit_shapes_records_like_ablation1_then_standardizes_and_dedupes(monkeypatch):
    mlm = _make_mlm(
        extraction_mode="direct",
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract all measurements.",
    )

    calls = []

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        name = response_format["json_schema"]["name"]
        calls.append(name)
        if name == "direct_extraction_list":
            assert len(message_sets) == 2  # one message per document
            return [
                '{"items": [{"name": "Lake A", "location": "WI", '
                '"attribute": "depth", "value": "3.2", "units": "m"}]}',
                '{"items": [{"name": "Lake B", "location": "MN", '
                '"attribute": "depth", "value": "5.0", "units": "m"}]}',
            ]
        if name == "standardize_response":
            assert len(message_sets) == 2  # one message per extracted triple
            return [
                '{"explanation": "ok", "value": "3.2", "units": "m"}',
                '{"explanation": "ok", "value": "5.0", "units": "m"}',
            ]
        raise AssertionError(f"unexpected _call_batch invocation: {name}")

    monkeypatch.setattr(MeasurementLM, "_call_batch", fake_call_batch)

    records = mlm.fit(["doc text A", "doc text B"])

    assert calls == ["direct_extraction_list", "standardize_response"]
    assert len(records) == 2

    records_by_name = {r["name"]: r for r in records}
    assert set(records_by_name) == {"Lake A", "Lake B"}

    for name, expected_value, doc_text in [
        ("Lake A", "3.2", "doc text A"),
        ("Lake B", "5.0", "doc text B"),
    ]:
        rec = records_by_name[name]
        assert rec["attribute"] == "depth"
        assert rec["value"] == expected_value
        assert rec["units"] == "m"
        assert rec["attribute_terms"] == []
        assert rec["entity_id"].startswith("doc_")
        # _deduplicate wraps provenance fields in singleton lists and is a
        # structural no-op here: _extract_triples assigns a unique entity_id
        # per record, so nothing ever collides (see build note point #4).
        assert rec["context"] == [doc_text]
        assert rec["page_number"] == [None]
        assert rec["source"] == [None]


def test_direct_mode_retries_and_recovers_from_out_of_vocabulary_attribute(monkeypatch):
    mlm = _make_mlm(
        extraction_mode="direct",
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract all measurements.",
    )
    mlm.data = [{"document_id": 0, "context": "DOC0 text"}]

    attempts = {"doc0": 0}

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        attempts["doc0"] += 1
        if attempts["doc0"] == 1:
            # First attempt: out-of-vocabulary attribute -> validator rejects, retried.
            return (
                '{"items": [{"name": "Lake A", "location": "WI", '
                '"attribute": "hardness", "value": "3.2", "units": "m"}]}'
            )
        # Retry: valid attribute -> accepted.
        return (
            '{"items": [{"name": "Lake A", "location": "WI", '
            '"attribute": "depth", "value": "3.2", "units": "m"}]}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    records = mlm._extract_triples()

    assert attempts["doc0"] == 2  # one retry, then recovered
    assert len(records) == 1
    assert records[0]["attribute"] == "depth"


def test_direct_mode_drops_and_counts_when_still_out_of_vocabulary_after_retries(monkeypatch, capsys):
    mlm = _make_mlm(
        extraction_mode="direct",
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract all measurements.",
    )
    mlm.data = [{"document_id": 0, "context": "DOC0 text"}]

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        # Always out-of-vocabulary, even after every retry is exhausted.
        return (
            '{"items": [{"name": "Lake A", "location": "WI", '
            '"attribute": "hardness", "value": "3.2", "units": "m"}]}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    records = mlm._extract_triples()

    assert records == []
    out = capsys.readouterr().out
    assert "hardness" in out
    assert "dropped 1 record" in out


def test_pipeline_mode_default_construction_dispatches_original_seven_steps(monkeypatch):
    mlm = _make_mlm()
    assert mlm.extraction_mode == "pipeline"

    called = []

    def stub(name, ret):
        def _stub(*args, **kwargs):
            called.append(name)
            return ret
        return _stub

    monkeypatch.setattr(mlm, "_extract_entities", stub("_extract_entities", []))
    monkeypatch.setattr(mlm, "_entity_provenance", stub("_entity_provenance", {}))
    monkeypatch.setattr(mlm, "_detect_attributes", stub("_detect_attributes", {}))
    monkeypatch.setattr(mlm, "_attribute_provenance", stub("_attribute_provenance", {}))
    monkeypatch.setattr(mlm, "_extract_values_from_text", stub("_extract_values_from_text", []))
    monkeypatch.setattr(mlm, "_extract_values_from_tables", stub("_extract_values_from_tables", []))
    monkeypatch.setattr(mlm, "_standardize", stub("_standardize", []))
    monkeypatch.setattr(mlm, "_deduplicate", lambda data: called.append("_deduplicate") or [])

    def _fail(*args, **kwargs):
        raise AssertionError("_extract_triples must not be called in pipeline mode")
    monkeypatch.setattr(mlm, "_extract_triples", _fail)

    result = mlm.fit(["doc text"])

    assert result == []
    assert called == [
        "_extract_entities",
        "_entity_provenance",
        "_detect_attributes",
        "_attribute_provenance",
        "_extract_values_from_text",
        "_extract_values_from_tables",
        "_standardize",
        "_deduplicate",
    ]
