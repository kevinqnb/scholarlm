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

import httpx
import pytest
from openai import BadRequestError
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.measurementlm import ContextLengthExceededError, MeasurementLM


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


def _make_bad_request_error(message: str, error_type: str = "BadRequestError") -> BadRequestError:
    """Build a real openai.BadRequestError shaped like vLLM's captured error body
    (see notes/scholarlm/builds/2026-08-20-per-document-isolation-01.md): the
    exception's .body is the *inner* dict, after openai's client unwraps the
    outer {"error": {...}} envelope -- i.e. exactly what _acall receives.
    """
    request = httpx.Request("POST", "http://localhost:0/v1/chat/completions")
    response = httpx.Response(400, request=request)
    body = {"message": message, "type": error_type, "param": None, "code": 400}
    return BadRequestError(f"Error code: 400 - {{'error': {body}}}", response=response, body=body)


def test_acall_raises_context_length_exceeded_only_for_matching_message(monkeypatch):
    """_acall must distinguish a context-length-exceeded 400 (re-raised as
    ContextLengthExceededError) from every other 400 (still swallowed to "",
    matching the pre-existing behavior for every other exception type)."""
    mlm = _make_mlm()

    async def raise_context_length(*args, **kwargs):
        raise _make_bad_request_error(
            "Input length (103770) exceeds model's maximum context length (81920)."
        )

    monkeypatch.setattr(mlm.async_client.chat.completions, "create", raise_context_length)
    with pytest.raises(ContextLengthExceededError):
        asyncio.run(mlm._acall([{"role": "user", "content": "x"}]))

    async def raise_other_bad_request(*args, **kwargs):
        raise _make_bad_request_error("The model's response was filtered.")

    monkeypatch.setattr(mlm.async_client.chat.completions, "create", raise_other_bad_request)
    result = asyncio.run(mlm._acall([{"role": "user", "content": "x"}]))
    assert result == ""


def test_call_batch_isolates_context_length_exceeded_without_retrying_it(monkeypatch):
    """A ContextLengthExceededError on one message set must not fail the batch,
    must not be retried (it's deterministic -- retrying wastes calls), and must
    not affect the sibling message set's result."""
    mlm = _make_mlm()
    call_counts = {"over_length": 0, "ok": 0}

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        text = messages[0]["content"]
        if text == "over_length":
            call_counts["over_length"] += 1
            raise ContextLengthExceededError("Input length exceeds model's maximum context length.")
        call_counts["ok"] += 1
        return '{"items": ["fine"]}'

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    results = mlm._call_batch(
        [
            [{"role": "user", "content": "over_length"}],
            [{"role": "user", "content": "ok"}],
        ],
        max_retries=3,
        validator=lambda r: __import__("json").loads(r),
    )

    assert isinstance(results[0], ContextLengthExceededError)
    assert results[1] == '{"items": ["fine"]}'
    # Exactly one attempt for the over-length document: never retried.
    assert call_counts["over_length"] == 1
    # The sibling document is unaffected by the other's failure.
    assert call_counts["ok"] == 1


def test_direct_mode_isolates_context_length_exceeded_document(monkeypatch):
    """End-to-end through _extract_triples (a real _call_batch call site): one
    document's context-length failure must not crash the batch, must not
    contaminate the sibling document's extracted record, and must be recorded
    in context_length_exceeded_docs rather than silently looking like an empty
    result."""
    mlm = _make_mlm(
        extraction_mode="direct",
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract all measurements.",
    )
    mlm.data = [
        {"document_id": 0, "context": "DOC0 too-long text"},
        {"document_id": 1, "context": "DOC1 text"},
    ]

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        if "DOC0" in messages[0]["content"]:
            raise ContextLengthExceededError("... exceeds model's maximum context length ...")
        return (
            '{"items": [{"name": "Lake B", "location": "MN", '
            '"attribute": "depth", "value": "5.0", "units": "m"}]}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    records = mlm._extract_triples()

    assert mlm.context_length_exceeded_docs == {0}
    assert len(records) == 1
    assert records[0]["name"] == "Lake B"


def test_standardize_isolates_context_length_exceeded_document(monkeypatch):
    """_standardize (a real _call_batch call site whose message_data_ids are
    positional indices into self.data, not document ids directly -- doc id is
    recovered via standardized_data[message_data_ids[i]]['document_id']) must
    record the failing record's document_id, leave its value/units unchanged
    (the existing validation-failure fallback), and not disturb the sibling
    record's standardized result."""
    mlm = _make_mlm()
    mlm.data = [
        {"document_id": 0, "entity_id": "doc_0_entity_0", "name": "Lake A",
         "location": "WI", "context": "DOC0 too-long text", "attribute": "depth",
         "value": "3.2", "units": "m", "attribute_terms": []},
        {"document_id": 1, "entity_id": "doc_1_entity_0", "name": "Lake B",
         "location": "MN", "context": "DOC1 text", "attribute": "depth",
         "value": "5.0", "units": "m", "attribute_terms": []},
    ]

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        if "DOC0" in messages[0]["content"]:
            raise ContextLengthExceededError("... exceeds model's maximum context length ...")
        return '{"explanation": "ok", "value": "6.0", "units": "m"}'

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    standardized = mlm._standardize()

    assert mlm.context_length_exceeded_docs == {0}
    by_doc = {r["document_id"]: r for r in standardized}
    # doc0's context-length failure falls back to its original, unstandardized value.
    assert by_doc[0]["value"] == "3.2"
    assert by_doc[0]["units"] == "m"
    # doc1 is unaffected and picks up the standardized value.
    assert by_doc[1]["value"] == "6.0"
    assert by_doc[1]["units"] == "m"


def test_extract_values_from_text_isolates_context_length_exceeded_document(monkeypatch):
    """_extract_values_from_text's message_ids are (event_record, page) pairs
    where event_record is a dict, not a positional index -- doc id must be
    recovered via pair_record['document_id'], not a tuple unpack. Verifies that
    recovery path directly, plus that the failing document's page is dropped
    from results (no has_value fallback exists for this site) while the
    sibling document's value is unaffected."""
    mlm = _make_mlm()

    entity_data = [
        {"document_id": 0, "entity_id": "doc_0_entity_0", "name": "Lake A",
         "location": "WI", "context": '<page number="0">too-long page</page>'},
        {"document_id": 1, "entity_id": "doc_1_entity_0", "name": "Lake B",
         "location": "MN", "context": '<page number="0">ok page</page>'},
    ]
    doc_attributes = {0: {"depth": []}, 1: {"depth": []}}
    entity_prov = {
        (0, "doc_0_entity_0"): [{"page": 0, "table": None}],
        (1, "doc_1_entity_0"): [{"page": 0, "table": None}],
    }
    attr_prov = {
        (0, "depth"): [{"page": 0, "table": None}],
        (1, "depth"): [{"page": 0, "table": None}],
    }

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        if "too-long page" in messages[0]["content"]:
            raise ContextLengthExceededError("... exceeds model's maximum context length ...")
        return '{"explanation": "ok", "has_value": true, "value": "3.2", "units": "m"}'

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    text_values = mlm._extract_values_from_text(entity_data, doc_attributes, entity_prov, attr_prov)

    assert mlm.context_length_exceeded_docs == {0}
    by_doc = {r["document_id"]: r for r in text_values}
    assert 0 not in by_doc  # over-length document contributes no value, doesn't crash the batch
    assert by_doc[1]["value"] == "3.2"


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
