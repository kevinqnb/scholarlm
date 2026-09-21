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

from scholarlm.measurementlm import (
    ContextLengthExceededError,
    MeasurementLM,
    check_quantity_consistency,
)


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
                '{"explanation": "ok", "units": "m"}',
                '{"explanation": "ok", "units": "m"}',
            ]
        if name == "parse_quantity_response":
            assert len(message_sets) == 2  # one message per standardized triple
            return [
                '{"explanation": "ok", "qualifiers": [], "point_value": "3.2", "lower": null, '
                '"upper": null, "list_values": null, "tolerance": null, "standard_deviation": null}',
                '{"explanation": "ok", "qualifiers": [], "point_value": "5.0", "lower": null, '
                '"upper": null, "list_values": null, "tolerance": null, "standard_deviation": null}',
            ]
        raise AssertionError(f"unexpected _call_batch invocation: {name}")

    monkeypatch.setattr(MeasurementLM, "_call_batch", fake_call_batch)

    records = mlm.fit(["doc text A", "doc text B"])

    assert calls == ["direct_extraction_list", "standardize_response", "parse_quantity_response"]
    assert len(records) == 2

    records_by_name = {r["name"]: r for r in records}
    assert set(records_by_name) == {"Lake A", "Lake B"}

    for name, expected_value, doc_text in [
        ("Lake A", "3.2", "doc text A"),
        ("Lake B", "5.0", "doc text B"),
    ]:
        rec = records_by_name[name]
        assert rec["attribute"] == "depth"
        assert rec["value"] == expected_value  # untouched by standardize; not overwritten by parse
        assert rec["units"] == "m"
        assert rec["qualifiers"] == []
        assert rec["point_value"] == expected_value
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


def _fake_response(content: str, prompt_tokens: int, completion_tokens: int):
    from types import SimpleNamespace
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


def test_acall_accumulates_token_usage_across_successful_and_failed_calls(monkeypatch):
    """token_usage must sum prompt/completion tokens over every successful call
    (not just track a running max, the way max_prompt_tokens does) and must
    count failed calls separately -- a compute-time report built only from
    token_usage's prompt/completion totals would otherwise silently omit the
    compute spent on calls that errored out."""
    mlm = _make_mlm()

    async def call_ok_1(*args, **kwargs):
        return _fake_response('{"a": 1}', prompt_tokens=100, completion_tokens=20)
    monkeypatch.setattr(mlm.async_client.chat.completions, "create", call_ok_1)
    asyncio.run(mlm._acall([{"role": "user", "content": "x"}]))

    async def call_ok_2(*args, **kwargs):
        return _fake_response('{"b": 2}', prompt_tokens=50, completion_tokens=10)
    monkeypatch.setattr(mlm.async_client.chat.completions, "create", call_ok_2)
    asyncio.run(mlm._acall([{"role": "user", "content": "y"}]))

    async def call_fails(*args, **kwargs):
        raise RuntimeError("connection reset")
    monkeypatch.setattr(mlm.async_client.chat.completions, "create", call_fails)
    result = asyncio.run(mlm._acall([{"role": "user", "content": "z"}]))
    assert result == ""

    assert mlm.token_usage == {
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "successful_calls": 2,
        "failed_calls": 1,
    }
    # max_prompt_tokens is unaffected -- still the largest single call, not a sum.
    assert mlm.max_prompt_tokens == 100


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
    record the failing record's document_id, leave its units unchanged (the
    existing validation-failure fallback), and not disturb the sibling record's
    standardized result. value is never touched by _standardize (it's now
    units-only; parsing value's shape is _parse_quantities()'s job) -- it stays
    "3.2"/"5.0" for both records regardless of context-length outcome."""
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
        return '{"explanation": "ok", "units": "m"}'

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    standardized = mlm._standardize()

    assert mlm.context_length_exceeded_docs == {0}
    by_doc = {r["document_id"]: r for r in standardized}
    # doc0's context-length failure falls back to its original, unstandardized units.
    assert by_doc[0]["value"] == "3.2"
    assert by_doc[0]["units"] == "m"
    # doc1 is unaffected and picks up the standardized units; value is untouched either way.
    assert by_doc[1]["value"] == "5.0"
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


def test_context_length_exceeded_docs_resets_between_fit_calls(monkeypatch):
    """context_length_exceeded_docs holds positions within the current batch's
    documents list, not stable document ids -- a second fit() call on the same
    instance must not carry the first batch's indices forward, or a caller
    reading the attribute after batch 2 would misattribute batch 1's failure
    to a document that succeeded in batch 2."""
    mlm = _make_mlm(
        extraction_mode="direct",
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract all measurements.",
    )
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    async def fake_acall_batch1(self, messages, response_format=None, temperature=None,
                                 max_tokens=None, timeout=600.0, extra_body=None):
        if "DOC0" in messages[0]["content"]:
            raise ContextLengthExceededError("... exceeds model's maximum context length ...")
        return '{"items": []}'

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall_batch1)
    mlm.fit(["DOC0 too-long text", "DOC1 text"])
    assert mlm.context_length_exceeded_docs == {0}

    async def fake_acall_batch2(self, messages, response_format=None, temperature=None,
                                 max_tokens=None, timeout=600.0, extra_body=None):
        return '{"items": []}'

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall_batch2)
    mlm.fit(["DOC2 fine text"])
    assert mlm.context_length_exceeded_docs == set()


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
    monkeypatch.setattr(mlm, "_parse_quantities", stub("_parse_quantities", []))
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
        "_parse_quantities",
        "_deduplicate",
    ]
    # Every ablation that reuses this base fit() unchanged (4, 5, 6) gets this
    # same per-step timing for free. values_text/values_tables are separate
    # (not one combined "values") so ablation 5's table-only change and
    # ablation 4's text+table change are each individually visible.
    assert set(mlm.step_seconds) == {
        "entities", "entity_prov", "attributes", "attribute_prov", "events",
        "values_text", "values_tables", "final",
    }


# ---------------------------------------------------------------------------
# _parse_quantities()
# ---------------------------------------------------------------------------

def _base_datapoint(document_id, value, units="m"):
    return {
        "document_id": document_id, "entity_id": f"doc_{document_id}_entity_0",
        "name": "Lake A", "location": "WI", "context": f"DOC{document_id} text",
        "attribute": "depth", "value": value, "units": units, "attribute_terms": [],
    }


def test_parse_quantities_populates_all_shapes(monkeypatch):
    """One fake response per reported shape -- plain point, two-sided range,
    one-sided inequality, list, and a mean given together with a range --
    verifying each lands in the field PARSE_QUANTITY_INSTRUCTIONS calls for
    and that every other quantity field stays null."""
    mlm = _make_mlm()
    mlm.data = [
        _base_datapoint(0, "12.3"),
        _base_datapoint(1, "3-7"),
        _base_datapoint(2, "< 5"),
        _base_datapoint(3, "2, 5, 9"),
        _base_datapoint(4, "5.2 (3.1-7.4)"),
    ]

    fake_responses = [
        '{"explanation": "plain point", "qualifiers": [], "point_value": "12.3", '
        '"lower": null, "upper": null, "list_values": null, "tolerance": null, '
        '"standard_deviation": null}',

        '{"explanation": "two-sided range", "qualifiers": ["IsRange"], "point_value": null, '
        '"lower": "3", "upper": "7", "list_values": null, "tolerance": null, '
        '"standard_deviation": null}',

        '{"explanation": "one-sided bound", "qualifiers": ["IsRange"], "point_value": null, '
        '"lower": null, "upper": "5", "list_values": null, "tolerance": null, '
        '"standard_deviation": null}',

        '{"explanation": "list", "qualifiers": ["IsList"], "point_value": null, "lower": null, '
        '"upper": null, "list_values": ["2", "5", "9"], "tolerance": null, '
        '"standard_deviation": null}',

        '{"explanation": "mean with range", "qualifiers": ["IsMean", "IsRange"], '
        '"point_value": "5.2", "lower": "3.1", "upper": "7.4", "list_values": null, '
        '"tolerance": null, "standard_deviation": null}',
    ]

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        for i in range(len(mlm.data)):
            if f"DOC{i}" in messages[0]["content"]:
                return fake_responses[i]
        raise AssertionError("unmatched document in parse-quantity prompt")

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    parsed = mlm._parse_quantities()
    by_doc = {r["document_id"]: r for r in parsed}

    assert by_doc[0]["qualifiers"] == []
    assert by_doc[0]["point_value"] == "12.3"
    assert by_doc[0]["lower"] is None and by_doc[0]["upper"] is None

    assert by_doc[1]["qualifiers"] == ["IsRange"]
    assert by_doc[1]["lower"] == "3" and by_doc[1]["upper"] == "7"
    assert by_doc[1]["point_value"] is None

    assert by_doc[2]["qualifiers"] == ["IsRange"]
    assert by_doc[2]["lower"] is None and by_doc[2]["upper"] == "5"

    assert by_doc[3]["qualifiers"] == ["IsList"]
    assert by_doc[3]["list_values"] == ["2", "5", "9"]

    assert by_doc[4]["qualifiers"] == ["IsMean", "IsRange"]
    assert by_doc[4]["point_value"] == "5.2"
    assert by_doc[4]["lower"] == "3.1" and by_doc[4]["upper"] == "7.4"

    # value/units are untouched by this step -- they were already set by extraction/standardize.
    for doc_id in by_doc:
        assert by_doc[doc_id]["units"] == "m"


def test_parse_quantities_isolates_context_length_exceeded_document(monkeypatch):
    """Mirrors test_standardize_isolates_context_length_exceeded_document: a
    context-length failure on one record must record its document_id and leave
    its quantity fields entirely unset, without disturbing the sibling record."""
    mlm = _make_mlm()
    mlm.data = [_base_datapoint(0, "3.2"), _base_datapoint(1, "5.0")]
    mlm.data[0]["context"] = "DOC0 too-long text"

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        if "DOC0" in messages[0]["content"]:
            raise ContextLengthExceededError("... exceeds model's maximum context length ...")
        return (
            '{"explanation": "ok", "qualifiers": [], "point_value": "5.0", "lower": null, '
            '"upper": null, "list_values": null, "tolerance": null, "standard_deviation": null}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    parsed = mlm._parse_quantities()

    assert mlm.context_length_exceeded_docs == {0}
    by_doc = {r["document_id"]: r for r in parsed}
    assert "qualifiers" not in by_doc[0]
    assert by_doc[1]["qualifiers"] == []
    assert by_doc[1]["point_value"] == "5.0"


def test_parse_quantities_logs_but_keeps_inconsistent_record(monkeypatch, capsys):
    """A response whose qualifiers don't match its populated fields (per-project
    policy, see CLAUDE.md's fail-loud rule and the explicit call not to drop or
    error on this class of mismatch) is still applied in full and only logged."""
    mlm = _make_mlm()
    mlm.data = [_base_datapoint(0, "3-7")]

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        # Tagged IsRange but lower/upper both left null -- inconsistent, not fatal.
        return (
            '{"explanation": "bad parse", "qualifiers": ["IsRange"], "point_value": null, '
            '"lower": null, "upper": null, "list_values": null, "tolerance": null, '
            '"standard_deviation": null}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    parsed = mlm._parse_quantities()

    assert parsed[0]["qualifiers"] == ["IsRange"]
    assert parsed[0]["lower"] is None and parsed[0]["upper"] is None
    out = capsys.readouterr().out
    assert "IsRange tagged but lower/upper both null" in out


# ---------------------------------------------------------------------------
# check_quantity_consistency()
# ---------------------------------------------------------------------------

def _quantity(**overrides):
    base = dict(qualifiers=[], point_value=None, lower=None, upper=None,
                list_values=None, tolerance=None, standard_deviation=None)
    base.update(overrides)
    return base


def test_check_quantity_consistency_clean_cases_produce_no_warnings():
    assert check_quantity_consistency(_quantity(point_value="12.3")) == []
    assert check_quantity_consistency(
        _quantity(qualifiers=["IsRange"], lower="3", upper="7")
    ) == []
    assert check_quantity_consistency(
        _quantity(qualifiers=["IsList"], list_values=["1", "2"])
    ) == []
    assert check_quantity_consistency(
        _quantity(qualifiers=["IsMean", "HasSD"], point_value="5", standard_deviation="0.4")
    ) == []


def test_check_quantity_consistency_flags_mismatches_without_raising():
    assert check_quantity_consistency(_quantity(qualifiers=["IsRange"])) == [
        "IsRange tagged but lower/upper both null"
    ]
    assert check_quantity_consistency(_quantity(lower="3")) == [
        "lower/upper populated but IsRange not tagged"
    ]
    assert check_quantity_consistency(_quantity(qualifiers=["IsList"])) == [
        "IsList tagged but list_values empty/null"
    ]
    assert check_quantity_consistency(_quantity(qualifiers=["HasTolerance"])) == [
        "HasTolerance tagged but tolerance null"
    ]
    assert check_quantity_consistency(_quantity(qualifiers=["IsMean"])) == [
        "central-tendency tag (IsMean/IsMedian/IsCount) but point_value null"
    ]
    # Multiple simultaneous mismatches are all reported, not just the first.
    warnings = check_quantity_consistency(_quantity(qualifiers=["IsRange", "HasSD"]))
    assert set(warnings) == {
        "IsRange tagged but lower/upper both null",
        "HasSD tagged but standard_deviation null",
    }
