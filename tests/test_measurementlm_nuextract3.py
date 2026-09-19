"""Unit tests for the NuExtract3 baseline adapter (measurementlm_nuextract3.py).

Uses hand-built two-document fixtures with stubbed `_call_batch`/`_acall` --
no network calls, no vLLM endpoint required. Covers the rung-1 checks called
out in `ISSUE-nuextract-baseline.md` and the build session, adapted to this
adapter's own design:

  * `attribute`/`units` are Literal enums in the guided-decoding schema, not
    plain `str` -- the actual fix for the old adapter's off-vocabulary hole.
  * `units` admits `None` (needed for attributes like `ph` with no units).
  * `fit()` never deduplicates or standardizes -- confirmed by feeding two
    literally-identical items back and asserting both survive, the opposite
    assertion from the old adapter's tests, since dedup here is a deliberate
    non-feature (see module docstring).
  * `ContextLengthExceededError` is isolated per document, not swallowed.
  * Validation failures are counted (`self.validation_failures`), not just
    silently dropped.
  * Out-of-vocabulary attributes still retry and, if unresolved, are dropped
    with a printed count -- the same safety net the base pipeline's own
    direct-extraction step uses, kept here as a backstop behind the Literal
    schema.
  * `seed`, when present in `sampling_params`, is forwarded via `extra_body`.
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from scholarlm.measurementlm import ContextLengthExceededError, MeasurementLM
from scholarlm.measurementlm_nuextract3 import (
    MeasurementLMNuExtract3,
    _build_example_messages,
    _build_nuextract_template,
    _build_response_schema,
    _units_vocabulary,
)


class _EntitySchema(BaseModel):
    name: str | None
    location: str | None


class _DirectSchema(BaseModel):
    name: str | None
    location: str | None
    date: str | None
    attribute: str
    value: str | None
    units: str | None


_ATTRIBUTE_INFO = {
    "depth": {"description": "Maximum depth", "units": ["m", "ft"]},
    "ph": {"description": "Water pH", "units": []},
    "tn": {"description": "Total nitrogen", "units": ["mg/L", "ug/L"]},
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
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract all measurements.",
        api_base="http://localhost:0/v1",
    )
    kwargs.update(overrides)
    return MeasurementLMNuExtract3(**kwargs)


# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


def test_units_vocabulary_unions_across_attributes_in_first_seen_order():
    assert _units_vocabulary(_ATTRIBUTE_INFO) == ["m", "ft", "mg/L", "ug/L"]


def test_units_vocabulary_empty_when_no_attribute_declares_units():
    # measeval-shaped: the whole dataset has no closed unit vocabulary --
    # not a config error, callers must fall back to open string typing.
    assert _units_vocabulary({"measurement": {"description": "any", "units": []}}) == []


def test_response_schema_falls_back_to_open_units_when_vocabulary_empty():
    schema = _build_response_schema(_DirectSchema, ["measurement"], [])
    json_schema = schema.model_json_schema()
    units_schema = json_schema["properties"]["units"]
    assert "enum" not in json.dumps(units_schema)
    item = schema(name="A", location=None, date=None, attribute="measurement", value="3.2", units="anything at all")
    assert item.units == "anything at all"


def test_build_nuextract_template_falls_back_to_verbatim_string_when_units_vocabulary_empty():
    template = _build_nuextract_template(_DirectSchema, ["measurement"], [])
    assert template["items"][0]["units"] == "verbatim-string"


def test_response_schema_has_literal_enum_for_attribute_and_units():
    schema = _build_response_schema(_DirectSchema, ["depth", "ph", "tn"], ["m", "ft", "mg/L"])
    json_schema = schema.model_json_schema()
    assert json_schema["properties"]["attribute"]["enum"] == ["depth", "ph", "tn"]
    # units must allow the enum values AND null (ph has no real units)
    units_schema = json_schema["properties"]["units"]
    branches = units_schema.get("anyOf", [units_schema])
    enum_branch = next(b for b in branches if "enum" in b)
    assert enum_branch["enum"] == ["m", "ft", "mg/L"]
    assert any(b.get("type") == "null" for b in branches)


def test_response_schema_units_field_accepts_none():
    schema = _build_response_schema(_DirectSchema, ["depth", "ph"], ["m"])
    item = schema(name="Lake A", location="WI", date=None, attribute="ph", value="6.9", units=None)
    assert item.units is None


def test_response_schema_rejects_out_of_vocabulary_attribute():
    schema = _build_response_schema(_DirectSchema, ["depth", "ph"], ["m"])
    with pytest.raises(Exception):
        schema(name="Lake A", location="WI", date=None, attribute="hardness", value="3.2", units="m")


def test_response_schema_single_attribute_dataset_still_builds():
    # measeval-shaped: exactly one attribute name -- Literal with one member
    # degrades to a JSON schema `const`, still a valid closed vocabulary.
    schema = _build_response_schema(_DirectSchema, ["measurement"], ["count"])
    json_schema = schema.model_json_schema()
    attr_schema = json_schema["properties"]["attribute"]
    assert attr_schema.get("const") == "measurement" or attr_schema.get("enum") == ["measurement"]


def test_build_nuextract_template_uses_enum_lists_for_attribute_and_units():
    template = _build_nuextract_template(_DirectSchema, ["depth", "ph"], ["m", "ft"])
    item = template["items"][0]
    assert item["attribute"] == ["depth", "ph"]
    assert item["units"] == ["m", "ft"]
    assert item["name"] == "verbatim-string"
    assert item["value"] == "verbatim-string"


def test_build_example_messages_wraps_dataset_examples_as_developer_messages():
    examples = [{"input": "Lake A is 3m deep.", "output": '{"items": []}'}]
    messages = _build_example_messages(examples)
    assert messages == [
        {
            "role": "developer",
            "content": [
                {"type": "text", "text": "Lake A is 3m deep."},
                {"type": "text", "text": '{"items": []}'},
            ],
        }
    ]


def test_build_example_messages_empty_when_none():
    assert _build_example_messages(None) == []


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------


def test_missing_direct_extraction_schema_raises():
    with pytest.raises(ValueError):
        _make_mlm(direct_extraction_schema=None)


def test_missing_direct_extraction_prompt_raises():
    with pytest.raises(ValueError):
        _make_mlm(direct_extraction_prompt=None)


def test_clean_tables_true_raises():
    with pytest.raises(ValueError):
        _make_mlm(clean_tables=True)


def test_use_extra_body_false_raises():
    # use_extra_body=False would silently drop template/instructions/seed at
    # the _acall layer while response_format still forces valid-looking JSON
    # out of the wrong (markdown/OCR) chat-template mode -- must fail loud
    # at construction instead.
    with pytest.raises(ValueError):
        _make_mlm(use_extra_body=False)


# ---------------------------------------------------------------------------
# fit(): one call per document, no standardize, no deduplication
# ---------------------------------------------------------------------------


def test_fit_calls_call_batch_once_with_one_message_set_per_document(monkeypatch):
    mlm = _make_mlm()
    calls = []

    def fake_call_batch(self, message_sets, response_format=None, extra_body=None, **kwargs):
        calls.append((response_format["json_schema"]["name"], len(message_sets), extra_body))
        return [
            '{"items": [{"name": "Lake A", "location": "WI", "date": null, '
            '"attribute": "depth", "value": "3.2", "units": "m"}]}',
            '{"items": []}',
        ]

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)

    records = mlm.fit(["doc text A", "doc text B"])

    assert len(calls) == 1
    name, n_message_sets, extra_body = calls[0]
    assert name == "direct_extraction_list"
    assert n_message_sets == 2
    assert "template" in extra_body["chat_template_kwargs"]
    assert "instructions" in extra_body["chat_template_kwargs"]

    assert len(records) == 1
    rec = records[0]
    assert rec["attribute"] == "depth"
    assert rec["value"] == "3.2"
    assert rec["units"] == "m"
    assert rec["attribute_terms"] == []
    assert rec["context"] == "doc text A"


def test_call_batch_pinned_to_max_concurrent_one(monkeypatch):
    """Concurrency, not just a missing seed, was named in the determinism
    issue (ISSUE-nuextract-baseline.md #3) -- must be pinned at the call
    site, not left at whatever the instance default happens to be."""
    mlm = _make_mlm()
    captured = {}

    def fake_call_batch(self, message_sets, max_concurrent=None, **kwargs):
        captured["max_concurrent"] = max_concurrent
        return ['{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)
    mlm.fit(["doc text A"])

    assert captured["max_concurrent"] == 1


def test_max_tokens_defaults_to_32768_when_unset_anywhere(monkeypatch):
    mlm = _make_mlm()
    captured = {}

    def fake_call_batch(self, message_sets, max_tokens=None, **kwargs):
        captured["max_tokens"] = max_tokens
        return ['{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)
    mlm.fit(["doc text A"])

    assert captured["max_tokens"] == 32768


def test_max_tokens_falls_back_to_sampling_params_when_constructor_unset(monkeypatch):
    mlm = _make_mlm(sampling_params={"temperature": 0.0, "max_tokens": 4096})
    captured = {}

    def fake_call_batch(self, message_sets, max_tokens=None, **kwargs):
        captured["max_tokens"] = max_tokens
        return ['{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)
    mlm.fit(["doc text A"])

    assert captured["max_tokens"] == 4096


def test_max_tokens_constructor_value_overrides_sampling_params(monkeypatch):
    mlm = _make_mlm(sampling_params={"temperature": 0.0, "max_tokens": 4096}, max_tokens=8192)
    captured = {}

    def fake_call_batch(self, message_sets, max_tokens=None, **kwargs):
        captured["max_tokens"] = max_tokens
        return ['{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)
    mlm.fit(["doc text A"])

    assert captured["max_tokens"] == 8192


def test_out_of_vocabulary_units_retried_then_dropped(monkeypatch, capsys):
    """Mirrors the attribute backstop: an off-vocabulary `units` value must
    not silently enter the record set just because the parse schema is
    lenient (see _build_response_schema's docstring)."""
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": "DOC0 text"}]

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        return (
            '{"items": [{"name": "Lake A", "location": "WI", "date": null, '
            '"attribute": "depth", "value": "3.2", "units": "furlongs"}]}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    records = mlm._extract_records()

    assert records == []
    out = capsys.readouterr().out
    assert "furlongs" in out
    assert "dropped 1 record" in out


def test_fit_never_deduplicates_identical_items(monkeypatch):
    """Two literally-identical items from the same document response must both
    survive -- deduplication is deliberately not this adapter's job."""
    mlm = _make_mlm()

    def fake_call_batch(self, message_sets, response_format=None, **kwargs):
        return [
            '{"items": ['
            '{"name": "Lake A", "location": "WI", "date": null, "attribute": "depth", "value": "3.2", "units": "m"},'
            '{"name": "Lake A", "location": "WI", "date": null, "attribute": "depth", "value": "3.2", "units": "m"}'
            ']}'
        ]

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)

    records = mlm.fit(["doc text A"])

    assert len(records) == 2
    assert records[0]["entity_id"] != records[1]["entity_id"]


def test_fit_does_not_call_standardize_or_clean_tables(monkeypatch):
    mlm = _make_mlm()

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("must not be called")

    monkeypatch.setattr(MeasurementLM, "_standardize", fail_if_called)
    monkeypatch.setattr(MeasurementLM, "_clean_tables", fail_if_called)
    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", lambda *a, **k: ['{"items": []}'])

    mlm.fit(["doc text A"])  # must not raise


# ---------------------------------------------------------------------------
# Seed forwarding
# ---------------------------------------------------------------------------


def test_seed_forwarded_when_present_in_sampling_params(monkeypatch):
    mlm = _make_mlm(sampling_params={"temperature": 0.0, "seed": 7})
    captured = {}

    def fake_call_batch(self, message_sets, extra_body=None, **kwargs):
        captured["extra_body"] = extra_body
        return ['{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)
    mlm.fit(["doc text A"])

    assert captured["extra_body"]["seed"] == 7


def test_seed_omitted_when_absent_from_sampling_params(monkeypatch):
    mlm = _make_mlm(sampling_params={"temperature": 0.0})
    captured = {}

    def fake_call_batch(self, message_sets, extra_body=None, **kwargs):
        captured["extra_body"] = extra_body
        return ['{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)
    mlm.fit(["doc text A"])

    assert "seed" not in captured["extra_body"]


# ---------------------------------------------------------------------------
# Context-length isolation and validation-failure counting
# ---------------------------------------------------------------------------


def test_context_length_exceeded_is_isolated_and_tracked(monkeypatch):
    mlm = _make_mlm()

    def fake_call_batch(self, message_sets, **kwargs):
        return [
            ContextLengthExceededError("too long"),
            '{"items": [{"name": "Lake B", "location": "MN", "date": null, '
            '"attribute": "tn", "value": "1.0", "units": "mg/L"}]}',
        ]

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)

    records = mlm.fit(["doc text A (too long)", "doc text B"])

    assert mlm.context_length_exceeded_docs == {0}
    assert len(records) == 1
    assert records[0]["context"] == "doc text B"


def test_validation_failures_are_counted_not_silently_swallowed(monkeypatch, capsys):
    mlm = _make_mlm()

    def fake_call_batch(self, message_sets, **kwargs):
        return ["not valid json at all", '{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)

    records = mlm.fit(["doc text A", "doc text B"])

    assert mlm.validation_failures == 1
    assert records == []
    assert "Validation error in NuExtract3 response" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# Out-of-vocabulary attribute retry/drop (exercises the real _call_batch retry
# loop via a stubbed _acall, mirroring test_measurementlm.py's equivalent test)
# ---------------------------------------------------------------------------


def test_out_of_vocabulary_attribute_retries_then_recovers(monkeypatch):
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": "DOC0 text"}]
    attempts = {"n": 0}

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return (
                '{"items": [{"name": "Lake A", "location": "WI", "date": null, '
                '"attribute": "hardness", "value": "3.2", "units": "m"}]}'
            )
        return (
            '{"items": [{"name": "Lake A", "location": "WI", "date": null, '
            '"attribute": "depth", "value": "3.2", "units": "m"}]}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    records = mlm._extract_records()

    assert attempts["n"] == 2
    assert len(records) == 1
    assert records[0]["attribute"] == "depth"



# ---------------------------------------------------------------------------
# Schema/template construction against every real dataset config -- catches
# field-name or units-vocabulary assumptions that a synthetic fixture can't.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["pond", "nfix", "supermat", "measeval"])
def test_builds_schema_and_template_for_every_real_dataset_config(dataset_name):
    from run_extraction import load_dataset_config

    cfg = load_dataset_config(dataset_name)
    attribute_names = list(cfg.attribute_info_dict.keys())
    unit_names = _units_vocabulary(cfg.attribute_info_dict)

    schema = _build_response_schema(cfg.direct_extraction_schema, attribute_names, unit_names)
    json_schema = schema.model_json_schema()
    attr_schema = json_schema["properties"]["attribute"]
    assert attr_schema.get("enum") == attribute_names or attr_schema.get("const") == attribute_names[0]

    template = _build_nuextract_template(cfg.direct_extraction_schema, attribute_names, unit_names)
    assert template["items"][0]["attribute"] == attribute_names
    if unit_names:
        assert template["items"][0]["units"] == unit_names
    else:
        assert template["items"][0]["units"] == "verbatim-string"

    # entity_identification_schema's fields must all be present on
    # direct_extraction_schema -- MeasurementLMAblation1's own assumption,
    # inherited here (see module docstring).
    entity_fields = set(cfg.entity_schema.model_fields.keys())
    assert entity_fields <= set(cfg.direct_extraction_schema.model_fields.keys())


def test_out_of_vocabulary_attribute_dropped_after_retries_exhausted(monkeypatch, capsys):
    mlm = _make_mlm()
    mlm.data = [{"document_id": 0, "context": "DOC0 text"}]

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        return (
            '{"items": [{"name": "Lake A", "location": "WI", "date": null, '
            '"attribute": "hardness", "value": "3.2", "units": "m"}]}'
        )

    monkeypatch.setattr(MeasurementLM, "_acall", fake_acall)
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)

    records = mlm._extract_records()

    assert records == []
    out = capsys.readouterr().out
    assert "hardness" in out
    assert "dropped 1 record" in out
