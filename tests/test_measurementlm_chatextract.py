"""Unit tests for the ChatExtract baseline adapter (measurementlm_chatextract.py).

Uses hand-built fixtures with a queued, stubbed `MeasurementLM._acall` -- no
network calls, no vLLM endpoint required. Covers the behaviors changed or
discussed in the 2026-09-19 review session:

  * `_is_yes`/`_is_no` anchor on the first word only, so "know"/"not"/"none"/
    "eyes" don't false-match a bare substring check the way the old
    `"no" in verdict.lower()` code did.
  * Table extraction (`_extract_multi`, `_process_table`) uses guided JSON
    decoding (`_TABLE_RESPONSE_FORMAT`) instead of CSV/pipe-splitting, so a
    comma inside a value (e.g. "1,000") survives intact.
  * `_prepare_document` never guesses a caption for a raw-OCR table with no
    `<caption>` element -- it passes the table through unchanged (the
    `_table_caption` heuristic was removed, not replaced).
  * `_extract_records` counts and labels failed work items in `self.failures`
    instead of swallowing them silently.
  * `fit()` skips `_deduplicate()` -- two identical mentions both survive,
    the same enforce-no-drop-of-nothing-to-drop convention as the NuExtract3
    and LangExtract baselines.
  * `TAB_Q` (part of the block claimed verbatim from ChatExtract.py) carries
    no JSON wording; `TABLE_EXTRACT_Q` (already flagged non-reference-script)
    states the JSON shape directly.
  * Every real dataset config supplies an explicit `chatextract_property_names`
    entry for every attribute and a `chatextract_entity_noun` -- regression
    test for the supermat gap (bare "tc" property phrase) found and fixed
    this session.
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

from scholarlm.measurementlm import MeasurementLM
from scholarlm.measurementlm_chatextract import (
    MeasurementLMChatExtract,
    TAB_Q,
    TABLE_EXTRACT_Q,
    _ChatExtractTableResponse,
    _TABLE_RESPONSE_FORMAT,
    _is_no,
    _is_yes,
)


class _EntitySchema(BaseModel):
    name: str | None
    location: str | None


_ATTRIBUTE_INFO = {
    "max_depth": {"description": "Maximum water depth", "units": ["m", "ft"]},
    "ph": {"description": "Water pH", "units": []},
    "tn": {"description": "Total nitrogen", "units": ["mg/L", "ug/L"]},
}


async def _fast_sleep(*args, **kwargs):
    """No-op stand-in for asyncio.sleep, to skip _acall_retry's backoff in tests."""
    return None


def _make_mlm(**overrides):
    kwargs = dict(
        model_name="test-model",
        entity_identification_prompt="Identify entities.",
        entity_identification_schema=_EntitySchema,
        attribute_info_dict=_ATTRIBUTE_INFO,
        api_base="http://localhost:0/v1",
    )
    kwargs.update(overrides)
    return MeasurementLMChatExtract(**kwargs)


def _queued_acall(responses):
    """Async stand-in for `MeasurementLM._acall` that returns queued responses
    in call order. Asserts every queued response is consumed and no extra
    call is made -- both signal a conversation taking a different branch than
    the test expects."""
    queue = list(responses)

    async def fake_acall(self, messages, response_format=None, temperature=None,
                          max_tokens=None, timeout=600.0, extra_body=None):
        assert queue, "ran out of queued responses -- conversation made an unexpected extra call"
        return queue.pop(0)

    return fake_acall


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# _is_yes / _is_no: anchored first-word matching
# ---------------------------------------------------------------------------


def test_is_yes_matches_first_word_case_insensitively():
    assert _is_yes("Yes.")
    assert _is_yes("yes, that is correct")
    assert not _is_yes("No.")


def test_is_yes_rejects_words_that_merely_contain_yes():
    assert not _is_yes("Eyes are not relevant here.")
    assert not _is_yes("Yesterday it rained.")


def test_is_no_matches_first_word_case_insensitively():
    assert _is_no("No.")
    assert _is_no("no, that is not present")
    assert not _is_no("Yes.")


def test_is_no_rejects_words_that_merely_contain_no():
    # The bug this replaces: `"no" in verdict.lower()` matched all of these.
    assert not _is_no("Know this is correct.")
    assert not _is_no("Not obviously wrong, but yes.")
    assert not _is_no("None of that matters here, yes.")


# ---------------------------------------------------------------------------
# Table JSON parsing (replaces the old CSV/pipe-splitting)
# ---------------------------------------------------------------------------


def test_parse_table_response_preserves_comma_inside_a_value():
    text = json.dumps({"rows": [{"material": "NaCl", "value": "1,000", "unit": "mg/L"}]})
    rows = MeasurementLMChatExtract._parse_table_response(text)
    assert rows == [("NaCl", "1,000", "mg/L")]


def test_parse_table_response_handles_multiple_rows_with_no_header_row():
    text = json.dumps({"rows": [
        {"material": "A", "value": "1", "unit": "m"},
        {"material": "B", "value": "2", "unit": "m"},
    ]})
    rows = MeasurementLMChatExtract._parse_table_response(text)
    assert rows == [("A", "1", "m"), ("B", "2", "m")]


def test_parse_table_response_raises_on_invalid_json():
    with pytest.raises(Exception):
        MeasurementLMChatExtract._parse_table_response("not valid json at all")


def test_table_response_format_requires_material_value_unit_fields():
    row_schema = _ChatExtractTableResponse.model_json_schema()["$defs"]["_ChatExtractTableRow"]
    assert set(row_schema["required"]) == {"material", "value", "unit"}
    assert _TABLE_RESPONSE_FORMAT["type"] == "json_schema"


# ---------------------------------------------------------------------------
# Prompt-text fidelity: TAB_Q stays verbatim, TABLE_EXTRACT_Q states JSON
# explicitly (see the 2026-09-19 review thread)
# ---------------------------------------------------------------------------


def test_tab_q_carries_no_json_wording():
    assert "json" not in TAB_Q.lower()


def test_table_extract_q_states_json_shape_explicitly():
    assert "JSON" in TABLE_EXTRACT_Q
    assert "material" in TABLE_EXTRACT_Q


# ---------------------------------------------------------------------------
# _slug / _build_passage / _property_items: pure helpers
# ---------------------------------------------------------------------------


def test_slug_normalizes_and_falls_back_to_none():
    assert MeasurementLMChatExtract._slug("Lake A!") == "lake_a"
    assert MeasurementLMChatExtract._slug(None) == "none"
    assert MeasurementLMChatExtract._slug("") == "none"


def test_build_passage_joins_title_prev_and_sentence():
    passage = MeasurementLMChatExtract._build_passage("My Paper", "Prev sentence.", "Target sentence.")
    assert passage == "My Paper. Prev sentence. Target sentence."


def test_build_passage_omits_missing_title_and_prev():
    passage = MeasurementLMChatExtract._build_passage("", "", "Target sentence.")
    assert passage == "Target sentence."


def test_property_items_uses_explicit_mapping_when_given():
    mlm = _make_mlm(attribute_property_names={"max_depth": "maximum water depth"})
    items = dict(mlm._property_items())
    assert items["max_depth"] == "maximum water depth"


def test_property_items_falls_back_to_key_with_underscores_replaced():
    mlm = _make_mlm(attribute_property_names={})
    items = dict(mlm._property_items())
    assert items["max_depth"] == "max depth"
    assert items["ph"] == "ph"


def test_no_max_tables_per_document_attribute():
    # Regression: the hardcoded cap was removed, not raised.
    mlm = _make_mlm()
    assert not hasattr(mlm, "max_tables_per_document")


# ---------------------------------------------------------------------------
# _prepare_document: page/table splitting, digit pre-filter, no caption guess
# ---------------------------------------------------------------------------


def test_prepare_document_applies_digit_prefilter_per_page():
    mlm = _make_mlm()
    context = (
        '<page number="0">Lake A is deep. The maximum depth is 3.2 m.</page>'
        '<page number="1">Lake A has clear water. No numbers here at all.</page>'
    )
    units = mlm._prepare_document(context, "Paper Title")
    kept = [s for s, _, _ in units["sentences"]]
    assert "The maximum depth is 3.2 m." in kept
    assert not any("clear water" in s for s in kept)
    assert not any("No numbers here at all." in s for s in kept)


def test_prepare_document_builds_passage_from_title_and_preceding_sentence():
    mlm = _make_mlm()
    context = '<page number="0">Lake A is deep. The maximum depth is 3.2 m.</page>'
    units = mlm._prepare_document(context, "Paper Title")
    sentence, passage, page_num = units["sentences"][0]
    assert page_num == 0
    assert passage == "Paper Title. Lake A is deep. The maximum depth is 3.2 m."
    assert sentence in passage


def test_prepare_document_tables_pass_through_without_caption_injection():
    # Regression: _table_caption used to grab the nearest preceding text line
    # as a fake caption for raw-OCR tables with no <caption> element. Removed
    # entirely -- the table now passes through exactly as OCR'd.
    mlm = _make_mlm()
    context = (
        '<page number="0">Table 1: This looks like a caption line.\n'
        '<table number="1">Material  Value\nMgB2  39</table></page>'
    )
    units = mlm._prepare_document(context, "Title")
    assert len(units["tables"]) == 1
    table_text, page_num = units["tables"][0]
    assert table_text == '<table number="1">Material  Value\nMgB2  39</table>'
    assert "caption line" not in table_text
    assert page_num == 0


def test_prepare_document_preserves_existing_caption_element_unchanged():
    mlm = _make_mlm()
    context = (
        '<page number="0"><table number="1"><caption>Table 1. Tc values.</caption>'
        "Material  Tc\nMgB2  39</table></page>"
    )
    units = mlm._prepare_document(context, "Title")
    table_text, _ = units["tables"][0]
    assert "<caption>Table 1. Tc values.</caption>" in table_text


def test_prepare_document_falls_back_to_single_untagged_page():
    mlm = _make_mlm()
    units = mlm._prepare_document("Plain text with a value of 5 mg/L in it.", "Title")
    assert len(units["sentences"]) == 1
    assert units["sentences"][0][2] is None


# ---------------------------------------------------------------------------
# _process_sentence: classify -> single/multi gate branching
# ---------------------------------------------------------------------------


def test_process_sentence_classify_no_returns_empty(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall(["No."]))
    records = _run(mlm._process_sentence(
        0, "Depth was measured.", "Passage.", "max_depth", "maximum water depth", 0
    ))
    assert records == []


def test_process_sentence_ambiguous_gate_returns_empty(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall(["Yes.", "Unclear."]))
    records = _run(mlm._process_sentence(0, "s", "p", "max_depth", "maximum water depth", 0))
    assert records == []


def test_process_sentence_single_value_path_builds_one_record(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall([
        "Yes.",    # classify
        "No.",     # single-vs-multi gate -> single
        "3.2",     # value
        "m",       # unit
        "Lake A",  # material
    ]))
    records = _run(mlm._process_sentence(
        0, "Depth was 3.2 m.", "Passage.", "max_depth", "maximum water depth", 0
    ))
    assert len(records) == 1
    rec = records[0]
    assert rec["value"] == "3.2"
    assert rec["units"] == "m"
    assert rec["name"] == "Lake A"
    assert rec["attribute"] == "max_depth"
    assert rec["page_number"] == 0


def test_process_sentence_multi_value_path_preserves_comma_in_value(monkeypatch):
    mlm = _make_mlm()
    table_json = json.dumps({"rows": [{"material": "Lake A", "value": "1,000", "unit": "mg/L"}]})
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall([
        "Yes.",       # classify
        "Yes.",       # gate -> multi
        table_json,   # TAB_Q structured response
        "Yes.",       # material verification
        "Yes.",       # value verification
        "Yes.",       # unit verification
    ]))
    records = _run(mlm._process_sentence(0, "s", "p", "tn", "total nitrogen", 0))
    assert len(records) == 1
    assert records[0]["value"] == "1,000"
    assert records[0]["name"] == "Lake A"
    assert records[0]["units"] == "mg/L"


# ---------------------------------------------------------------------------
# _extract_single: value/unit/material extraction and optional verification
# ---------------------------------------------------------------------------


def test_extract_single_drops_record_when_value_is_none(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall(["None", "m", "Lake A"]))
    messages = [{"role": "system", "content": ""}]
    records = _run(mlm._extract_single(messages, 0, "passage", "max_depth", "maximum water depth", 0))
    assert records == []


def test_extract_single_verification_nulls_out_rejected_field_only(monkeypatch):
    mlm = _make_mlm(include_single_verification=True)
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall([
        "3.2", "Yes.",                          # value + verification: accepted
        "m", "No, that doesn't look right.",    # unit + verification: rejected
        "Lake A", "Yes.",                       # material + verification: accepted
    ]))
    messages = [{"role": "system", "content": ""}]
    records = _run(mlm._extract_single(messages, 0, "passage", "max_depth", "maximum water depth", 0))
    assert len(records) == 1
    assert records[0]["value"] == "3.2"
    assert records[0]["units"] is None
    assert records[0]["name"] == "Lake A"


def test_extract_single_empty_response_after_retries_is_treated_as_absent(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(asyncio, "sleep", _fast_sleep)
    # 3 empty retries for the value question, then unit/material never asked
    # since the loop still runs but this exercises the retry-exhaustion path.
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall(["", "", "", "m", "Lake A"]))
    messages = [{"role": "system", "content": ""}]
    records = _run(mlm._extract_single(messages, 0, "passage", "max_depth", "maximum water depth", 0))
    assert records == []


# ---------------------------------------------------------------------------
# _extract_multi: table JSON extraction and per-cell verification
# ---------------------------------------------------------------------------


def test_extract_multi_verification_no_drops_row_and_short_circuits_unit(monkeypatch):
    mlm = _make_mlm()
    table_json = json.dumps({"rows": [{"material": "Lake A", "value": "3.2", "unit": "m"}]})
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall([
        table_json,
        "Yes.",  # material verification: accepted
        "No.",   # value verification: rejected -> row dropped, unit never asked
    ]))
    messages = [{"role": "system", "content": ""}]
    records = _run(mlm._extract_multi(messages, 0, "passage", "max_depth", "maximum water depth", 0))
    assert records == []


def test_extract_multi_accepts_row_when_all_cells_verify(monkeypatch):
    mlm = _make_mlm()
    table_json = json.dumps({"rows": [{"material": "Lake A", "value": "3.2", "unit": "m"}]})
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall([
        table_json, "Yes.", "Yes.", "Yes.",
    ]))
    messages = [{"role": "system", "content": ""}]
    records = _run(mlm._extract_multi(messages, 0, "passage", "max_depth", "maximum water depth", 0))
    assert len(records) == 1
    assert records[0]["name"] == "Lake A"


# ---------------------------------------------------------------------------
# _process_table: real-document-table workflow (no verification)
# ---------------------------------------------------------------------------


def test_process_table_classify_no_returns_empty(monkeypatch):
    mlm = _make_mlm()
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall(["No."]))
    records = _run(mlm._process_table(0, '<table number="1">...</table>', "max_depth", "maximum water depth", 0))
    assert records == []


def test_process_table_drops_rows_with_no_value_but_keeps_null_material(monkeypatch):
    mlm = _make_mlm()
    table_json = json.dumps({"rows": [
        {"material": "Lake A", "value": "3.2", "unit": "m"},
        {"material": "None", "value": "None", "unit": "None"},
        {"material": "None", "value": "5.0", "unit": "m"},
    ]})
    monkeypatch.setattr(MeasurementLM, "_acall", _queued_acall(["Yes.", table_json]))
    records = _run(mlm._process_table(0, '<table number="1">...</table>', "max_depth", "maximum water depth", 0))
    assert len(records) == 2
    assert records[0]["value"] == "3.2"
    assert records[0]["name"] == "Lake A"
    # A missing/None material doesn't drop the row (only a missing value does).
    assert records[1]["value"] == "5.0"
    assert records[1]["name"] is None


# ---------------------------------------------------------------------------
# _extract_records: failures are counted and labeled, not silently swallowed
# ---------------------------------------------------------------------------


def test_extract_records_counts_failures_instead_of_raising(monkeypatch, capsys):
    mlm = _make_mlm(attribute_info_dict={"max_depth": {"description": "d", "units": ["m"]}})

    async def raising_process_sentence(self, *args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(MeasurementLMChatExtract, "_process_sentence", raising_process_sentence)

    records = mlm._extract_records(["Site A has a depth of 3.2 m."], ["Title"])

    assert records == []
    assert len(mlm.failures) == 1
    assert "RuntimeError: boom" in mlm.failures[0]["error"]
    assert "kind=sentence" in mlm.failures[0]["item"]
    out = capsys.readouterr().out
    assert "1 of 1 work items failed" in out


# ---------------------------------------------------------------------------
# fit(): skips _deduplicate(), validates titles/documents length
# ---------------------------------------------------------------------------


def test_fit_raises_on_mismatched_titles_length():
    mlm = _make_mlm()
    with pytest.raises(ValueError):
        mlm.fit(["doc1", "doc2"], ["only one title"])


def test_fit_skips_deduplicate_keeps_duplicate_mentions(monkeypatch):
    mlm = _make_mlm()
    template = mlm._make_record(0, "max_depth", "Lake A", "3.2", "m", 0)

    def fake_extract_records(self, documents, titles):
        return [dict(template), dict(template)]

    monkeypatch.setattr(MeasurementLMChatExtract, "_extract_records", fake_extract_records)

    records = mlm.fit(["doc text"], ["Title"])

    assert len(records) == 2


# ---------------------------------------------------------------------------
# Real dataset configs: every attribute must have an explicit ChatExtract
# property phrase and entity noun -- regression test for the supermat gap
# (bare "tc" property phrase) found and fixed 2026-09-19.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["pond", "nfix", "supermat", "measeval"])
def test_every_dataset_supplies_explicit_chatextract_property_names_and_noun(dataset_name):
    from run_extraction import load_dataset_config

    cfg = load_dataset_config(dataset_name)

    assert cfg.chatextract_property_names, f"{dataset_name} has no chatextract_property_names at all"
    missing = set(cfg.attribute_info_dict) - set(cfg.chatextract_property_names)
    assert not missing, (
        f"{dataset_name}: attributes {missing} fall back to the bare attribute key "
        "in ChatExtract's <PROPERTY> phrase"
    )
    assert cfg.chatextract_entity_noun, f"{dataset_name} has no chatextract_entity_noun set"
