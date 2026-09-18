"""Unit tests for the langextract baseline adapter (measurementlm_langextract.py).

Rung 1 of the staged-gate ladder (see CLAUDE.md): hand-built fixtures,
`langextract.extract` monkeypatched to a canned result -- no network calls, no
vLLM endpoint, no real langextract inference. Covers:

  * `_prompt_description` drops the JSON-envelope line and keeps the rest, and
    raises loud if that line ever goes missing from the source constant.
  * `_build_examples` translates nuextract_examples into langextract
    ExampleData, using each item's `value` as the grounded extraction_text,
    dropping Nones from attributes, and raising on an item with no value to
    ground on.
  * `_page_for_offset` maps a char offset back to the OCR <page number="N">
    tag that contains it, straight against the tag-included text (no
    stripping/offset-index step).
  * `fit()` end-to-end: out-of-vocabulary attributes are dropped with a
    printed count, ungrounded extractions (char_interval=None) are kept with
    page_number=None and counted, and the record key set matches exactly
    direct_extraction_schema's fields plus (document_id, entity_id,
    attribute_terms, page_number) -- the same shape _extract_triples produces.
"""
import json
import sys
from pathlib import Path

import langextract
import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import scholarlm.measurementlm_langextract as lex_mod
from scholarlm.measurementlm_langextract import (
    MeasurementLMLangExtract,
    _build_examples,
    _page_for_offset,
    _prompt_description,
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
}

_NUEXTRACT_EXAMPLES = [
    {
        "input": "Lake A is 3.2 m deep with a pH of 6.9.",
        "output": json.dumps({"items": [
            {"name": "Lake A", "location": None, "date": None, "attribute": "depth", "value": "3.2", "units": "m"},
            {"name": "Lake A", "location": None, "date": None, "attribute": "ph", "value": "6.9", "units": None},
        ]}),
    }
]


def _make_mlm(**overrides):
    kwargs = dict(
        model_name="test-model",
        entity_identification_prompt="Identify entities.",
        entity_identification_schema=_EntitySchema,
        attribute_info_dict=_ATTRIBUTE_INFO,
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt=_FAKE_DIRECT_EXTRACTION_PROMPT,
        nuextract_examples=_NUEXTRACT_EXAMPLES,
        api_base="http://localhost:0/v1",
        use_extra_body=False,
        max_char_buffer=4000,
        extraction_passes=1,
        max_workers=1,
        batch_length=1,
        use_schema_constraints=False,
        fence_output=True,
    )
    kwargs.update(overrides)
    return MeasurementLMLangExtract(**kwargs)


# ---------------------------------------------------------------------------
# _prompt_description
# ---------------------------------------------------------------------------


_FAKE_DIRECT_EXTRACTION_PROMPT = """Entity fields:
- name: the entity name.

Output format requirements:
- Output must be valid, strictly parseable JSON.
- The top-level object must have this form:
{ "items": [...] }
"""


def test_prompt_description_drops_json_envelope_line_keeps_rest():
    result = _prompt_description(_FAKE_DIRECT_EXTRACTION_PROMPT)
    assert '"items" list' not in result
    assert "expert in data extraction" in result  # first sentence of the shared guidelines survives
    assert "name: the entity name." in result  # substantive dataset content survives
    # The dataset prompt's own conflicting output-envelope section is gone too.
    assert "Output format requirements" not in result
    assert '"items": [...]' not in result


def test_prompt_description_raises_if_envelope_line_missing(monkeypatch):
    monkeypatch.setattr(lex_mod, "DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS", "Some other text entirely.")
    with pytest.raises(AssertionError):
        _prompt_description(_FAKE_DIRECT_EXTRACTION_PROMPT)


def test_prompt_description_raises_if_dataset_prompt_has_no_output_format_heading():
    with pytest.raises(AssertionError):
        _prompt_description("Entity fields:\n- name: the entity name.\n")


# ---------------------------------------------------------------------------
# _build_examples
# ---------------------------------------------------------------------------


def test_build_examples_uses_value_as_extraction_text_and_drops_nones():
    examples = _build_examples(_NUEXTRACT_EXAMPLES)
    assert len(examples) == 1
    ex = examples[0]
    assert ex.text == _NUEXTRACT_EXAMPLES[0]["input"]
    assert len(ex.extractions) == 2
    depth, ph = ex.extractions
    assert depth.extraction_text == "3.2"
    assert depth.attributes == {"name": "Lake A", "attribute": "depth", "value": "3.2", "units": "m"}
    assert ph.extraction_text == "6.9"
    assert "location" not in ph.attributes  # None fields dropped
    assert "units" not in ph.attributes


def test_build_examples_empty_when_none():
    assert _build_examples(None) == []


def test_build_examples_raises_on_item_with_no_value():
    bad = [{"input": "text", "output": json.dumps({"items": [
        {"name": "Lake A", "attribute": "depth", "value": None, "units": "m"}
    ]})}]
    with pytest.raises(ValueError):
        _build_examples(bad)


# ---------------------------------------------------------------------------
# _page_for_offset
# ---------------------------------------------------------------------------


def test_page_for_offset_finds_containing_page_tag():
    context = (
        '<page number="0">first page text</page>'
        '<page number="1">second page text</page>'
    )
    page0_pos = context.index("first page")
    page1_pos = context.index("second page")
    assert _page_for_offset(context, page0_pos) == 0
    assert _page_for_offset(context, page1_pos) == 1


def test_page_for_offset_none_when_char_pos_none():
    assert _page_for_offset('<page number="0">x</page>', None) is None


def test_page_for_offset_none_when_untagged():
    assert _page_for_offset("no tags here", 3) is None


# ---------------------------------------------------------------------------
# fit(): OOV attributes, ungrounded extractions, record schema parity
# ---------------------------------------------------------------------------


def _canned_annotated_document(context):
    return langextract.data.AnnotatedDocument(
        text=context,
        extractions=[
            langextract.data.Extraction(  # grounded, in vocabulary
                extraction_class="measurement",
                extraction_text="3.2",
                char_interval=langextract.data.CharInterval(
                    start_pos=context.index("3.2"), end_pos=context.index("3.2") + 3
                ),
                attributes={"name": "Lake A", "location": None, "date": None,
                            "attribute": "depth", "value": "3.2", "units": "m"},
            ),
            langextract.data.Extraction(  # ungrounded, in vocabulary
                extraction_class="measurement",
                extraction_text="6.9",
                char_interval=None,
                attributes={"name": "Lake A", "location": None, "date": None,
                            "attribute": "ph", "value": "6.9", "units": None},
            ),
            langextract.data.Extraction(  # out-of-vocabulary attribute
                extraction_class="measurement",
                extraction_text="12",
                char_interval=langextract.data.CharInterval(start_pos=0, end_pos=2),
                attributes={"name": "Lake A", "location": None, "date": None,
                            "attribute": "hardness", "value": "12", "units": None},
            ),
        ],
    )


def test_fit_drops_oov_attribute_keeps_ungrounded_with_null_page(monkeypatch, capsys):
    mlm = _make_mlm()
    context = '<page number="0">Lake A depth 3.2 m, pH 6.9.</page>'

    monkeypatch.setattr(langextract, "extract", lambda *a, **k: _canned_annotated_document(context))

    records = mlm.fit([context])

    assert len(records) == 2
    by_attr = {r["attribute"]: r for r in records}
    # _deduplicate (inherited, reused as-is) wraps provenance fields into
    # per-duplicate-group lists -- a single grounded/ungrounded record here
    # becomes a one-element list, same as every other baseline's output.
    assert by_attr["depth"]["page_number"] == [0]
    assert by_attr["ph"]["page_number"] == [None]

    out = capsys.readouterr().out
    assert "dropped 1 record" in out
    assert "'hardness': 1" in out  # which attribute got dropped, not just a bare count
    assert "1 record(s) had no character-grounded span" in out


def test_fit_record_key_set_matches_direct_extraction_schema_plus_provenance(monkeypatch):
    mlm = _make_mlm()
    context = '<page number="0">Lake A depth 3.2 m.</page>'

    monkeypatch.setattr(langextract, "extract", lambda *a, **k: _canned_annotated_document(context))

    records = mlm.fit([context])
    # _deduplicate (inherited) additionally injects the other provenance
    # fields it aggregates alongside page_number -- same shape as every other
    # baseline's final.json record (see e.g. ChatExtract's real output).
    expected_keys = (
        set(_DirectSchema.model_fields)
        | {"document_id", "entity_id", "attribute_terms", "page_number"}
        | {"table_number", "row_index", "column_index", "source", "context"}
    )
    assert set(records[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------


def test_missing_direct_extraction_schema_raises():
    with pytest.raises(ValueError):
        _make_mlm(direct_extraction_schema=None)


def test_missing_direct_extraction_prompt_raises():
    with pytest.raises(ValueError):
        _make_mlm(direct_extraction_prompt=None)
