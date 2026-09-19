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
  * `_attribute_object_schema`/`_build_output_schema` build a JSON Schema
    enum-constraining `attribute` to `attribute_info_dict`'s vocabulary --
    the generation-time constraint langextract's own example-inferred schema
    doesn't provide (it only types by Python type, never by string value).
  * `fit()` end-to-end: out-of-vocabulary attributes are kept, not dropped
    (this baseline polices nothing about its own output -- see its
    docstring), ungrounded extractions (char_interval=None) are kept with
    page_number=None and counted, the record key set matches exactly
    direct_extraction_schema's fields plus (document_id, entity_id,
    attribute_terms, page_number) -- the same shape _extract_triples
    produces -- and `output_schema` is only passed to `lx.extract` when
    `use_schema_constraints` is set.
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
    _attribute_object_schema,
    _build_examples,
    _build_output_schema,
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
# _attribute_object_schema / _build_output_schema
# ---------------------------------------------------------------------------


def test_attribute_object_schema_enum_constrains_attribute_only():
    schema = _attribute_object_schema(_DirectSchema, _ATTRIBUTE_INFO)
    assert schema["properties"]["attribute"] == {"type": "string", "enum": ["depth", "ph"]}
    # Every other field is a plain nullable string, no enum.
    assert schema["properties"]["name"] == {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert "enum" not in schema["properties"]["value"]
    assert set(schema["required"]) == set(_DirectSchema.model_fields)
    assert schema["additionalProperties"] is False


def test_build_output_schema_uses_langextracts_own_key_constants():
    import langextract as lx

    schema = _build_output_schema(_DirectSchema, _ATTRIBUTE_INFO)
    extractions = schema["properties"][lx.data.EXTRACTIONS_KEY]
    item = extractions["items"]
    attributes_key = f"measurement{lx.data.ATTRIBUTE_SUFFIX}"
    assert set(item["properties"]) == {"measurement", attributes_key}
    nested = item["properties"][attributes_key]["anyOf"][0]
    assert nested["properties"]["attribute"]["enum"] == ["depth", "ph"]


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


def test_fit_keeps_out_of_vocabulary_attribute_records(monkeypatch, capsys):
    """This baseline polices nothing about `attribute` itself -- an
    out-of-vocabulary value survives into the record, unfiltered (see
    fit()'s docstring: that's a matching-time concern, not extraction-time)."""
    mlm = _make_mlm()
    context = '<page number="0">Lake A depth 3.2 m, pH 6.9.</page>'

    monkeypatch.setattr(langextract, "extract", lambda *a, **k: _canned_annotated_document(context))

    records = mlm.fit([context])

    assert len(records) == 3
    by_attr = {r["attribute"]: r for r in records}
    assert by_attr["hardness"]["value"] == "12"  # kept, not dropped
    # No _deduplicate call (deliberately -- see fit()'s docstring): page_number
    # stays a plain scalar, not wrapped into a list.
    assert by_attr["depth"]["page_number"] == 0
    assert by_attr["ph"]["page_number"] is None

    out = capsys.readouterr().out
    assert "dropped" not in out
    assert "1 record(s) had no character-grounded span" in out


def test_fit_record_key_set_matches_direct_extraction_schema_plus_provenance(monkeypatch):
    mlm = _make_mlm()
    context = '<page number="0">Lake A depth 3.2 m.</page>'

    monkeypatch.setattr(langextract, "extract", lambda *a, **k: _canned_annotated_document(context))

    records = mlm.fit([context])
    # No _deduplicate call, so no extra provenance-aggregation fields --
    # exactly direct_extraction_schema's own fields plus bookkeeping, the same
    # shape _extract_triples produces.
    expected_keys = set(_DirectSchema.model_fields) | {
        "document_id", "entity_id", "attribute_terms", "page_number",
    }
    assert set(records[0].keys()) == expected_keys


def test_fit_passes_output_schema_only_when_schema_constraints_enabled(monkeypatch):
    context = '<page number="0">Lake A depth 3.2 m.</page>'
    captured = {}

    def fake_extract(*a, **k):
        captured["output_schema"] = k.get("output_schema")
        return langextract.data.AnnotatedDocument(text=context, extractions=[])

    monkeypatch.setattr(langextract, "extract", fake_extract)

    _make_mlm(use_schema_constraints=False).fit([context])
    assert captured["output_schema"] is None

    _make_mlm(use_schema_constraints=True, fence_output=False).fit([context])
    assert captured["output_schema"] is not None
    assert captured["output_schema"] == _build_output_schema(_DirectSchema, _ATTRIBUTE_INFO)


def test_fit_threads_max_tokens_from_sampling_params_into_language_model_params(monkeypatch):
    """langextract has no fallback to the model config's own completion
    budget -- must be threaded through explicitly or per-chunk calls fall
    back to an unverified provider default."""
    context = '<page number="0">Lake A depth 3.2 m.</page>'
    captured = {}

    def fake_extract(*a, **k):
        captured["language_model_params"] = k.get("language_model_params")
        return langextract.data.AnnotatedDocument(text=context, extractions=[])

    monkeypatch.setattr(langextract, "extract", fake_extract)

    _make_mlm(sampling_params={"temperature": 0.6, "max_tokens": 8192}).fit([context])
    assert captured["language_model_params"] == {"max_output_tokens": 8192}


def test_fit_does_not_merge_duplicate_mentions(monkeypatch):
    """Same entity+attribute+event extracted twice must survive as two
    records -- deduplication is deliberately not this baseline's job (that's
    langextract's own chunking/resolution to decide, not ours to redo)."""
    mlm = _make_mlm()
    context = '<page number="0">Lake A depth 3.2 m. Lake A depth 3.2 m.</page>'
    first = context.index("3.2")
    second = context.index("3.2", first + 1)

    def fake_extract(*a, **k):
        item = {"name": "Lake A", "location": None, "date": None,
                "attribute": "depth", "value": "3.2", "units": "m"}
        return langextract.data.AnnotatedDocument(
            text=context,
            extractions=[
                langextract.data.Extraction(
                    extraction_class="measurement", extraction_text="3.2",
                    char_interval=langextract.data.CharInterval(start_pos=first, end_pos=first + 3),
                    attributes=item,
                ),
                langextract.data.Extraction(
                    extraction_class="measurement", extraction_text="3.2",
                    char_interval=langextract.data.CharInterval(start_pos=second, end_pos=second + 3),
                    attributes=item,
                ),
            ],
        )

    monkeypatch.setattr(langextract, "extract", fake_extract)

    records = mlm.fit([context])
    assert len(records) == 2
    assert records[0]["entity_id"] != records[1]["entity_id"]


# ---------------------------------------------------------------------------
# Constructor guards
# ---------------------------------------------------------------------------


def test_missing_direct_extraction_schema_raises():
    with pytest.raises(ValueError):
        _make_mlm(direct_extraction_schema=None)


def test_missing_direct_extraction_prompt_raises():
    with pytest.raises(ValueError):
        _make_mlm(direct_extraction_prompt=None)
