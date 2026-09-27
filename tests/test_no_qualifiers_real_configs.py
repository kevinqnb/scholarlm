"""End-to-end (stubbed-model-call) check that the *real* dataset configs'
no-qualifiers material -- not a hand-built fixture -- never leaks a qualifier
field name into what NuExtract3/LangExtract actually send, for all three
datasets (pond, nfix, supermat).

Rung 1 of the staged-gate ladder: `_call_batch`/`langextract.extract` are
monkeypatched, no network calls, no vLLM endpoint. Each no-qualifiers check
has a paired positive-control test using the *qualifier-bearing* (default)
material, proving the detector actually sees qualifier-field mentions in
prose rather than passing vacuously -- otherwise a detector that matches
nothing would make every no-qualifiers assertion pass for the wrong reason.

The prose detector matches a quoted token (`"point_value"`, no trailing colon
required -- the shared instructions list them as `"point_value", "lower"` in
running text, not as JSON keys) or a line-start bullet (`- point_value:`,
the dataset prompt's own per-field style). A stricter JSON-key-only match
would miss both of those and silently pass when it shouldn't -- confirmed by
running the positive control against the naive JSON-key check.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments"))

from scholarlm.instruction_prompts import DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS  # noqa: E402
from scholarlm.measurementlm_langextract import (  # noqa: E402
    JSON_ENVELOPE_LINE_NO_QUALIFIERS,
    MeasurementLMLangExtract,
)
from scholarlm.measurementlm_nuextract3 import MeasurementLMNuExtract3  # noqa: E402
from experiments.run_extraction import load_dataset_config  # noqa: E402

_QUALIFIER_FIELDS = (
    "qualifiers", "point_value", "lower", "upper",
    "list_values", "tolerance", "standard_deviation",
)


def _mentions_any_field(text: str) -> set[str]:
    """Qualifier fields mentioned in free text, as a quoted token
    (`"point_value"`, however it's punctuated after the quote) or a
    line-start bullet (`- point_value:`) -- broader than a JSON-key-shaped
    match, since prose (the shared instructions, a dataset prompt's per-field
    bullets) names these fields without necessarily following JSON-key
    syntax."""
    found = set()
    for field in _QUALIFIER_FIELDS:
        if f'"{field}"' in text:
            found.add(field)
        elif re.search(rf'^\s*-\s*{re.escape(field)}\b', text, re.MULTILINE):
            found.add(field)
    return found


def _schema_property_names(json_schema) -> set[str]:
    """Every property name anywhere in a JSON Schema dict, recursively --
    NuExtract3's schema nests through `$defs`, LangExtract's own
    `_build_output_schema` nests through `extractions.items.properties.
    attributes.properties`, so this walks the whole tree rather than assuming
    either shape."""
    names: set[str] = set()
    if isinstance(json_schema, dict):
        names |= set(json_schema.get("properties", {}))
        for value in json_schema.values():
            names |= _schema_property_names(value)
    elif isinstance(json_schema, list):
        for item in json_schema:
            names |= _schema_property_names(item)
    return names


def _run_nuextract3(dataset_config, *, no_qualifiers: bool, monkeypatch):
    kwargs = dict(
        model_name="test-model",
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        api_base="http://localhost:0/v1",
    )
    if no_qualifiers:
        kwargs.update(
            direct_extraction_schema=dataset_config.direct_extraction_schema_no_qualifiers,
            direct_extraction_prompt=dataset_config.direct_extraction_prompt_no_qualifiers,
            direct_extraction_instructions=DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS,
            examples=dataset_config.nuextract_examples_no_qualifiers,
        )
    else:
        kwargs.update(
            direct_extraction_schema=dataset_config.direct_extraction_schema,
            direct_extraction_prompt=dataset_config.direct_extraction_prompt,
            examples=dataset_config.nuextract_examples,
        )
    mlm = MeasurementLMNuExtract3(**kwargs)

    captured = {}

    def fake_call_batch(self, message_sets, extra_body=None, response_format=None, **kw):
        captured["extra_body"] = extra_body
        captured["response_format"] = response_format
        return ['{"items": []}']

    monkeypatch.setattr(MeasurementLMNuExtract3, "_call_batch", fake_call_batch)
    mlm.fit(["doc text A"])

    chat_kwargs = captured["extra_body"]["chat_template_kwargs"]
    schema_props = _schema_property_names(captured["response_format"]["json_schema"]["schema"])
    mentioned = (
        _mentions_any_field(chat_kwargs["template"])
        | _mentions_any_field(chat_kwargs["instructions"])
        | schema_props
    )
    return mentioned, schema_props


def _run_langextract(dataset_config, *, no_qualifiers: bool, monkeypatch):
    import langextract

    kwargs = dict(
        model_name="test-model",
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        api_base="http://localhost:0/v1",
        use_extra_body=False,
        max_char_buffer=4000,
        extraction_passes=1,
        max_workers=1,
        batch_length=1,
        use_schema_constraints=True,
        fence_output=False,
    )
    if no_qualifiers:
        kwargs.update(
            direct_extraction_schema=dataset_config.direct_extraction_schema_no_qualifiers,
            direct_extraction_prompt=dataset_config.direct_extraction_prompt_no_qualifiers,
            direct_extraction_instructions=DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS,
            json_envelope_line=JSON_ENVELOPE_LINE_NO_QUALIFIERS,
            nuextract_examples=dataset_config.nuextract_examples_no_qualifiers,
        )
    else:
        kwargs.update(
            direct_extraction_schema=dataset_config.direct_extraction_schema,
            direct_extraction_prompt=dataset_config.direct_extraction_prompt,
            nuextract_examples=dataset_config.nuextract_examples,
        )
    mlm = MeasurementLMLangExtract(**kwargs)

    context = '<page number="0">Synthetic smoke-test passage.</page>'
    captured = {}

    def fake_extract(*a, **k):
        captured["prompt_description"] = k.get("prompt_description")
        captured["examples"] = k.get("examples")
        captured["output_schema"] = k.get("output_schema")
        return langextract.data.AnnotatedDocument(text=context, extractions=[])

    monkeypatch.setattr(langextract, "extract", fake_extract)
    mlm.fit([context])

    schema_props = _schema_property_names(captured["output_schema"]) if captured["output_schema"] else set()
    example_attr_keys = {
        key
        for example in captured["examples"]
        for extraction in example.extractions
        for key in (extraction.attributes or {})
    }
    mentioned = _mentions_any_field(captured["prompt_description"]) | schema_props | example_attr_keys
    return mentioned, schema_props


# ---------------------------------------------------------------------------
# NuExtract3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["pond", "nfix", "supermat", "measeval"])
def test_nuextract3_no_qualifiers_material_has_no_qualifier_fields(dataset_name, monkeypatch):
    dataset_config = load_dataset_config(dataset_name)
    assert dataset_config.direct_extraction_schema_no_qualifiers is not None
    assert dataset_config.direct_extraction_prompt_no_qualifiers is not None
    assert dataset_config.nuextract_examples_no_qualifiers is not None

    mentioned, schema_props = _run_nuextract3(dataset_config, no_qualifiers=True, monkeypatch=monkeypatch)

    leaked = mentioned & set(_QUALIFIER_FIELDS)
    assert leaked == set(), f"{dataset_name}: qualifier field(s) leaked into no-qualifiers material: {leaked}"
    assert {"value", "units"} <= schema_props


@pytest.mark.parametrize("dataset_name", ["pond", "nfix", "supermat", "measeval"])
def test_nuextract3_qualifier_bearing_material_is_detected_positive_control(dataset_name, monkeypatch):
    """Proves the detector above can actually see qualifier fields in prose --
    without this, an always-empty `mentioned` set would make the no-qualifiers
    test above pass for the wrong reason."""
    dataset_config = load_dataset_config(dataset_name)
    mentioned, _ = _run_nuextract3(dataset_config, no_qualifiers=False, monkeypatch=monkeypatch)
    assert mentioned & set(_QUALIFIER_FIELDS), (
        f"{dataset_name}: detector found no qualifier mentions in qualifier-bearing "
        f"material -- the no-qualifiers test above may be passing vacuously"
    )


# ---------------------------------------------------------------------------
# LangExtract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset_name", ["pond", "nfix", "supermat", "measeval"])
def test_langextract_no_qualifiers_material_has_no_qualifier_fields(dataset_name, monkeypatch):
    dataset_config = load_dataset_config(dataset_name)
    assert dataset_config.direct_extraction_schema_no_qualifiers is not None
    assert dataset_config.direct_extraction_prompt_no_qualifiers is not None
    assert dataset_config.nuextract_examples_no_qualifiers is not None

    mentioned, schema_props = _run_langextract(dataset_config, no_qualifiers=True, monkeypatch=monkeypatch)

    leaked = mentioned & set(_QUALIFIER_FIELDS)
    assert leaked == set(), f"{dataset_name}: qualifier field(s) leaked into no-qualifiers material: {leaked}"
    assert {"value", "units"} <= schema_props


@pytest.mark.parametrize("dataset_name", ["pond", "nfix", "supermat", "measeval"])
def test_langextract_qualifier_bearing_material_is_detected_positive_control(dataset_name, monkeypatch):
    """Proves the detector above can actually see qualifier fields in prose --
    without this, an always-empty `mentioned` set would make the no-qualifiers
    test above pass for the wrong reason."""
    dataset_config = load_dataset_config(dataset_name)
    mentioned, _ = _run_langextract(dataset_config, no_qualifiers=False, monkeypatch=monkeypatch)
    assert mentioned & set(_QUALIFIER_FIELDS), (
        f"{dataset_name}: detector found no qualifier mentions in qualifier-bearing "
        f"material -- the no-qualifiers test above may be passing vacuously"
    )
