"""Ablation 1's no-qualifiers variant (params.include_qualifiers=false in
run_ablation.py): drops the qualifier/shape fields (qualifiers, point_value,
lower, upper, list_values, tolerance, standard_deviation) from the direct
extraction schema, prompt, and instructions -- and nothing else. This is the
rung-1 check for the "nothing else should change" requirement: the
no-qualifiers constants must be identical to their originals except for
exactly those 7 fields.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pydantic
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments"))

from scholarlm.instruction_prompts import (  # noqa: E402
    DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS,
    DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS,
)
from scholarlm.config import DatasetConfig, ModelConfig  # noqa: E402
from scholarlm.measurementlm import MeasurementLM  # noqa: E402
from experiments.run_extraction import load_dataset_config  # noqa: E402
import run_ablation  # noqa: E402

_QUALIFIER_FIELDS = {
    "qualifiers", "point_value", "lower", "upper",
    "list_values", "tolerance", "standard_deviation",
}


# ---------------------------------------------------------------------------
# Schema: exactly the 7 qualifier fields removed, everything else identical
# ---------------------------------------------------------------------------


def test_pond_no_qualifiers_schema_drops_exactly_the_qualifier_fields():
    pond = load_dataset_config("pond")
    with_fields = pond.direct_extraction_schema.model_fields
    without_fields = pond.direct_extraction_schema_no_qualifiers.model_fields

    assert set(with_fields) - set(without_fields) == _QUALIFIER_FIELDS
    for name in without_fields:
        assert with_fields[name].annotation == without_fields[name].annotation, name
    # Order of the surviving fields is unchanged.
    assert list(without_fields) == [n for n in with_fields if n not in _QUALIFIER_FIELDS]


# ---------------------------------------------------------------------------
# Shared instructions: only the qualifier bullets and the trailing field
# list differ
# ---------------------------------------------------------------------------


def test_no_qualifiers_instructions_diff_is_exactly_the_qualifier_bullets():
    with_lines = DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS.splitlines()
    without_lines = DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS.splitlines()

    removed_bullet_prefixes = (
        '- qualifiers:', '  - "IsCount"', '  - "IsApproximate"', '  - "IsList"',
        '  - "IsRange"', '  - "IsMean"', '  - "IsMedian"', '  - "HasTolerance"',
        '  - "HasSD"', '- point_value:', '- lower / upper:', '- list_values:',
        '- tolerance:', '- standard_deviation:',
    )
    expected_without_lines = [
        line for line in with_lines if not line.startswith(removed_bullet_prefixes)
    ]
    assert len(expected_without_lines) == len(without_lines)

    diff_indices = [
        i for i, (a, b) in enumerate(zip(expected_without_lines, without_lines)) if a != b
    ]
    # The one remaining line that differs is the trailing field-list sentence.
    field_list_indices = [
        i for i, line in enumerate(without_lines) if line.startswith("- Structure your response")
    ]
    assert diff_indices == field_list_indices
    trimmed_line = without_lines[diff_indices[0]]
    for field in _QUALIFIER_FIELDS:
        assert f'"{field}"' not in trimmed_line
    assert '"attribute", "value", and "units"' in trimmed_line


# ---------------------------------------------------------------------------
# Pond's dataset-specific prompt: only the JSON example's qualifier keys
# differ
# ---------------------------------------------------------------------------


def test_pond_no_qualifiers_prompt_diff_is_exactly_the_json_example_keys():
    pond = load_dataset_config("pond")
    with_lines = pond.direct_extraction_prompt.splitlines()
    without_lines = pond.direct_extraction_prompt_no_qualifiers.splitlines()

    removed_key_prefixes = tuple(f'      "{f}":' for f in _QUALIFIER_FIELDS)
    expected_without_lines = [
        line for line in with_lines if not line.startswith(removed_key_prefixes)
    ]
    # "units" loses its trailing comma once it's the last key in the JSON
    # example -- a cosmetic JSON-formatting difference, not a content change.
    assert [line.rstrip(",") for line in expected_without_lines] == [
        line.rstrip(",") for line in without_lines
    ]

    # Independent check via the JSON example's own quoted keys (the example
    # itself isn't strict JSON -- "qualifiers": [...] uses a bare ellipsis --
    # so this greps keys rather than json.loads-ing it).
    with_keys = set(re.findall(r'"(\w+)":', pond.direct_extraction_prompt))
    without_keys = set(re.findall(r'"(\w+)":', pond.direct_extraction_prompt_no_qualifiers))
    assert with_keys - without_keys == _QUALIFIER_FIELDS


# ---------------------------------------------------------------------------
# run_ablation.py: fails loud on misuse
# ---------------------------------------------------------------------------


def _make_fixture(tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "doc1.txt").write_text("Site A depth 10m.")
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text('{"doc1": {}}')

    from pydantic import BaseModel

    class _EntitySchema(BaseModel):
        name: str | None = None

    dataset_config = DatasetConfig(
        name="fixture",
        data_dir=str(tmp_path),
        metadata_file=str(metadata_file),
        entity_schema=_EntitySchema,
        entity_identification_prompt="Identify entities.",
        entity_type_description="site",
        attribute_info_dict={"depth": {"description": "Depth", "units": ["m"]}},
    )
    model_config = ModelConfig(name="fixture-model", model_id="fixture/fixture-model", sampling_params={})
    return dataset_config, model_config, str(ocr_dir)


def test_include_qualifiers_false_rejected_for_non_ablation1(tmp_path):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="only applies to ablation '1'"):
        run_ablation.run_ablation(
            dataset_config, model_config, "4", tmp_path / "out",
            ocr_dir=ocr_dir, include_qualifiers=False,
        )


def test_include_qualifiers_false_rejected_without_dataset_variant(tmp_path):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    assert dataset_config.direct_extraction_schema_no_qualifiers is None
    with pytest.raises(ValueError, match="direct_extraction_schema_no_qualifiers"):
        run_ablation.run_ablation(
            dataset_config, model_config, "1", tmp_path / "out",
            ocr_dir=ocr_dir, include_qualifiers=False,
        )


def test_include_qualifiers_non_bool_rejected(tmp_path):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="must be a bool"):
        run_ablation.run_ablation(
            dataset_config, model_config, "1", tmp_path / "out",
            ocr_dir=ocr_dir, include_qualifiers="false",
        )


# ---------------------------------------------------------------------------
# run_ablation.py: the model call itself gets the right material, in both
# directions -- this is the regression guard for every existing ablation-1
# config (nfix, supermat, and pond's own qualifier-bearing runs) as well as
# the new no-qualifiers path.
# ---------------------------------------------------------------------------


def _make_ablation1_fixture(tmp_path):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)

    class _DirectSchema(pydantic.BaseModel):
        name: str | None = None
        attribute: str
        value: str | None = None
        units: str | None = None
        qualifiers: list[str] = []
        point_value: str | float | None = None
        lower: str | float | None = None
        upper: str | float | None = None
        list_values: list[str] | None = None
        tolerance: str | float | None = None
        standard_deviation: str | float | None = None

    class _DirectSchemaNoQualifiers(pydantic.BaseModel):
        name: str | None = None
        attribute: str
        value: str | None = None
        units: str | None = None

    dataset_config.direct_extraction_schema = _DirectSchema
    dataset_config.direct_extraction_prompt = "WITH QUALIFIERS PROMPT"
    dataset_config.direct_extraction_schema_no_qualifiers = _DirectSchemaNoQualifiers
    dataset_config.direct_extraction_prompt_no_qualifiers = "NO QUALIFIERS PROMPT"
    return dataset_config, model_config, ocr_dir


def _all_schema_field_names(schema_json: dict) -> set[str]:
    """Every property name anywhere in a model_json_schema() dict, including
    nested $defs -- response_format's schema is the actual contract sent to
    the model, so this is what must (not) contain the qualifier fields."""
    names: set[str] = set()
    for defn in [schema_json, *schema_json.get("$defs", {}).values()]:
        names |= set(defn.get("properties", {}))
    return names


def _run_and_capture_call(monkeypatch, tmp_path, *, include_qualifiers):
    dataset_config, model_config, ocr_dir = _make_ablation1_fixture(tmp_path)
    captured = {}

    def _fake_call_batch(self, message_sets, response_format=None, **kwargs):
        captured["messages"] = message_sets
        captured["response_format"] = response_format
        return ['{"items": []}'] * len(message_sets)

    monkeypatch.setattr(MeasurementLM, "_call_batch", _fake_call_batch)
    output_dir = tmp_path / "out"
    run_ablation.run_ablation(
        dataset_config, model_config, "1", output_dir,
        ocr_dir=ocr_dir, include_qualifiers=include_qualifiers,
    )
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    return captured, metadata


def test_ablation1_no_qualifiers_call_uses_no_qualifiers_material(monkeypatch, tmp_path):
    captured, metadata = _run_and_capture_call(monkeypatch, tmp_path, include_qualifiers=False)

    prompt = captured["messages"][0][0]["content"]
    assert DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS in prompt
    assert DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS not in prompt
    assert "NO QUALIFIERS PROMPT" in prompt
    assert "WITH QUALIFIERS PROMPT" not in prompt

    schema_fields = _all_schema_field_names(captured["response_format"]["json_schema"]["schema"])
    assert schema_fields & _QUALIFIER_FIELDS == set()
    assert {"attribute", "value", "units"} <= schema_fields

    assert metadata["include_qualifiers"] is False


def test_main_reads_include_qualifiers_from_params_and_forwards_it(monkeypatch, tmp_path):
    """main()'s own params.get("include_qualifiers", True) plumbing -- separate
    from run_ablation()'s handling of the value once received."""
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "id: cfg\nproject: scholarlm\ndescription: test\nseed: 342\n"
        "params:\n  dataset: pond\n  model: gemma-3-27b\n  ablation: \"1\"\n"
        "  include_qualifiers: false\n"
    )
    captured = {}
    monkeypatch.setattr(run_ablation, "load_dataset_config", lambda name: object())
    monkeypatch.setattr(run_ablation, "get_model_config", lambda name: object())
    monkeypatch.setattr(run_ablation.paths, "result_dir", lambda *a, **kw: tmp_path / "out")
    monkeypatch.setattr(
        run_ablation, "run_ablation", lambda **kwargs: captured.update(kwargs)
    )

    run_ablation.main([str(config_path)])

    assert captured["include_qualifiers"] is False


def test_ablation1_default_call_is_unchanged_qualifier_bearing_material(monkeypatch, tmp_path):
    captured, metadata = _run_and_capture_call(monkeypatch, tmp_path, include_qualifiers=True)

    prompt = captured["messages"][0][0]["content"]
    assert DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS in prompt
    assert "WITH QUALIFIERS PROMPT" in prompt

    schema_fields = _all_schema_field_names(captured["response_format"]["json_schema"]["schema"])
    assert _QUALIFIER_FIELDS <= schema_fields

    assert metadata["include_qualifiers"] is True
