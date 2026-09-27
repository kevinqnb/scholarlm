"""Unit tests for run_baseline_nuextract3.py's `include_qualifiers` wiring.

Rung 1 of the staged-gate ladder: `MeasurementLMNuExtract3.__init__`/`fit` are
monkeypatched to no-ops that record the kwargs the instance was constructed
with -- no network calls, no vLLM endpoint, no real NuExtract3 inference.
Mirrors run_ablation.py's own include_qualifiers tests
(tests/test_ablation1_no_qualifiers.py) and
tests/test_run_baseline_langextract.py's include_qualifiers tests.
"""
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import run_baseline_nuextract3
from scholarlm.config import DatasetConfig, ModelConfig


class _EntitySchema(BaseModel):
    name: str | None


class _DirectSchema(BaseModel):
    name: str | None
    attribute: str
    value: str | None
    units: str | None
    point_value: str | float | None = None


class _DirectSchemaNoQualifiers(BaseModel):
    name: str | None
    attribute: str
    value: str | None
    units: str | None


def _make_fixture(tmp_path, *, with_no_qualifiers_variant=False):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "doc1.txt").write_text('<page number="0">Lake A depth 3.2 m.</page>')
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"doc1": {}}))

    kwargs = dict(
        name="fixture",
        data_dir=str(tmp_path),
        metadata_file=str(metadata_file),
        entity_schema=_EntitySchema,
        entity_identification_prompt="Identify entities.",
        entity_type_description="site",
        attribute_info_dict={"depth": {"description": "Depth", "units": ["m"]}},
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract depth measurements.",
        nuextract_examples=[{"input": "x", "output": '{"items": []}'}],
    )
    if with_no_qualifiers_variant:
        kwargs["direct_extraction_schema_no_qualifiers"] = _DirectSchemaNoQualifiers
        kwargs["direct_extraction_prompt_no_qualifiers"] = "Extract depth measurements, no qualifiers."
        kwargs["nuextract_examples_no_qualifiers"] = [{"input": "x", "output": '{"items": []}'}]

    dataset_config = DatasetConfig(**kwargs)
    model_config = ModelConfig(
        name="fixture-model",
        model_id="fixture/fixture-model",
        sampling_params={"temperature": 0.0, "max_tokens": 8192},
    )
    return dataset_config, model_config, str(ocr_dir)


def _stub_fit(monkeypatch):
    captured = {}

    def fake_init(self, **kwargs):
        captured["kwargs"] = kwargs
        self.max_prompt_tokens = 0
        self.token_usage = {}
        self.validation_failures = 0
        self.context_length_exceeded_docs = set()

    def fake_fit(self, documents):
        return []

    monkeypatch.setattr(run_baseline_nuextract3.MeasurementLMNuExtract3, "__init__", fake_init)
    monkeypatch.setattr(run_baseline_nuextract3.MeasurementLMNuExtract3, "fit", fake_fit)
    return captured


def test_include_qualifiers_non_bool_rejected(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    _stub_fit(monkeypatch)
    with pytest.raises(ValueError, match="must be a bool"):
        run_baseline_nuextract3.run_baseline_nuextract3(
            dataset_config, model_config, tmp_path / "out", ocr_dir=ocr_dir, include_qualifiers="false",
        )


def test_include_qualifiers_false_rejected_without_dataset_variant(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    _stub_fit(monkeypatch)
    assert dataset_config.direct_extraction_schema_no_qualifiers is None
    with pytest.raises(ValueError, match="direct_extraction_schema_no_qualifiers"):
        run_baseline_nuextract3.run_baseline_nuextract3(
            dataset_config, model_config, tmp_path / "out", ocr_dir=ocr_dir, include_qualifiers=False,
        )


def test_include_qualifiers_false_selects_no_qualifiers_material(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path, with_no_qualifiers_variant=True)
    captured = _stub_fit(monkeypatch)
    output_dir = tmp_path / "out"

    run_baseline_nuextract3.run_baseline_nuextract3(
        dataset_config, model_config, output_dir, ocr_dir=ocr_dir, include_qualifiers=False,
    )

    kwargs = captured["kwargs"]
    assert kwargs["direct_extraction_schema"] is _DirectSchemaNoQualifiers
    assert kwargs["direct_extraction_prompt"] == dataset_config.direct_extraction_prompt_no_qualifiers
    assert kwargs["direct_extraction_instructions"] == (
        run_baseline_nuextract3.DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS
    )
    assert kwargs["examples"] == dataset_config.nuextract_examples_no_qualifiers

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["include_qualifiers"] is False


def test_include_qualifiers_default_true_selects_qualifier_bearing_material(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    captured = _stub_fit(monkeypatch)
    output_dir = tmp_path / "out"

    run_baseline_nuextract3.run_baseline_nuextract3(
        dataset_config, model_config, output_dir, ocr_dir=ocr_dir,
    )

    kwargs = captured["kwargs"]
    assert kwargs["direct_extraction_schema"] is dataset_config.direct_extraction_schema
    assert kwargs["direct_extraction_instructions"] == run_baseline_nuextract3.DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS
    assert kwargs["examples"] == dataset_config.nuextract_examples

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["include_qualifiers"] is True
