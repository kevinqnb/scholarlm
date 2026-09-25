"""Unit test for run_baseline_langextract.py's `temperature_override` wiring.

Rung 1 of the staged-gate ladder: `MeasurementLMLangExtract.fit` is
monkeypatched to a no-op that records the `sampling_params` dict its instance
was constructed with -- no network calls, no vLLM endpoint, no real
langextract inference.

Context (2026-09-21 seed-determinism follow-up): LangExtract's raw
pre-parse output showed a reproducible repetition collapse (a valid
extraction record followed by an unbounded run of blank lines until
max_tokens truncation) at temperature 0.2 -- the same temperature that made
MeasurementLMv2 fully deterministic. Testing whether a higher temperature
avoids the collapse requires overriding just this baseline's temperature
without moving it for extraction v1/v2/ablations/table_cleaning, which all
read the same experiments/model-configs/extraction/<model>.yaml. Hence
`params.temperature` as a per-experiment override, plumbed through
`run_baseline_langextract()`'s `temperature_override` kwarg.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from pydantic import BaseModel

import run_baseline_langextract
from scholarlm.config import DatasetConfig, ModelConfig


class _EntitySchema(BaseModel):
    name: str | None


class _DirectSchema(BaseModel):
    name: str | None
    attribute: str
    value: str | None
    units: str | None


def _make_fixture(tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "doc1.txt").write_text('<page number="0">Lake A depth 3.2 m.</page>')
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"doc1": {}}))

    dataset_config = DatasetConfig(
        name="fixture",
        data_dir=str(tmp_path),
        metadata_file=str(metadata_file),
        entity_schema=_EntitySchema,
        entity_identification_prompt="Identify entities.",
        entity_type_description="site",
        attribute_info_dict={"depth": {"description": "Depth", "units": ["m"]}},
        direct_extraction_schema=_DirectSchema,
        direct_extraction_prompt="Extract depth measurements.\n\nOutput format requirements:\n- x",
        nuextract_examples=[],
    )
    model_config = ModelConfig(
        name="fixture-model",
        model_id="fixture/fixture-model",
        sampling_params={"temperature": 0.2, "top_p": 0.95, "max_tokens": 8192},
    )
    return dataset_config, model_config, str(ocr_dir)


def _stub_fit(monkeypatch):
    captured = {}

    def fake_init(self, **kwargs):
        captured["sampling_params"] = kwargs["sampling_params"]
        self.data = []

    def fake_fit(self, documents):
        return []

    monkeypatch.setattr(run_baseline_langextract.MeasurementLMLangExtract, "__init__", fake_init)
    monkeypatch.setattr(run_baseline_langextract.MeasurementLMLangExtract, "fit", fake_fit)
    return captured


def test_temperature_override_replaces_model_configs_temperature(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    captured = _stub_fit(monkeypatch)

    run_baseline_langextract.run_baseline_langextract(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=tmp_path / "out",
        max_char_buffer=5000,
        extraction_passes=1,
        max_workers=1,
        batch_length=1,
        use_schema_constraints=False,
        fence_output=True,
        ocr_dir=ocr_dir,
        temperature_override=1.0,
    )

    assert captured["sampling_params"]["temperature"] == 1.0
    # Untouched keys survive the override.
    assert captured["sampling_params"]["top_p"] == 0.95
    assert captured["sampling_params"]["max_tokens"] == 8192
    # The model config's own dict is not mutated in place.
    assert model_config.sampling_params["temperature"] == 0.2


def test_no_override_leaves_model_configs_temperature_untouched(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    captured = _stub_fit(monkeypatch)

    run_baseline_langextract.run_baseline_langextract(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=tmp_path / "out",
        max_char_buffer=5000,
        extraction_passes=1,
        max_workers=1,
        batch_length=1,
        use_schema_constraints=False,
        fence_output=True,
        ocr_dir=ocr_dir,
    )

    assert captured["sampling_params"]["temperature"] == 0.2


def test_repetition_penalty_override_sets_sampling_params_and_metadata(tmp_path, monkeypatch):
    """Mirrors the temperature_override tests above -- repetition_penalty is
    a property of this baseline's own prompting/chunking (see
    run_baseline_langextract()'s docstring), not the shared per-model
    sampling defaults, so it must not mutate model_config.sampling_params in
    place, and it must show up in run_metadata.json or a run with a penalty
    would be indistinguishable from one without."""
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    captured = _stub_fit(monkeypatch)
    output_dir = tmp_path / "out"

    run_baseline_langextract.run_baseline_langextract(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        max_char_buffer=5000,
        extraction_passes=1,
        max_workers=1,
        batch_length=1,
        use_schema_constraints=False,
        fence_output=True,
        ocr_dir=ocr_dir,
        repetition_penalty_override=1.3,
    )

    assert captured["sampling_params"]["repetition_penalty"] == 1.3
    assert model_config.sampling_params.get("repetition_penalty") is None

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["repetition_penalty"] == 1.3


def test_no_repetition_penalty_override_leaves_it_absent(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    captured = _stub_fit(monkeypatch)
    output_dir = tmp_path / "out"

    run_baseline_langextract.run_baseline_langextract(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        max_char_buffer=5000,
        extraction_passes=1,
        max_workers=1,
        batch_length=1,
        use_schema_constraints=False,
        fence_output=True,
        ocr_dir=ocr_dir,
    )

    assert captured["sampling_params"].get("repetition_penalty") is None
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["repetition_penalty"] is None
