"""Unit tests for run_baseline_gliner.py's `ocr_dir`/`include_qualifiers` wiring.

Rung 1 of the staged-gate ladder: `MeasurementLMGliner.__init__`/`fit` are
monkeypatched to no-ops that record the kwargs the instance was constructed
with -- no real GLiNER2 model load, no network calls. Unlike Ablation 1/
NuExtract3/LangExtract, GLiNER needs no dataset-config no-qualifiers variant
(see measurementlm_gliner.py's own docstring), so there's no fail-loud path
to test here beyond the bool type-check.

`ocr_dir` regression guard: until 2026-09-25 `run_baseline_gliner()` accepted
no `ocr_dir` parameter at all and `main()` never read `params.ocr_dir` --
every experiment config that set it (e.g. to a table-cleaning-based
drop_references corpus) was silently ignored, and the run always read
`{data_dir}/ocr_output_raw` instead. Fixed to match the same override
convention as run_baseline_nuextract3.py/run_baseline_langextract.py.
"""
import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import run_baseline_gliner
from scholarlm.config import DatasetConfig, ModelConfig


class _EntitySchema(BaseModel):
    name: str | None


def _make_fixture(tmp_path):
    ocr_dir = tmp_path / "ocr_output_raw"
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
    )
    model_config = ModelConfig(
        name="fixture-model",
        model_id="fixture/fixture-model",
        sampling_params={},
        device="cpu",
    )
    return dataset_config, model_config


def _stub_fit(monkeypatch):
    captured = {}

    def fake_init(self, **kwargs):
        captured["kwargs"] = kwargs
        self.max_prompt_tokens = 0
        self.token_usage = {}

    def fake_fit(self, documents):
        return []

    monkeypatch.setattr(run_baseline_gliner.MeasurementLMGliner, "__init__", fake_init)
    monkeypatch.setattr(run_baseline_gliner.MeasurementLMGliner, "fit", fake_fit)
    return captured


def test_include_qualifiers_non_bool_rejected(tmp_path, monkeypatch):
    dataset_config, model_config = _make_fixture(tmp_path)
    _stub_fit(monkeypatch)
    with pytest.raises(ValueError, match="must be a bool"):
        run_baseline_gliner.run_baseline_gliner(
            dataset_config, model_config, tmp_path / "out", include_qualifiers="false", device="cpu",
        )


def test_include_qualifiers_false_forwarded_to_constructor_and_metadata(tmp_path, monkeypatch):
    dataset_config, model_config = _make_fixture(tmp_path)
    captured = _stub_fit(monkeypatch)
    output_dir = tmp_path / "out"

    run_baseline_gliner.run_baseline_gliner(
        dataset_config, model_config, output_dir, include_qualifiers=False, device="cpu",
    )

    assert captured["kwargs"]["include_qualifiers"] is False
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["include_qualifiers"] is False


def test_include_qualifiers_default_true_forwarded(tmp_path, monkeypatch):
    dataset_config, model_config = _make_fixture(tmp_path)
    captured = _stub_fit(monkeypatch)
    output_dir = tmp_path / "out"

    run_baseline_gliner.run_baseline_gliner(dataset_config, model_config, output_dir, device="cpu")

    assert captured["kwargs"]["include_qualifiers"] is True
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["include_qualifiers"] is True


# ---------------------------------------------------------------------------
# ocr_dir
# ---------------------------------------------------------------------------


def _stub_fit_and_load_papers(monkeypatch):
    captured = _stub_fit(monkeypatch)

    def fake_load_papers(dataset_config, ocr_dir, paper_subset_override):
        captured["ocr_dir"] = ocr_dir
        return [], {}

    monkeypatch.setattr(run_baseline_gliner, "load_papers", fake_load_papers)
    return captured


def test_ocr_dir_default_is_data_dir_ocr_output_raw(tmp_path, monkeypatch):
    dataset_config, model_config = _make_fixture(tmp_path)
    captured = _stub_fit_and_load_papers(monkeypatch)
    output_dir = tmp_path / "out"

    run_baseline_gliner.run_baseline_gliner(dataset_config, model_config, output_dir, device="cpu")

    expected = str(Path(dataset_config.data_dir) / "ocr_output_raw")
    assert captured["ocr_dir"] == expected
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["ocr_dir"] == expected


def test_ocr_dir_override_reaches_load_papers_and_metadata(tmp_path, monkeypatch):
    dataset_config, model_config = _make_fixture(tmp_path)
    captured = _stub_fit_and_load_papers(monkeypatch)
    output_dir = tmp_path / "out"
    override_dir = str(tmp_path / "drop_references")

    run_baseline_gliner.run_baseline_gliner(
        dataset_config, model_config, output_dir, ocr_dir=override_dir, device="cpu",
    )

    assert captured["ocr_dir"] == override_dir
    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata["ocr_dir"] == override_dir


def test_main_reads_ocr_dir_from_params(tmp_path, monkeypatch):
    dataset_config, model_config = _make_fixture(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(
        "id: cfg\nproject: scholarlm\ndescription: test\nseed: 0\n"
        "params:\n  dataset: fixture\n  model: fixture-model\n"
        "  ocr_dir: /custom/ocr/path\n"
    )
    captured = {}
    monkeypatch.setattr(run_baseline_gliner, "load_dataset_config", lambda name: dataset_config)
    monkeypatch.setattr(run_baseline_gliner.paths, "get_model_config", lambda kind, name: model_config)
    monkeypatch.setattr(run_baseline_gliner.paths, "result_dir", lambda *a, **kw: tmp_path / "out")
    monkeypatch.setattr(
        run_baseline_gliner, "run_baseline_gliner", lambda **kwargs: captured.update(kwargs)
    )

    run_baseline_gliner.main([str(config_path)])

    assert captured["ocr_dir"] == "/custom/ocr/path"
