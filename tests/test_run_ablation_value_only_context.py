"""run_ablation.py forwards parse_quantities_context / standardize_context to the
ablation's MeasurementLM and records them in run_metadata.json; ablations that never
run _standardize()/_parse_quantities() (1 and 7) raise rather than silently ignore them.

Predicted result before running: ablation 4 with both set to "value_only" -> the
constructed object carries "value_only" for both and run_metadata.json records
them; with neither passed -> both "full"; ablation 1 and 7 with either set to
"value_only" -> ValueError; an invalid value -> ValueError from MeasurementLM.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import run_ablation
from scholarlm.measurementlm_ablation4 import MeasurementLMAblation4
from test_run_ablation_metadata_wiring import _make_fixture


def _record_fit(monkeypatch):
    seen = {}

    def fit(self, text):
        seen["parse"] = self.parse_quantities_context
        seen["std"] = self.standardize_context
        return []

    monkeypatch.setattr(MeasurementLMAblation4, "fit", fit)
    return seen


def test_value_only_forwarded_and_recorded(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    seen = _record_fit(monkeypatch)
    out = tmp_path / "results"
    run_ablation.run_ablation(
        dataset_config, model_config, "4", out, ocr_dir=ocr_dir,
        parse_quantities_context="value_only", standardize_context="value_only",
    )
    assert seen == {"parse": "value_only", "std": "value_only"}
    meta = json.loads((out / "run_metadata.json").read_text())
    assert meta["parse_quantities_context"] == "value_only"
    assert meta["standardize_context"] == "value_only"


def test_defaults_are_full(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    seen = _record_fit(monkeypatch)
    run_ablation.run_ablation(dataset_config, model_config, "4", tmp_path / "r", ocr_dir=ocr_dir)
    assert seen == {"parse": "full", "std": "full"}


@pytest.mark.parametrize("ablation", ["1", "7"])
@pytest.mark.parametrize("kw", ["parse_quantities_context", "standardize_context"])
def test_raises_where_steps_not_run(tmp_path, ablation, kw):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="only apply to ablations"):
        run_ablation.run_ablation(
            dataset_config, model_config, ablation, tmp_path / "r", ocr_dir=ocr_dir,
            **{kw: "value_only"},
        )


def test_invalid_value_raises(tmp_path):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    with pytest.raises(ValueError, match="must be 'full' or 'value_only'"):
        run_ablation.run_ablation(
            dataset_config, model_config, "4", tmp_path / "r", ocr_dir=ocr_dir,
            standardize_context="bogus",
        )
