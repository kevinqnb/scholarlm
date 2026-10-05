"""Rung-1 unit tests for analysis/synthetic_probe_train.py: the id-addressed
run loader (_load_synthetic_run), and main() on a tiny hand-built judge_interp run
-- where it writes (analysis/results/synthetic_probe/<analysis config id>/, never
the judge_interp run dir), which artifacts the USE_PLATT_SCALING flag selects, and
same-seed determinism.

Hand-built fixtures under tmp_path, monkeypatching utils.EXPERIMENT_CONFIGS_ROOT,
utils.RESULTS_ROOT and spt.RESULTS_ROOT so no real repo data is touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import yaml  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import utils as paths  # noqa: E402
from analysis import synthetic_probe_train as spt  # noqa: E402

EXP_ID = "2026-09-16-test-synthetic-judge-train-01"
PROBE_CFG_ID = "2026-10-02-test-synthetic-probe-01"


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_experiment_config(root: Path, dataset: str, exp_id: str, judge: str) -> Path:
    path = root / dataset / "judge_interp" / exp_id / f"{exp_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "id": exp_id, "project": "scholarlm", "description": "test", "seed": 342,
        "params": {"dataset": dataset, "judge": judge, "synthetic_file": "fixture.json"},
    }
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _write_run_output(
    root: Path, dataset: str, exp_id: str, judge_model: str,
    n_docs: int = 12, rows_per_doc: int = 2, n_layers: int = 2, n_heads: int = 2, head_dim: int = 4,
    *, mismatched_judge_model: str | None = None, drop_file: str | None = None,
    extra_activation_id: bool = False,
) -> Path:
    run_dir = root / dataset / "judge_interp" / exp_id
    run_dir.mkdir(parents=True, exist_ok=True)

    responses = []
    mid = 0
    for doc_idx in range(n_docs):
        for row in range(rows_per_doc):
            responses.append({
                "measurement_id": mid,
                "document_id": f"doc{doc_idx}",
                "label": "valid" if mid % 2 == 0 else "invalid",
                "judgement_p_true": 0.9 if mid % 2 == 0 else 0.1,
            })
            mid += 1

    if drop_file != "responses.json":
        with open(run_dir / "responses.json", "w") as f:
            json.dump(responses, f)

    rng = np.random.RandomState(0)
    attn = {str(r["measurement_id"]): rng.randn(n_layers, n_heads, head_dim).astype(np.float32) for r in responses}
    if extra_activation_id:
        attn["9999"] = rng.randn(n_layers, n_heads, head_dim).astype(np.float32)
    if drop_file != "attention_outputs.npz":
        np.savez_compressed(run_dir / "attention_outputs.npz", **attn)

    layer_out = {str(r["measurement_id"]): rng.randn(n_layers, head_dim).astype(np.float32) for r in responses}
    if drop_file != "layer_outputs.npz":
        np.savez_compressed(run_dir / "layer_outputs.npz", **layer_out)

    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump({"judge_model": mismatched_judge_model or judge_model}, f)

    return run_dir


@pytest.fixture
def fixture_roots(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment-configs"
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)
    return exp_root, results_root


# ---------------------------------------------------------------------------
# _load_synthetic_run
# ---------------------------------------------------------------------------


def test_load_synthetic_run_happy_path(fixture_roots):
    exp_root, results_root = fixture_roots
    _write_experiment_config(exp_root, "pond", EXP_ID, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", EXP_ID, "qwen-2.5-7b")

    judge_model, syn_responses, syn_activations, syn_layer_outputs = spt._load_synthetic_run(EXP_ID, "pond")
    assert judge_model == "qwen-2.5-7b"
    assert len(syn_responses) == 24
    assert set(syn_activations.files) == {str(r["measurement_id"]) for r in syn_responses}
    assert set(syn_layer_outputs.files) == set(syn_activations.files)


def test_load_synthetic_run_dataset_mismatch_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    _write_experiment_config(exp_root, "pond", EXP_ID, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", EXP_ID, "qwen-2.5-7b")
    with pytest.raises(ValueError, match="analysis config says dataset='nfix'"):
        spt._load_synthetic_run(EXP_ID, "nfix")


def test_load_synthetic_run_missing_judge_param_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    path = exp_root / "pond" / "judge_interp" / EXP_ID / f"{EXP_ID}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"id": EXP_ID, "project": "scholarlm", "description": "x",
                         "seed": 342, "params": {"dataset": "pond"}}, f)
    with pytest.raises(ValueError, match=r"missing required params key\(s\): \['judge'\]"):
        spt._load_synthetic_run(EXP_ID, "pond")


def test_load_synthetic_run_missing_file_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    _write_experiment_config(exp_root, "pond", EXP_ID, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", EXP_ID, "qwen-2.5-7b", drop_file="layer_outputs.npz")
    with pytest.raises(FileNotFoundError, match="layer_outputs.npz"):
        spt._load_synthetic_run(EXP_ID, "pond")


def test_load_synthetic_run_judge_model_mismatch_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    _write_experiment_config(exp_root, "pond", EXP_ID, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", EXP_ID, "qwen-2.5-7b", mismatched_judge_model="llama-3.1-8b")
    with pytest.raises(ValueError, match="disagrees with"):
        spt._load_synthetic_run(EXP_ID, "pond")


def test_load_synthetic_run_measurement_id_mismatch_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    _write_experiment_config(exp_root, "pond", EXP_ID, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", EXP_ID, "qwen-2.5-7b", extra_activation_id=True)
    with pytest.raises(ValueError, match="measurement_id set mismatch"):
        spt._load_synthetic_run(EXP_ID, "pond")


# ---------------------------------------------------------------------------
# main(): one positional analysis config, output under spt.RESULTS_ROOT
# ---------------------------------------------------------------------------


@pytest.fixture
def run_main(tmp_path, monkeypatch, fixture_roots):
    """Build a tiny pond run, point spt at tmp dirs, return a callable that runs
    main() with the given USE_PLATT_SCALING and returns the output dirs."""
    exp_root, results_root = fixture_roots
    _write_experiment_config(exp_root, "pond", EXP_ID, "qwen-2.5-7b")
    run_dir = _write_run_output(results_root, "pond", EXP_ID, "qwen-2.5-7b")

    probe_results_root = tmp_path / "synthetic_probe"
    monkeypatch.setattr(spt, "RESULTS_ROOT", probe_results_root)

    cfg_path = tmp_path / "analysis-configs" / f"{PROBE_CFG_ID}.yaml"
    cfg_path.parent.mkdir()
    with open(cfg_path, "w") as f:
        yaml.safe_dump({
            "id": PROBE_CFG_ID, "project": "scholarlm", "description": "test", "seed": 7,
            "params": {"dataset": "pond", "judge_interp_id": EXP_ID},
        }, f)

    def _run(use_platt: bool):
        monkeypatch.setattr(spt, "USE_PLATT_SCALING", use_platt)
        monkeypatch.setattr(sys, "argv", ["synthetic_probe_train.py", str(cfg_path)])
        spt.main()
        out_dir = probe_results_root / PROBE_CFG_ID
        return out_dir, out_dir / "trained_probe", run_dir

    return _run


def test_main_noplatt_saves_plain_pipeline_under_analysis_results(run_main):
    out_dir, probe_dir, run_dir = run_main(False)

    assert (probe_dir / "head_probe_noplatt.pkl").exists()
    assert (probe_dir / "ntp_calibrator_noplatt.pkl").exists()
    assert not (probe_dir / "head_probe.pkl").exists()
    assert not (run_dir / "trained_probe").exists()  # never the judge_interp run dir

    probe_data = joblib.load(probe_dir / "head_probe_noplatt.pkl")
    assert type(probe_data["probe"]) is Pipeline  # one fit, no CalibratedClassifierCV ensemble
    assert probe_data["dataset"] == "pond"
    assert probe_data["judge_model"] == "qwen-2.5-7b"
    assert type(joblib.load(probe_dir / "ntp_calibrator_noplatt.pkl")["calibrator"]) is Pipeline

    results = json.loads((out_dir / "results.json").read_text())
    assert results["use_platt_scaling"] is False
    assert Path(results["probe_path"]).parent == probe_dir
    assert (out_dir / "head_scores.npz").exists()


def test_main_platt_saves_calibrated_ensemble_unsuffixed(run_main):
    out_dir, probe_dir, run_dir = run_main(True)

    assert (probe_dir / "head_probe.pkl").exists()
    assert (probe_dir / "ntp_calibrator.pkl").exists()
    assert not (probe_dir / "head_probe_noplatt.pkl").exists()
    assert not (run_dir / "trained_probe").exists()

    probe_data = joblib.load(probe_dir / "head_probe.pkl")
    assert type(probe_data["probe"]) is CalibratedClassifierCV
    assert len(probe_data["probe"].calibrated_classifiers_) == spt.N_FOLDS
    assert json.loads((out_dir / "results.json").read_text())["use_platt_scaling"] is True


def test_main_same_seed_is_deterministic(run_main):
    _, probe_dir, _ = run_main(False)
    first = joblib.load(probe_dir / "head_probe_noplatt.pkl")
    first_coef = first["probe"].named_steps["clf"].coef_.copy()
    first_heads = first["top_k_heads"]

    _, probe_dir, _ = run_main(False)
    second = joblib.load(probe_dir / "head_probe_noplatt.pkl")
    assert second["top_k_heads"] == first_heads
    assert np.array_equal(second["probe"].named_steps["clf"].coef_, first_coef)
