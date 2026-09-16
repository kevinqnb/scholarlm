"""Rung-1 unit tests for the id-addressed synthetic-probe training path added
2026-09-16 (analysis/synthetic_probe_train.py's --judge-run-ids mode, and
run_judge_interp.py's migration of synthetic_file output onto
experiments/results/{dataset}/judge_interp/{id}/).

Hand-built fixtures under tmp_path, monkeypatching utils.EXPERIMENT_CONFIGS_ROOT
and utils.RESULTS_ROOT so no real repo data is touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import utils as paths  # noqa: E402
from analysis import synthetic_probe_train as spt  # noqa: E402


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
    n = mid

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
    exp_id = "2026-09-16-test-synthetic-judge-train-01"
    _write_experiment_config(exp_root, "pond", exp_id, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", exp_id, "qwen-2.5-7b")

    dataset, judge_model, syn_responses, syn_activations, syn_layer_outputs, probe_dir = (
        spt._load_synthetic_run(exp_id)
    )
    assert dataset == "pond"
    assert judge_model == "qwen-2.5-7b"
    assert len(syn_responses) == 24
    assert set(syn_activations.files) == {str(r["measurement_id"]) for r in syn_responses}
    assert probe_dir == results_root / "pond" / "judge_interp" / exp_id / "trained_probe"


def test_load_synthetic_run_missing_judge_param_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    exp_id = "2026-09-16-test-synthetic-judge-train-01"
    path = exp_root / "pond" / "judge_interp" / exp_id / f"{exp_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump({"id": exp_id, "project": "scholarlm", "description": "x",
                         "seed": 342, "params": {"dataset": "pond"}}, f)
    with pytest.raises(ValueError, match=r"missing required params key\(s\): \['judge'\]"):
        spt._load_synthetic_run(exp_id)


def test_load_synthetic_run_missing_file_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    exp_id = "2026-09-16-test-synthetic-judge-train-01"
    _write_experiment_config(exp_root, "pond", exp_id, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", exp_id, "qwen-2.5-7b", drop_file="layer_outputs.npz")
    with pytest.raises(FileNotFoundError, match="layer_outputs.npz"):
        spt._load_synthetic_run(exp_id)


def test_load_synthetic_run_judge_model_mismatch_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    exp_id = "2026-09-16-test-synthetic-judge-train-01"
    _write_experiment_config(exp_root, "pond", exp_id, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", exp_id, "qwen-2.5-7b", mismatched_judge_model="llama-3.1-8b")
    with pytest.raises(ValueError, match="disagrees with"):
        spt._load_synthetic_run(exp_id)


def test_load_synthetic_run_measurement_id_mismatch_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    exp_id = "2026-09-16-test-synthetic-judge-train-01"
    _write_experiment_config(exp_root, "pond", exp_id, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", exp_id, "qwen-2.5-7b", extra_activation_id=True)
    with pytest.raises(ValueError, match="measurement_id set mismatch"):
        spt._load_synthetic_run(exp_id)


# ---------------------------------------------------------------------------
# _select_judge_run_ids
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("SYNTHETIC_PROBE_DATASETS", "SYNTHETIC_PROBE_JUDGES",
              "SYNTHETIC_PROBE_JUDGE_DATE", "SYNTHETIC_PROBE_SOURCE",
              "SYNTHETIC_PROBE_JUDGE_RUN_IDS"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(sys, "argv", ["synthetic_probe_train.py"])


def test_select_judge_run_ids_none_by_default():
    assert spt._select_judge_run_ids() is None


def test_select_judge_run_ids_from_cli(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["synthetic_probe_train.py", "--judge-run-ids", "id-a", "id-b"])
    assert spt._select_judge_run_ids() == ["id-a", "id-b"]


def test_select_judge_run_ids_from_env(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_PROBE_JUDGE_RUN_IDS", "id-a id-b")
    assert spt._select_judge_run_ids() == ["id-a", "id-b"]


def test_select_judge_run_ids_cli_beats_env(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_PROBE_JUDGE_RUN_IDS", "id-env")
    monkeypatch.setattr(sys, "argv", ["synthetic_probe_train.py", "--judge-run-ids", "id-cli"])
    assert spt._select_judge_run_ids() == ["id-cli"]


# ---------------------------------------------------------------------------
# main() mode dispatch: mutual exclusivity
# ---------------------------------------------------------------------------


def test_main_rejects_run_ids_mixed_with_legacy_cli_flags(monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "synthetic_probe_train.py", "--judge-run-ids", "id-a", "--datasets", "pond",
    ])
    with pytest.raises(ValueError, match="mutually exclusive"):
        spt.main()


def test_main_rejects_run_ids_mixed_with_legacy_env(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["synthetic_probe_train.py", "--judge-run-ids", "id-a"])
    monkeypatch.setenv("SYNTHETIC_PROBE_SOURCE", "v2")
    with pytest.raises(ValueError, match="mutually exclusive"):
        spt.main()


# ---------------------------------------------------------------------------
# End-to-end: id mode trains and saves to the id-addressed tree
# ---------------------------------------------------------------------------


def test_main_run_ids_mode_trains_and_saves(tmp_path, monkeypatch, fixture_roots):
    exp_root, results_root = fixture_roots
    exp_id = "2026-09-16-test-synthetic-judge-train-01"
    _write_experiment_config(exp_root, "pond", exp_id, "qwen-2.5-7b")
    _write_run_output(results_root, "pond", exp_id, "qwen-2.5-7b")

    monkeypatch.setattr(sys, "argv", ["synthetic_probe_train.py", "--judge-run-ids", exp_id])
    monkeypatch.chdir(tmp_path)  # FIGURES_DIR is written relative to cwd

    spt.main()

    probe_dir = results_root / "pond" / "judge_interp" / exp_id / "trained_probe"
    assert (probe_dir / "head_probe.pkl").exists()
    assert (probe_dir / "ntp_calibrator.pkl").exists()

    import joblib
    probe_data = joblib.load(probe_dir / "head_probe.pkl")
    assert probe_data["dataset"] == "pond"
    assert probe_data["judge_model"] == "qwen-2.5-7b"
