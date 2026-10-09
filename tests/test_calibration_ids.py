"""Rung-1 unit tests for analysis/calibration_ids.py's run-id resolution
helpers (resolve_run, pinned_run_dir). Hand-built fixtures under tmp_path, monkeypatching
utils.EXPERIMENT_CONFIGS_ROOT and utils.RESULTS_ROOT so no real repo data is
touched (same pattern as tests/test_synthetic_probe_run_ids.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import utils as paths  # noqa: E402
from analysis import calibration_ids as cids  # noqa: E402


def _write_config(root: Path, dataset: str, exp_type: str, exp_id: str) -> Path:
    path = root / dataset / exp_type / exp_id / f"{exp_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342, "params": {}}
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


@pytest.fixture
def fixture_roots(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment-configs"
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)
    return exp_root, results_root


# ---------------------------------------------------------------------------
# resolve_run
# ---------------------------------------------------------------------------


def test_resolve_run_happy_path(fixture_roots):
    exp_root, _ = fixture_roots
    _write_config(exp_root, "pond", "judge_interp", "2026-09-16-test-run-01")
    assert cids.resolve_run("2026-09-16-test-run-01") == ("pond", "judge_interp")


def test_resolve_run_missing_raises(fixture_roots):
    with pytest.raises(FileNotFoundError):
        cids.resolve_run("2026-09-16-does-not-exist-01")


# ---------------------------------------------------------------------------
# pinned_run_dir
# ---------------------------------------------------------------------------


def test_pinned_run_dir_happy_path(fixture_roots):
    exp_root, results_root = fixture_roots
    _write_config(exp_root, "pond", "extraction", "2026-05-05-pond-gemma-3-27b-extraction-01")
    d = cids.pinned_run_dir("2026-05-05-pond-gemma-3-27b-extraction-01", "pond", "extraction")
    assert d == results_root / "pond" / "extraction" / "2026-05-05-pond-gemma-3-27b-extraction-01"


def test_pinned_run_dir_wrong_dataset_raises(fixture_roots):
    exp_root, _ = fixture_roots
    _write_config(exp_root, "nfix", "extraction", "2026-05-06-nfix-gemma-3-27b-extraction-01")
    with pytest.raises(ValueError, match="dataset='pond'.*dataset='nfix'|nfix.*pond"):
        cids.pinned_run_dir("2026-05-06-nfix-gemma-3-27b-extraction-01", "pond", "extraction")


def test_pinned_run_dir_wrong_type_raises(fixture_roots):
    exp_root, _ = fixture_roots
    _write_config(exp_root, "pond", "ablation", "2026-05-03-pond-gpt-oss-120b-ablation1-01")
    with pytest.raises(ValueError, match="type='ablation'|ablation.*extraction"):
        cids.pinned_run_dir("2026-05-03-pond-gpt-oss-120b-ablation1-01", "pond", "extraction")
