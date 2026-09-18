"""Rung-1 unit tests for analysis/calibration_ids.py -- the id-resolution
helpers calibration_updated.py uses under the 2026-09-16 experiment-contract
migration. Hand-built fixtures under tmp_path, monkeypatching
utils.EXPERIMENT_CONFIGS_ROOT and utils.RESULTS_ROOT so no real repo data is
touched (same pattern as tests/test_synthetic_probe_run_ids.py).
"""
from __future__ import annotations

import json
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


def _write_run_metadata(root: Path, dataset: str, result_type: str, run_id: str, meta_dataset: str) -> Path:
    run_dir = root / dataset / result_type / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump({"dataset": meta_dataset}, f)
    return run_dir


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


# ---------------------------------------------------------------------------
# pinned_extraction_dir
# ---------------------------------------------------------------------------


def test_pinned_extraction_dir_happy_path(fixture_roots):
    _, results_root = fixture_roots
    run_id = "2026-05-05-pond-gemma-3-27b-extraction-01"
    _write_run_metadata(results_root, "pond", "extraction", run_id, meta_dataset="pond")
    d = cids.pinned_extraction_dir("pond", "extraction", run_id)
    assert d == results_root / "pond" / "extraction" / run_id


def test_pinned_extraction_dir_missing_metadata_raises(fixture_roots):
    with pytest.raises(FileNotFoundError):
        cids.pinned_extraction_dir("pond", "extraction", "2026-05-05-does-not-exist-01")


def test_pinned_extraction_dir_dataset_mismatch_raises(fixture_roots):
    _, results_root = fixture_roots
    run_id = "2026-05-06-nfix-gemma-3-27b-extraction-01"
    # Placed under pond/, but its own run_metadata.json says nfix -- a registry typo.
    _write_run_metadata(results_root, "pond", "extraction", run_id, meta_dataset="nfix")
    with pytest.raises(ValueError, match="nfix"):
        cids.pinned_extraction_dir("pond", "extraction", run_id)


# ---------------------------------------------------------------------------
# select_setting
# ---------------------------------------------------------------------------


def test_select_setting_default():
    setting, entry, datasets = cids.select_setting(None)
    assert setting == cids.DEFAULT_SETTING
    assert entry is cids.SETTINGS[cids.DEFAULT_SETTING]
    assert datasets == cids.ALL_DATASETS


def test_select_setting_explicit():
    setting, entry, _ = cids.select_setting("gpt-oss-120b-ablation1")
    assert setting == "gpt-oss-120b-ablation1"
    assert entry["result_type"] == "ablation"


def test_select_setting_unknown_raises():
    with pytest.raises(ValueError, match="Unknown setting"):
        cids.select_setting("not-a-real-setting")


def test_select_setting_narrows_datasets():
    _, _, datasets = cids.select_setting(None, datasets=["nfix"])
    assert datasets == ["nfix"]


def test_select_setting_unknown_dataset_raises():
    with pytest.raises(ValueError, match="Unknown dataset"):
        cids.select_setting(None, datasets=["not-a-real-dataset"])


def test_select_setting_narrowed_datasets_stay_in_canonical_order():
    _, _, datasets = cids.select_setting(None, datasets=["supermat", "pond"])
    assert datasets == ["pond", "supermat"]


# ---------------------------------------------------------------------------
# Registry shape invariant -- every setting must cover every dataset, or a
# per-dataset id lookup elsewhere in calibration_updated.py KeyErrors far from
# the actual mistake (an edited registry entry missing a dataset).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("setting_name", list(cids.SETTINGS))
def test_registry_entry_covers_every_dataset(setting_name):
    entry = cids.SETTINGS[setting_name]
    for key in ("extraction_id", "judge_interp_id", "judge_combine_id"):
        assert set(entry[key]) == set(cids.ALL_DATASETS), (
            f"{setting_name}.{key} covers {sorted(entry[key])}, "
            f"expected exactly {cids.ALL_DATASETS}"
        )


def test_train_datasets_registry_shape():
    """TRAIN_DATASETS/SYN_TRAIN_IDS/SYN_TEST_IDS must all agree on which
    datasets have a migrated synthetic probe -- a dataset added to one but
    not the others would KeyError deep inside compute_predictions rather
    than at settings-resolution time."""
    assert set(cids.SYN_TRAIN_IDS) == set(cids.TRAIN_DATASETS)
    assert set(cids.SYN_TEST_IDS) == set(cids.TRAIN_DATASETS)


@pytest.mark.parametrize("train_ds", cids.TRAIN_DATASETS)
def test_syn_test_ids_cover_primary_and_diag(train_ds):
    assert set(cids.SYN_TEST_IDS[train_ds]) == set(cids.SYN_SPLITS) == {"primary", "diag"}
