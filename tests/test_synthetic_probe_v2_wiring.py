"""Rung-1 unit tests for the ``--source v2`` wiring that lets the probe /
calibration pipeline train and evaluate on the augmented ``_v2`` synthetic
corpus without overwriting the baseline artifacts every committed calibration
number depends on.

Experiment: ``2026-09-08-probe-v2-calibration-01``.

Covered:
  * ``paths.trained_probe_dir`` — default path byte-for-byte unchanged; a
    ``source`` routes to the parallel ``synthetic_probe_<source>`` tree; a
    malformed source is a hard error.
  * ``loaders.load_trained_probe`` / ``load_trained_ntp_calibrator`` — the
    ``source`` kwarg reaches ``trained_probe_dir`` (checked via the path named
    in the FileNotFoundError, and via a real round-trip through a fake artifact).
  * ``synthetic_probe_train._select_run_config`` — ``baseline`` resolves to the
    exact prior behavior (``syn_name`` / ``probe_source`` both ``None``); ``v2``
    flips both; missing required config is a hard error.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import paths  # noqa: E402
from analysis import loaders  # noqa: E402
from analysis import synthetic_probe_train as spt  # noqa: E402


# ── paths.trained_probe_dir ──────────────────────────────────────────────────


def test_trained_probe_dir_default_unchanged():
    p = paths.trained_probe_dir("pond", "qwen-2.5-7b")
    assert p == paths.EXPERIMENTS_ROOT / "pond" / "synthetic_probe" / "qwen-2.5-7b" / "trained_probe"
    # explicit source=None must be identical to omitting it
    assert paths.trained_probe_dir("pond", "qwen-2.5-7b", source=None) == p


def test_trained_probe_dir_v2_parallel_tree():
    p = paths.trained_probe_dir("pond", "qwen-2.5-7b", source="v2")
    assert p.parts[-4:] == ("pond", "synthetic_probe_v2", "qwen-2.5-7b", "trained_probe")
    # the v2 trained-probe dir sits alongside the v2 judge run dirs
    judge_run = paths.synthetic_probe_named("pond", "v2", "qwen-2.5-7b", "2026_09_08")
    assert judge_run.parent == p.parent


def test_trained_probe_dir_rejects_bad_source():
    for bad in ("V2", "v2/x", "has space", "-x", ""):
        with pytest.raises(ValueError):
            paths.trained_probe_dir("pond", "qwen-2.5-7b", source=bad)


# ── loaders pass `source` through ────────────────────────────────────────────


def test_load_trained_probe_source_in_error_path(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"synthetic_probe_v2/.*trained_probe/head_probe\.pkl"):
        loaders.load_trained_probe("pond", "qwen-2.5-7b", source="v2")
    with pytest.raises(FileNotFoundError) as ei:
        loaders.load_trained_probe("pond", "qwen-2.5-7b", source=None)
    assert "synthetic_probe_v2" not in str(ei.value)


def test_load_trained_probe_roundtrip_from_v2_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path)
    d = tmp_path / "pond" / "synthetic_probe_v2" / "qwen-2.5-7b" / "trained_probe"
    d.mkdir(parents=True)
    sentinel = {"probe": "SENTINEL", "dataset": "pond"}
    joblib.dump(sentinel, d / "head_probe.pkl")
    got = loaders.load_trained_probe("pond", "qwen-2.5-7b", source="v2")
    assert got["probe"] == "SENTINEL"


def test_load_ntp_calibrator_source_in_error_path(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError, match=r"synthetic_probe_v2/.*ntp_calibrator\.pkl"):
        loaders.load_trained_ntp_calibrator("pond", "qwen-2.5-7b", source="v2")


# ── synthetic_probe_train._select_run_config ─────────────────────────────────


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("SYNTHETIC_PROBE_DATASETS", "SYNTHETIC_PROBE_JUDGES",
              "SYNTHETIC_PROBE_JUDGE_DATE", "SYNTHETIC_PROBE_SOURCE"):
        monkeypatch.delenv(k, raising=False)
    # _select_run_config reads sys.argv via parse_known_args
    monkeypatch.setattr(sys, "argv", ["synthetic_probe_train.py"])


def _set_required(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_PROBE_DATASETS", "pond nfix supermat")
    monkeypatch.setenv("SYNTHETIC_PROBE_JUDGES", "qwen-2.5-7b")
    monkeypatch.setenv("SYNTHETIC_PROBE_JUDGE_DATE", "2026_09_08")


def test_select_baseline_is_prior_behavior(monkeypatch):
    _set_required(monkeypatch)
    datasets, judges, judge_date, syn_name, probe_source = spt._select_run_config()
    assert datasets == ["pond", "nfix", "supermat"]
    assert judges == ["qwen-2.5-7b"]
    assert judge_date == "2026_09_08"
    assert syn_name is None and probe_source is None


def test_select_v2_flips_both_selectors(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("SYNTHETIC_PROBE_SOURCE", "v2")
    *_, syn_name, probe_source = spt._select_run_config()
    assert syn_name == "v2" and probe_source == "v2"


def test_select_arbitrary_source_name(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("SYNTHETIC_PROBE_SOURCE", "rung3")
    *_, syn_name, probe_source = spt._select_run_config()
    assert syn_name == "rung3" and probe_source == "rung3"


def test_select_missing_config_is_hard_error(monkeypatch):
    monkeypatch.setenv("SYNTHETIC_PROBE_JUDGES", "qwen-2.5-7b")
    monkeypatch.setenv("SYNTHETIC_PROBE_JUDGE_DATE", "2026_09_08")
    # no SYNTHETIC_PROBE_DATASETS
    with pytest.raises(ValueError, match="--datasets"):
        spt._select_run_config()


def test_select_cli_flags_beat_env(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("SYNTHETIC_PROBE_SOURCE", "baseline")
    monkeypatch.setattr(
        sys, "argv",
        ["synthetic_probe_train.py", "--source", "v2", "--datasets", "pond",
         "--judges", "qwen-2.5-7b", "--judge-date", "2026_09_08"],
    )
    datasets, judges, judge_date, syn_name, probe_source = spt._select_run_config()
    assert datasets == ["pond"] and syn_name == "v2" and probe_source == "v2"


def test_select_bad_source_is_hard_error(monkeypatch):
    _set_required(monkeypatch)
    monkeypatch.setenv("SYNTHETIC_PROBE_SOURCE", "V2")
    with pytest.raises(ValueError, match="source"):
        spt._select_run_config()
