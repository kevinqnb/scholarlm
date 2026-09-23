"""Rung-1 unit tests for analysis/analysis_config.py, on hand-built YAML
fixtures under tmp_path so no real repo config is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from analysis import analysis_config as ac  # noqa: E402

GOOD_SEED = 342  # an arbitrary bootstrap seed -- not checked against defaults.seed


def _write(tmp_path: Path, name: str, cfg: dict) -> Path:
    path = tmp_path / f"{name}.yaml"
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _base_cfg(**overrides) -> dict:
    cfg = {
        "id": "2026-09-23-test-analysis-01",
        "project": "scholarlm",
        "description": "test",
        "seed": GOOD_SEED,
        "params": {"experiment_ids": ["2026-01-01-pond-model-extraction-01"]},
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# load_analysis_config
# ---------------------------------------------------------------------------


def test_load_analysis_config_happy_path(tmp_path):
    cfg = _base_cfg()
    path = _write(tmp_path, cfg["id"], cfg)
    loaded = ac.load_analysis_config(path)
    assert loaded["params"]["experiment_ids"] == ["2026-01-01-pond-model-extraction-01"]


@pytest.mark.parametrize("missing_key", ["id", "project", "description", "seed", "params"])
def test_load_analysis_config_missing_envelope_key_raises(tmp_path, missing_key):
    cfg = _base_cfg()
    del cfg[missing_key]
    path = _write(tmp_path, "2026-09-23-test-analysis-01", cfg)
    with pytest.raises(ValueError, match="missing required key"):
        ac.load_analysis_config(path)


def test_load_analysis_config_id_filename_mismatch_raises(tmp_path):
    cfg = _base_cfg(id="2026-09-23-different-id-01")
    path = _write(tmp_path, "2026-09-23-test-analysis-01", cfg)
    with pytest.raises(ValueError, match="does not match filename stem"):
        ac.load_analysis_config(path)


def test_load_analysis_config_params_not_a_mapping_raises(tmp_path):
    cfg = _base_cfg(params=["not", "a", "mapping"])
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="params must be a mapping"):
        ac.load_analysis_config(path)


@pytest.mark.parametrize("bad_experiment_ids", [None, [], "a-string", [1, 2], {"a": "b"}])
def test_load_analysis_config_bad_experiment_ids_raises(tmp_path, bad_experiment_ids):
    cfg = _base_cfg()
    if bad_experiment_ids is None:
        del cfg["params"]["experiment_ids"]
    else:
        cfg["params"]["experiment_ids"] = bad_experiment_ids
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="experiment_ids"):
        ac.load_analysis_config(path)


def test_load_analysis_config_seed_need_not_match_defaults_seed(tmp_path):
    # Unlike an experiments/experiment-configs/ entry, this envelope's seed
    # is the bootstrap RNG seed (recovery_validity.py), not a model-generation
    # seed -- it's never checked against experiments/config.yaml's
    # defaults.seed (see load_analysis_config's own docstring).
    cfg = _base_cfg(seed=GOOD_SEED + 1)
    path = _write(tmp_path, cfg["id"], cfg)
    loaded = ac.load_analysis_config(path)
    assert loaded["seed"] == GOOD_SEED + 1


def test_load_analysis_config_duplicate_experiment_ids_raises(tmp_path):
    cfg = _base_cfg()
    cfg["params"]["experiment_ids"] = ["id-a", "id-b", "id-a"]
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="duplicate"):
        ac.load_analysis_config(path)


def test_load_analysis_config_unknown_top_level_param_key_raises(tmp_path):
    cfg = _base_cfg()
    cfg["params"]["fuzzy_threshold"] = 0.3  # e.g. a value meant for a section, dropped at top level
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="unexpected top-level key"):
        ac.load_analysis_config(path)


def test_load_analysis_config_known_section_key_is_allowed(tmp_path):
    cfg = _base_cfg()
    cfg["params"]["recovery_validity"] = {"n_resamples": 2000}
    path = _write(tmp_path, cfg["id"], cfg)
    loaded = ac.load_analysis_config(path)
    assert loaded["params"]["recovery_validity"] == {"n_resamples": 2000}


# ---------------------------------------------------------------------------
# get_section
# ---------------------------------------------------------------------------


def test_get_section_happy_path():
    cfg = {"params": {"recovery_validity": {"n_resamples": 2000, "alpha": 0.05}}}
    section = ac.get_section(cfg, "recovery_validity", ("n_resamples", "alpha"))
    assert section == {"n_resamples": 2000, "alpha": 0.05}


def test_get_section_missing_section_raises():
    cfg = {"params": {}}
    with pytest.raises(KeyError, match="required but missing"):
        ac.get_section(cfg, "recovery_validity", ("n_resamples",))


def test_get_section_not_a_mapping_raises():
    cfg = {"params": {"recovery_validity": ["not", "a", "mapping"]}}
    with pytest.raises(KeyError, match="must be a mapping"):
        ac.get_section(cfg, "recovery_validity", ())


def test_get_section_missing_required_key_raises():
    cfg = {"params": {"recovery_validity": {"alpha": 0.05}}}
    with pytest.raises(KeyError, match="missing required"):
        ac.get_section(cfg, "recovery_validity", ("n_resamples", "alpha"))


def test_get_section_unexpected_key_raises():
    cfg = {"params": {"recovery_validity": {"n_resamples": 2000, "fuzzy_threshold": 0.3}}}
    with pytest.raises(KeyError, match="unexpected"):
        ac.get_section(cfg, "recovery_validity", ("n_resamples",))


def test_get_section_optional_key_allowed_but_not_required():
    cfg = {"params": {"recovery_validity": {"n_resamples": 2000}}}
    section = ac.get_section(
        cfg, "recovery_validity", ("n_resamples",), optional_keys=("judge_combine_ids",)
    )
    assert section == {"n_resamples": 2000}

    cfg2 = {
        "params": {
            "recovery_validity": {"n_resamples": 2000, "judge_combine_ids": {"a": "b"}}
        }
    }
    section2 = ac.get_section(
        cfg2, "recovery_validity", ("n_resamples",), optional_keys=("judge_combine_ids",)
    )
    assert section2 == {"n_resamples": 2000, "judge_combine_ids": {"a": "b"}}
