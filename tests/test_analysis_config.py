"""Rung-1 unit tests for analysis/common/config.py, on hand-built YAML
fixtures under tmp_path so no real repo config is touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from analysis.common import config as ac  # noqa: E402

GOOD_SEED = 342  # an arbitrary bootstrap seed -- not checked against defaults.seed


def _write(tmp_path: Path, name: str, cfg: dict, analysis_type: str = "recovery-validity") -> Path:
    path = tmp_path / analysis_type / f"{name}.yaml"
    path.parent.mkdir(exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _ground_truth_file(tmp_path: Path) -> Path:
    path = tmp_path / "ground_truth.json"
    path.write_text("[]")
    return path


def _base_cfg(tmp_path: Path, **overrides) -> dict:
    cfg = {
        "id": "2026-09-23-test-analysis-01",
        "project": "scholarlm",
        "description": "test",
        "seed": GOOD_SEED,
        "params": {
            "experiment_ids": ["2026-01-01-pond-model-extraction-01"],
            "ground_truth_file": str(_ground_truth_file(tmp_path)),
        },
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# load_analysis_config
# ---------------------------------------------------------------------------


def test_load_analysis_config_happy_path(tmp_path):
    cfg = _base_cfg(tmp_path)
    path = _write(tmp_path, cfg["id"], cfg)
    loaded = ac.load_analysis_config(path, "recovery-validity")
    assert loaded["params"]["experiment_ids"] == ["2026-01-01-pond-model-extraction-01"]


@pytest.mark.parametrize("missing_key", ["id", "project", "description", "seed", "params"])
def test_load_analysis_config_missing_envelope_key_raises(tmp_path, missing_key):
    cfg = _base_cfg(tmp_path)
    del cfg[missing_key]
    path = _write(tmp_path, "2026-09-23-test-analysis-01", cfg)
    with pytest.raises(ValueError, match="missing required key"):
        ac.load_analysis_config(path, "recovery-validity")


def test_load_analysis_config_id_filename_mismatch_raises(tmp_path):
    cfg = _base_cfg(tmp_path, id="2026-09-23-different-id-01")
    path = _write(tmp_path, "2026-09-23-test-analysis-01", cfg)
    with pytest.raises(ValueError, match="does not match filename stem"):
        ac.load_analysis_config(path, "recovery-validity")


def test_load_analysis_config_params_not_a_mapping_raises(tmp_path):
    cfg = _base_cfg(tmp_path, params=["not", "a", "mapping"])
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="params must be a mapping"):
        ac.load_analysis_config(path, "recovery-validity")


@pytest.mark.parametrize("bad_experiment_ids", [None, [], "a-string", [1, 2], {"a": "b"}])
def test_load_analysis_config_bad_experiment_ids_raises(tmp_path, bad_experiment_ids):
    cfg = _base_cfg(tmp_path)
    if bad_experiment_ids is None:
        del cfg["params"]["experiment_ids"]
    else:
        cfg["params"]["experiment_ids"] = bad_experiment_ids
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="experiment_ids"):
        ac.load_analysis_config(path, "recovery-validity")


@pytest.mark.parametrize("bad_ground_truth_file", [None, "", 123, ["a"]])
def test_load_analysis_config_bad_ground_truth_file_raises(tmp_path, bad_ground_truth_file):
    cfg = _base_cfg(tmp_path)
    if bad_ground_truth_file is None:
        del cfg["params"]["ground_truth_file"]
    else:
        cfg["params"]["ground_truth_file"] = bad_ground_truth_file
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="ground_truth_file"):
        ac.load_analysis_config(path, "recovery-validity")


def test_load_analysis_config_nonexistent_ground_truth_file_raises(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["params"]["ground_truth_file"] = str(tmp_path / "no_such_file.json")
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="does not exist"):
        ac.load_analysis_config(path, "recovery-validity")


def test_get_ground_truth_path_resolves_repo_relative_path():
    cfg = {"params": {"ground_truth_file": "data/pond/ground_truth_review.json"}}
    resolved = ac.get_ground_truth_path(cfg)
    assert resolved.is_absolute()
    assert resolved.parts[-3:] == ("data", "pond", "ground_truth_review.json")


def test_get_ground_truth_path_keeps_absolute_path(tmp_path):
    gt_path = _ground_truth_file(tmp_path)
    cfg = {"params": {"ground_truth_file": str(gt_path)}}
    assert ac.get_ground_truth_path(cfg) == gt_path


def test_load_analysis_config_seed_need_not_match_defaults_seed(tmp_path):
    # Unlike an experiments/experiment-configs/ entry, this envelope's seed
    # is the bootstrap RNG seed (recovery_validity.py), not a model-generation
    # seed -- it's never checked against experiments/config.yaml's
    # defaults.seed (see load_analysis_config's own docstring).
    cfg = _base_cfg(tmp_path, seed=GOOD_SEED + 1)
    path = _write(tmp_path, cfg["id"], cfg)
    loaded = ac.load_analysis_config(path, "recovery-validity")
    assert loaded["seed"] == GOOD_SEED + 1


def test_load_analysis_config_duplicate_experiment_ids_raises(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["params"]["experiment_ids"] = ["id-a", "id-b", "id-a"]
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="duplicate"):
        ac.load_analysis_config(path, "recovery-validity")


def test_load_analysis_config_unknown_top_level_param_key_raises(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["params"]["fuzzy_threshold"] = 0.3  # e.g. a value meant for a section, dropped at top level
    path = _write(tmp_path, cfg["id"], cfg)
    with pytest.raises(ValueError, match="unexpected top-level key"):
        ac.load_analysis_config(path, "recovery-validity")


def test_load_analysis_config_known_section_key_is_allowed(tmp_path):
    cfg = _base_cfg(tmp_path)
    cfg["params"]["recovery_validity"] = {"n_resamples": 2000}
    path = _write(tmp_path, cfg["id"], cfg)
    loaded = ac.load_analysis_config(path, "recovery-validity")
    assert loaded["params"]["recovery_validity"] == {"n_resamples": 2000}


# ---------------------------------------------------------------------------
# analysis types: config / results paths and the type-directory check
# ---------------------------------------------------------------------------


def test_analysis_config_path_is_typed():
    assert ac.analysis_config_path("platt-scaling", "x-01") == ac.ANALYSIS_CONFIGS_ROOT / "platt-scaling" / "x-01.yaml"


@pytest.mark.parametrize("bad", ["platt_scaling", "match-cache", "results"])
def test_analysis_config_path_rejects_unknown_type(bad):
    with pytest.raises(ValueError, match="unknown analysis type"):
        ac.analysis_config_path(bad, "x-01")


def test_analysis_results_dir_accepts_results_only_types():
    assert ac.analysis_results_dir("match-cache") == ac.ANALYSIS_RESULTS_ROOT / "match-cache"
    with pytest.raises(ValueError, match="unknown analysis type"):
        ac.analysis_results_dir("match_cache")


def test_load_analysis_config_wrong_type_dir_raises(tmp_path):
    cfg = _base_cfg(tmp_path)
    path = _write(tmp_path, cfg["id"], cfg, "calibration")
    with pytest.raises(ValueError, match="must live in analysis-configs/recovery-validity/"):
        ac.load_analysis_config(path, "recovery-validity")


def test_load_analysis_config_measeval_type(tmp_path):
    cfg = _base_cfg(tmp_path)
    path = _write(tmp_path, cfg["id"], cfg, "measeval")
    assert ac.load_analysis_config(path, "measeval")["id"] == cfg["id"]


def test_load_analysis_config_rejects_other_types(tmp_path):
    cfg = _base_cfg(tmp_path)
    path = _write(tmp_path, cfg["id"], cfg, "calibration")
    with pytest.raises(ValueError, match="serves recovery-validity and measeval"):
        ac.load_analysis_config(path, "calibration")


def test_synthetic_probe_config_in_calibration_dir_raises(tmp_path):
    path = _write(tmp_path, "probe-01", _probe_cfg(), "calibration")
    with pytest.raises(ValueError, match="must live in analysis-configs/synthetic-probe/"):
        ac.load_synthetic_probe_config(path)


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


# ── load_synthetic_probe_config ───────────────────────────────────────────

def _write_probe(tmp_path, name, cfg):
    return _write(tmp_path, name, cfg, "synthetic-probe")


def _probe_cfg(**params_override):
    params = {"dataset": "pond", "judge_interp_id": "2026-09-30-pond-v3-qwen-2.5-7b-synthetic-judge-train-01",
              "use_platt_scaling": False}
    params.update(params_override)
    return {"id": "probe-01", "project": "scholarlm", "description": "d", "seed": GOOD_SEED, "params": params}


def test_synthetic_probe_config_loads(tmp_path):
    cfg = ac.load_synthetic_probe_config(_write_probe(tmp_path, "probe-01", _probe_cfg()))
    assert cfg["params"]["dataset"] == "pond"


def test_synthetic_probe_config_rejects_extra_key(tmp_path):
    path = _write_probe(tmp_path, "probe-01", _probe_cfg(experiment_ids=["x"]))
    with pytest.raises(ValueError, match="must be exactly"):
        ac.load_synthetic_probe_config(path)


def test_synthetic_probe_config_rejects_missing_key(tmp_path):
    cfg = _probe_cfg()
    del cfg["params"]["dataset"]
    with pytest.raises(ValueError, match="must be exactly"):
        ac.load_synthetic_probe_config(_write_probe(tmp_path, "probe-01", cfg))


def test_synthetic_probe_config_rejects_missing_use_platt_scaling(tmp_path):
    cfg = _probe_cfg()
    del cfg["params"]["use_platt_scaling"]
    with pytest.raises(ValueError, match="must be exactly"):
        ac.load_synthetic_probe_config(_write_probe(tmp_path, "probe-01", cfg))


def test_synthetic_probe_config_rejects_non_bool_use_platt_scaling(tmp_path):
    with pytest.raises(ValueError, match="use_platt_scaling must be a bool"):
        ac.load_synthetic_probe_config(_write_probe(tmp_path, "probe-01", _probe_cfg(use_platt_scaling="true")))


def test_synthetic_probe_config_rejects_id_mismatch(tmp_path):
    with pytest.raises(ValueError, match="does not match filename stem"):
        ac.load_synthetic_probe_config(_write_probe(tmp_path, "other-name", _probe_cfg()))


def test_synthetic_probe_config_rejects_non_int_seed(tmp_path):
    cfg = _probe_cfg()
    cfg["seed"] = "342"
    with pytest.raises(ValueError, match="seed must be an int"):
        ac.load_synthetic_probe_config(_write_probe(tmp_path, "probe-01", cfg))
