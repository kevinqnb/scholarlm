"""Rung-1 unit tests for load_calibration_config (analysis/analysis_config.py) and
calibration_ids.resolve_calibration_inputs. Hand-built three-dataset fixture under
tmp_path, with every companion-run cross-check exercised in both directions.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import utils as paths  # noqa: E402
from analysis import analysis_config as ac  # noqa: E402
from analysis import calibration_ids as cids  # noqa: E402

DATASETS = ("pond", "nfix", "supermat")


def _write_exp(root, ds, typ, run_id, params):
    p = root / ds / typ / run_id / f"{run_id}.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"id": run_id, "project": "scholarlm", "description": "t", "seed": 342, "params": params}
    p.write_text(yaml.safe_dump(cfg))


def _ids(ds):
    return {
        "ext": f"2026-09-21-{ds}-extraction-01",
        "interp": f"2026-09-27-{ds}-extraction-qwen7b-judge-interp-01",
        "combine": f"2026-09-27-{ds}-extraction-judge-combine-01",
        "train": f"2026-09-30-{ds}-v3-qwen-train-01",
        "primary": f"2026-09-30-{ds}-v3-qwen-test-primary-01",
        "diag": f"2026-09-30-{ds}-v3-qwen-test-diag-01",
        "probe_cfg": f"2026-10-02-{ds}-synthetic-probe-qwen-v3-01",
    }


@pytest.fixture
def world(tmp_path, monkeypatch):
    exp_root, res_root = tmp_path / "experiment-configs", tmp_path / "results"
    ac_root, probe_res = tmp_path / "analysis-configs", tmp_path / "synthetic_probe"
    ac_root.mkdir()
    monkeypatch.setattr(paths, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(paths, "RESULTS_ROOT", res_root)
    monkeypatch.setattr(cids, "ANALYSIS_CONFIGS_ROOT", ac_root)
    monkeypatch.setattr(cids, "_SYNTHETIC_PROBE_RESULTS_ROOT", probe_res)

    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    datasets = {}
    for ds in DATASETS:
        i = _ids(ds)
        _write_exp(exp_root, ds, "extraction", i["ext"], {})
        _write_exp(exp_root, ds, "judge_interp", i["interp"], {"judge": "qwen-2.5-7b", "extraction_id": i["ext"]})
        _write_exp(exp_root, ds, "judge_combine", i["combine"], {"judge_ids": ["x", i["interp"]]})
        _write_exp(exp_root, ds, "judge_interp", i["train"],
                   {"judge": "qwen-2.5-7b", "synthetic_file": f"data/{ds}/probe_dataset_v3.json"})
        for split, f in (("primary", "probe_dataset_test_v3.json"), ("diag", "probe_dataset_test_v3_diag.json")):
            _write_exp(exp_root, ds, "judge_interp", i[split], {"judge": "qwen-2.5-7b", "synthetic_file": f"data/{ds}/{f}"})
        (ac_root / f"{i['probe_cfg']}.yaml").write_text(yaml.safe_dump({
            "id": i["probe_cfg"], "project": "scholarlm", "description": "t", "seed": 1,
            "params": {"dataset": ds, "judge_interp_id": i["train"]}}))
        rd = probe_res / i["probe_cfg"]
        rd.mkdir(parents=True)
        probe_dir = rd / "trained_probe"
        (rd / "results.json").write_text(json.dumps({"probe_path": str(probe_dir / "head_probe.pkl")}))
        datasets[ds] = {
            "extraction_id": i["ext"], "judge_interp_id": i["interp"], "judge_combine_id": i["combine"],
            "ground_truth_file": str(gt), "synthetic_probe_config": i["probe_cfg"],
            "syn_test_ids": {"primary": i["primary"], "diag": i["diag"]},
            "use_matching_labels": True,
        }
    cfg = {"id": "2026-10-03-cal-test-01", "project": "scholarlm", "description": "t", "seed": 0,
           "params": {"probe_type": "head", "probe_variant": "platt", "syn_split": "primary",
                      "pi_te_estimate": 0.5, "datasets": datasets}}
    return cfg, exp_root


def _load(cfg, tmp_path):
    p = tmp_path / f"{cfg['id']}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return ac.load_calibration_config(p)


def test_happy_path(world, tmp_path):
    cfg, _ = world
    out = cids.resolve_calibration_inputs(_load(cfg, tmp_path))
    assert out["judge_model"] == "qwen-2.5-7b"
    assert out["datasets"]["nfix"]["syn_test_id"] == _ids("nfix")["primary"]
    assert out["datasets"]["pond"]["probe_dir"] == (
        tmp_path / "synthetic_probe" / _ids("pond")["probe_cfg"] / "trained_probe")
    assert out["datasets"]["pond"]["extraction_dir"].parts[-2] == "extraction"


def test_diag_split_selects_diag(world, tmp_path):
    cfg, _ = world
    cfg["params"]["syn_split"] = "diag"
    out = cids.resolve_calibration_inputs(_load(cfg, tmp_path))
    assert out["datasets"]["supermat"]["syn_test_id"] == _ids("supermat")["diag"]


def test_null_pi_te_is_explicit_off(world, tmp_path):
    cfg, _ = world
    cfg["params"]["pi_te_estimate"] = None
    assert _load(cfg, tmp_path)["params"]["pi_te_estimate"] is None


@pytest.mark.parametrize("mutate", [
    lambda c: c["params"].pop("pi_te_estimate"),
    lambda c: c["params"].pop("syn_split"),
    lambda c: c["params"].update(probe_type="both"),
    lambda c: c["params"].update(pi_te_estimate=1.5),
    lambda c: c["params"]["datasets"].pop("supermat"),
    lambda c: c["params"]["datasets"]["pond"].pop("judge_combine_id"),
    lambda c: c["params"]["datasets"]["pond"].update(extra="x"),
    lambda c: c["params"]["datasets"]["pond"].pop("use_matching_labels"),
    lambda c: c["params"]["datasets"]["pond"].update(use_matching_labels="yes"),
    lambda c: c["params"]["datasets"]["pond"]["syn_test_ids"].pop("diag"),
    lambda c: c["params"]["datasets"]["pond"].update(ground_truth_file="/nonexistent.json"),
    lambda c: c.update(seed="0"),
])
def test_loader_rejects_malformed(world, tmp_path, mutate):
    cfg, _ = world
    mutate(cfg)
    with pytest.raises(ValueError):
        _load(cfg, tmp_path)


def test_loader_rejects_id_filename_mismatch(world, tmp_path):
    cfg, _ = world
    p = tmp_path / "other-name.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        ac.load_calibration_config(p)


def test_extraction_under_wrong_dataset(world, tmp_path):
    cfg, _ = world
    cfg["params"]["datasets"]["pond"]["extraction_id"] = _ids("nfix")["ext"]
    with pytest.raises(ValueError, match="listed under"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_interp_run_for_different_extraction(world, tmp_path):
    cfg, exp_root = world
    i = _ids("pond")
    _write_exp(exp_root, "pond", "judge_interp", i["interp"], {"judge": "qwen-2.5-7b", "extraction_id": "2026-01-01-other-01"})
    with pytest.raises(ValueError, match="extraction_id"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_combine_missing_interp_id(world, tmp_path):
    cfg, exp_root = world
    _write_exp(exp_root, "nfix", "judge_combine", _ids("nfix")["combine"], {"judge_ids": ["x"]})
    with pytest.raises(ValueError, match="judge_ids"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_judge_model_disagreement(world, tmp_path):
    cfg, exp_root = world
    i = _ids("supermat")
    _write_exp(exp_root, "supermat", "judge_interp", i["interp"], {"judge": "llama-3.1-8b", "extraction_id": i["ext"]})
    with pytest.raises(ValueError, match="judge model differs"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_synthetic_corpus_version_mismatch(world, tmp_path):
    cfg, exp_root = world
    i = _ids("pond")
    _write_exp(exp_root, "pond", "judge_interp", i["primary"],
               {"judge": "qwen-2.5-7b", "synthetic_file": "data/pond/probe_dataset_test_v2.json"})
    with pytest.raises(ValueError, match="synthetic corpus"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_probe_config_dataset_mismatch(world, tmp_path):
    cfg, _ = world
    cfg["params"]["datasets"]["pond"]["synthetic_probe_config"] = _ids("nfix")["probe_cfg"]
    with pytest.raises(ValueError, match="params.dataset"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_missing_probe_results_json(world, tmp_path):
    cfg, _ = world
    (cids._SYNTHETIC_PROBE_RESULTS_ROOT / _ids("pond")["probe_cfg"] / "results.json").unlink()
    with pytest.raises(FileNotFoundError, match="results.json"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_results_json_probe_path_elsewhere(world, tmp_path):
    cfg, _ = world
    (cids._SYNTHETIC_PROBE_RESULTS_ROOT / _ids("pond")["probe_cfg"] / "results.json").write_text(
        json.dumps({"probe_path": "/somewhere/else/head_probe.pkl"}))
    with pytest.raises(ValueError, match="not inside"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


# ---------------------------------------------------------------------------
# edges_to_judged_rows
# ---------------------------------------------------------------------------


def _frames():
    # 3 judged rows; the extraction list-expanded row 1 into two rows (mid 1, 1).
    judged = pd.DataFrame({"measurement_id": [0, 1, 2], "document_id": ["a", "a", "b"],
                           "attribute": ["x", "y", "x"]})
    ext = pd.DataFrame({"measurement_id": [0, 1, 1, 2], "document_id": ["a", "a", "a", "b"],
                        "attribute": ["x", "y", "y", "x"]})
    return judged, ext


def test_edges_map_through_measurement_id_and_collapse():
    judged, ext = _frames()
    # gt 7 matches both expanded copies of judged row 1; gt 8 matches ext row 3 -> judged row 2.
    out = cids.edges_to_judged_rows([(7, 1), (7, 2), (8, 3)], ext, judged)
    assert out == [(7, 1), (8, 2)]


def test_edges_identity_when_no_expansion():
    judged, _ = _frames()
    assert cids.edges_to_judged_rows([(0, 2), (1, 0)], judged.copy(), judged) == [(0, 2), (1, 0)]


def test_edges_reject_nonrange_judged_ids():
    judged, ext = _frames()
    judged["measurement_id"] = [0, 2, 1]
    with pytest.raises(ValueError, match="range"):
        cids.edges_to_judged_rows([], ext, judged)


def test_edges_reject_ext_mid_outside_judged():
    judged, ext = _frames()
    ext.loc[3, "measurement_id"] = 9
    with pytest.raises(ValueError, match="outside"):
        cids.edges_to_judged_rows([], ext, judged)


def test_edges_reject_attribute_disagreement():
    judged, ext = _frames()
    ext.loc[3, "attribute"] = "z"
    with pytest.raises(ValueError, match="attribute"):
        cids.edges_to_judged_rows([], ext, judged)


def test_edges_reject_out_of_range_edge():
    judged, ext = _frames()
    with pytest.raises(ValueError, match="out of range"):
        cids.edges_to_judged_rows([(0, 4)], ext, judged)


# ── load_calibration_v2_config: per-dataset pi_te_estimate ───────────────────
def _to_v2(cfg, pi=0.5):
    cfg["params"].pop("pi_te_estimate")
    for block in cfg["params"]["datasets"].values():
        block["pi_te_estimate"] = pi
    return cfg


def _load_v2(cfg, tmp_path):
    p = tmp_path / f"{cfg['id']}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return ac.load_calibration_v2_config(p)


def test_v2_happy_path_per_dataset_pi(world, tmp_path):
    cfg, _ = world
    _to_v2(cfg)
    for pi, ds in zip((0.2, 0.5, 0.9), ("pond", "nfix", "supermat")):
        cfg["params"]["datasets"][ds]["pi_te_estimate"] = pi
    out = _load_v2(cfg, tmp_path)
    assert [out["params"]["datasets"][d]["pi_te_estimate"] for d in ("pond", "nfix", "supermat")] == [0.2, 0.5, 0.9]
    cids.resolve_calibration_inputs(out)  # extra per-dataset key must not trip id resolution


@pytest.mark.parametrize("mutate", [
    lambda c: c["params"].update(pi_te_estimate=0.5),                       # v1 global key not allowed
    lambda c: c["params"]["datasets"]["pond"].pop("pi_te_estimate"),        # missing per-dataset
    lambda c: c["params"]["datasets"]["nfix"].update(pi_te_estimate=None),  # null/off not allowed
    lambda c: c["params"]["datasets"]["nfix"].update(pi_te_estimate=1.0),
    lambda c: c["params"]["datasets"]["supermat"].update(pi_te_estimate=0),
    lambda c: c["params"]["datasets"]["pond"].update(pi_te_estimate=True),
    lambda c: c["params"]["datasets"]["pond"].update(pi_te_estimate="0.5"),
])
def test_v2_loader_rejects_malformed(world, tmp_path, mutate):
    cfg, _ = world
    _to_v2(cfg)
    mutate(cfg)
    with pytest.raises(ValueError):
        _load_v2(cfg, tmp_path)


def test_v1_loader_rejects_v2_config(world, tmp_path):
    cfg, _ = world
    _to_v2(cfg)
    with pytest.raises(ValueError):
        _load(cfg, tmp_path)


# ── load_calibration_v3_config: platt_n, no pi_te ────────────────────────────
def _to_v3(cfg, n=100):
    cfg["params"].pop("pi_te_estimate")
    cfg["params"]["platt_n"] = n
    return cfg


def _load_v3(cfg, tmp_path):
    p = tmp_path / f"{cfg['id']}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return ac.load_calibration_v3_config(p)


def test_v3_happy_path(world, tmp_path):
    cfg, _ = world
    _to_v3(cfg, 7)
    out = _load_v3(cfg, tmp_path)
    assert out["params"]["platt_n"] == 7
    cids.resolve_calibration_inputs(out)


@pytest.mark.parametrize("mutate", [
    lambda c: c["params"].pop("platt_n"),
    lambda c: c["params"].update(platt_n=0),
    lambda c: c["params"].update(platt_n=-5),
    lambda c: c["params"].update(platt_n=True),
    lambda c: c["params"].update(platt_n=10.0),
    lambda c: c["params"].update(pi_te_estimate=0.5),                      # v1 key not allowed
    lambda c: c["params"]["datasets"]["pond"].update(pi_te_estimate=0.5),  # v2 key not allowed
])
def test_v3_loader_rejects_malformed(world, tmp_path, mutate):
    cfg, _ = world
    _to_v3(cfg)
    mutate(cfg)
    with pytest.raises(ValueError):
        _load_v3(cfg, tmp_path)
