"""Rung-1 unit tests for the calibration config loaders (analysis/common/config.py) and
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
from analysis.common import config as ac  # noqa: E402
from analysis.common import calibration_ids as cids  # noqa: E402
from analysis.common import matching  # noqa: E402

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
    ac_root, probe_res = tmp_path / "analysis-configs", tmp_path / "synthetic-probe"
    (ac_root / "synthetic-probe").mkdir(parents=True)
    monkeypatch.setattr(paths, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(paths, "RESULTS_ROOT", res_root)
    monkeypatch.setattr(ac, "ANALYSIS_CONFIGS_ROOT", ac_root)
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
        (ac_root / "synthetic-probe" / f"{i['probe_cfg']}.yaml").write_text(yaml.safe_dump({
            "id": i["probe_cfg"], "project": "scholarlm", "description": "t", "seed": 1,
            "params": {"dataset": ds, "judge_interp_id": i["train"],
                       "use_platt_scaling": True}}))
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
    """The world config, through the v3-schema loader (the v1 loader it was written
    against is retired) -- the cross-run tests below exercise resolve_calibration_inputs,
    which is loader-independent."""
    return _load_v3(_to_v3(cfg), tmp_path)


def test_happy_path(world, tmp_path):
    cfg, _ = world
    out = cids.resolve_calibration_inputs(_load(cfg, tmp_path))
    assert out["judge_model"] == "qwen-2.5-7b"
    assert out["datasets"]["nfix"]["syn_test_id"] == _ids("nfix")["primary"]
    assert out["datasets"]["pond"]["probe_dir"] == (
        tmp_path / "synthetic-probe" / _ids("pond")["probe_cfg"] / "trained_probe")
    assert out["datasets"]["pond"]["extraction_dir"].parts[-2] == "extraction"


def test_diag_split_selects_diag(world, tmp_path):
    cfg, _ = world
    cfg["params"]["syn_split"] = "diag"
    out = cids.resolve_calibration_inputs(_load(cfg, tmp_path))
    assert out["datasets"]["supermat"]["syn_test_id"] == _ids("supermat")["diag"]


@pytest.mark.parametrize("mutate", [
    lambda c: c["params"].pop("syn_split"),
    lambda c: c["params"].update(probe_type="both"),
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
    p = tmp_path / "calibration" / "other-name.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump(_to_v3(cfg)))
    with pytest.raises(ValueError):
        ac.load_calibration_v3_config(p)


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


@pytest.mark.parametrize("name, version", [
    ("probe_dataset_v3.json", "v3"),
    ("probe_dataset_test_v3_diag.json", "v3"),
    ("probe_dataset_v3s.json", "v3s"),
    ("probe_dataset_test_v3s.json", "v3s"),
    ("probe_dataset_test_v3s_diag.json", "v3s"),
])
def test_synthetic_file_version(name, version):
    assert cids._synthetic_file_version("r", {"params": {"synthetic_file": f"data/supermat/{name}"}}) == version


def test_synthetic_file_without_version_raises():
    with pytest.raises(ValueError, match="no _v<N>"):
        cids._synthetic_file_version("r", {"params": {"synthetic_file": "data/supermat/probe_dataset.json"}})


def _set_supermat_corpus(exp_root, train_file, test_file, diag_file):
    i = _ids("supermat")
    for run, f in ((i["train"], train_file), (i["primary"], test_file), (i["diag"], diag_file)):
        _write_exp(exp_root, "supermat", "judge_interp", run,
                   {"judge": "qwen-2.5-7b", "synthetic_file": f"data/supermat/{f}"})


def test_v3s_corpus_resolves(world, tmp_path):
    cfg, exp_root = world
    _set_supermat_corpus(exp_root, "probe_dataset_v3s.json", "probe_dataset_test_v3s.json",
                         "probe_dataset_test_v3s_diag.json")
    cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_v3s_train_with_v3_test_rejected(world, tmp_path):
    cfg, exp_root = world
    _set_supermat_corpus(exp_root, "probe_dataset_v3s.json", "probe_dataset_test_v3.json",
                         "probe_dataset_test_v3s_diag.json")
    with pytest.raises(ValueError, match="synthetic corpus 'v3' != .*'v3s'"):
        cids.resolve_calibration_inputs(_load(cfg, tmp_path))


def test_config_in_wrong_type_dir_rejected(world, tmp_path):
    cfg, _ = world
    p = tmp_path / "platt-scaling" / f"{cfg['id']}.yaml"
    p.parent.mkdir()
    p.write_text(yaml.safe_dump(_to_v3(cfg)))
    with pytest.raises(ValueError, match="must live in analysis-configs/calibration/"):
        ac.load_calibration_v3_config(p)


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
    out = matching.edges_to_judged_rows([(7, 1), (7, 2), (8, 3)], ext, judged)
    assert out == [(7, 1), (8, 2)]


def test_edges_identity_when_no_expansion():
    judged, _ = _frames()
    assert matching.edges_to_judged_rows([(0, 2), (1, 0)], judged.copy(), judged) == [(0, 2), (1, 0)]


def test_edges_reject_nonrange_judged_ids():
    judged, ext = _frames()
    judged["measurement_id"] = [0, 2, 1]
    with pytest.raises(ValueError, match="range"):
        matching.edges_to_judged_rows([], ext, judged)


def test_edges_reject_ext_mid_outside_judged():
    judged, ext = _frames()
    ext.loc[3, "measurement_id"] = 9
    with pytest.raises(ValueError, match="outside"):
        matching.edges_to_judged_rows([], ext, judged)


def test_edges_reject_attribute_disagreement():
    judged, ext = _frames()
    ext.loc[3, "attribute"] = "z"
    with pytest.raises(ValueError, match="attribute"):
        matching.edges_to_judged_rows([], ext, judged)


def test_edges_reject_out_of_range_edge():
    judged, ext = _frames()
    with pytest.raises(ValueError, match="out of range"):
        matching.edges_to_judged_rows([(0, 4)], ext, judged)


# ── load_calibration_v3_config: platt_n, no pi_te ────────────────────────────
BOOTSTRAP = {"n_fit_samples": 3, "n_doc_boot": 4, "n_syn_boot": 5}


def _to_v3(cfg, n=100, recalibration="platt_fit"):
    cfg["params"].pop("pi_te_estimate")
    cfg["params"]["platt_n"] = n
    cfg["params"]["recalibration"] = recalibration
    cfg["params"].update(BOOTSTRAP)
    return cfg


def _load_v3(cfg, tmp_path):
    p = tmp_path / "calibration" / f"{cfg['id']}.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return ac.load_calibration_v3_config(p)


def test_v3_happy_path(world, tmp_path):
    cfg, _ = world
    _to_v3(cfg, 7)
    out = _load_v3(cfg, tmp_path)
    assert out["params"]["platt_n"] == 7
    cids.resolve_calibration_inputs(out)


@pytest.mark.parametrize("method", ["prior_shift", "intercept_fit", "platt_fit"])
def test_v3_accepts_every_recalibration_method(world, tmp_path, method):
    cfg, _ = world
    _to_v3(cfg, recalibration=method)
    assert _load_v3(cfg, tmp_path)["params"]["recalibration"] == method


def test_recalibration_methods_match_library():
    from scholarlm.utils.calibration import RECALIBRATION_METHODS
    assert ac.RECALIBRATION_METHODS == RECALIBRATION_METHODS


@pytest.mark.parametrize("mutate", [
    lambda c: c["params"].pop("platt_n"),
    lambda c: c["params"].update(platt_n=0),
    lambda c: c["params"].update(platt_n=-5),
    lambda c: c["params"].update(platt_n=True),
    lambda c: c["params"].update(platt_n=10.0),
    lambda c: c["params"].pop("recalibration"),                            # no default method
    lambda c: c["params"].update(recalibration="platt"),
    lambda c: c["params"].update(recalibration=None),
    lambda c: c["params"].update(pi_te_estimate=0.5),                      # v1 key not allowed
    lambda c: c["params"]["datasets"]["pond"].update(pi_te_estimate=0.5),  # v2 key not allowed
    *[lambda c, k=k: c["params"].pop(k) for k in BOOTSTRAP],               # no default replicate counts
    *[lambda c, k=k: c["params"].update({k: 0}) for k in BOOTSTRAP],
    *[lambda c, k=k: c["params"].update({k: True}) for k in BOOTSTRAP],
    *[lambda c, k=k: c["params"].update({k: 2.0}) for k in BOOTSTRAP],
])
def test_v3_loader_rejects_malformed(world, tmp_path, mutate):
    cfg, _ = world
    _to_v3(cfg)
    mutate(cfg)
    with pytest.raises(ValueError):
        _load_v3(cfg, tmp_path)


# ── load_calibration_validated_config: pond+supermat only, pinned validation sha256 ──
def _to_validated(cfg, tmp_path, monkeypatch):
    """v3 shape, minus nfix, plus a validations dir whose files the config pins."""
    import hashlib
    cfg["params"].pop("pi_te_estimate")
    cfg["params"]["platt_n"] = 100
    cfg["params"]["recalibration"] = "platt_fit"
    cfg["params"].update(BOOTSTRAP)
    cfg["params"]["datasets"].pop("nfix")
    vdir = tmp_path / "validations"
    vdir.mkdir()
    for ds, block in cfg["params"]["datasets"].items():
        (vdir / f"{ds}.json").write_text(json.dumps({"dataset": ds, "n": 1}))
        block["validation_sha256"] = hashlib.sha256((vdir / f"{ds}.json").read_bytes()).hexdigest()
    monkeypatch.setenv(ac.VALIDATIONS_ENV, str(vdir))
    return cfg, vdir


def _load_validated(cfg, tmp_path):
    p = tmp_path / "calibration-validated" / f"{cfg['id']}.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return ac.load_calibration_validated_config(p)


def test_validated_happy_path(world, tmp_path, monkeypatch):
    cfg, _ = world
    _to_validated(cfg, tmp_path, monkeypatch)
    out = cids.resolve_calibration_inputs(_load_validated(cfg, tmp_path))
    assert set(out["datasets"]) == {"pond", "supermat"}


def test_validated_rejects_sha_mismatch(world, tmp_path, monkeypatch):
    cfg, vdir = _to_validated(world[0], tmp_path, monkeypatch)
    (vdir / "pond.json").write_text(json.dumps({"dataset": "pond", "n": 2}))  # rebuilt after pinning
    with pytest.raises(ValueError, match="sha256"):
        _load_validated(cfg, tmp_path)


def test_validated_env_unset_is_hard_error(world, tmp_path, monkeypatch):
    cfg, _ = world
    _to_validated(cfg, tmp_path, monkeypatch)
    monkeypatch.delenv(ac.VALIDATIONS_ENV)
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: None)
    with pytest.raises(KeyError, match=ac.VALIDATIONS_ENV):
        _load_validated(cfg, tmp_path)


def test_validated_missing_file_is_hard_error(world, tmp_path, monkeypatch):
    cfg, vdir = _to_validated(world[0], tmp_path, monkeypatch)
    (vdir / "supermat.json").unlink()
    with pytest.raises(FileNotFoundError):
        _load_validated(cfg, tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda c: c["params"]["datasets"]["pond"].pop("validation_sha256"),
    lambda c: c["params"]["datasets"]["pond"].update(validation_sha256="abc"),
    lambda c: c["params"]["datasets"].pop("supermat"),
    lambda c: c["params"]["datasets"].update(nfix=dict(c["params"]["datasets"]["pond"])),  # no validations for nfix
    lambda c: c["params"].pop("platt_n"),
    lambda c: c["params"].pop("recalibration"),
    lambda c: c["params"].update(recalibration="intercept"),
    lambda c: c["params"].pop("n_fit_samples"),
    lambda c: c["params"].update(n_doc_boot=0),
])
def test_validated_loader_rejects_malformed(world, tmp_path, monkeypatch, mutate):
    cfg, _ = _to_validated(world[0], tmp_path, monkeypatch)
    mutate(cfg)
    with pytest.raises(ValueError):
        _load_validated(cfg, tmp_path)
