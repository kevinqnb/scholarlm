"""Unit tests for calibration_baseline: config loading, run resolution, row checks and metrics."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
for _p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "experiments"):
    sys.path.insert(0, str(_p))

import utils as paths  # noqa: E402
from analysis import calibration_baseline as cb  # noqa: E402
from analysis.common import calibration_ids as cids  # noqa: E402
from analysis.common import config as ac  # noqa: E402

CFG_ID = "2026-01-01-calibration-baseline-test-01"
CAL_ID = "2026-01-01-calibration-test-01"


def _run_id(ds, method):
    return f"2026-01-01-{ds}-{method}-01"


@pytest.fixture
def cfg():
    return {
        "id": CFG_ID, "project": "scholarlm", "description": "test", "seed": 0,
        "params": {
            "calibration_config_id": CAL_ID, "n_boot": 10,
            "datasets": {ds: {m: _run_id(ds, m) for m in ac.CALIBRATION_BASELINE_METHODS}
                         for ds in ac.CALIBRATION_DATASETS},
        },
    }


def _load(cfg, tmp_path, analysis_type="calibration-baseline"):
    path = tmp_path / analysis_type / f"{cfg['id']}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))
    return ac.load_calibration_baseline_config(path)


# ── Config loader ────────────────────────────────────────────────────────────


def test_load_happy_path(cfg, tmp_path):
    assert _load(copy.deepcopy(cfg), tmp_path) == cfg


def test_load_rejects_wrong_directory(cfg, tmp_path):
    with pytest.raises(ValueError, match="must live in"):
        _load(cfg, tmp_path, analysis_type="calibration")


@pytest.mark.parametrize("mutate", [
    lambda c: c["params"].pop("n_boot"),
    lambda c: c["params"].update(recalibration="platt_fit"),
    lambda c: c["params"].update(n_boot=0),
    lambda c: c["params"].update(n_boot=True),
    lambda c: c["params"].update(calibration_config_id=""),
    lambda c: c["params"]["datasets"].pop("nfix"),
    lambda c: c["params"]["datasets"]["pond"].pop("ntp"),
    lambda c: c["params"]["datasets"]["pond"].update(laplace="x"),
    lambda c: c["params"]["datasets"]["pond"].update(ntp=None),
    lambda c: c["params"]["datasets"]["pond"].update(ntp=_run_id("pond", "sampling")),  # listed twice
    lambda c: c.update(seed="0"),
])
def test_load_rejects(cfg, tmp_path, mutate):
    mutate(cfg)
    with pytest.raises(ValueError):
        _load(cfg, tmp_path)


# ── Run resolution ───────────────────────────────────────────────────────────


@pytest.fixture
def world(cfg, tmp_path, monkeypatch):
    """Nine baseline runs on disk, with the calibration config's own resolution stubbed out."""
    monkeypatch.setattr(paths, "EXPERIMENT_CONFIGS_ROOT", tmp_path / "configs")
    monkeypatch.setattr(paths, "RESULTS_ROOT", tmp_path / "results")
    cal_path = tmp_path / "cal.yaml"
    cal_path.write_text("")
    cal_cfg = {"seed": 0, "params": {"datasets": {ds: {"extraction_id": f"ext-{ds}"} for ds in ac.CALIBRATION_DATASETS}}}
    monkeypatch.setattr(cids, "analysis_config_path", lambda *_: cal_path)
    monkeypatch.setattr(cids, "load_calibration_v4_config", lambda _: cal_cfg)
    monkeypatch.setattr(cids, "resolve_calibration_inputs", lambda _: {"judge_model": "j"})
    monkeypatch.setattr(cids, "analysis_results_dir", lambda _: tmp_path / "cal-results")
    (tmp_path / "cal-results" / CAL_ID).mkdir(parents=True)
    (tmp_path / "cal-results" / CAL_ID / "predictions.pkl").write_bytes(b"")

    for ds in ac.CALIBRATION_DATASETS:
        for m in ac.CALIBRATION_BASELINE_METHODS:
            params = {"dataset": ds, "baseline": m, "extraction_id": f"ext-{ds}", "judge": f"judge-{m}", "ocr_dir": "o"}
            _write_run(tmp_path, ds, m, params, {})
    return tmp_path


def _write_run(root, ds, method, params, meta_overrides):
    rid = _run_id(ds, method)
    cdir = root / "configs" / ds / "judge_baselines" / rid
    cdir.mkdir(parents=True, exist_ok=True)
    (cdir / f"{rid}.yaml").write_text(yaml.safe_dump(
        {"id": rid, "project": "p", "description": "d", "seed": 342, "params": params}))
    rdir = root / "results" / ds / "judge_baselines" / rid
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "responses.json").write_text("[]")
    meta = {k: params[k] for k in ("dataset", "baseline", "extraction_id", "ocr_dir")} | {
        "judge_key": params["judge"], "seed": 342, "row_indices": None, "n_failed_rows": 0} | meta_overrides
    (rdir / "run_metadata.json").write_text(json.dumps(meta))


def test_resolve_happy_path(cfg, world):
    out = cids.resolve_calibration_baseline_inputs(cfg)
    assert out["judges"] == {m: f"judge-{m}" for m in ac.CALIBRATION_BASELINE_METHODS}
    assert out["runs"]["nfix"]["ntp"].name == _run_id("nfix", "ntp")


_PARAMS = {"dataset": "pond", "baseline": "ntp", "extraction_id": "ext-pond", "judge": "judge-ntp", "ocr_dir": "o"}


@pytest.mark.parametrize("params, meta, match", [
    ({"baseline": "sampling"}, {}, "baseline"),
    ({"extraction_id": "ext-other"}, {}, "extraction_id"),
    ({"judge": "other-judge"}, {}, "judge differs"),
    ({}, {"n_failed_rows": 3}, "n_failed_rows"),
    ({}, {"row_indices": [0, 1]}, "row_indices"),
    ({}, {"judge_key": "other"}, "judge_key"),
])
def test_resolve_rejects(cfg, world, params, meta, match):
    _write_run(world, "pond", "ntp", _PARAMS | params, meta)
    with pytest.raises(ValueError, match=match):
        cids.resolve_calibration_baseline_inputs(cfg)


def test_resolve_rejects_swapped_datasets(cfg, world):
    d = cfg["params"]["datasets"]
    d["pond"]["ntp"], d["nfix"]["ntp"] = d["nfix"]["ntp"], d["pond"]["ntp"]
    with pytest.raises(ValueError, match="dataset="):
        cids.resolve_calibration_baseline_inputs(cfg)


def test_resolve_rejects_seed_mismatch(cfg, world):
    cfg["seed"] = 1
    with pytest.raises(ValueError, match="seed"):
        cids.resolve_calibration_baseline_inputs(cfg)


# ── Scores ───────────────────────────────────────────────────────────────────


def _rows():
    final = [{"measurement_id": i, "document_id": f"d{i}", "attribute": "a", "name": "n", "value": float(i),
              "units": None} for i in range(3)]
    responses = [r | {"judgement_p_true": p, "judgement_failed": False} for r, p in zip(final, (0.1, 0.5, 1.0))]
    return responses, final


def test_scores_follow_requested_id_order():
    responses, final = _rows()
    assert cb.baseline_scores(responses, final, [2, 0]).tolist() == [1.0, 0.1]


def test_scores_nan_fields_match_nan():
    responses, final = _rows()
    responses[0]["units"] = final[0]["units"] = float("nan")
    assert cb.baseline_scores(responses, final, [0]).tolist() == [0.1]


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop(),
    lambda r: r[0].update(value=9.0),
    lambda r: r[1].update(document_id="other"),
    lambda r: r[1].update(judgement_failed=True),
    lambda r: r[1].update(judgement_p_true=None),
    lambda r: r[1].update(judgement_p_true=1.2),
    lambda r: r.reverse(),
])
def test_scores_reject(mutate):
    responses, final = _rows()
    mutate(responses)
    with pytest.raises(ValueError):
        cb.baseline_scores(responses, final, [0])


def test_scores_unknown_id_raises():
    responses, final = _rows()
    with pytest.raises(KeyError):
        cb.baseline_scores(responses, final, [7])


# ── Metrics table ────────────────────────────────────────────────────────────


def test_metrics_known_answers():
    """Perfect, uninformative and inverted scores have known AUROC and BSS."""
    labels = np.array([True, False] * 10)
    docs = np.repeat(np.arange(5), 4)
    probs = {"verbalized": labels.astype(float), "sampling": np.tile([0.4, 0.4, 0.6, 0.6], 5), "ntp": 1.0 - labels}
    preds = {ds: {"measurement_ids": np.arange(20), "document_ids": docs, "labels": labels, "probs": probs}
             for ds in ac.CALIBRATION_DATASETS}
    boot = cb.bootstrap_predictions(preds, n_boot=3, seed=0)
    df = cb.compute_metrics(preds, boot, {"verbalized": "a", "sampling": "a", "ntp": "b"},
                            {ds: {"b"} for ds in preds}).set_index(["Test dataset", "Method"])
    perfect, flat, inverted = (df.loc[("pond", m)] for m in ("verbalized", "sampling", "ntp"))
    assert (perfect["AUROC"], perfect["BSS"], perfect["ECE"]) == (1.0, 1.0, 0.0)
    assert flat["AUROC"] == 0.5
    assert flat["BSS"] == pytest.approx(1 - 0.26 / 0.25)  # mean((p - y)^2) = (0.36 + 0.16) / 2
    assert (inverted["AUROC"], inverted["BSS"]) == (0.0, -3.0)
    assert not perfect["Judge votes on labels"] and inverted["Judge votes on labels"]
