"""Rung-1 unit tests for analysis/platt_scaling.py helpers and load_platt_sweep_v2_config."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import relplot
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

from analysis.common import config as ac  # noqa: E402
from analysis import platt_scaling as ps2  # noqa: E402

CFG_ID = "2026-01-01-platt-scaling-test-01"


def _base_cfg(gt: Path) -> dict:
    block = {"extraction_id": "e", "judge_interp_id": "j", "judge_combine_id": "c", "ground_truth_file": str(gt),
             "synthetic_probe_config": "p", "use_matching_labels": True,
             "syn_test_ids": {"primary": "tp", "diag": "td"}}
    return {"id": CFG_ID, "project": "scholarlm", "description": "t", "seed": 0,
            "params": {"probe_type": "head", "probe_variant": "platt", "syn_split": "primary",
                       "platt_ns": [0, 100, 500], "recalibration": "intercept_fit", "n_train_resamples": 2000,
                       "datasets": {ds: dict(block) for ds in ac.CALIBRATION_DATASETS}}}


def _write(tmp_path: Path, cfg: dict, analysis_type: str = "platt-scaling") -> Path:
    p = tmp_path / analysis_type / f"{cfg['id']}.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return p


@pytest.fixture
def base_cfg(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    return _base_cfg(gt)


def test_summarize_hand_built():
    # 0.00, 0.01, ..., 1.00: mean 0.5; numpy-linear quantiles at 2.5% / 97.5% are 0.025 / 0.975.
    s = ps2.summarize(np.linspace(0, 1, 101))
    assert s["SmECE"] == pytest.approx(0.5)
    assert s["SmECE_lo"] == pytest.approx(0.025) and s["SmECE_hi"] == pytest.approx(0.975)
    # Constant values: degenerate interval at the value.
    assert ps2.summarize([0.2] * 7) == pytest.approx({"SmECE": 0.2, "SmECE_lo": 0.2, "SmECE_hi": 0.2})


@pytest.mark.parametrize("bad", [[], [0.1, np.nan], [[0.1, 0.2]]])
def test_summarize_fails_loud(bad):
    with pytest.raises(AssertionError):
        ps2.summarize(bad)


def test_smece_is_doc_bootstrap_point():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 300)
    y = rng.random(300) < p
    resamples = ps2.db.document_resamples(np.repeat(np.arange(10), 30), 2, rng)
    assert ps2.smece(p, y) == ps2.db.doc_bootstrap_calibration(p, y, resamples)["point"]["SmECE"]
    assert ps2.smece(p, y) == relplot.smECE(p, y.astype(float), eps=0.001)


def test_smece_known_answers():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 20000)
    assert ps2.smece(p, rng.random(20000) < p) < 0.02                  # perfectly calibrated
    assert ps2.smece(np.full(2000, 0.9), np.zeros(2000, bool)) == pytest.approx(0.9, abs=0.01)


def test_fit_map_known_answers():
    from scipy.special import logit
    from scholarlm.utils.calibration import apply_platt, fit_platt
    rng = np.random.default_rng(3)
    p = rng.uniform(0.05, 0.95, 400)
    y = rng.random(400) < p
    # prior_shift: slope 1, intercept logit(label rate) - logit(pi_tr).
    assert ps2.fit_map("prior_shift", p, y, 0.3) == pytest.approx((1.0, logit(y.mean()) - logit(0.3)))
    # intercept_fit: slope 1, mapped rows average to their label rate.
    coef, icpt = ps2.fit_map("intercept_fit", p, y, 0.3)
    assert coef == 1.0 and apply_platt(p, coef, icpt).mean() == pytest.approx(y.mean(), abs=1e-8)
    # platt_fit: scholarlm's fit_platt; pi_tr unused.
    assert ps2.fit_map("platt_fit", p, y, 0.3) == fit_platt(p, y)
    with pytest.raises(ValueError):
        ps2.fit_map("isotonic", p, y, 0.3)


def test_config_loads(tmp_path, base_cfg):
    p = ac.load_platt_sweep_v2_config(_write(tmp_path, base_cfg))["params"]
    assert p["platt_ns"] == [0, 100, 500] and p["recalibration"] == "intercept_fit"
    assert p["n_train_resamples"] == 2000


def test_config_under_calibration_dir_rejected(tmp_path, base_cfg):
    # platt-scaling is its own analysis type, no longer filed under calibration/
    with pytest.raises(ValueError, match="must live in analysis-configs/platt-scaling/"):
        ac.load_platt_sweep_v2_config(_write(tmp_path, base_cfg, "calibration"))


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("platt_ns"),
    lambda p: p.pop("recalibration"),
    lambda p: p.pop("n_train_resamples"),
    lambda p: p.update(platt_ns=[]),
    lambda p: p.update(platt_ns=[10, 10, 50]),
    lambda p: p.update(platt_ns=[50, 10]),
    lambda p: p.update(platt_ns=[-1, 10]),
    lambda p: p.update(platt_ns=[0]),
    lambda p: p.update(platt_ns=[0, 0, 50]),
    lambda p: p.update(n_train_resamples=0),
    lambda p: p.update(n_train_resamples=True),
    lambda p: p.update(n_train_resamples=2000.0),
    lambda p: p.update(recalibration="isotonic"),
    lambda p: p.update(n_fit_samples=100),   # nested-bootstrap keys not allowed
    lambda p: p.update(n_doc_boot=100),
    lambda p: p.update(platt_n=100),
])
def test_loader_rejects_malformed(tmp_path, base_cfg, mutate):
    cfg = copy.deepcopy(base_cfg)
    mutate(cfg["params"])
    with pytest.raises(ValueError):
        ac.load_platt_sweep_v2_config(_write(tmp_path, cfg))


class _FakeInputs:
    """Hand-built SweepInputs stand-in: one dataset, 6 pool docs, 4 test rows."""

    def __init__(self):
        import pandas as pd
        self.datasets, self.seed = ["pond"], 0
        n_pool = 24
        docs = np.r_[np.repeat(np.arange(6), 4), [10, 10, 11, 11]]
        self.labels = np.r_[np.tile([True, False], 12), [True, False, True, True]]
        self.data = {"pond": {
            "real_df": pd.DataFrame({"document_id": docs}),
            "labels": self.labels, "pool_docs": set(range(6)),
            "pool_idx": np.arange(n_pool), "test_idx": np.arange(n_pool, n_pool + 4)}}
        self.probe = {"pond": {"train_prevalence": 0.5, "syn_document_ids": list(range(6))}}
        rng = np.random.default_rng(0)
        self.raw = {"probe": rng.uniform(0.1, 0.9, len(docs)), "ntp": rng.uniform(0.1, 0.9, len(docs))}

    def score_rows(self, train_ds, test_ds, idx):
        return {k: v[idx] for k, v in self.raw.items()}


def test_run_sweep_baseline_is_raw_scores():
    inp = _FakeInputs()
    summary, samples_df = ps2.run_sweep(inp, [0, 8], "intercept_fit", 5)
    base = summary[summary["Platt N"] == 0]
    assert len(base) == 2 and (base["Recalibration"] == "none").all() and (base["Train resamples"] == 0).all()
    test = inp.data["pond"]["test_idx"]
    for _, row in base.iterrows():
        key = row["Type"].lower()
        v = ps2.smece(inp.raw[key][test], inp.labels[test])
        assert row["SmECE"] == row["SmECE_lo"] == row["SmECE_hi"] == v
        assert np.isnan(row["Fit label rate"])
    # n > 0 rows unaffected: 5 resamples each, and the baseline has no per-resample rows.
    assert (summary[summary["Platt N"] == 8]["Train resamples"] == 5).all()
    assert len(samples_df) == 2 * 5 and (samples_df["Platt N"] == 8).all()


class _ConstProbe:
    """predict_proba stand-in: constant 0.7 for every row."""

    def predict_proba(self, X):
        return np.tile([0.3, 0.7], (len(X), 1))


class _FakeHeadActs:
    def features(self, act_dir, mids, top):
        return np.zeros((len(mids), 2), dtype=np.float32)


def test_score_rows_ntp_is_raw_judge_p_true():
    # Known answer: with no trained NTP model, the NTP score is the judge's p(true) itself,
    # including the boundary values 0 and 1.
    import pandas as pd
    inp = object.__new__(ps2.SweepInputs)
    inp.judge_model, inp.probe_type = "j", "head"
    p_true = np.array([0.0, 0.25, 0.5, 0.9, 1.0])
    inp.data = {"pond": {"real_df": pd.DataFrame({"measurement_id": list("abcde"), "judgement_p_true_j": p_true})}}
    inp.inputs = {"datasets": {"pond": {"judge_interp_dir": Path("/nonexistent")}}}
    inp.probe = {"pond": {"probe": _ConstProbe(), "top_k_heads": [(0, 0)]}}
    inp.head_acts = _FakeHeadActs()
    idx = np.array([4, 0, 2])
    out = inp.score_rows("pond", "pond", idx)
    assert np.array_equal(out["ntp"], p_true[idx])
    assert np.array_equal(out["probe"], np.full(3, 0.7))


def test_score_rows_rejects_ntp_outside_unit_interval():
    import pandas as pd
    inp = object.__new__(ps2.SweepInputs)
    inp.judge_model, inp.probe_type = "j", "head"
    inp.data = {"pond": {"real_df": pd.DataFrame({"measurement_id": ["a"], "judgement_p_true_j": [1.2]})}}
    inp.inputs = {"datasets": {"pond": {"judge_interp_dir": Path("/nonexistent")}}}
    inp.probe = {"pond": {"probe": _ConstProbe(), "top_k_heads": [(0, 0)]}}
    inp.head_acts = _FakeHeadActs()
    with pytest.raises(AssertionError):
        inp.score_rows("pond", "pond", np.array([0]))


def test_fit_map_prior_shift_without_training_prevalence_raises():
    with pytest.raises(ValueError, match="training prevalence"):
        ps2.fit_map("prior_shift", np.array([0.2, 0.8]), np.array([False, True]), None)
    # The probe (which has a training prevalence) is unaffected.
    assert ps2.fit_map("prior_shift", np.array([0.2, 0.8]), np.array([False, True]), 0.5) == (1.0, 0.0)
