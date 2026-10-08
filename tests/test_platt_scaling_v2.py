"""Rung-1 unit tests for analysis/platt_scaling_v2.py helpers and load_platt_sweep_v2_config."""
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

from analysis import analysis_config as ac  # noqa: E402
from analysis import platt_scaling_v2 as ps2  # noqa: E402

_CFG = _REPO / "analysis/analysis-configs/2026-10-08-platt-scaling-v2-intercept-fit-tiny-01.yaml"


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


def test_committed_config_loads():
    cfg = ac.load_platt_sweep_v2_config(_CFG)
    p = cfg["params"]
    assert p["platt_ns"] == [10, 50, 100, 250, 500, 1000] and p["recalibration"] == "intercept_fit"
    assert p["n_train_resamples"] == 100


@pytest.mark.parametrize("slug", ["intercept-fit", "platt-fit", "prior-shift"])
def test_full_configs_load(slug):
    cfg = ac.load_platt_sweep_v2_config(
        _REPO / f"analysis/analysis-configs/2026-10-08-platt-scaling-v2-gemma27b-qwen-2.5-7b-{slug}-01.yaml")
    assert cfg["params"]["n_train_resamples"] == 2000
    assert cfg["params"]["recalibration"] == slug.replace("-", "_")


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("platt_ns"),
    lambda p: p.pop("recalibration"),
    lambda p: p.pop("n_train_resamples"),
    lambda p: p.update(platt_ns=[]),
    lambda p: p.update(platt_ns=[10, 10, 50]),
    lambda p: p.update(platt_ns=[50, 10]),
    lambda p: p.update(platt_ns=[0, 10]),
    lambda p: p.update(n_train_resamples=0),
    lambda p: p.update(n_train_resamples=True),
    lambda p: p.update(n_train_resamples=2000.0),
    lambda p: p.update(recalibration="isotonic"),
    lambda p: p.update(n_fit_samples=100),   # nested-bootstrap keys not allowed
    lambda p: p.update(n_doc_boot=100),
    lambda p: p.update(platt_n=100),
])
def test_loader_rejects_malformed(tmp_path, mutate):
    cfg = copy.deepcopy(yaml.safe_load(_CFG.read_text()))
    mutate(cfg["params"])
    p = tmp_path / f"{cfg['id']}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        ac.load_platt_sweep_v2_config(p)
