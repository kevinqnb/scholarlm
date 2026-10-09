"""Rung-1 tests for calibration v4: load_calibration_v4_config's recalibration /
fit_source / fit_n / fit_seed / pi_te_estimate consistency rules, and
analysis/common/recalibration.py on hand-built fixtures whose answers can be checked by
inspection."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis.common import config as ac  # noqa: E402
from analysis.common.recalibration import (  # noqa: E402
    intercept_fit_map, platt_fit_map, prior_shift_map, uniform_fit_sample,
)
from scholarlm.utils.calibration import apply_platt, fit_platt, fit_prior_shift  # noqa: E402

CFG_ID = "2026-01-01-calibration-v4-test-01"


@pytest.fixture
def base_cfg(tmp_path):
    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    block = {
        "extraction_id": "e", "judge_interp_id": "j", "judge_combine_id": "c",
        "ground_truth_file": str(gt), "synthetic_probe_config": "p",
        "use_matching_labels": True, "pi_te_estimate": None,
        "syn_test_ids": {"primary": "tp", "diag": "td"},
    }
    return {
        "id": CFG_ID, "project": "scholarlm", "description": "test", "seed": 0,
        "params": {
            "probe_type": "head", "probe_variant": "platt", "syn_split": "primary",
            "n_boot": 10, "recalibration": "prior_shift", "fit_source": "oracle",
            "fit_n": None, "fit_seed": None,
            "datasets": {ds: copy.deepcopy(block) for ds in ac.CALIBRATION_DATASETS},
        },
    }


def _load(cfg, tmp_path):
    path = tmp_path / f"{cfg['id']}.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return ac.load_calibration_v4_config(path)


def _manual(p):
    p["fit_source"] = "manual"
    for i, block in enumerate(p["datasets"].values()):
        block["pi_te_estimate"] = 0.3 + 0.1 * i


@pytest.mark.parametrize("recal,mutate", [
    ("prior_shift", lambda p: None),                                                  # oracle
    ("prior_shift", lambda p: p.update(fit_source="sample", fit_n=100, fit_seed=0)),
    ("prior_shift", _manual),
    ("intercept_fit", lambda p: None),                                                # oracle
    ("intercept_fit", lambda p: p.update(fit_source="sample", fit_n=100, fit_seed=3)),
    ("platt_fit", lambda p: None),                                                    # oracle
    ("platt_fit", lambda p: p.update(fit_source="sample", fit_n=100, fit_seed=0)),
])
def test_accepts_valid(base_cfg, tmp_path, recal, mutate):
    base_cfg["params"]["recalibration"] = recal
    mutate(base_cfg["params"])
    _load(base_cfg, tmp_path)


@pytest.mark.parametrize("mutate", [
    lambda p: p.update(recalibration="platt_fit") or _manual(p),           # manual is prior_shift only
    lambda p: p.update(recalibration="platt_scaling"),                     # unknown method
    lambda p: p.update(recalibration="intercept_fit") or _manual(p),       # manual is prior_shift only
    lambda p: p.update(fit_source="sample"),                               # sample without n / seed
    lambda p: p.update(fit_source="sample", fit_n=100),                    # sample without seed
    lambda p: p.update(fit_source="sample", fit_seed=0),                   # sample without n
    lambda p: p.update(fit_source="sample", fit_n=0, fit_seed=0),
    lambda p: p.update(fit_source="sample", fit_n=True, fit_seed=0),
    lambda p: p.update(fit_source="sample", fit_n=10, fit_seed=-1),
    lambda p: p.update(fit_n=50),                                          # n given for oracle
    lambda p: p.update(fit_seed=0),                                        # seed given for oracle
    lambda p: p.update(fit_source="manual"),                               # manual without pi_te
    lambda p: p["datasets"]["pond"].update(pi_te_estimate=0.4),            # pi_te given for oracle
    lambda p: p.update(fit_source="sample", fit_n=50, fit_seed=0) or p["datasets"]["nfix"].update(pi_te_estimate=0.4),
    lambda p: p.update(fit_source="bogus"),
    lambda p: p.update(n_boot=0),
    lambda p: p.pop("n_boot"),
    lambda p: p.pop("fit_seed"),                                           # null is required, absence is not
    lambda p: p.update(prior_source="oracle"),                             # stale key name
    lambda p: p["datasets"]["pond"].pop("pi_te_estimate"),
])
def test_rejects_inconsistent(base_cfg, tmp_path, mutate):
    mutate(base_cfg["params"])
    with pytest.raises(ValueError):
        _load(base_cfg, tmp_path)


def test_manual_rejects_out_of_range(base_cfg, tmp_path):
    _manual(base_cfg["params"])
    base_cfg["params"]["datasets"]["supermat"]["pi_te_estimate"] = 1.0
    with pytest.raises(ValueError):
        _load(base_cfg, tmp_path)


def test_prior_shift_map_known_answer():
    # pi_te = pi_tr: identity map.
    assert prior_shift_map(0.3, 0.3) == (1.0, 0.0)
    # pi_tr = 0.5, pi_te = 0.75: intercept logit(0.75) = ln 3; p = 0.5 maps to 0.75.
    coef, icpt = prior_shift_map(0.75, 0.5)
    assert coef == 1.0 and icpt == pytest.approx(np.log(3))
    assert apply_platt(np.array([0.5]), coef, icpt)[0] == pytest.approx(0.75)
    # Odds scale by (0.75/0.25) / (0.5/0.5) = 3: p = 0.2 (odds 1/4) -> odds 3/4 -> 3/7.
    assert apply_platt(np.array([0.2]), coef, icpt)[0] == pytest.approx(3 / 7)
    for bad in [(0.0, 0.5), (0.5, 1.0), (True, 0.5)]:
        with pytest.raises(ValueError):
            prior_shift_map(*bad)


def test_prior_shift_map_matches_library_fit():
    labels = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=bool)  # rate 0.25
    assert fit_prior_shift(labels, 0.6) == prior_shift_map(0.25, 0.6)


def test_intercept_fit_known_answer():
    # All scores 0.5 (logit 0), label rate 0.25: the MLE intercept is logit(0.25) = -ln 3.
    coef, icpt = intercept_fit_map(np.full(8, 0.5), np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=bool))
    assert coef == 1.0 and icpt == pytest.approx(-np.log(3), abs=1e-9)
    # Varied scores: the mapped mean equals the label rate (asserted inside, rechecked here).
    p = np.array([0.1, 0.3, 0.6, 0.9, 0.8, 0.2])
    y = np.array([0, 0, 1, 1, 0, 0], dtype=bool)
    coef, icpt = intercept_fit_map(p, y)
    assert apply_platt(p, coef, icpt).mean() == pytest.approx(y.mean(), abs=1e-8)
    with pytest.raises(ValueError):
        intercept_fit_map(p, np.zeros(6, dtype=bool))


def _two_level_rows(n_per, rate_lo, rate_hi):
    """n_per rows at p = expit(-1) and n_per at p = expit(+1), with the given label
    rates; positives first within each level, so the rows are not separable."""
    from scipy.special import expit
    k_lo, k_hi = round(n_per * rate_lo), round(n_per * rate_hi)
    p = np.r_[np.full(n_per, expit(-1.0)), np.full(n_per, expit(1.0))]
    y = np.r_[np.arange(n_per) < k_lo, np.arange(n_per) < k_hi]
    return p, y


def test_platt_fit_known_answer():
    # Logits z in {-1, +1} with label rates r_lo, r_hi: the MLE makes each level's
    # mapped probability its own rate, expit(-a + b) = r_lo and expit(a + b) = r_hi, so
    # a = (logit r_hi - logit r_lo) / 2 and b = (logit r_hi + logit r_lo) / 2.
    p, y = _two_level_rows(40, 0.25, 0.75)                 # a = ln 3, b = 0
    coef, icpt = platt_fit_map(p, y)
    assert coef == pytest.approx(np.log(3), abs=1e-3) and icpt == pytest.approx(0.0, abs=1e-3)
    p, y = _two_level_rows(40, 0.2, 0.5)                   # a = ln 2, b = -ln 2
    coef, icpt = platt_fit_map(p, y)
    assert coef == pytest.approx(np.log(2), abs=1e-3) and icpt == pytest.approx(-np.log(2), abs=1e-3)
    # Rates already matching the raw scores (expit(-1), expit(1)) would give the identity,
    # (1, 0); same-rate levels give slope 0 (scores carry no information).
    p, y = _two_level_rows(40, 0.5, 0.5)
    coef, icpt = platt_fit_map(p, y)
    assert coef == pytest.approx(0.0, abs=1e-3) and icpt == pytest.approx(0.0, abs=1e-3)


def test_platt_fit_map_matches_library_fit():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.01, 0.99, 200)
    y = rng.random(200) < p ** 2
    assert platt_fit_map(p, y) == fit_platt(p, y)


def test_platt_fit_map_rejects_separable_and_single_class():
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    with pytest.raises(ValueError, match="separable"):
        platt_fit_map(p, np.array([0, 0, 0, 1, 1, 1], dtype=bool))
    with pytest.raises(ValueError, match="separable"):         # reversed direction
        platt_fit_map(p, np.array([1, 1, 1, 0, 0, 0], dtype=bool))
    with pytest.raises(ValueError, match="separable"):         # quasi: overlap only at a tie
        platt_fit_map(np.array([0.1, 0.5, 0.5, 0.9]), np.array([0, 0, 1, 1], dtype=bool))
    with pytest.raises(ValueError, match="both classes"):
        platt_fit_map(p, np.zeros(6, dtype=bool))
    # One overlapping pair is enough for a finite MLE.
    platt_fit_map(p, np.array([0, 0, 1, 0, 1, 1], dtype=bool))


def test_uniform_fit_sample():
    pos = uniform_fit_sample(20, 8, fit_seed=7)
    # Distinct, sorted, in range: a plain subset (no resampling, so no repeats).
    assert len(pos) == 8 and len(set(pos.tolist())) == 8 and np.all(np.diff(pos) > 0)
    assert pos.min() >= 0 and pos.max() < 20
    # Exactly numpy's draw for this seed, and deterministic.
    assert np.array_equal(pos, np.sort(np.random.default_rng(7).choice(20, 8, replace=False)))
    assert np.array_equal(pos, uniform_fit_sample(20, 8, fit_seed=7))
    assert not np.array_equal(pos, uniform_fit_sample(20, 8, fit_seed=8))
    # n = pool size takes every row.
    assert np.array_equal(uniform_fit_sample(5, 5, fit_seed=0), np.arange(5))
    with pytest.raises(AssertionError):
        uniform_fit_sample(5, 6, fit_seed=0)
