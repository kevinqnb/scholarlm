"""Rung-1 unit tests for analysis/platt_scaling.py helpers and load_platt_sweep_config."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

from analysis import analysis_config as ac  # noqa: E402
from analysis import platt_scaling as ps  # noqa: E402

_CFG = _REPO / "analysis/analysis-configs/2026-10-08-platt-scaling-intercept-fit-nested-tiny-01.yaml"


def _pool(n_docs=8, rows=5, seed=0):
    rng = np.random.default_rng(seed)
    docs = np.repeat(np.arange(n_docs), rows)
    y = rng.random(len(docs)) < 0.4
    return docs, y


def test_draw_fit_samples_reproduces_v3_draws():
    # No single-class draw at this n, so draw r is exactly v3's fit sample r.
    docs, y = _pool()
    kept, n_draws = ps.draw_fit_samples(docs, y, 20, 6, seed=3)
    assert n_draws == 6 and [r for r, _ in kept] == list(range(6))
    for r, s in kept:
        assert np.array_equal(s, ps.nb.pool_resampled_fit_sample(docs, 20, ps.nb.fit_sample_rng(3, r)))


def test_draw_fit_samples_skips_single_class_and_keeps_going():
    # docs 0..3 all valid, docs 4..7 all invalid, one row each: a size-2 sample is
    # single-class whenever both rows come from the same half.
    docs = np.arange(8)
    y = docs < 4
    kept, n_draws = ps.draw_fit_samples(docs, y, 2, 30, seed=0)
    assert len(kept) == 30 and n_draws > 30
    assert all(0 < y[s].sum() < 2 for _, s in kept)
    rs = [r for r, _ in kept]
    assert rs == sorted(rs) and len(set(rs)) == 30
    skipped = sorted(set(range(n_draws)) - set(rs))
    assert all(y[ps.nb.pool_resampled_fit_sample(docs, 2, ps.nb.fit_sample_rng(0, r))].sum() in (0, 2)
               for r in skipped)
    assert ps.draw_fit_samples(docs, y, 2, 30, seed=0)[1] == n_draws          # deterministic


def test_draw_fit_samples_fails_loud_when_pool_is_single_class():
    docs = np.arange(10)
    with pytest.raises(RuntimeError):
        ps.draw_fit_samples(docs, np.zeros(10, dtype=bool), 3, 4, seed=0)


@pytest.mark.parametrize("recalibration", ["platt_fit", "intercept_fit", "prior_shift"])
def test_recalibrated_sets_match_fit_recalibration(recalibration):
    from scholarlm.utils.calibration import apply_platt, fit_recalibration
    docs, y = _pool(n_docs=20, rows=10, seed=1)
    rng = np.random.default_rng(2)
    pool_raw = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.15, len(y)), 0.01, 0.99)
    test_raw = rng.uniform(0.05, 0.95, 50)
    kept, _ = ps.draw_fit_samples(docs, y, 40, 3, seed=0)
    sets, maps = ps.recalibrated_sets(pool_raw, y, test_raw, kept, recalibration, 0.5)
    assert len(sets) == len(maps) == 3
    for (r, s), p, m in zip(kept, sets, maps):
        coef, icpt = fit_recalibration(recalibration, pool_raw[s], y[s], 0.5)
        assert (m["Fit sample"], m["coef"], m["intercept"]) == (r, coef, icpt)
        assert m["Label rate"] == y[s].mean()
        assert np.array_equal(p, apply_platt(test_raw, coef, icpt))
        if recalibration == "prior_shift":
            # pi_tr = 0.5: intercept is logit(sample label rate)
            assert icpt == pytest.approx(np.log(y[s].mean() / (1 - y[s].mean())))


def test_committed_config_loads():
    cfg = ac.load_platt_sweep_config(_CFG)
    p = cfg["params"]
    assert p["platt_ns"] == [10, 100] and p["recalibration"] == "intercept_fit"
    assert p["n_fit_samples"] == 10 and p["n_doc_boot"] == 10


@pytest.mark.parametrize("mutate", [
    lambda p: p.pop("platt_ns"),
    lambda p: p.pop("recalibration"),
    lambda p: p.pop("n_fit_samples"),
    lambda p: p.pop("n_doc_boot"),
    lambda p: p.update(platt_ns=[]),
    lambda p: p.update(platt_ns=[10, 10, 50]),
    lambda p: p.update(platt_ns=[50, 10]),
    lambda p: p.update(platt_ns=[0, 10]),
    lambda p: p.update(platt_ns=[True, 10]),
    lambda p: p.update(n_fit_samples=0),
    lambda p: p.update(n_doc_boot=True),
    lambda p: p.update(recalibration="isotonic"),
    lambda p: p.update(n_trials=100),   # pre-nested-bootstrap keys not allowed
    lambda p: p.update(ci=0.95),
    lambda p: p.update(single_class_policy="drop"),
    lambda p: p.update(n_syn_boot=100),   # real cells only
    lambda p: p.update(platt_n=100),   # v3 key not allowed
])
def test_loader_rejects_malformed(tmp_path, mutate):
    cfg = copy.deepcopy(yaml.safe_load(_CFG.read_text()))
    mutate(cfg["params"])
    p = tmp_path / f"{cfg['id']}.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        ac.load_platt_sweep_config(p)
