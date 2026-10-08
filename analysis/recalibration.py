"""Fit-row sample and recalibration maps for analysis/calibration_updated_v4.py.

Import-side-effect free so it can be unit tested on a hand-built fixture.

Both maps have slope 1, in apply_platt's ``(coef, intercept)`` format:
  - prior_shift: under label shift (P(x|y) fixed between the scorer's training data and
    the test data), Bayes' rule gives ``logit p_te(x) = logit p_tr(x) + logit pi_te -
    logit pi_tr``, so the intercept is ``logit pi_te - logit pi_tr``.
  - intercept_fit: the intercept MLE on labelled fit rows (scholarlm's fit_intercept),
    so the fit rows' mapped probabilities average to their label rate.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logit

from scholarlm.utils.calibration import apply_platt, fit_intercept


def prior_shift_map(pi_te: float, pi_tr: float) -> tuple[float, float]:
    """``(1.0, logit(pi_te) - logit(pi_tr))``; both prevalences must be in (0, 1)."""
    for name, v in (("pi_te", pi_te), ("pi_tr", pi_tr)):
        if isinstance(v, bool) or not 0 < v < 1:
            raise ValueError(f"{name} must be in (0, 1), got {v!r}")
    return 1.0, float(logit(pi_te) - logit(pi_tr))


def intercept_fit_map(probs, labels) -> tuple[float, float]:
    """fit_intercept on these rows, checked against its score equation."""
    coef, icpt = fit_intercept(probs, labels)
    assert coef == 1.0, coef
    mean = float(apply_platt(np.asarray(probs, dtype=float), coef, icpt).mean())
    rate = float(np.mean(labels))
    assert abs(mean - rate) < 1e-8, (mean, rate)
    return coef, icpt


def uniform_fit_sample(n_pool: int, n: int, fit_seed: int) -> np.ndarray:
    """``n`` distinct sorted positions in ``range(n_pool)``, drawn uniformly without
    replacement by ``np.random.default_rng(fit_seed)``. One draw, no document resampling:
    the fit rows are a plain random subset of the pool."""
    assert n_pool > 0 and 0 < n <= n_pool, (n_pool, n)
    pos = np.sort(np.random.default_rng(fit_seed).choice(n_pool, n, replace=False))
    assert len(np.unique(pos)) == n
    return pos
