"""Fit-row sample and recalibration maps for analysis/calibration.py.

Import-side-effect free so it can be unit tested on a hand-built fixture.

Maps are in apply_platt's ``(coef, intercept)`` format. The first two have slope 1:
  - prior_shift: under label shift (P(x|y) fixed between the scorer's training data and
    the test data), Bayes' rule gives ``logit p_te(x) = logit p_tr(x) + logit pi_te -
    logit pi_tr``, so the intercept is ``logit pi_te - logit pi_tr``.
  - intercept_fit: the intercept MLE on labelled fit rows (scholarlm's fit_intercept),
    so the fit rows' mapped probabilities average to their label rate.
  - platt_fit: slope and intercept by unregularized logistic MLE on labelled fit rows
    (scholarlm's fit_platt).
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning

from scholarlm.utils.calibration import apply_platt, fit_intercept, fit_platt

# fit_platt's lbfgs stops at sklearn's default gradient tolerance; its score equations
# hold to ~1e-4 (mean over fit rows), so they are checked at 1e-3.
PLATT_SCORE_TOL = 1e-3


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


def platt_fit_map(probs, labels, eps: float = 1e-6) -> tuple[float, float]:
    """fit_platt on these rows, checked against both Platt score equations.

    On (quasi-)separable fit rows -- every positive's clipped logit on one side of every
    negative's, ties included -- the unregularized slope has no finite MLE, and lbfgs
    reports convergence anyway at an arbitrary large slope (e.g. 15 on a 6-row toy), so
    that is checked up front and raised. A ConvergenceWarning is also raised as an error.
    ``eps`` must be fit_platt's / apply_platt's clipping (their default).
    """
    labels = np.asarray(labels, dtype=bool)
    z = logit(np.clip(np.asarray(probs, dtype=float), eps, 1 - eps))
    if labels.all() or not labels.any():
        raise ValueError(f"platt_fit needs both classes; got {int(labels.sum())}/{len(labels)} positive")
    if z[labels].min() >= z[~labels].max() or z[labels].max() <= z[~labels].min():
        raise ValueError(
            f"platt_fit: fit rows are (quasi-)separable in logit(p) (positives in "
            f"[{z[labels].min():.4g}, {z[labels].max():.4g}], negatives in "
            f"[{z[~labels].min():.4g}, {z[~labels].max():.4g}]); the Platt MLE does not exist")
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        coef, icpt = fit_platt(probs, labels)
    resid = expit(coef * z + icpt) - labels
    assert np.allclose(expit(coef * z + icpt), apply_platt(probs, coef, icpt), rtol=0, atol=1e-12)
    # MLE score equations: d/d(intercept) and d/d(slope) of the log-likelihood vanish.
    assert abs(float(resid.mean())) < PLATT_SCORE_TOL, float(resid.mean())
    assert abs(float((resid * z).mean())) < PLATT_SCORE_TOL, float((resid * z).mean())
    return coef, icpt


def uniform_fit_sample(n_pool: int, n: int, fit_seed: int) -> np.ndarray:
    """``n`` distinct sorted positions in ``range(n_pool)``, drawn uniformly without
    replacement by ``np.random.default_rng(fit_seed)``. One draw, no document resampling:
    the fit rows are a plain random subset of the pool."""
    assert n_pool > 0 and 0 < n <= n_pool, (n_pool, n)
    pos = np.sort(np.random.default_rng(fit_seed).choice(n_pool, n, replace=False))
    assert len(np.unique(pos)) == n
    return pos
