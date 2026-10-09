"""Recalibration maps and the fit-row sampler used by analysis/calibration.py.

Each map returns apply_platt's ``(coef, intercept)``:
  - prior_shift: slope 1, intercept ``logit pi_te - logit pi_tr`` (Bayes under label shift).
  - intercept_fit: slope 1, intercept fit so mapped fit-row probs average to their label rate.
  - platt_fit: slope and intercept by unregularized logistic MLE on fit rows.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.special import expit, logit
from sklearn.exceptions import ConvergenceWarning

from scholarlm.utils.calibration import apply_platt, fit_intercept, fit_platt

# Tolerance for the Platt score-equation check; lbfgs at sklearn's default tol gets ~1e-4.
PLATT_SCORE_TOL = 1e-3


def prior_shift_map(pi_te: float, pi_tr: float) -> tuple[float, float]:
    """Label-shift correction: keep the slope, shift the intercept by the prior log-odds.

    Args:
        pi_te: Positive rate in the test population, in (0, 1).
        pi_tr: Positive rate in the scorer's training data, in (0, 1).

    Returns:
        ``(1.0, logit(pi_te) - logit(pi_tr))``.

    Raises:
        ValueError: A prevalence is outside (0, 1) or is a bool.
    """
    for name, v in (("pi_te", pi_te), ("pi_tr", pi_tr)):
        if isinstance(v, bool) or not 0 < v < 1:
            raise ValueError(f"{name} must be in (0, 1), got {v!r}")
    return 1.0, float(logit(pi_te) - logit(pi_tr))


def intercept_fit_map(probs, labels) -> tuple[float, float]:
    """Fit an intercept-only recalibration and check mapped probs match the label rate.

    Args:
        probs: Fit-row predicted probabilities.
        labels: Fit-row boolean labels.

    Returns:
        ``(1.0, intercept)``.
    """
    coef, icpt = fit_intercept(probs, labels)
    assert coef == 1.0, coef
    mean = float(apply_platt(np.asarray(probs, dtype=float), coef, icpt).mean())
    rate = float(np.mean(labels))
    assert abs(mean - rate) < 1e-8, (mean, rate)
    return coef, icpt


def platt_fit_map(probs, labels, eps: float = 1e-6) -> tuple[float, float]:
    """Fit Platt scaling and check both MLE score equations hold.

    Separable fit rows have no finite MLE, but lbfgs still "converges" to an arbitrary
    slope, so separability is rejected up front.

    Args:
        probs: Fit-row predicted probabilities.
        labels: Fit-row boolean labels.
        eps: Logit clipping; must equal fit_platt / apply_platt's default.

    Returns:
        ``(coef, intercept)``.

    Raises:
        ValueError: Only one class present, or rows are (quasi-)separable in logit(p).
        ConvergenceWarning: lbfgs did not converge (promoted to an error).
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
    """Draw the recalibration fit rows: a uniform random subset of the pool.

    Args:
        n_pool: Number of candidate rows.
        n: Number of fit rows to draw (1..n_pool).
        fit_seed: Seed for ``np.random.default_rng``.

    Returns:
        Sorted array of ``n`` distinct positions in ``range(n_pool)``.
    """
    assert n_pool > 0 and 0 < n <= n_pool, (n_pool, n)
    pos = np.sort(np.random.default_rng(fit_seed).choice(n_pool, n, replace=False))
    assert len(np.unique(pos)) == n
    return pos
