"""
Calibration metrics for judge model probabilities.

Supports two probability sources:
1. **Probe probabilities** — ``predict_proba`` output from a trained probe
   (``scholarlm.utils.probe``).
2. **Next-token probabilities** — ``judgement_p_true`` values stored by
   ``run_judge_local.py`` in ``responses.json`` (local judges) or frontier judge
   response files.

Typical usage
-------------
    from scholarlm.utils.calibration import compute_ece, reliability_diagram_data
    import numpy as np

    probs = np.array([0.9, 0.7, 0.3, 0.1, ...])  # predicted P(valid)
    labels = np.array([1, 1, 0, 0, ...])           # ground truth

    ece = compute_ece(probs, labels)
    diag = reliability_diagram_data(probs, labels)
"""
from __future__ import annotations

import numpy as np


def rescale_probabilities_em(
    probs: np.ndarray,
    train_labels: np.ndarray | None = None,
    *,
    pi_tr: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
    init_pi_te: float | None = None,
    return_history: bool = False,
    eps: float = 1e-12,
) -> tuple:
    """Rescale predicted probabilities under label shift via Saerens et al. (2002) EM.

    Estimates the test-set prevalence from unlabeled test predictions and adjusts
    the probabilities accordingly.  Assumes label shift (P(X|Y) constant across
    domains) and that the source-domain probabilities are calibrated.

    Args:
        probs: Predicted probabilities P_train(Y=1 | x) on the test set. Shape ``(n_test,)``.
        train_labels: Binary training labels (0/1). Used only to estimate the
            training prevalence.  Mutually exclusive with ``pi_tr``.
        pi_tr: Training prevalence as a scalar in ``(0, 1)``.  Use this when
            the full label array is unavailable (e.g., when loading a saved
            probe).  Mutually exclusive with ``train_labels``.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on the change in estimated test prevalence.
        init_pi_te: Initial guess for test prevalence. Defaults to the training
            prevalence (neutral starting point).
        return_history: If True, also return the list of pi_te estimates per iteration.
        eps: Numerical stability constant.

    Returns:
        ``(rescaled, pi_te_hat)`` — rescaled probabilities and estimated test
        prevalence.  If ``return_history=True``, returns ``(rescaled, pi_te_hat,
        history)``.

    Reference:
        Saerens, Latinne, and Decaestecker (2002), Neural Computation 14(1):21-41.
    """
    if train_labels is None and pi_tr is None:
        raise ValueError("Provide exactly one of 'train_labels' or 'pi_tr'.")
    if train_labels is not None and pi_tr is not None:
        raise ValueError("Provide exactly one of 'train_labels' or 'pi_tr', not both.")

    probs = np.asarray(probs, dtype=float)

    if pi_tr is None:
        pi_tr = float(np.mean(np.asarray(train_labels)))
    if not (eps < pi_tr < 1 - eps):
        raise ValueError(
            f"Training prevalence pi_tr={pi_tr:.4g} is degenerate; cannot rescale."
        )

    probs_clipped = np.clip(probs, eps, 1 - eps)

    pi_te = float(init_pi_te) if init_pi_te is not None else pi_tr
    if init_pi_te is not None and not (eps < pi_te < 1 - eps):
        raise ValueError(f"init_pi_te={pi_te:.4g} must be strictly between 0 and 1.")

    history = [pi_te]

    for _ in range(max_iter):
        num = probs_clipped * (pi_te / pi_tr)
        den = num + (1 - probs_clipped) * ((1 - pi_te) / (1 - pi_tr))
        rescaled = num / den
        pi_te_new = float(np.mean(rescaled))
        history.append(pi_te_new)
        if abs(pi_te_new - pi_te) < tol:
            pi_te = pi_te_new
            break
        pi_te = pi_te_new

    num = probs_clipped * (pi_te / pi_tr)
    den = num + (1 - probs_clipped) * ((1 - pi_te) / (1 - pi_tr))
    rescaled = num / den

    if return_history:
        return rescaled, pi_te, history
    return rescaled, pi_te


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute the Expected Calibration Error (ECE).

    Partitions predictions into ``n_bins`` equal-width confidence bins and
    computes the weighted average of |confidence - accuracy| per bin.

    Args:
        probs: Predicted probabilities for the positive class, shape ``(n,)``.
            Values should be in ``[0, 1]``.
        labels: Binary ground truth labels, shape ``(n,)``.  ``1`` / ``True``
            is positive.
        n_bins: Number of equal-width bins in ``[0, 1]``.

    Returns:
        Scalar ECE value in ``[0, 1]``.  Lower is better.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        # Include upper boundary in the last bin
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        if not mask.any():
            continue
        bin_conf = probs[mask].mean()
        bin_acc = labels[mask].mean()
        ece += mask.sum() * abs(bin_conf - bin_acc)

    return float(ece / n)


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute data for a reliability (calibration) diagram.

    Args:
        probs: Predicted probabilities for the positive class, shape ``(n,)``.
        labels: Binary ground truth labels, shape ``(n,)``.
        n_bins: Number of equal-width bins.

    Returns:
        Dict with keys:
        - ``"bin_centers"``  — midpoint of each bin, shape ``(n_bins,)``.
        - ``"bin_accuracy"`` — mean label in each bin (fraction positive),
          shape ``(n_bins,)``.  ``np.nan`` for empty bins.
        - ``"bin_confidence"`` — mean predicted probability in each bin,
          shape ``(n_bins,)``.  ``np.nan`` for empty bins.
        - ``"bin_counts"``   — number of samples in each bin, shape ``(n_bins,)``.
        - ``"ece"``          — scalar ECE value.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_accuracy = np.full(n_bins, np.nan)
    bin_confidence = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            bin_accuracy[i] = labels[mask].mean()
            bin_confidence[i] = probs[mask].mean()

    return {
        "bin_centers": bin_centers,
        "bin_accuracy": bin_accuracy,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
        "ece": compute_ece(probs, labels, n_bins=n_bins),
    }
