"""
Calibration metrics for judge model probabilities.

Supports two probability sources:
1. **Probe probabilities** — ``predict_proba`` output from a trained probe
   (``scholarlm.analysis.probe``).
2. **Next-token probabilities** — ``judgement_p_true`` values stored by
   ``run_judge.py`` in ``responses.json`` (local judges) or frontier judge
   response files.

Typical usage
-------------
    from scholarlm.analysis.calibration import compute_ece, reliability_diagram_data
    import numpy as np

    probs = np.array([0.9, 0.7, 0.3, 0.1, ...])  # predicted P(valid)
    labels = np.array([1, 1, 0, 0, ...])           # ground truth

    ece = compute_ece(probs, labels)
    diag = reliability_diagram_data(probs, labels)
"""
from __future__ import annotations

import numpy as np


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
