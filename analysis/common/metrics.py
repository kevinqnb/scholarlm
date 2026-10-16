"""Recovery and validity rates, with optional Wilson 95% CIs.

``recovery_rate`` / ``validity_rate`` match against ground truth; the ``_from_labels``
variants take precomputed edges and labels.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
from statsmodels.stats.proportion import proportion_confint

from .loaders import cached_match


def recovery_rate(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    cache_path: Path | None = None,
    return_ci: bool = False,
) -> float | tuple[float, float, float]:
    """Fraction of ground-truth rows with at least one matching extraction.

    Matching is computed (or read from cache) at threshold 0.0; edges are then kept
    if their weight is >= ``fuzzy_threshold``.

    Args:
        ground_truth_df: Manual ground truth, one row per measurement.
        extraction_df: Extracted measurements, one row per measurement.
        strict_matching: Exact-match column mapping.
        fuzzy_matching: Fuzzy-match column mapping.
        fuzzy_threshold: Minimum edge weight that counts as a match.
        cache_path: Match cache to read or write (see ``cached_match``).
        return_ci: Also return a Wilson 95% CI.

    Returns:
        Recovery rate, or ``(rate, lower, upper)`` if ``return_ci``.
    """
    matching, edges, edge_weights = cached_match(
        ground_truth_df,
        extraction_df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=0.0,
        cache_path=cache_path,
    )

    gt_edge_exists = np.zeros(len(ground_truth_df), dtype = bool)
    for i, (gt_idx, ex_idx) in enumerate(edges):
        if edge_weights[i] >= fuzzy_threshold:
            gt_edge_exists[gt_idx] = True

    rate = float(np.mean(gt_edge_exists))
    if not return_ci:
        return rate
    n = len(gt_edge_exists)
    k = int(np.sum(gt_edge_exists))
    lower, upper = proportion_confint(k, n, alpha=0.05, method='wilson')
    return rate, float(lower), float(upper)


def recovery_rate_from_labels(
    n_ground_truth: int,
    edges: list[tuple[int, int]],
    predicted_labels: np.ndarray,
    return_ci: bool = False,
) -> float | tuple[float, float, float]:
    """Fraction of ground-truth rows matched by at least one predicted-valid extraction.

    Args:
        n_ground_truth: Number of ground-truth rows.
        edges: (gt_idx, ex_idx) pairs, already filtered by threshold.
        predicted_labels: Boolean per extraction, True = predicted valid.
        return_ci: Also return a Wilson 95% CI.

    Returns:
        Recovery rate, or ``(rate, lower, upper)`` if ``return_ci``.
    """
    ground_truth_matched = np.zeros(n_ground_truth, dtype = bool)
    for gt_idx, ex_idx in edges:
        if predicted_labels[ex_idx]:
            ground_truth_matched[gt_idx] = True
            
    rate = float(np.mean(ground_truth_matched))
    if not return_ci:
        return rate
    n = len(ground_truth_matched)
    k = int(np.sum(ground_truth_matched))
    lower, upper = proportion_confint(k, n, alpha=0.05, method='wilson')
    return rate, float(lower), float(upper)


def validity_rate(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    judged_df: pd.DataFrame | None = None,
    cache_path: Path | None = None,
    label_col: str = "judgement_combined",
    return_ci: bool = False,
    denominator_n: int | None = None,
) -> float | tuple[float, float, float]:
    """Fraction of extractions that are valid (1 - hallucination rate).

    A row is valid if it matches ground truth or the judge marked it valid.
    Returns 0.0 if the denominator is 0.

    Args:
        ground_truth_df: Manual ground truth, one row per measurement.
        extraction_df: Extracted measurements, one row per measurement.
        strict_matching: Exact-match column mapping.
        fuzzy_matching: Fuzzy-match column mapping.
        fuzzy_threshold: Minimum edge weight that counts as a match.
        judged_df: Judge output aligned to ``extraction_df``; None means no judge labels.
        cache_path: Match cache to read or write (see ``cached_match``).
        label_col: Boolean judge-label column in ``judged_df``.
        return_ci: Also return a Wilson 95% CI.
        denominator_n: Row count before the caller filtered out unusable rows, so
            those rows count as invalid instead of being excused. None means
            ``len(extraction_df)``.

    Returns:
        Validity rate, or ``(rate, lower, upper)`` if ``return_ci``.

    Raises:
        ValueError: ``judged_df`` length differs from ``extraction_df``, or
            ``denominator_n`` is smaller than the rows scored.
    """
    if judged_df is not None and len(judged_df) != len(extraction_df):
        raise ValueError(
            f"judged_df length ({len(judged_df)}) must match extraction_df length ({len(extraction_df)})"
        )

    matching, edges, edge_weights = cached_match(
        ground_truth_df,
        extraction_df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=0.0,
        cache_path=cache_path,
    )

    ex_edge_exists = np.zeros(len(extraction_df), dtype = bool)
    for i, (gt_idx, ex_idx) in enumerate(edges):
        if edge_weights[i] >= fuzzy_threshold:
            ex_edge_exists[ex_idx] = True

    if judged_df is not None:
        jlabels = judged_df[label_col].to_numpy(dtype = bool)
    else:
        jlabels = np.zeros(len(extraction_df), dtype = bool)

    labels = jlabels | ex_edge_exists

    n = len(labels) if denominator_n is None else denominator_n
    if n < len(labels):
        raise ValueError(
            f"denominator_n ({n}) is smaller than the number of rows scored "
            f"({len(labels)})"
        )
    k = int(np.sum(labels))  # valid = matched or judged valid

    rate = k / n if n else 0.0
    if not return_ci:
        return rate
    lower, upper = proportion_confint(k, n, alpha=0.05, method='wilson')
    return float(rate), float(lower), float(upper)


def validity_rate_from_labels(
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    return_ci: bool = False,
) -> float | tuple[float, float, float]:
    """Precision of predicted-valid extractions: TP / (TP + FP).

    Returns 0.0 (and a (0, 0) CI) when nothing is predicted valid.

    Args:
        labels: Boolean per extraction, True = actually valid.
        predicted_labels: Boolean per extraction, True = predicted valid.
        return_ci: Also return a Wilson 95% CI.

    Returns:
        Validity rate, or ``(rate, lower, upper)`` if ``return_ci``.

    Raises:
        ValueError: The two arrays differ in length.
    """
    if len(labels) != len(predicted_labels):
        raise ValueError(
            f"ground_truth_labels length ({len(labels)}) must match predicted_labels length ({len(predicted_labels)})"
        )

    n = int(np.sum(predicted_labels))
    if n == 0:
        return (0.0, 0.0, 0.0) if return_ci else 0.0
    k = int(np.sum(labels & predicted_labels))
    rate = k / n
    if not return_ci:
        return float(rate)
    lower, upper = proportion_confint(k, n, alpha=0.05, method='wilson')
    return float(rate), float(lower), float(upper)


