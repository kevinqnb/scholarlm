"""Recovery and hallucination metrics for ScholarlM extraction evaluation.

Typical usage
-------------
    from analysis.metrics import recovery_rate, hallucination_rate

    recall = recovery_rate(extraction_df, ground_truth_df, strict_matching={"entity": ["name"]})
    print(recall)  # 0.82

    hallucination = hallucination_rate(extraction_df, ground_truth_df, judged_df, strict_matching={"entity": ["name"]})
    print(hallucination)  # 0.15
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np


def recovery_rate(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    cache_path: Path | None = None,
) -> float:
    """Run ``match_datasets`` and return recall/precision statistics.

    Args:
        extraction_df: Extracted measurements (rows = measurements).
        ground_truth_df: Manual ground truth (rows = measurements).
        strict_matching: Exact-match column mapping passed to ``match_datasets``.
        fuzzy_matching: Fuzzy-match column mapping passed to ``match_datasets``.
        fuzzy_threshold: Minimum fuzzy score for a match.
        cache_path: Optional path for a disk-cached result (see ``cached_match``).

    Returns:
        Recovery rate (float) representing the fraction of ground truth items matched.
    """
    from .loaders import cached_match

    matching, edges, edge_weights = cached_match(
        ground_truth_df,
        extraction_df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=fuzzy_threshold,
        cache_path=cache_path,
    )

    gt_edge_exists = np.zeros(len(ground_truth_df), dtype = bool)
    for i, (gt_idx, ex_idx) in enumerate(edges):
        if edge_weights[i] > fuzzy_threshold:
            gt_edge_exists[gt_idx] = True

    return np.mean(gt_edge_exists)


def hallucination_rate(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    judged_df: pd.DataFrame,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    cache_path: Path | None = None,
    label_col: str = "judgement_combined",
) -> float:
    """Compute hallucination rate from judged extraction results.

    Args:
        extraction_df: Extracted measurements (rows = measurements).
        ground_truth_df: Manual ground truth (rows = measurements).
        judged_df: DataFrame with a boolean ``label_col`` column (``True`` = valid).
        strict_matching: Exact-match column mapping passed to ``match_datasets``.
        fuzzy_matching: Fuzzy-match column mapping passed to ``match_datasets``.
        fuzzy_threshold: Minimum fuzzy score for a match.
        cache_path: Optional path for a disk-cached result (see ``cached_match``).
        label_col: Column name for the combined judgement label.

    Returns:
        Hallucination rate (float) representing the fraction of extracted items that are hallucinated.
    """
    if len(judged_df) != len(extraction_df):
        raise ValueError(
            f"judged_df length ({len(judged_df)}) must match extraction_df length ({len(extraction_df)})"
        )

    from .loaders import cached_match

    matching, edges, edge_weights = cached_match(
        ground_truth_df,
        extraction_df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=fuzzy_threshold,
        cache_path=cache_path,
    )

    ex_edge_exists = np.zeros(len(extraction_df), dtype = bool)
    for i, (gt_idx, ex_idx) in enumerate(edges):
        if edge_weights[i] > fuzzy_threshold:
            ex_edge_exists[ex_idx] = True

    jlabels = judged_df[label_col].to_numpy(dtype = bool)
    labels = jlabels | ex_edge_exists

    return 1 - np.mean(labels)


def per_paper_metrics(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    judged_df: pd.DataFrame | None = None,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    cache_path: Path | None = None,
    paper_col: str = "document_id",
    label_col: str = "judgement_combined",
) -> pd.DataFrame:
    """Per-paper recovery and (optionally) hallucination summary.

    Args:
        extraction_df: Extracted measurements with a ``paper_col`` column.
        ground_truth_df: Ground truth measurements with a ``paper_col`` column.
        judged_df: Optional judged DataFrame (enables hallucination column).
        strict_matching: Passed to ``match_datasets``.
        fuzzy_matching: Passed to ``match_datasets``.
        fuzzy_threshold: Minimum fuzzy score for a match.
        cache_path: Optional path for a disk-cached result.
        paper_col: Column identifying which paper each row belongs to.
        label_col: Judgement column in ``judged_df``.

    Returns:
        DataFrame indexed by paper, with columns for recovery, hallucination,
        n_extracted, and n_gt.
    """
    if judged_df is not None and len(judged_df) != len(extraction_df):
        raise ValueError(
            f"judged_df length ({len(judged_df)}) must match extraction_df length ({len(extraction_df)})"
        )

    from .loaders import cached_match

    matching, edges, edge_weights = cached_match(
        ground_truth_df,
        extraction_df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=fuzzy_threshold,
        cache_path=cache_path,
    )

    ex_edge_exists = np.zeros(len(extraction_df), dtype = bool)
    gt_edge_exists = np.zeros(len(ground_truth_df), dtype = bool)
    for i, (gt_idx, ex_idx) in enumerate(edges):
        if edge_weights[i] > fuzzy_threshold:
            ex_edge_exists[ex_idx] = True
            gt_edge_exists[gt_idx] = True

    papers = sorted(set(extraction_df[paper_col].unique()) | set(ground_truth_df[paper_col].unique()))
    rows = []
    for paper in papers:
        row: dict = {"paper": paper}
        ext_idxs = extraction_df[extraction_df[paper_col] == paper].index
        gt_idxs = ground_truth_df[ground_truth_df[paper_col] == paper].index
        if len(ext_idxs) == 0 and len(gt_idxs) == 0:
            continue
        elif len(ext_idxs) == 0:
            row.update({"recovery": 0.0, "hallucination": 0.0, "n_extracted": 0, "n_gt": len(gt_idxs)})
        elif len(gt_idxs) == 0:
            row.update({"recovery": 0.0, "hallucination": 1.0, "n_extracted": len(ext_idxs), "n_gt": 0})
        else:
            if judged_df is not None:
                labels = judged_df[label_col].to_numpy(dtype = bool)
                labels = labels[ext_idxs] | ex_edge_exists[ext_idxs]
            else:
                labels = ex_edge_exists[ext_idxs]

            stats = {
                "recovery": np.mean(gt_edge_exists[gt_idxs]),
                "hallucination": 1 - np.mean(ex_edge_exists[ext_idxs]),
                "n_extracted": len(ext_idxs),
                "n_gt": len(gt_idxs),
            }
            row.update(stats)
        rows.append(row)
    return pd.DataFrame(rows).set_index("paper")
