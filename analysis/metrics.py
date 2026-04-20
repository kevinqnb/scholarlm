"""Recovery and hallucination metrics for ScholarlM extraction evaluation.

Typical usage
-------------
    from analysis.metrics import recovery_rate, hallucination_rate

    stats = recovery_rate(extraction_df, ground_truth_df, strict_matching={"entity": ["name"]})
    print(stats)  # {"recall": 0.82, "precision": 0.91, "n_extracted": 450, "n_gt": 310}
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def recovery_rate(
    extraction_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    cache_path: Path | None = None,
) -> dict:
    """Run ``match_datasets`` and return recall/precision statistics.

    Args:
        extraction_df: Extracted measurements (rows = measurements).
        ground_truth_df: Manual ground truth (rows = measurements).
        strict_matching: Exact-match column mapping passed to ``match_datasets``.
        fuzzy_matching: Fuzzy-match column mapping passed to ``match_datasets``.
        fuzzy_threshold: Minimum fuzzy score for a match.
        cache_path: Optional path for a disk-cached result (see ``cached_match``).

    Returns:
        Dict with keys ``recall``, ``precision``, ``n_extracted``, ``n_gt``.
    """
    from .loaders import cached_match

    matching, _edges, _edge_weights = cached_match(
        extraction_df,
        ground_truth_df,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        fuzzy_threshold=fuzzy_threshold,
        cache_path=cache_path,
    )
    tp = len(matching)
    n_ext = len(extraction_df)
    n_gt = len(ground_truth_df)
    return {
        "recall": tp / n_gt if n_gt > 0 else 0.0,
        "precision": tp / n_ext if n_ext > 0 else 0.0,
        "n_extracted": n_ext,
        "n_gt": n_gt,
    }


def hallucination_rate(
    judged_df: pd.DataFrame,
    label_col: str = "judgement_combined",
) -> dict:
    """Compute hallucination rate from judged extraction results.

    Args:
        judged_df: DataFrame with a boolean ``label_col`` column (``True`` = valid).
        label_col: Column name for the combined judgement label.

    Returns:
        Dict with keys ``hallucination_rate``, ``n_valid``, ``n_total``.
    """
    n_total = len(judged_df)
    n_valid = int(judged_df[label_col].sum())
    rate = 1.0 - (n_valid / n_total) if n_total > 0 else float("nan")
    return {"hallucination_rate": rate, "n_valid": n_valid, "n_total": n_total}


def per_paper_metrics(
    extraction_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    judged_df: pd.DataFrame | None = None,
    *,
    paper_col: str = "document_id",
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    label_col: str = "judgement_combined",
) -> pd.DataFrame:
    """Per-paper recovery and (optionally) hallucination summary.

    Args:
        extraction_df: Extracted measurements with a ``paper_col`` column.
        ground_truth_df: Ground truth measurements with a ``paper_col`` column.
        judged_df: Optional judged DataFrame (enables hallucination column).
        paper_col: Column identifying which paper each row belongs to.
        strict_matching: Passed to ``match_datasets``.
        fuzzy_matching: Passed to ``match_datasets``.
        fuzzy_threshold: Minimum fuzzy score for a match.
        label_col: Judgement column in ``judged_df``.

    Returns:
        DataFrame indexed by paper, with columns for recall, precision,
        n_extracted, n_gt, and (if ``judged_df`` is provided) hallucination_rate.
    """
    papers = sorted(set(extraction_df[paper_col].unique()) | set(ground_truth_df[paper_col].unique()))
    rows = []
    for paper in papers:
        ext = extraction_df[extraction_df[paper_col] == paper].reset_index(drop=True)
        gt = ground_truth_df[ground_truth_df[paper_col] == paper].reset_index(drop=True)
        if len(ext) == 0 and len(gt) == 0:
            continue
        row: dict = {"paper": paper}
        if len(ext) > 0 and len(gt) > 0:
            stats = recovery_rate(
                ext, gt,
                strict_matching=strict_matching,
                fuzzy_matching=fuzzy_matching,
                fuzzy_threshold=fuzzy_threshold,
            )
            row.update(stats)
        else:
            row.update({"recall": 0.0, "precision": 0.0, "n_extracted": len(ext), "n_gt": len(gt)})
        if judged_df is not None:
            judged_paper = judged_df[judged_df[paper_col] == paper]
            row.update(hallucination_rate(judged_paper, label_col))
        rows.append(row)
    return pd.DataFrame(rows).set_index("paper")
