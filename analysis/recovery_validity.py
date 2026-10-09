"""Recovery and validity per experiment id, with paper-clustered bootstrap CIs.

- recovery: fraction of ground-truth rows with any match-cache edge of weight >=
  the dataset's fuzzy_threshold. ``recovery_max_weight_matching`` is the stricter
  1-1 matching version, always <= recovery.
- validity: fraction of extracted rows that are matched OR judged valid by their
  judge_combine run, found through the judges' committed configs.

Reads existing match caches (never builds them) and refuses stale or mismatched
ones. CIs resample whole papers because rows within a paper are correlated. Point
estimates are cross-checked against analysis/common/metrics.py on every call.

An optional ``fuzzy_threshold_curve`` config block sweeps the threshold for chosen
ids and writes a CSV and figure per id under analysis/results/recovery-validity/<config>/.

Usage
-----
    python analysis/recovery_validity.py <id> [...] --n-resamples N --seed S \\
        --ground-truth-file <path> [--alpha 0.05] [--skip-validity] [--output PATH]
    python analysis/recovery_validity.py --config analysis/analysis-configs/recovery-validity/<id>.yaml

--config is mutually exclusive with ids and every flag; its params.recovery_validity
section requires n_resamples, alpha, compute_validity and output, plus an optional
judge_combine_ids map (still verified) and fuzzy_threshold_curve.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from analysis.common import matching, provenance
from analysis.common.config import analysis_results_dir, get_ground_truth_path, get_section, load_analysis_config
from analysis.common.recovery import (
    ext_matched_mask, filter_edges_by_threshold, gt_recovered_mask, load_checked_inputs, resolve_judged_labels,
    verify_recovery, verify_validity,
)

def count_split_rows(extraction_df: pd.DataFrame) -> int:
    """Count rows that inherited their judge label from a split parent row.

    Args:
        extraction_df: Extraction rows.

    Returns:
        Number of rows sharing (document_id, measurement_id) with another row.
    """
    return int(extraction_df.duplicated(["document_id", "measurement_id"], keep=False).sum())


# ---------------------------------------------------------------------------
# Maximum-weight 1-1 matching recovery (over threshold-filtered edges)
# ---------------------------------------------------------------------------


# Integer edge weight = round(w * 1e6) * (max matching size + 1) + 1, so total weight
# decides first and cardinality only breaks exact ties.
_WEIGHT_SCALE = 10**6


def max_weight_matching_recovered(
    n_gt: int, n_ext: int, edges: list[tuple[int, int]], edge_weights: list[float],
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Recover ground-truth rows through a maximum-weight 1-1 matching of surviving edges.

    Solved fresh on the filtered edges (the cache's own matching used all
    threshold-0 edges). Weight comes first; cardinality only breaks ties.

    Args:
        n_gt: Number of ground-truth rows.
        n_ext: Number of extraction rows.
        edges: Threshold-filtered (gt_idx, ex_idx) pairs.
        edge_weights: Weight per edge, in [0, 1].

    Returns:
        ``(recovered, matching)``: boolean mask over ground-truth rows, and the
        matching as sorted (gt_idx, ex_idx) pairs.

    Raises:
        ValueError: Length mismatch, duplicate edge, weight outside [0, 1], or an
            index out of range.
    """
    import networkx as nx

    if len(edges) != len(edge_weights):
        raise ValueError(f"{len(edges)} edges vs {len(edge_weights)} weights")
    if len(set(edges)) != len(edges):
        raise ValueError("duplicate (gt_idx, ex_idx) edges -- the graph would silently keep only one weight")
    multiplier = min(n_gt, n_ext) + 1

    G = nx.Graph()
    for (gt_idx, ex_idx), w in zip(edges, edge_weights):
        if not (0 <= gt_idx < n_gt and 0 <= ex_idx < n_ext):
            raise ValueError(f"edge ({gt_idx}, {ex_idx}) out of range for n_gt={n_gt}, n_ext={n_ext}")
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"edge ({gt_idx}, {ex_idx}) has weight {w!r} outside [0, 1]")
        # Namespaced nodes: bare ints would merge gt row 0 with extraction row 0.
        G.add_edge(f"L_{gt_idx}", f"R_{ex_idx}", weight=round(w * _WEIGHT_SCALE) * multiplier + 1)

    matching: list[tuple[int, int]] = []
    for u, v in nx.max_weight_matching(G, maxcardinality=False):
        left, right = (u, v) if u.startswith("L_") else (v, u)
        matching.append((int(left[2:]), int(right[2:])))
    matching.sort()

    recovered = np.zeros(n_gt, dtype=bool)
    for gt_idx, _ex_idx in matching:
        recovered[gt_idx] = True
    return recovered, matching


def _verify_matching(
    matching: list[tuple[int, int]],
    edges: list[tuple[int, int]],
    recovered: np.ndarray,
    any_edge_recovered: np.ndarray,
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
) -> None:
    """Check the matching is 1-1, drawn from ``edges``, and never crosses papers.

    No cross-paper edges is what makes resampling papers after one global matching valid.

    Args:
        matching: (gt_idx, ex_idx) pairs from ``max_weight_matching_recovered``.
        edges: Threshold-filtered edges.
        recovered: Matching-based recovered mask.
        any_edge_recovered: Any-edge recovered mask (must be a superset).
        ground_truth_df: Ground-truth frame.
        extraction_df: Extraction frame.

    Raises:
        AssertionError: Any check fails.
    """
    gts = [g for g, _ in matching]
    exs = [e for _, e in matching]
    if len(set(gts)) != len(gts) or len(set(exs)) != len(exs):
        raise AssertionError("matching uses a ground truth or extraction row more than once")
    edge_set = set(edges)
    not_edges = [m for m in matching if m not in edge_set]
    if not_edges:
        raise AssertionError(f"matching contains pair(s) not among the filtered edges: {not_edges[:5]}")
    if int(recovered.sum()) != len(matching):
        raise AssertionError(f"recovered count {int(recovered.sum())} != matching size {len(matching)}")
    if (recovered & ~any_edge_recovered).any():
        raise AssertionError("matching recovers a ground truth row with no surviving edge")

    gt_docs = ground_truth_df["document_id"].astype(str).to_numpy()
    ex_docs = extraction_df["document_id"].astype(str).to_numpy()
    cross = [(g, e) for g, e in edges if gt_docs[g] != ex_docs[e]]
    if cross:
        raise AssertionError(
            f"{len(cross)} edge(s) join different document_ids (first: {cross[0]}) -- "
            f"the matching no longer decomposes per paper, so the paper-clustered "
            f"bootstrap over it is not valid"
        )


# ---------------------------------------------------------------------------
# Paper-clustered percentile bootstrap
# ---------------------------------------------------------------------------


def bootstrap_cluster_rate(
    labels: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap CI of a proportion, resampling whole clusters (papers).

    Args:
        labels: Boolean per row.
        clusters: Cluster id per row.
        n_resamples: Number of resamples.
        seed: RNG seed.
        alpha: CI level; returns the alpha/2 and 1 - alpha/2 percentiles.

    Returns:
        ``(point, ci_lo, ci_hi)``; ``point`` is the plain rate over all rows.

    Raises:
        ValueError: Length mismatch or zero rows.
    """
    labels = np.asarray(labels, dtype=bool)
    clusters = np.asarray(clusters)
    if len(labels) != len(clusters):
        raise ValueError(
            f"labels ({len(labels)}) and clusters ({len(clusters)}) length mismatch"
        )
    if len(labels) == 0:
        raise ValueError("cannot compute a rate over zero rows")

    point = float(labels.mean())

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    n_true_by_cluster = np.array([
        int(labels[clusters == c].sum()) for c in unique_clusters
    ])
    n_total_by_cluster = np.array([
        int((clusters == c).sum()) for c in unique_clusters
    ])

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    boot_true = n_true_by_cluster[draws].sum(axis=1)
    boot_total = n_total_by_cluster[draws].sum(axis=1)
    boot_rates = boot_true / boot_total

    lo, hi = np.percentile(boot_rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-id orchestration
# ---------------------------------------------------------------------------


def compute_metrics_for_id(
    experiment_id: str,
    *,
    ground_truth_path: Path,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
    compute_validity: bool = True,
    judge_combine_id: str | None = None,
) -> dict:
    """Recovery and (optionally) validity with paper-clustered CIs for one experiment.

    Args:
        experiment_id: Extraction experiment id.
        ground_truth_path: Ground-truth file (must match the cache's sidecar).
        n_resamples: Bootstrap resamples.
        seed: Bootstrap seed (shared by all three metrics).
        alpha: CI level.
        compute_validity: If False, skip judge resolution; validity fields are None.
        judge_combine_id: Declared combine id (verified), or None to search.

    Returns:
        Output row dict: ids, input files, counts, threshold, ``recovery``,
        ``recovery_max_weight_matching`` and ``validity`` with ``_ci_lo`` / ``_ci_hi``,
        judge ids, and ``n_split_rows_inheriting_judgement``.

    Raises:
        FileNotFoundError, KeyError, RuntimeError, ValueError, AssertionError: A
            missing or mismatched cache, judge resolution failure, or disagreement
            with analysis/common/metrics.py.
    """
    inputs = load_checked_inputs(experiment_id, ground_truth_path)
    dataset = inputs["dataset"]
    ground_truth_df, extraction_df = inputs["ground_truth_df"], inputs["extraction_df"]
    extraction_file_path, ground_truth_path = inputs["extraction_file_path"], inputs["ground_truth_path"]
    cfg, cache_path = inputs["cfg"], inputs["cache_path"]
    raw_edges, raw_weights = inputs["raw_edges"], inputs["raw_weights"]
    threshold = cfg["fuzzy_threshold"]
    n_gt, n_ext = len(ground_truth_df), len(extraction_df)

    # Dataset-threshold edges: the only edges recovery and validity ever use.
    threshold_edges, threshold_weights = filter_edges_by_threshold(raw_edges, raw_weights, threshold)
    matched = ext_matched_mask(n_ext, threshold_edges)
    recovered = gt_recovered_mask(n_gt, threshold_edges)
    verify_recovery(ground_truth_df, extraction_df, cfg, threshold, cache_path, recovered)

    recovered_mwm, matching = max_weight_matching_recovered(n_gt, n_ext, threshold_edges, threshold_weights)
    _verify_matching(matching, threshold_edges, recovered_mwm, recovered, ground_truth_df, extraction_df)

    judge_ids = None
    judged_labels = None
    if compute_validity:
        judge_combine_id, judge_ids, judged_labels = resolve_judged_labels(
            dataset, experiment_id, judge_combine_id, extraction_df,
        )

    gt_docs = ground_truth_df["document_id"].to_numpy()
    recovery_point, recovery_lo, recovery_hi = bootstrap_cluster_rate(
        recovered, gt_docs, n_resamples=n_resamples, seed=seed, alpha=alpha,
    )
    mwm_point, mwm_lo, mwm_hi = bootstrap_cluster_rate(
        recovered_mwm, gt_docs, n_resamples=n_resamples, seed=seed, alpha=alpha,
    )

    row = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "ground_truth_file": provenance.repo_relative(ground_truth_path),
        "extraction_file": provenance.repo_relative(extraction_file_path),
        "n_gt": n_gt,
        "n_ext": n_ext,
        "fuzzy_threshold": threshold,
        "n_surviving_edges": len(threshold_edges),
        "n_resamples": n_resamples,
        "seed": seed,
        "bootstrap_unit": "paper",
        "recovery": recovery_point,
        "recovery_ci_lo": recovery_lo,
        "recovery_ci_hi": recovery_hi,
        "recovery_max_weight_matching": mwm_point,
        "recovery_max_weight_matching_ci_lo": mwm_lo,
        "recovery_max_weight_matching_ci_hi": mwm_hi,
        "judge_combine_id": None,
        "judge_ids": None,
        "n_split_rows_inheriting_judgement": None,
        "validity": None,
        "validity_ci_lo": None,
        "validity_ci_hi": None,
    }

    if not compute_validity:
        return row

    validity_labels = verify_validity(
        ground_truth_df, extraction_df, cfg, threshold, cache_path, matched, judged_labels,
    )

    validity_point, validity_lo, validity_hi = bootstrap_cluster_rate(
        validity_labels, extraction_df["document_id"].to_numpy(),
        n_resamples=n_resamples, seed=seed, alpha=alpha,
    )

    row.update({
        "judge_combine_id": judge_combine_id,
        "judge_ids": ";".join(judge_ids),
        "n_split_rows_inheriting_judgement": count_split_rows(extraction_df),
        "validity": validity_point,
        "validity_ci_lo": validity_lo,
        "validity_ci_hi": validity_hi,
    })
    return row


# ---------------------------------------------------------------------------
# Validity/recovery operating curves for score thresholds (used by other scripts).
# Callers must pass threshold-filtered edges indexed like ``probs``; not checked here.
# ---------------------------------------------------------------------------


def validity_recovery_curve(probs, labels, n_ground_truth, edges, thresholds):
    """Validity and recovery of the rows kept by ``probs > t``, for each threshold t.

    Args:
        probs: Score per extraction row.
        labels: Boolean validity label per extraction row.
        n_ground_truth: Number of ground-truth rows.
        edges: Threshold-filtered (gt_idx, ex_idx) pairs, ex_idx indexing ``probs``.
        thresholds: Score thresholds.

    Returns:
        ``(validity, recovery, thresholds)`` arrays, skipping thresholds that keep no rows.
    """
    from analysis.common.metrics import recovery_rate_from_labels, validity_rate_from_labels

    probs = np.asarray(probs)
    labels = np.asarray(labels, dtype=bool)
    v, r, ts = [], [], []
    for t in thresholds:
        preds = probs > t
        if preds.sum() == 0:
            continue
        v.append(validity_rate_from_labels(labels, preds))
        r.append(recovery_rate_from_labels(n_ground_truth, edges, preds))
        ts.append(t)
    return np.array(v), np.array(r), np.array(ts)


def plot_validity_recovery(curves, labels, n_ground_truth, edges, out_path, *, thresholds, n_random, seed):
    """Plot validity (y) vs recovery (x) curves plus a random-score baseline.

    Points are coloured by threshold and the point nearest 0.5 is ringed. The dotted
    baseline averages ``n_random`` uniform random score draws.

    Args:
        curves: List of (probs, linestyle), drawn in order (later on top).
        labels: Boolean validity label per extraction row.
        n_ground_truth: Number of ground-truth rows.
        edges: Threshold-filtered (gt_idx, ex_idx) pairs.
        out_path: Figure path.
        thresholds: Score thresholds.
        n_random: Number of random baseline draws.
        seed: Seed for the baseline.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from analysis.common.metrics import recovery_rate_from_labels, validity_rate_from_labels

    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    labels = np.asarray(labels, dtype=bool)
    thresholds = np.asarray(thresholds)

    fig, ax = plt.subplots(figsize=(4.0, 3.8))

    def plot_vr_curve(probs, linestyle, zorder_base):
        """Draw one curve: grey line, ~10 threshold-coloured points, ring at t=0.5."""
        v, r, ts = validity_recovery_curve(probs, labels, n_ground_truth, edges, thresholds)
        if len(ts) == 0:
            return
        ax.plot(r, v, linestyle, color='grey', lw=3.0, zorder=zorder_base)
        n = len(ts)
        stride = max(1, n // 10)
        idx = sorted({0, n - 1} | set(range(0, n, stride)))
        ax.scatter(r[idx], v[idx], c=ts[idx], cmap=cmap, norm=norm, s=45, zorder=zorder_base + 1)
        idx0 = int(np.argmin(np.abs(ts - 0.5)))
        ax.scatter([r[idx0]], [v[idx0]], s=60, c='none',
                   edgecolors='k', linewidths=1.1, zorder=zorder_base + 2, marker='o')

    # Each curve sits 3 above the previous, so the last-listed is on top.
    for i, (probs, linestyle) in enumerate(curves):
        plot_vr_curve(np.asarray(probs), linestyle, zorder_base=3 + 3 * i)

    # Random baseline: average validity/recovery over repeated uniform draws.
    rng = np.random.default_rng(seed)
    n_items = len(labels)
    rand_v = np.full((n_random, len(thresholds)), np.nan)
    rand_r = np.full((n_random, len(thresholds)), np.nan)
    for i in range(n_random):
        rand_probs_i = rng.uniform(0, 1, n_items)
        for j, t in enumerate(thresholds):
            preds = rand_probs_i > t
            if preds.sum() > 0:
                rand_v[i, j] = validity_rate_from_labels(labels, preds)
                rand_r[i, j] = recovery_rate_from_labels(n_ground_truth, edges, preds)
    avg_v = np.nanmean(rand_v, axis=0)
    avg_r = np.nanmean(rand_r, axis=0)
    valid_rand = ~(np.isnan(avg_v) | np.isnan(avg_r))
    if valid_rand.any():
        ax.plot(avg_r[valid_rand], avg_v[valid_rand], ':', color='grey', lw=2.0, zorder=2)

    ax.set_xlim(left=-0.02)
    ax.set_ylim(top=1.02)
    ax.set_xlabel('Recovery')
    ax.set_ylabel('Validity')
    ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def save_validity_recovery_colorbar(out_path):
    """Save the shared 0..1 threshold colorbar for ``plot_validity_recovery`` figures.

    Args:
        out_path: Figure path.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=mcolors.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(0.35, 3.2))
    plt.colorbar(sm, cax=ax, label='Threshold')
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def save_validity_recovery_legend(out_path):
    """Save the standalone Probe (solid) / NTP (dashed) / Random (dotted) legend.

    Args:
        out_path: Figure path.
    """
    import matplotlib.lines as mlines
    import matplotlib.pyplot as plt

    handles = [
        mlines.Line2D([], [], color='grey', lw=2, linestyle='-', label='Probe'),
        mlines.Line2D([], [], color='grey', lw=2, linestyle='--', label='NTP'),
        mlines.Line2D([], [], color='grey', lw=2, linestyle=':', label='Random'),
    ]
    fig, ax = plt.subplots(figsize=(4.0, 0.35))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=3, fontsize=13, frameon=False, handlelength=2.0)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Fuzzy-threshold sweep: recovery/validity as the matching threshold varies
# ---------------------------------------------------------------------------


def _validate_threshold_grid(thresholds, dataset_threshold: float) -> list[float]:
    """Validate a fuzzy-threshold sweep grid.

    Must include the dataset threshold exactly, so the sweep can be checked against
    the headline row. Write values as explicit decimals: a generated 0.7000000000000001
    would drop edges scoring exactly 0.7.

    Args:
        thresholds: Candidate grid.
        dataset_threshold: The dataset's fuzzy_threshold.

    Returns:
        The grid as floats.

    Raises:
        ValueError: Empty, non-numeric, outside [0, 1], not strictly increasing, or
            missing the dataset threshold.
    """
    if not isinstance(thresholds, list) or not thresholds or not all(
        isinstance(t, (int, float)) and not isinstance(t, bool) for t in thresholds
    ):
        raise ValueError(f"thresholds must be a non-empty list of numbers, got {thresholds!r}")
    thresholds = [float(t) for t in thresholds]
    if any(not (0.0 <= t <= 1.0) for t in thresholds):
        raise ValueError(f"thresholds must all lie in [0, 1], got {thresholds}")
    if any(b <= a for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError(f"thresholds must be strictly increasing, got {thresholds}")
    if dataset_threshold not in thresholds:
        raise ValueError(
            f"thresholds must include the dataset's own fuzzy_threshold "
            f"{dataset_threshold!r} exactly (the known-answer point checked against "
            f"compute_metrics_for_id), got {thresholds}"
        )
    return thresholds


def fuzzy_threshold_curve(
    experiment_id: str,
    *,
    ground_truth_path: Path,
    thresholds: list[float],
    judge_combine_id: str | None = None,
) -> pd.DataFrame:
    """Recovery and validity point estimates at each fuzzy threshold (no CIs).

    Same guards, judge resolution and metrics.py cross-checks as
    ``compute_metrics_for_id``.

    Args:
        experiment_id: Extraction experiment id.
        ground_truth_path: Ground-truth file.
        thresholds: Grid accepted by ``_validate_threshold_grid``.
        judge_combine_id: Declared combine id, or None to search.

    Returns:
        One row per threshold: ids, ``fuzzy_threshold``, ``is_dataset_threshold``,
        counts, ``recovery``, ``validity``.

    Raises:
        AssertionError: A metric increases with the threshold (impossible).
    """
    inputs = load_checked_inputs(experiment_id, ground_truth_path)
    dataset = inputs["dataset"]
    ground_truth_df, extraction_df = inputs["ground_truth_df"], inputs["extraction_df"]
    cfg, cache_path = inputs["cfg"], inputs["cache_path"]
    raw_edges, raw_weights = inputs["raw_edges"], inputs["raw_weights"]
    n_gt, n_ext = len(ground_truth_df), len(extraction_df)

    thresholds = _validate_threshold_grid(thresholds, cfg["fuzzy_threshold"])
    judge_combine_id, _judge_ids, judged_labels = resolve_judged_labels(
        dataset, experiment_id, judge_combine_id, extraction_df,
    )

    rows = []
    for t in thresholds:
        edges, _weights = filter_edges_by_threshold(raw_edges, raw_weights, t)
        recovered = gt_recovered_mask(n_gt, edges)
        verify_recovery(ground_truth_df, extraction_df, cfg, t, cache_path, recovered)
        matched = ext_matched_mask(n_ext, edges)
        validity_labels = verify_validity(
            ground_truth_df, extraction_df, cfg, t, cache_path, matched, judged_labels,
        )
        rows.append({
            "experiment_id": experiment_id,
            "dataset": dataset,
            "judge_combine_id": judge_combine_id,
            "fuzzy_threshold": t,
            "is_dataset_threshold": t == cfg["fuzzy_threshold"],
            "n_gt": n_gt,
            "n_ext": n_ext,
            "n_surviving_edges": len(edges),
            "recovery": float(recovered.mean()),
            "validity": float(validity_labels.mean()),
        })
    df = pd.DataFrame(rows)

    for col in ("recovery", "validity"):
        if (np.diff(df[col].to_numpy()) > 0).any():
            raise AssertionError(
                f"{experiment_id}: {col} increases with fuzzy threshold -- impossible, "
                f"raising the threshold only removes edges: {df[col].tolist()}"
            )
    return df


def _assert_curve_matches_row(curve: pd.DataFrame, row: dict) -> None:
    """Known-answer check: the sweep at the dataset threshold equals the headline row.

    Args:
        curve: Output of ``fuzzy_threshold_curve``.
        row: Output of ``compute_metrics_for_id`` for the same id.

    Raises:
        AssertionError: The two disagree, or there isn't exactly one dataset-threshold point.
    """
    at = curve[curve["is_dataset_threshold"]]
    if len(at) != 1:
        raise AssertionError(f"{row['experiment_id']}: {len(at)} curve rows at the dataset threshold, expected 1")
    at = at.iloc[0]
    for col in ("recovery", "validity", "n_surviving_edges", "fuzzy_threshold", "judge_combine_id"):
        if at[col] != row[col]:
            raise AssertionError(
                f"{row['experiment_id']}: fuzzy-threshold curve {col}={at[col]!r} at the "
                f"dataset threshold disagrees with compute_metrics_for_id's {row[col]!r}"
            )


_FUZZY_THRESHOLD_NORM = (0.0, 1.0)


def plot_fuzzy_threshold_curve(curve: pd.DataFrame, out_path: Path) -> None:
    """Plot validity (y) vs recovery (x) across the sweep, coloured by threshold.

    The dataset threshold is a diamond; others are circles.

    Args:
        curve: Output of ``fuzzy_threshold_curve``.
        out_path: Figure path.

    Raises:
        ValueError: Not exactly one dataset-threshold row.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    is_selected = curve["is_dataset_threshold"].to_numpy(dtype=bool)
    if is_selected.sum() != 1:
        raise ValueError(f"expected exactly one is_dataset_threshold row, got {int(is_selected.sum())}")

    norm = mcolors.Normalize(*_FUZZY_THRESHOLD_NORM)
    fig, ax = plt.subplots(figsize=(4.0, 3.8))
    ax.plot(curve["recovery"], curve["validity"], "-", color="grey", lw=3.0, zorder=2)
    ax.scatter(
        curve["recovery"][~is_selected], curve["validity"][~is_selected], c=curve["fuzzy_threshold"][~is_selected],
        cmap=plt.cm.coolwarm, norm=norm, marker="o", s=45, zorder=3,
    )
    # Dark edge: dataset thresholds sit near coolwarm's white midpoint.
    ax.scatter(
        curve["recovery"][is_selected], curve["validity"][is_selected], c=curve["fuzzy_threshold"][is_selected],
        cmap=plt.cm.coolwarm, norm=norm, marker="D", s=60, zorder=4, edgecolors="k", linewidths=0.9,
    )
    ax.set_xlabel("Recovery")
    ax.set_ylabel("Validity")
    ax.grid(alpha=0.25, linestyle="-", linewidth=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_fuzzy_threshold_colorbar(out_path: Path) -> None:
    """Save the 0..1 "Fuzzy Threshold" colorbar for ``plot_fuzzy_threshold_curve`` figures.

    Args:
        out_path: Figure path.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=mcolors.Normalize(*_FUZZY_THRESHOLD_NORM))
    sm.set_array([])
    fig, ax = plt.subplots(figsize=(0.35, 3.2))
    plt.colorbar(sm, cax=ax, label="Fuzzy Threshold")
    fig.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def fuzzy_threshold_figures_dir(analysis_config_id: str) -> Path:
    """Figure directory for a recovery-validity config.

    Args:
        analysis_config_id: Config id.

    Returns:
        ``analysis/results/recovery-validity/<config id>/figures``.
    """
    return analysis_results_dir("recovery-validity") / analysis_config_id / "figures"


def _parse_fuzzy_threshold_curve_section(section, experiment_ids: list[str], compute_validity: bool, where: str) -> dict:
    """Validate the ``fuzzy_threshold_curve`` config block.

    Args:
        section: The block: exactly ``experiment_ids`` and ``thresholds``.
        experiment_ids: The config's experiment ids (curve ids must be a subset).
        compute_validity: Must be True, since the curve plots validity.
        where: Config path (for error messages).

    Returns:
        ``{"experiment_ids", "thresholds"}``; thresholds are checked later per dataset.

    Raises:
        ValueError: Any check fails.
    """
    if not isinstance(section, dict) or set(section) != {"experiment_ids", "thresholds"}:
        raise ValueError(
            f"{where}: params.recovery_validity.fuzzy_threshold_curve must be a mapping with "
            f"exactly the keys experiment_ids and thresholds, got {section!r}"
        )
    if not compute_validity:
        raise ValueError(
            f"{where}: params.recovery_validity.fuzzy_threshold_curve is set but compute_validity "
            f"is false -- the curve plots validity, which needs judge_combine coverage"
        )
    curve_ids = section["experiment_ids"]
    if not isinstance(curve_ids, list) or not curve_ids or not all(isinstance(x, str) for x in curve_ids):
        raise ValueError(f"{where}: fuzzy_threshold_curve.experiment_ids must be a non-empty list of strings")
    if len(set(curve_ids)) != len(curve_ids):
        raise ValueError(f"{where}: fuzzy_threshold_curve.experiment_ids has duplicates: {curve_ids}")
    unknown = set(curve_ids) - set(experiment_ids)
    if unknown:
        raise ValueError(
            f"{where}: fuzzy_threshold_curve.experiment_ids not in params.experiment_ids: {sorted(unknown)}"
        )
    return {"experiment_ids": curve_ids, "thresholds": section["thresholds"]}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """CLI argument parser (``--help`` shows the module docstring)."""
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment_ids", nargs="*", help="Extraction/ablation/baseline experiment ids to score.")
    p.add_argument(
        "--config", type=Path, default=None,
        help="analysis-configs/recovery-validity/<id>.yaml providing params.experiment_ids, "
             "params.ground_truth_file, and params.recovery_validity -- mutually "
             "exclusive with experiment_ids and every flag below.",
    )
    p.add_argument(
        "--ground-truth-file", type=Path, default=None,
        help="Ground-truth CSV/JSON experiment_ids are scored against -- must match "
             "the file the corresponding match_cache.py run used. Required unless "
             "--config is given.",
    )
    p.add_argument(
        "--n-resamples", type=int, default=None,
        help="Number of paper-cluster bootstrap resamples. Required unless --config is given.",
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help="Bootstrap RNG seed. Required unless --config is given.",
    )
    p.add_argument(
        "--alpha", type=float, default=None,
        help="CI significance level (default: 0.05, a 95%% interval; ignored with --config, "
             "which requires it explicit in params.recovery_validity.alpha).",
    )
    p.add_argument(
        "--skip-validity", action="store_true",
        help="Report recovery only -- skip judge_combine resolution and validity entirely "
             "(for an id with no judge coverage yet). Ignored with --config, which requires "
             "params.recovery_validity.compute_validity explicit instead.",
    )
    p.add_argument(
        "--output", type=Path, default=None,
        help="CSV path to write (default: analysis/results/recovery-validity/recovery_validity.csv, matching "
             "analysis/ablation.py's/baselines.py's own results/ output convention; "
             "ignored with --config, which requires params.recovery_validity.output explicit).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    """CLI: score every id, write the CSV, and run any configured threshold sweeps.

    Args:
        argv: Arguments (None = sys.argv).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    cli_flags_given = (
        args.experiment_ids or args.n_resamples is not None or args.seed is not None
        or args.alpha is not None or args.skip_validity or args.output is not None
        or args.ground_truth_file is not None
    )
    if args.config and cli_flags_given:
        parser.error("--config is mutually exclusive with experiment_ids and every other flag")
    if not args.config and not (
        args.experiment_ids and args.n_resamples is not None and args.seed is not None
        and args.ground_truth_file is not None
    ):
        parser.error(
            "experiment_ids, --n-resamples, --seed and --ground-truth-file are "
            "required unless --config is given"
        )

    judge_combine_overrides: dict[str, str] = {}
    if args.config:
        cfg = load_analysis_config(args.config, "recovery-validity")
        experiment_ids = cfg["params"]["experiment_ids"]
        ground_truth_path = get_ground_truth_path(cfg)
        # Unknown keys (e.g. the removed edge_filter) fail here.
        section = get_section(
            cfg, "recovery_validity",
            required_keys=("n_resamples", "alpha", "compute_validity", "output"),
            optional_keys=("judge_combine_ids", "fuzzy_threshold_curve"),
        )
        n_resamples = section["n_resamples"]
        seed = cfg["seed"]
        alpha = section["alpha"]
        compute_validity = section["compute_validity"]
        if not isinstance(compute_validity, bool):
            raise ValueError(
                f"{args.config}: params.recovery_validity.compute_validity must be a "
                f"bool, got {compute_validity!r}"
            )
        output = Path(section["output"])
        if not output.is_absolute():
            output = _REPO_ROOT / output

        judge_combine_overrides = section.get("judge_combine_ids", {})
        if not isinstance(judge_combine_overrides, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in judge_combine_overrides.items()
        ):
            raise ValueError(
                f"{args.config}: params.recovery_validity.judge_combine_ids must be a "
                f"string-to-string mapping, got {judge_combine_overrides!r}"
            )
        if judge_combine_overrides and not compute_validity:
            raise ValueError(
                f"{args.config}: params.recovery_validity.judge_combine_ids is set but "
                f"compute_validity is false -- it would never be used"
            )
        unknown_overrides = set(judge_combine_overrides) - set(experiment_ids)
        if unknown_overrides:
            raise ValueError(
                f"{args.config}: params.recovery_validity.judge_combine_ids has key(s) "
                f"not in params.experiment_ids: {sorted(unknown_overrides)}"
            )
        curve_section = None
        if "fuzzy_threshold_curve" in section:
            curve_section = _parse_fuzzy_threshold_curve_section(
                section["fuzzy_threshold_curve"], experiment_ids, compute_validity, str(args.config),
            )
        analysis_config_id = cfg["id"]
    else:
        curve_section = None
        experiment_ids = args.experiment_ids
        ground_truth_path = args.ground_truth_file
        if not ground_truth_path.is_absolute():
            ground_truth_path = _REPO_ROOT / ground_truth_path
        if not ground_truth_path.exists():
            parser.error(f"--ground-truth-file {ground_truth_path} does not exist")
        n_resamples = args.n_resamples
        seed = args.seed
        alpha = args.alpha if args.alpha is not None else 0.05
        compute_validity = not args.skip_validity
        output = args.output if args.output is not None else analysis_results_dir("recovery-validity") / "recovery_validity.csv"
        analysis_config_id = None

    rows = []
    for experiment_id in experiment_ids:
        print(f"Processing {experiment_id} ...")
        row = compute_metrics_for_id(
            experiment_id, ground_truth_path=ground_truth_path,
            n_resamples=n_resamples, seed=seed, alpha=alpha,
            compute_validity=compute_validity,
            judge_combine_id=judge_combine_overrides.get(experiment_id),
        )
        row["analysis_config_id"] = analysis_config_id
        rows.append(row)
        validity_str = (
            f"validity={row['validity']:.3f} [{row['validity_ci_lo']:.3f}, {row['validity_ci_hi']:.3f}] "
            f"(judge_combine={row['judge_combine_id']})"
            if row["validity"] is not None else "validity=skipped"
        )
        print(
            f"  recovery={row['recovery']:.3f} "
            f"[{row['recovery_ci_lo']:.3f}, {row['recovery_ci_hi']:.3f}]  "
            f"recovery_max_weight_matching={row['recovery_max_weight_matching']:.3f}  {validity_str}"
        )

    df = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    print(f"\nWrote {len(df)} row(s) to {output}")

    if curve_section is None:
        return
    figures_dir = fuzzy_threshold_figures_dir(analysis_config_id)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows_by_id = {row["experiment_id"]: row for row in rows}
    for experiment_id in curve_section["experiment_ids"]:
        print(f"Fuzzy-threshold curve for {experiment_id} ...")
        curve = fuzzy_threshold_curve(
            experiment_id, ground_truth_path=ground_truth_path,
            thresholds=curve_section["thresholds"],
            judge_combine_id=judge_combine_overrides.get(experiment_id),
        )
        _assert_curve_matches_row(curve, rows_by_id[experiment_id])
        curve["analysis_config_id"] = analysis_config_id
        curve.to_csv(figures_dir.parent / f"{experiment_id}-fuzzy-threshold-curve.csv", index=False)
        plot_fuzzy_threshold_curve(curve, figures_dir / f"{experiment_id}.pdf")
        print(f"  wrote {figures_dir / f'{experiment_id}.pdf'}")
    save_fuzzy_threshold_colorbar(figures_dir / "fuzzy_threshold_colorbar.pdf")
    print(f"  wrote {figures_dir / 'fuzzy_threshold_colorbar.pdf'}")


if __name__ == "__main__":
    main()
