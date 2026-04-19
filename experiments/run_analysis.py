"""
Analysis runner for the ScholarlM experiment framework.

Provides three analyses over the structured results from run_extraction.py
and run_judge.py:

  probe-heatmap     — For each (extraction_model, local_judge_model) pair,
                      train a mechanistic probe on JudgementLM activations
                      and record accuracy.  Output: CSV heatmap.

  calibration       — For a given extraction model, compute calibration metrics
                      (ECE, reliability diagram data) for every available judge
                      model.  Output: CSV + per-judge NPZ files.

  cross-dataset     — For each (judge_model, train_dataset, test_dataset) triple,
                      train a probe on the training set and evaluate on the test
                      set.  Output: CSV cross-dataset accuracy matrix.

All outputs land in:
  data/experiments/{dataset}/analysis/         (probe-heatmap, calibration)
  data/experiments/cross_dataset/              (cross-dataset)

Usage
-----
    python experiments/run_analysis.py probe-heatmap \\
        --dataset pond \\
        --extraction-models gemma-3-27b qwen-2.5-72b \\
        --judge-models llama-3.1-8b qwen-3-8b

    python experiments/run_analysis.py calibration \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge-models llama-3.1-8b openai anthropic

    python experiments/run_analysis.py cross-dataset \\
        --judge-model llama-3.1-8b \\
        --datasets pond nfix \\
        --extraction-model gemma-3-27b
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd

from scholarlm.utils.probe import build_feature_matrix, train_probe, eval_probe
from scholarlm.utils.calibration import compute_ece, reliability_diagram_data
from analysis.cross_dataset import (
    cross_dataset_probe_matrix,
    load_activations_and_labels,
)

_EXPERIMENTS_ROOT = _REPO_ROOT / "data" / "experiments"
import paths

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_most_recent(directory: Path, filename: str) -> Path | None:
    """Return the most recent ``filename`` under any date-stamped subdirectory."""
    if not directory.exists():
        return None
    for date_dir in sorted(directory.iterdir(), reverse=True):
        candidate = date_dir / filename
        if candidate.exists():
            return candidate
    return None


def _load_combined(dataset: str, extraction_model: str, extraction_date: str) -> list[dict]:
    """Load the combined judge results for a (dataset, extraction_model, extraction_date) triple."""
    path = paths.find_combined(dataset, extraction_model, extraction_date)
    with open(path) as f:
        return json.load(f)


def _get_judge_response_file(
    dataset: str, extraction_model: str, extraction_date: str, judge_model: str
) -> Path | None:
    """Find the most recent responses.json for a judge model."""
    judge_dir = paths.judge_base(dataset, extraction_model, extraction_date) / judge_model
    return _find_most_recent(judge_dir, "responses.json")


def _get_activations_file(
    dataset: str, extraction_model: str, extraction_date: str, judge_model: str
) -> Path | None:
    """Find the most recent attention_outputs.npz for a judge model."""
    judge_dir = paths.judge_base(dataset, extraction_model, extraction_date) / judge_model
    return _find_most_recent(judge_dir, "attention_outputs.npz")


# ---------------------------------------------------------------------------
# Analysis 1: Probe heatmap
# ---------------------------------------------------------------------------


def run_probe_heatmap(
    dataset: str,
    extraction_models: list[str],
    extraction_dates: list[str],
    judge_models: list[str],
) -> pd.DataFrame:
    """Train probes for each (extraction_model, judge_model) pair.

    For each pair:
    1. Loads activations (``attention_outputs.npz``) and ground truth labels
       (``combined.json``) for the extraction model.
    2. Splits into 80/20 train/test.
    3. Trains a logistic-regression probe.
    4. Records test accuracy.

    Args:
        dataset: Dataset name.
        extraction_models: List of extraction model short names.
        extraction_dates: Extraction date tags (``YYYY_mm_dd``), one per model.
        judge_models: List of local judge model short names.

    Returns:
        DataFrame with extraction models as rows and judge models as columns,
        values are probe accuracy.  Saved to
        ``data/experiments/{dataset}/analysis/probe_heatmap.csv``.
    """
    if len(extraction_dates) != len(extraction_models):
        raise ValueError("extraction_dates must have the same length as extraction_models")
    from sklearn.model_selection import train_test_split

    results: dict[str, dict[str, float]] = {em: {} for em in extraction_models}

    for extraction_model, extraction_date in zip(extraction_models, extraction_dates):
        combined = _load_combined(dataset, extraction_model, extraction_date)
        ground_truth: dict[int, bool] = {
            r["measurement_id"]: r["judgement_combined"] for r in combined
        }

        for judge_model in judge_models:
            activations_path = _get_activations_file(dataset, extraction_model, extraction_date, judge_model)
            if activations_path is None:
                print(f"  Skipping ({extraction_model}, {judge_model}): no activations found.")
                results[extraction_model][judge_model] = float("nan")
                continue

            activations = np.load(activations_path)
            measurement_ids = [
                int(k) for k in activations.files if int(k) in ground_truth
            ]
            if not measurement_ids:
                results[extraction_model][judge_model] = float("nan")
                continue

            labels = np.array([ground_truth[mid] for mid in measurement_ids], dtype=bool)
            X = build_feature_matrix(activations, measurement_ids)

            if len(X) < 10:
                results[extraction_model][judge_model] = float("nan")
                continue

            X_train, X_test, y_train, y_test = train_test_split(
                X, labels, test_size=0.2, random_state=42, stratify=labels
            )
            probe = train_probe(X_train, y_train)
            result = eval_probe(probe, X_test, y_test)
            acc = result["accuracy"]
            results[extraction_model][judge_model] = acc
            print(f"  ({extraction_model}, {judge_model}): accuracy={acc:.4f}")

    df = pd.DataFrame(results).T  # rows=extraction_models, cols=judge_models
    df.index.name = "extraction_model"

    out_dir = _EXPERIMENTS_ROOT / dataset / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "probe_heatmap.csv"
    df.to_csv(out_path)
    print(f"\nProbe heatmap saved to {out_path}")
    return df


# ---------------------------------------------------------------------------
# Analysis 2: Calibration
# ---------------------------------------------------------------------------


def run_calibration(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    judge_models: list[str],
) -> dict[str, dict]:
    """Compute calibration metrics for each judge model.

    For local judge models: uses ``judgement_p_true`` from ``responses.json``
    as the predicted probability.  For frontier judges: uses
    ``judgement_prob`` (the raw API probability if available).

    Ground truth labels come from ``combined.json`` (``judgement_combined``).

    Args:
        dataset: Dataset name.
        extraction_model: Extraction model short name.
        extraction_date: Extraction date tag (``YYYY_mm_dd``).
        judge_models: List of judge model keys.

    Returns:
        Dict mapping judge model key → reliability diagram data dict.
        Summary ECE values are saved to
        ``data/experiments/{dataset}/analysis/calibration_{extraction_model}.csv``.
    """
    combined = _load_combined(dataset, extraction_model, extraction_date)
    ground_truth: dict[int, bool] = {
        r["measurement_id"]: r["judgement_combined"] for r in combined
    }

    all_results: dict[str, dict] = {}
    summary_rows: list[dict] = []

    for judge_model in judge_models:
        responses_path = _get_judge_response_file(dataset, extraction_model, extraction_date, judge_model)
        if responses_path is None:
            print(f"  Skipping {judge_model}: no responses.json found.")
            continue

        with open(responses_path) as f:
            responses: list[dict] = json.load(f)

        probs: list[float] = []
        labels: list[bool] = []

        for r in responses:
            mid = r.get("measurement_id")
            if mid not in ground_truth:
                continue
            # Prefer p_true (local judges); fall back to prob (frontier judges)
            p = r.get("judgement_p_true") or r.get("judgement_prob")
            if p is None:
                continue
            probs.append(float(p))
            labels.append(ground_truth[mid])

        if not probs:
            print(f"  Skipping {judge_model}: no aligned probability data.")
            continue

        probs_arr = np.array(probs, dtype=np.float64)
        labels_arr = np.array(labels, dtype=bool)

        diag = reliability_diagram_data(probs_arr, labels_arr)
        all_results[judge_model] = diag
        summary_rows.append({"judge_model": judge_model, "ece": diag["ece"], "n": len(probs)})
        print(f"  {judge_model}: ECE={diag['ece']:.4f}  n={len(probs)}")

    if summary_rows:
        out_dir = _EXPERIMENTS_ROOT / dataset / "analysis"
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = out_dir / f"calibration_{extraction_model}.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nCalibration summary saved to {summary_path}")

        # Save per-judge reliability diagram data as NPZ
        for judge_model, diag in all_results.items():
            npz_path = out_dir / f"calibration_{extraction_model}_{judge_model}.npz"
            np.savez(
                npz_path,
                bin_centers=diag["bin_centers"],
                bin_accuracy=diag["bin_accuracy"],
                bin_confidence=diag["bin_confidence"],
                bin_counts=diag["bin_counts"],
                ece=np.array([diag["ece"]]),
            )

    return all_results


# ---------------------------------------------------------------------------
# Analysis 3: Cross-dataset probing
# ---------------------------------------------------------------------------


def run_cross_dataset(
    judge_model: str,
    datasets: list[str],
    extraction_model: str,
    extraction_dates: list[str],
) -> pd.DataFrame:
    """Build a cross-dataset probe accuracy matrix.

    Trains a probe on each dataset and evaluates on all others using
    ``analysis.cross_dataset.cross_dataset_probe_matrix``.

    Args:
        judge_model: Local judge model short name.
        datasets: List of dataset names.
        extraction_model: Extraction model short name.
        extraction_dates: Extraction date tags (``YYYY_mm_dd``), one per dataset.

    Returns:
        DataFrame saved to
        ``data/experiments/cross_dataset/probe_matrix_{judge_model}_{extraction_model}.csv``.
    """
    df = cross_dataset_probe_matrix(
        judge_model=judge_model,
        datasets=datasets,
        extraction_model=extraction_model,
        extraction_dates=extraction_dates,
    )

    out_dir = paths.cross_dataset_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"probe_matrix_{judge_model}_{extraction_model}.csv"
    df.to_csv(out_path)
    print(f"\nCross-dataset matrix saved to {out_path}")
    print(df.to_string())
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ScholarlM analyses (probe heatmap, calibration, cross-dataset).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── probe-heatmap ─────────────────────────────────────────────────────────
    ph = sub.add_parser("probe-heatmap", help="Probe accuracy heatmap across model pairs.")
    ph.add_argument("--dataset", required=True, help="Dataset name.")
    ph.add_argument(
        "--extraction-models", nargs="+", required=True,
        help="Extraction model short names.",
    )
    ph.add_argument(
        "--extraction-dates", nargs="+", required=True,
        help="Extraction date tags (YYYY_mm_dd), one per --extraction-models entry.",
    )
    ph.add_argument(
        "--judge-models", nargs="+", required=True,
        help="Local judge model short names.",
    )

    # ── calibration ───────────────────────────────────────────────────────────
    cal = sub.add_parser("calibration", help="Calibration metrics for each judge model.")
    cal.add_argument("--dataset", required=True, help="Dataset name.")
    cal.add_argument("--extraction-model", required=True, help="Extraction model short name.")
    cal.add_argument("--extraction-date", required=True, help="Extraction date tag (YYYY_mm_dd).")
    cal.add_argument(
        "--judge-models", nargs="+", required=True,
        help="Judge model keys (local or frontier).",
    )

    # ── cross-dataset ─────────────────────────────────────────────────────────
    cd = sub.add_parser("cross-dataset", help="Cross-dataset probe accuracy matrix.")
    cd.add_argument("--judge-model", required=True, help="Local judge model short name.")
    cd.add_argument(
        "--datasets", nargs="+", required=True,
        help="Dataset names to include in the matrix.",
    )
    cd.add_argument("--extraction-model", required=True, help="Extraction model short name.")
    cd.add_argument(
        "--extraction-dates", nargs="+", required=True,
        help="Extraction date tags (YYYY_mm_dd), one per --datasets entry.",
    )

    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.command == "probe-heatmap":
        run_probe_heatmap(
            dataset=args.dataset,
            extraction_models=args.extraction_models,
            extraction_dates=args.extraction_dates,
            judge_models=args.judge_models,
        )
    elif args.command == "calibration":
        run_calibration(
            dataset=args.dataset,
            extraction_model=args.extraction_model,
            extraction_date=args.extraction_date,
            judge_models=args.judge_models,
        )
    elif args.command == "cross-dataset":
        run_cross_dataset(
            judge_model=args.judge_model,
            datasets=args.datasets,
            extraction_model=args.extraction_model,
            extraction_dates=args.extraction_dates,
        )


if __name__ == "__main__":
    main()
