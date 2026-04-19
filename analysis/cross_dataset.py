"""
Cross-dataset probing: train a probe on one dataset's activations and evaluate
on another's.

For each ``(judge_model, train_dataset, test_dataset)`` triple this module:
1. Loads activations and ground-truth labels for both datasets.
2. Trains a logistic-regression probe on the training set.
3. Evaluates it on the test set.
4. Returns a DataFrame of probe accuracies (rows = train dataset,
   columns = test dataset).

The probe uses the same feature representation as ``scholarlm.utils.probe``
(mean over layers, then flatten over heads × head_dim).

Typical usage
-------------
    from analysis.cross_dataset import cross_dataset_probe_matrix

    df = cross_dataset_probe_matrix(
        judge_model="llama-3.1-8b",
        datasets=["pond", "nfix"],
        extraction_model="gemma-3-27b",
        extraction_dates=["2026_04_01", "2026_04_01"],
    )
    print(df)
    df.to_csv("data/experiments/cross_dataset/probe_matrix_llama-3.1-8b.csv")
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scholarlm.utils.probe import build_feature_matrix, train_probe, eval_probe

_EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import paths as _paths


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def load_activations_and_labels(
    dataset: str,
    extraction_model: str,
    judge_model: str,
    extraction_date: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Load activation feature matrix and aligned ground-truth labels.

    Args:
        dataset: Dataset name.
        extraction_model: Extraction model short name.
        judge_model: Local judge model short name (must have activation files).
        extraction_date: Date tag of the extraction run (``YYYY_mm_dd``).

    Returns:
        A tuple ``(X, y, measurement_ids)`` where:
        - ``X`` is a float32 array of shape ``(n, n_features)``.
        - ``y`` is a boolean array of shape ``(n,)`` (ground truth labels).
        - ``measurement_ids`` is the ordered list of measurement IDs.
    """
    import json

    activations_file = _paths.find_activations(dataset, extraction_model, extraction_date, judge_model)
    combined_file = _paths.find_combined(dataset, extraction_model, extraction_date)

    activations = np.load(activations_file)

    with open(combined_file) as f:
        combined_data: list[dict] = json.load(f)

    # Only keep records that have activations
    records_with_activations = [
        r for r in combined_data if str(r["measurement_id"]) in activations
    ]

    measurement_ids = [r["measurement_id"] for r in records_with_activations]
    labels = np.array([r["judgement_combined"] for r in records_with_activations], dtype=bool)

    X = build_feature_matrix(activations, measurement_ids)
    return X, labels, measurement_ids


# ---------------------------------------------------------------------------
# Cross-dataset matrix
# ---------------------------------------------------------------------------


def cross_dataset_probe_matrix(
    judge_model: str,
    datasets: list[str],
    extraction_model: str,
    extraction_dates: list[str],
) -> pd.DataFrame:
    """Train a probe on each dataset and evaluate on all others.

    For each ``(train_dataset, test_dataset)`` pair, trains a logistic-regression
    probe on ``train_dataset``'s activations + ground truth, then evaluates on
    ``test_dataset``'s activations + ground truth.

    Args:
        judge_model: Local judge model short name (e.g. ``"llama-3.1-8b"``).
            Must be the same model across all datasets for activations to have
            compatible shapes.
        datasets: List of dataset names to include in the matrix.
        extraction_model: Extraction model short name (same across all datasets).
        extraction_dates: Extraction date tags (``YYYY_mm_dd``), one per dataset,
            positionally aligned with ``datasets``.

    Returns:
        ``pd.DataFrame`` with ``datasets`` as both index (train) and columns (test).
        Values are probe accuracy (float in [0, 1]).  Diagonal is in-domain
        accuracy (same dataset for train and test, evaluated via a held-out split).
    """
    if len(extraction_dates) != len(datasets):
        raise ValueError(
            f"extraction_dates must have the same length as datasets "
            f"({len(extraction_dates)} vs {len(datasets)})"
        )
    from sklearn.model_selection import train_test_split

    # Load activations + labels for every dataset
    dataset_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for ds, ext_date in zip(datasets, extraction_dates):
        X, y, _ = load_activations_and_labels(ds, extraction_model, judge_model, ext_date)
        dataset_data[ds] = (X, y)

    # Build accuracy matrix
    n = len(datasets)
    matrix = np.full((n, n), np.nan)

    for i, train_ds in enumerate(datasets):
        X_train_full, y_train_full = dataset_data[train_ds]

        for j, test_ds in enumerate(datasets):
            if i == j:
                # In-domain: split train dataset into 80/20
                if len(X_train_full) < 10:
                    matrix[i, j] = np.nan
                    continue
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_train_full, y_train_full,
                    test_size=0.2, random_state=42, stratify=y_train_full
                )
                probe = train_probe(X_tr, y_tr)
                result = eval_probe(probe, X_te, y_te)
            else:
                # Cross-domain: train on full train_ds, test on full test_ds
                X_test, y_test = dataset_data[test_ds]
                probe = train_probe(X_train_full, y_train_full)
                result = eval_probe(probe, X_test, y_test)

            matrix[i, j] = result["accuracy"]

    return pd.DataFrame(matrix, index=datasets, columns=datasets)
