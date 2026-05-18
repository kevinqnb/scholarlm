"""
Logistic-regression probe utilities for JudgementLM attention activations.
"""
from __future__ import annotations

from typing import Generator

import numpy as np


def grouped_holdout_split(
    groups: np.ndarray,
    train_frac: float = 0.70,
    cal_frac: float = 0.15,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group-aware three-way split into train (CV pool), calibration, and test indices.

    No group appears in more than one split.  Groups are greedily assigned to
    each partition in order until the target fraction is reached.

    Args:
        groups: Array of group labels (e.g. ``paper__page`` strings) for each
            sample, aligned with X and y.
        train_frac: Target fraction for the training / CV pool partition.
        cal_frac: Target fraction for the calibration partition.  The test
            fraction is ``1 - train_frac - cal_frac``.
        random_state: Random seed for reproducibility.

    Returns:
        ``(train_idx, cal_idx, test_idx)`` integer index arrays.
    """
    rng = np.random.RandomState(random_state)
    groups = np.asarray(groups)
    unique_groups = np.array(sorted(set(groups)))
    rng.shuffle(unique_groups)

    n = len(groups)
    group_counts = {g: int((groups == g).sum()) for g in unique_groups}
    target_train = round(n * train_frac)
    target_cal   = round(n * cal_frac)

    train_groups: list = []
    cal_groups:   list = []
    test_groups:  list = []
    train_count, cal_count = 0, 0

    for g in unique_groups:
        gc = group_counts[g]
        if train_count < target_train:
            train_groups.append(g)
            train_count += gc
        elif cal_count < target_cal:
            cal_groups.append(g)
            cal_count += gc
        else:
            test_groups.append(g)

    train_idx = np.where(np.isin(groups, train_groups))[0]
    cal_idx   = np.where(np.isin(groups, cal_groups))[0]
    test_idx  = np.where(np.isin(groups, test_groups))[0]
    return train_idx, cal_idx, test_idx


def grouped_kfold_split(
    groups: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
    """Group-aware k-fold split: no group appears in both train and test folds.

    Greedily assigns groups to folds to keep fold sizes balanced.  Every sample
    belongs to exactly one test fold across the full iteration.

    Args:
        groups: Array of group labels (e.g., paper titles) for each sample.
        n_splits: Number of folds.
        random_state: Random seed for reproducibility.

    Yields:
        ``(train_idx, test_idx)`` integer arrays for each fold.
    """
    rng = np.random.RandomState(random_state)
    groups = np.asarray(groups)
    unique_groups = np.array(sorted(set(groups)))
    rng.shuffle(unique_groups)

    group_counts = {g: int((groups == g).sum()) for g in unique_groups}
    fold_groups: list[set] = [set() for _ in range(n_splits)]
    fold_sizes = [0] * n_splits

    for g in unique_groups:
        best_fold = min(range(n_splits), key=lambda f: fold_sizes[f])
        fold_groups[best_fold].add(g)
        fold_sizes[best_fold] += group_counts[g]

    for i in range(n_splits):
        test_mask = np.isin(groups, list(fold_groups[i]))
        yield np.where(~test_mask)[0], np.where(test_mask)[0]
