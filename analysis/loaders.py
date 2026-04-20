"""Loaders for ScholarlM experiment outputs.

All path resolution delegates to ``experiments/paths.py``; callers never
build paths by hand.

Typical usage
-------------
    from analysis.loaders import load_extraction, load_combined_judgements

    records = load_extraction("pond", "gemma-3-27b", "2026_04_01")
    judgements = load_combined_judgements("pond", "gemma-3-27b", "2026_04_01")
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

_EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import paths as _paths


def load_extraction(
    dataset: str, model: str, date: str | None = None
) -> list[dict]:
    """Load final.json for a full extraction run."""
    final = _paths.find_extraction_final(dataset, model, date)
    with open(final) as f:
        return json.load(f)


def load_ablation(
    dataset: str, ablation_n: str | int, model: str, date: str | None = None
) -> list[dict]:
    """Load final.json for an ablation run."""
    final = _paths.find_extraction_final(dataset, model, date, ablation=str(ablation_n))
    with open(final) as f:
        return json.load(f)


def load_combined_judgements(
    dataset: str, extraction_model: str, extraction_date: str
) -> list[dict]:
    """Load combined.json for a judged extraction run."""
    combined = _paths.find_combined(dataset, extraction_model, extraction_date)
    with open(combined) as f:
        return json.load(f)


def load_ground_truth(config) -> "pd.DataFrame":
    """Load the manual ground-truth dataset using ``config.ground_truth_file``.

    Args:
        config: A ``DatasetConfig`` with ``ground_truth_file`` set.

    Returns:
        DataFrame loaded from the CSV or JSON file.

    Raises:
        ValueError: If ``config.ground_truth_file`` is ``None``.
        FileNotFoundError: If the file does not exist.
    """
    import pandas as pd

    if config.ground_truth_file is None:
        raise ValueError(
            f"DatasetConfig for '{config.name}' has no ground_truth_file set."
        )
    path = Path(config.ground_truth_file)
    if not path.is_absolute():
        path = Path(__file__).parent.parent / path
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"Unsupported ground truth file format: {path.suffix} (expected .csv or .json)")


def load_activations(
    dataset: str, extraction_model: str, extraction_date: str, judge_model: str
) -> "np.lib.npyio.NpzFile":
    """Load attention_outputs.npz for a given (dataset, extraction, judge) triple."""
    path = _paths.find_activations(dataset, extraction_model, extraction_date, judge_model)
    return np.load(path)


# ---------------------------------------------------------------------------
# Cached match_datasets wrapper
# ---------------------------------------------------------------------------


def cached_match(
    df_left: "pd.DataFrame",
    df_right: "pd.DataFrame",
    strict_matching: dict,
    fuzzy_matching: dict | None = None,
    fuzzy_threshold: float = 0.0,
    cache_path: Path | None = None,
) -> tuple:
    """Wrapper around ``match_datasets`` with optional disk caching.

    Cache key: absolute path of the left final.json + its modification time.
    If ``cache_path`` is ``None``, caching is disabled and ``match_datasets`` is
    called directly.

    Args:
        df_left: Left DataFrame (typically extraction results).
        df_right: Right DataFrame (typically ground truth).
        strict_matching: Passed to ``match_datasets``.
        fuzzy_matching: Passed to ``match_datasets``.
        fuzzy_threshold: Passed to ``match_datasets``.
        cache_path: If given, the result is loaded from this path when it
            exists; otherwise computed and saved there.

    Returns:
        The ``(matching, edges, edge_weights)`` tuple from ``match_datasets``.
    """
    from scholarlm.utils.data import match_datasets

    if cache_path is not None and cache_path.exists():
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    result = match_datasets(
        df_left,
        df_right,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching or {},
        fuzzy_threshold=fuzzy_threshold,
    )

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(result, f)

    return result
