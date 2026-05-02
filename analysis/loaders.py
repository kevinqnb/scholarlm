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

_UTILS_DIR = Path(__file__).parent.parent / "experiments"
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

import utils as _utils


def load_run_metadata(output_dir: Path) -> dict | None:
    """Load run_metadata.json from an output directory, or return None if absent."""
    return _utils.load_run_metadata(output_dir)


def check_run_for_issues(
    dataset: str,
    model: str,
    date: str | None = None,
    ablation: str | None = None,
    *,
    raise_on_warning: bool = False,
) -> list[str]:
    """Load run_metadata.json and return any recorded compatibility warnings.

    Args:
        dataset: Dataset name.
        model: Extraction model short name.
        date: Optional date tag.
        ablation: Optional ablation number.
        raise_on_warning: If True, raise RuntimeError when warnings are present.

    Returns:
        List of warning strings from ``gpu_compatibility_warnings``; empty if clean.
    """
    import warnings

    if ablation is not None:
        base = _paths.EXPERIMENTS_ROOT / dataset / "ablations" / f"ablation{ablation}" / model
    else:
        base = _paths.EXPERIMENTS_ROOT / dataset / "extraction" / model

    if date:
        output_dir = base / date
    else:
        date_dirs = sorted(base.iterdir(), reverse=True) if base.exists() else []
        output_dir = next(
            (d for d in date_dirs if (d / "final.json").exists()),
            base,
        )

    meta = load_run_metadata(output_dir)
    if meta is None:
        return []

    issues = meta.get("gpu_compatibility_warnings", [])
    for w in issues:
        msg = f"Run {dataset}/{model}/{output_dir.name}: {w}"
        if raise_on_warning:
            raise RuntimeError(msg)
        warnings.warn(msg, stacklevel=2)

    return issues


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
        return pd.read_json(path, orient="records")
    raise ValueError(f"Unsupported ground truth file format: {path.suffix} (expected .csv or .json)")


def load_activations(
    dataset: str, extraction_model: str, extraction_date: str, judge_model: str, judge_date: str | None = None
) -> "np.lib.npyio.NpzFile":
    """Load attention_outputs.npz for a given (dataset, extraction, judge) triple."""
    path = _paths.find_activations(dataset, extraction_model, extraction_date, judge_model, judge_date)
    return np.load(path)

def load_layer_outputs(
    dataset: str, extraction_model: str, extraction_date: str, judge_model: str, judge_date: str | None = None
) -> "np.lib.npyio.NpzFile":
    """Load layer_outputs.npz for a given (dataset, extraction, judge) triple."""
    path = _paths.find_layer_outputs(dataset, extraction_model, extraction_date, judge_model, judge_date)
    return np.load(path)


def load_synthetic_responses(
    dataset: str, judge_model: str, judge_date: str | None = None
) -> list[dict]:
    """Load responses.json from a synthetic probe run."""
    path = _paths.find_synthetic_responses(dataset, judge_model, judge_date)
    with open(path) as f:
        return json.load(f)


def load_synthetic_activations(
    dataset: str, judge_model: str, judge_date: str | None = None
) -> "np.lib.npyio.NpzFile":
    """Load attention_outputs.npz from a synthetic probe run."""
    path = _paths.find_synthetic_activations(dataset, judge_model, judge_date)
    return np.load(path)


def load_synthetic_layer_outputs(
    dataset: str, judge_model: str, judge_date: str | None = None
) -> "np.lib.npyio.NpzFile":
    """Load layer_outputs.npz from a synthetic probe run."""
    path = _paths.find_synthetic_layer_outputs(dataset, judge_model, judge_date)
    return np.load(path)


def load_trained_probe(dataset: str, judge_model: str) -> dict:
    """Load a trained head probe saved by synthetic_probe_analysis.ipynb.

    Returns a dict with keys:
        ``probe``            — fitted sklearn Pipeline (StandardScaler + LogisticRegression)
        ``top_k_heads``      — list of (layer, head) tuples used by the probe
        ``train_prevalence`` — fraction of positive labels in the training set
        ``syn_document_ids`` — list of paper IDs in the synthetic training set
        ``judge_model``      — judge model name
        ``dataset``          — training dataset name
        ``n_layers``         — number of layers in the judge model
        ``n_heads``          — number of attention heads per layer
        ``head_dim``         — dimension of each attention head

    Raises:
        FileNotFoundError: If no probe has been saved for this (dataset, judge_model).
    """
    import joblib

    path = _paths.trained_probe_dir(dataset, judge_model) / "head_probe.pkl"
    if not path.exists():
        raise FileNotFoundError(
            f"Trained probe not found: {path}. "
            f"Run synthetic_probe_analysis.ipynb for dataset='{dataset}' judge='{judge_model}' first."
        )
    return joblib.load(path)


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
