"""Loaders for extraction outputs, ground truth, trained probes and match caches.

Paths are resolved through ``experiments/utils.py``, never built by hand.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import utils as _paths



def load_extraction(
    dataset: str, model: str, date: str | None = None
) -> list[dict]:
    """Load final.json of an extraction run from the legacy ``data/experiments`` tree.

    Args:
        dataset: Dataset name.
        model: Extraction model name.
        date: Run date folder; None lets ``find_extraction_final`` choose.

    Returns:
        List of extracted measurement records.
    """
    final = _paths.find_extraction_final(dataset, model, date)
    with open(final) as f:
        return json.load(f)


def load_ground_truth_file(path: Path) -> "pd.DataFrame":
    """Load a manual ground-truth CSV or JSON from an explicit path.

    The path comes from an analysis config, not a DatasetConfig, so results stay
    pinned to the file they were computed on.

    Args:
        path: Ground-truth file (``.csv`` or ``.json`` records).

    Returns:
        Ground-truth DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file's suffix isn't ``.csv`` or ``.json``.
    """
    import pandas as pd

    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    if path.suffix == ".csv":
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path, orient="records")
    raise ValueError(f"Unsupported ground truth file format: {path.suffix} (expected .csv or .json)")


def load_trained_ntp_calibrator(
    dataset: str, judge_model: str, variant: str | None = None, source: str | None = None
) -> dict:
    """Load an NTP calibrator saved by synthetic_probe_train.py.

    Args:
        dataset: Training dataset name.
        judge_model: Judge model name.
        variant: None for the Platt-scaled calibrator, ``"noplatt"`` for the unwrapped one.
        source: Synthetic corpus (e.g. ``"v2"``); None for the default ``trained_probe/``.

    Returns:
        Dict with ``calibrator``, ``train_prevalence``, ``syn_document_ids``,
        ``judge_model``, ``dataset``.

    Raises:
        ValueError: If variant is not one of (None, "noplatt").
        FileNotFoundError: If no calibrator has been saved for this (dataset, judge_model, variant).
    """
    import joblib

    if variant not in (None, "noplatt"):
        raise ValueError(f"Unknown variant {variant!r}; expected None or 'noplatt'")
    filename = "ntp_calibrator.pkl" if variant is None else "ntp_calibrator_noplatt.pkl"
    path = _paths.trained_probe_dir(dataset, judge_model, source=source) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"NTP calibrator not found: {path}. "
            f"Run synthetic_probe_train.py for dataset='{dataset}' judge='{judge_model}' "
            f"variant={variant!r} source={source!r} first."
        )
    return joblib.load(path)


def load_trained_probe(
    dataset: str, judge_model: str, ptype: str = "head",
    variant: str | None = None, source: str | None = None,
) -> dict:
    """Load a trained head or layer probe saved by synthetic_probe_train.py.

    Args:
        dataset: Training dataset name.
        judge_model: Judge model name.
        ptype: ``"head"`` or ``"layer"``.
        variant: None for the Platt-scaled probe, ``"noplatt"`` for the unwrapped one
            (head probes only).
        source: Synthetic corpus (e.g. ``"v2"``); None for the default ``trained_probe/``.

    Returns:
        Dict with ``probe`` (sklearn Pipeline, possibly Platt-wrapped), ``top_k_heads``,
        ``train_prevalence``, ``syn_document_ids``, ``judge_model``, ``dataset``,
        ``n_layers``, ``n_heads``, ``head_dim``.

    Raises:
        ValueError: If variant is not one of (None, "noplatt"), or variant="noplatt" with ptype="layer".
        FileNotFoundError: If no probe has been saved for this (dataset, judge_model, ptype, variant).
    """
    import joblib

    if variant not in (None, "noplatt"):
        raise ValueError(f"Unknown variant {variant!r}; expected None or 'noplatt'")
    if variant == "noplatt" and ptype == "layer":
        raise ValueError("variant='noplatt' is only defined for ptype='head'")

    if ptype == "layer":
        filename = "layer_probe.pkl"
    else:
        filename = "head_probe.pkl" if variant is None else "head_probe_noplatt.pkl"
    path = _paths.trained_probe_dir(dataset, judge_model, source=source) / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Trained probe not found: {path}. "
            f"Run synthetic_probe_train.py for dataset='{dataset}' judge='{judge_model}' "
            f"ptype={ptype!r} variant={variant!r} source={source!r} first."
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
    """Run ``match_datasets``, or return a pickled result from ``cache_path`` if one exists.

    An existing cache is returned as-is: its inputs are not checked against the
    arguments. Freshness is the caller's job (see match_cache.py's sidecar).

    Args:
        df_left: Left DataFrame (ground truth, in current callers).
        df_right: Right DataFrame (extractions, in current callers).
        strict_matching: Exact-match column mapping.
        fuzzy_matching: Fuzzy-match column mapping.
        fuzzy_threshold: Minimum fuzzy score, used only when computing.
        cache_path: Pickle to read, or to write after computing; None disables caching.

    Returns:
        ``(matching, edges, edge_weights)`` from ``match_datasets``.
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


def load_probe_artifact(probe_dir: Path, filename: str, dataset: str, judge_model: str) -> dict:
    """Load a trained probe or NTP calibrator and check it was built for this dataset/judge.

    Args:
        probe_dir: synthetic_probe_train.py's ``trained_probe`` directory.
        filename: Artifact file name in ``probe_dir``.
        dataset: Expected training dataset.
        judge_model: Expected judge model.

    Returns:
        The artifact dict.

    Raises:
        FileNotFoundError: The artifact does not exist.
    """
    import joblib
    path = probe_dir / filename
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist. Run analysis/synthetic_probe_train.py '
            f'on the {dataset} synthetic-probe config first.'
        )
    artifact = joblib.load(path)
    assert artifact['judge_model'] == judge_model, (
        f'{path}: judge_model {artifact["judge_model"]!r} != {judge_model!r}')
    assert artifact['dataset'] == dataset, f'{path}: dataset {artifact["dataset"]!r} != {dataset!r}'
    return artifact
