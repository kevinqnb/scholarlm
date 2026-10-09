"""Loaders for ScholarlM experiment outputs.

All path resolution delegates to ``experiments/utils.py``; callers never
build paths by hand.

Typical usage
-------------
    from analysis.common.loaders import load_ground_truth_file, load_trained_probe
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
    """Load final.json for a full extraction run."""
    final = _paths.find_extraction_final(dataset, model, date)
    with open(final) as f:
        return json.load(f)


def load_ground_truth_file(path: Path) -> "pd.DataFrame":
    """Load a manual ground-truth CSV/JSON from an explicit path.

    The path-based counterpart to ``load_ground_truth`` -- used by
    analysis/match_cache.py and analysis/recovery_validity.py, which resolve
    the ground truth file from an analysis config's own
    ``params.ground_truth_file`` (see analysis/common/config.py) rather
    than from a DatasetConfig, so an analysis run's ground truth is pinned
    explicitly and doesn't silently drift if the DatasetConfig's own
    ``ground_truth_file`` is later repointed.

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
    """Load the NTP Platt calibrator saved by synthetic_probe_train.py.

    Args:
        variant: ``None`` (default) loads the Platt-scaled baseline
            (``ntp_calibrator.pkl``). ``"noplatt"`` loads the
            CalibratedClassifierCV-free variant (``ntp_calibrator_noplatt.pkl``)
            saved when the synthetic-probe config's ``params.use_platt_scaling``
            is ``false``.
        source: synthetic training corpus the calibrator was fit on. ``None``
            (default) is the baseline ``trained_probe/`` directory, unchanged.
            A non-``None`` value (e.g. ``"v2"``) reads from the parallel
            ``synthetic_probe_<source>/`` tree. See ``paths.trained_probe_dir``.

    Returns a dict with keys:
        ``calibrator``       — fitted calibrator (CalibratedClassifierCV, or the
                                base Pipeline directly for variant="noplatt")
        ``train_prevalence`` — fraction of positive labels in the synthetic training set
        ``syn_document_ids`` — list of paper IDs in the synthetic training set
        ``judge_model``      — judge model name
        ``dataset``          — training dataset name

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
    """Load a trained head probe saved by synthetic_probe_train.py.

    Args:
        variant: ``None`` (default) loads the Platt-scaled baseline
            (``head_probe.pkl``). ``"noplatt"`` loads the
            CalibratedClassifierCV-free variant (``head_probe_noplatt.pkl``)
            saved when the synthetic-probe config's ``params.use_platt_scaling``
            is ``false``. Only defined for ``ptype="head"`` — the layer
            probe has no no-Platt variant (out of scope for
            2026-08-10-no-platt-scaling-01).
        source: synthetic training corpus the probe was fit on. ``None``
            (default) is the baseline ``trained_probe/`` directory, unchanged.
            A non-``None`` value (e.g. ``"v2"``) reads from the parallel
            ``synthetic_probe_<source>/`` tree. See ``paths.trained_probe_dir``.

    Returns a dict with keys:
        ``probe``            — fitted sklearn Pipeline (StandardScaler + LogisticRegression),
                                or CalibratedClassifierCV wrapping one for the Platt-scaled variant
        ``top_k_heads``      — list of (layer, head) tuples used by the probe
        ``train_prevalence`` — fraction of positive labels in the training set
        ``syn_document_ids`` — list of paper IDs in the synthetic training set
        ``judge_model``      — judge model name
        ``dataset``          — training dataset name
        ``n_layers``         — number of layers in the judge model
        ``n_heads``          — number of attention heads per layer
        ``head_dim``         — dimension of each attention head

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


def load_probe_artifact(probe_dir: Path, filename: str, dataset: str, judge_model: str) -> dict:
    """A trained synthetic-probe artifact (head probe or NTP calibrator) from
    analysis/synthetic_probe_train.py's ``probe_dir``, asserting it was trained for this
    ``dataset`` and ``judge_model``."""
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
