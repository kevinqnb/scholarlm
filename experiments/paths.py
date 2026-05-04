"""Canonical path helpers for the ScholarlM experiment framework.

All experiment scripts import from this module instead of building paths by
string concatenation.  Every directory layout is defined exactly once here.

Directory schema
----------------
data/experiments/
  {dataset}/
    extraction/{model}/{YYYY_mm_dd}/
    ablations/ablation{N}/{model}/{YYYY_mm_dd}/
    judge/{ext_model}/{ext_date}/{judge_model}/{judge_date}/
    judge/{ext_model}/{ext_date}/combined/
    analysis/
    analysis/figures/
  cross_dataset/
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
EXPERIMENTS_ROOT = _REPO_ROOT / "data" / "experiments"


def today() -> str:
    return datetime.now().strftime("%Y_%m_%d")


# ---------------------------------------------------------------------------
# Output directory helpers
# ---------------------------------------------------------------------------


def extraction(dataset: str, model: str, date: str | None = None) -> Path:
    """data/experiments/{dataset}/extraction/{model}/{date}/"""
    return EXPERIMENTS_ROOT / dataset / "extraction" / model / (date or today())


def ablation(
    dataset: str, ablation_n: str | int, model: str, date: str | None = None
) -> Path:
    """data/experiments/{dataset}/ablations/ablation{N}/{model}/{date}/"""
    return (
        EXPERIMENTS_ROOT / dataset / "ablations" / f"ablation{ablation_n}" / model / (date or today())
    )


def judge(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    judge_model: str,
    judge_date: str | None = None,
    ablation: str | None = None,
) -> Path:
    """Full judge output directory (with or without ablation).

    Without ablation:
        data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_model}/{judge_date}/
    With ablation:
        data/experiments/{dataset}/ablations/ablation{N}/{extraction_model}/{extraction_date}/judge/{judge_model}/{judge_date}/
    """
    jdate = judge_date or today()
    if ablation is not None:
        return (
            EXPERIMENTS_ROOT
            / dataset / "ablations" / f"ablation{ablation}"
            / extraction_model / extraction_date / "judge" / judge_model / jdate
        )
    return (
        EXPERIMENTS_ROOT
        / dataset / "judge"
        / extraction_model / extraction_date / judge_model / jdate
    )


def judge_base(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    ablation: str | None = None,
) -> Path:
    """Parent directory of per-judge-model subdirs; used by run_judge_combine.

    Without ablation: data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/
    With ablation:    data/experiments/{dataset}/ablations/ablation{N}/{extraction_model}/{extraction_date}/judge/
    """
    if ablation is not None:
        return (
            EXPERIMENTS_ROOT
            / dataset / "ablations" / f"ablation{ablation}"
            / extraction_model / extraction_date / "judge"
        )
    return EXPERIMENTS_ROOT / dataset / "judge" / extraction_model / extraction_date


def judge_combined(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    ablation: str | None = None,
) -> Path:
    """Directory that contains combined.json."""
    return judge_base(dataset, extraction_model, extraction_date, ablation) / "combined"


def analysis_dir(dataset: str) -> Path:
    """data/experiments/{dataset}/analysis/"""
    return EXPERIMENTS_ROOT / dataset / "analysis"


def figures_dir(dataset: str) -> Path:
    """data/experiments/{dataset}/analysis/figures/"""
    return EXPERIMENTS_ROOT / dataset / "analysis" / "figures"


def cross_dataset_dir() -> Path:
    """data/experiments/cross_dataset/"""
    return EXPERIMENTS_ROOT / "cross_dataset"


# ---------------------------------------------------------------------------
# Lookup helpers
# ---------------------------------------------------------------------------


def find_extraction_final(
    dataset: str,
    model: str,
    date: str | None = None,
    ablation: str | None = None,
) -> Path:
    """Return path to the most-recent (or date-pinned) final.json.

    Raises:
        FileNotFoundError: If no matching final.json exists.
    """
    if ablation is not None:
        base = (
            EXPERIMENTS_ROOT / dataset / "ablations" / f"ablation{ablation}" / model
        )
    else:
        base = EXPERIMENTS_ROOT / dataset / "extraction" / model

    if date:
        candidate = base / date / "final.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Extraction results not found: {candidate}")
        return candidate

    date_dirs = sorted(base.iterdir(), reverse=True) if base.exists() else []
    for d in date_dirs:
        candidate = d / "final.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No extraction results found for dataset='{dataset}' model='{model}' "
        f"under {base}. Run run_extraction.py first."
    )


def find_activations(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    judge_model: str,
    judge_date: str | None = None,
) -> Path:
    """Return path to the most-recent attention_outputs.npz for the given extraction date.

    Raises:
        FileNotFoundError: If no attention_outputs.npz exists.
    """
    judge_dir = (
        EXPERIMENTS_ROOT / dataset / "judge" / extraction_model / extraction_date / judge_model
    )
    if not judge_dir.exists():
        raise FileNotFoundError(f"No judge directory: {judge_dir}")
    if judge_date is None:
        for date_dir in sorted(judge_dir.iterdir(), reverse=True):
            candidate = date_dir / "attention_outputs.npz"
            if candidate.exists():
                return candidate
    else:
        candidate = judge_dir / judge_date / "attention_outputs.npz"
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(
        f"No attention_outputs.npz for dataset='{dataset}' "
        f"extraction_model='{extraction_model}' extraction_date='{extraction_date}' "
        f"judge='{judge_model}' under {judge_dir}"
    )


def find_layer_outputs(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    judge_model: str,
    judge_date: str | None = None,
) -> Path:
    """Return path to the most-recent layer_outputs.npz for the given extraction date.

    Raises:
        FileNotFoundError: If no layer_outputs.npz exists.
    """
    judge_dir = (
        EXPERIMENTS_ROOT / dataset / "judge" / extraction_model / extraction_date / judge_model
    )
    if not judge_dir.exists():
        raise FileNotFoundError(f"No judge directory: {judge_dir}")
    if judge_date is None:
        for date_dir in sorted(judge_dir.iterdir(), reverse=True):
            candidate = date_dir / "layer_outputs.npz"
            if candidate.exists():
                return candidate
    else:
        candidate = judge_dir / judge_date / "layer_outputs.npz"
        if candidate.exists():
            return candidate
    
    raise FileNotFoundError(
        f"No layer_outputs.npz for dataset='{dataset}' "
        f"extraction_model='{extraction_model}' extraction_date='{extraction_date}' "
        f"judge='{judge_model}' under {judge_dir}"
    )


def synthetic_probe(
    dataset: str,
    judge_model: str,
    judge_date: str | None = None,
) -> Path:
    """data/experiments/{dataset}/synthetic_probe/{judge_model}/{judge_date}/"""
    return EXPERIMENTS_ROOT / dataset / "synthetic_probe" / judge_model / (judge_date or today())


def synthetic_probe_test(
    dataset: str,
    judge_model: str,
    judge_date: str | None = None,
) -> Path:
    """data/experiments/{dataset}/synthetic_probe_test/{judge_model}/{judge_date}/"""
    return EXPERIMENTS_ROOT / dataset / "synthetic_probe_test" / judge_model / (judge_date or today())


def trained_probe_dir(dataset: str, judge_model: str) -> Path:
    """data/experiments/{dataset}/synthetic_probe/{judge_model}/trained_probe/"""
    return EXPERIMENTS_ROOT / dataset / "synthetic_probe" / judge_model / "trained_probe"


def find_synthetic_activations(
    dataset: str,
    judge_model: str,
    judge_date: str | None = None,
    split: str = "train",
) -> Path:
    """Return path to the most-recent attention_outputs.npz in synthetic_probe."""
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split} (expected 'train' or 'test')")
    if split == "test":
        judge_dir = EXPERIMENTS_ROOT / dataset / "synthetic_probe_test" / judge_model
    else:
        judge_dir = EXPERIMENTS_ROOT / dataset / "synthetic_probe" / judge_model
    if not judge_dir.exists():
        raise FileNotFoundError(f"No synthetic probe directory: {judge_dir}")
    if judge_date is not None:
        candidate = judge_dir / judge_date / "attention_outputs.npz"
        if candidate.exists():
            return candidate
    else:
        for date_dir in sorted(judge_dir.iterdir(), reverse=True):
            candidate = date_dir / "attention_outputs.npz"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No attention_outputs.npz for dataset='{dataset}' judge='{judge_model}' under {judge_dir}"
    )


def find_synthetic_layer_outputs(
    dataset: str,
    judge_model: str,
    judge_date: str | None = None,
    split: str = "train",
) -> Path:
    """Return path to the most-recent layer_outputs.npz in synthetic_probe."""
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split} (expected 'train' or 'test')")
    if split == "test":
        judge_dir = EXPERIMENTS_ROOT / dataset / "synthetic_probe_test" / judge_model
    else:
        judge_dir = EXPERIMENTS_ROOT / dataset / "synthetic_probe" / judge_model
    if not judge_dir.exists():
        raise FileNotFoundError(f"No synthetic probe directory: {judge_dir}")
    if judge_date is not None:
        candidate = judge_dir / judge_date / "layer_outputs.npz"
        if candidate.exists():
            return candidate
    else:
        for date_dir in sorted(judge_dir.iterdir(), reverse=True):
            candidate = date_dir / "layer_outputs.npz"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No layer_outputs.npz for dataset='{dataset}' judge='{judge_model}' under {judge_dir}"
    )


def find_synthetic_responses(
    dataset: str,
    judge_model: str,
    judge_date: str | None = None,
    split: str = "train",
) -> Path:
    """Return path to the most-recent responses.json in synthetic_probe."""
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split} (expected 'train' or 'test')")
    if split == "test":
        judge_dir = EXPERIMENTS_ROOT / dataset / "synthetic_probe_test" / judge_model
    else:
        judge_dir = EXPERIMENTS_ROOT / dataset / "synthetic_probe" / judge_model
    if not judge_dir.exists():
        raise FileNotFoundError(f"No synthetic probe directory: {judge_dir}")
    if judge_date is not None:
        candidate = judge_dir / judge_date / "responses.json"
        if candidate.exists():
            return candidate
    else:
        for date_dir in sorted(judge_dir.iterdir(), reverse=True):
            candidate = date_dir / "responses.json"
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No responses.json for dataset='{dataset}' judge='{judge_model}' under {judge_dir}"
    )


def find_combined(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
) -> Path:
    """Return path to combined.json for the given extraction date.

    Raises:
        FileNotFoundError: If combined.json does not exist.
    """
    path = judge_combined(dataset, extraction_model, extraction_date) / "combined.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Combined judge file not found: {path}. Run run_judge_combine.py first."
        )
    return path
