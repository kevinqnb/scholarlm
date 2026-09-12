"""Shared utilities for the ScholarlM experiment framework.

Merges what used to be two separate modules:

  - Seeding, git introspection, GPU detection, compatibility checks,
    run-metadata persistence, and repo-wide config loading (formerly this
    file).
  - Every path helper the framework needs (formerly experiments/paths.py) --
    both the legacy ``(dataset, model, date)`` "most recent" addressing that
    analysis/*.py still uses unchanged against ``data/experiments/`` (that
    tree and its addressing scheme are explicitly out of scope for this
    restructure -- see the plan note in notes/scholarlm/), and the new
    experiment-id addressing that ``experiments/run_{type}.py`` runners use
    against ``experiments/results/``.
  - Model-config loading (``experiments/model-configs/{kind}/{model}.yaml``),
    replacing ``scripts/_resolve_model.py``'s eval'd-shell-variable approach
    with a plain Python function ``experiments/submit.sh`` calls directly.

Import from this module rather than duplicating logic across runner scripts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
from scholarlm.config import ModelConfig
_CONFIG_PATH = Path(__file__).parent / "config.yaml"

# ---------------------------------------------------------------------------
# Models known to degrade silently on specific GPU architectures
# ---------------------------------------------------------------------------

# AWQ marlin kernel is broken on A100; vLLM falls back silently → bad outputs.
_A100_INCOMPATIBLE_MODELS: set[str] = {
    "gaunernst/gemma-3-27b-it-int4-awq",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(path: Path | str = _CONFIG_PATH) -> dict:
    """Load experiments/config.yaml and resolve ${VAR} placeholders from env.

    Args:
        path: Path to the YAML config file.  Defaults to experiments/config.yaml.

    Returns:
        Parsed config dict with environment variable references expanded.
    """
    with open(path) as f:
        raw = f.read()

    def _resolve(m: re.Match) -> str:
        var, _, default = m.group(1).partition(":-")
        return os.environ.get(var, default)

    expanded = re.sub(r"\$\{([^}]+)\}", _resolve, raw)
    return yaml.safe_load(expanded)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def set_seeds(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (if available).

    Args:
        seed: Integer seed value.
    """
    import random
    random.seed(seed)

    import numpy as np
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# Git introspection
# ---------------------------------------------------------------------------


def get_git_info() -> dict[str, str | bool]:
    """Return the current git commit hash and working-tree cleanliness.

    Returns:
        Dict with keys:
            ``commit``  — short SHA (7 chars), or ``"unknown"`` if git fails.
            ``dirty``   — True if there are uncommitted changes.
    """
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=_REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(status)
    except Exception:
        dirty = False

    if dirty:
        import warnings
        warnings.warn(
            "Git working tree has uncommitted changes. "
            "For reproducible paper results, run from a clean tree.",
            stacklevel=3,
        )

    return {"commit": commit, "dirty": dirty}


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------


def get_gpu_info() -> list[dict[str, Any]]:
    """Return a list of GPU descriptors for all visible CUDA devices.

    Uses PyTorch when available; falls back to nvidia-smi for extraction
    runners that don't load torch.

    Returns:
        List of dicts, one per device, with keys:
            ``index``, ``name``, ``memory_total_gib``.
        Empty list if no GPU is visible.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        gpus = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpus.append({
                "index": i,
                "name": props.name,
                "memory_total_gib": round(props.total_memory / (1024 ** 3), 1),
            })
        return gpus
    except ImportError:
        pass

    # Fallback: nvidia-smi
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        gpus = []
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 3:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_gib": round(int(parts[2]) / 1024, 1),
                })
        return gpus
    except Exception:
        return []


# ---------------------------------------------------------------------------
# GPU–model compatibility check
# ---------------------------------------------------------------------------


def check_gpu_model_compatibility(model_id: str) -> list[str]:
    """Raise if model_id is known to silently degrade on a visible GPU.

    Currently detects: Gemma 3 27B AWQ on A100 (AWQ marlin kernel issue) --
    vLLM does not error on this combination, it silently produces degraded
    output, which is exactly the "runs cleanly, produces a quietly wrong
    number" failure mode this repo optimizes against. A crash here is
    strictly preferable to a run that completes and writes results that look
    fine but aren't.

    Previously this only ``warnings.warn``'d and let the run continue; a
    warning that doesn't stop a run whose results are known to be wrong is
    the anti-pattern, not a safety net.

    Args:
        model_id: HuggingFace model ID string.

    Returns:
        Always ``[]`` on return -- a genuine incompatibility raises instead
        of returning issue strings. Kept as a list return (rather than
        ``None``) purely for call-site compatibility: every runner does
        ``gpu_warnings = check_gpu_model_compatibility(...)`` and forwards it
        to ``write_run_metadata(gpu_compatibility_warnings=gpu_warnings)``,
        and ``analysis/loaders.py``'s reader does
        ``meta.get("gpu_compatibility_warnings", [])`` then iterates it --
        returning ``None`` on success would write ``null`` into future
        run_metadata.json files and crash that (out-of-scope, unchanged)
        reader on a ``NoneType`` iteration.

    Raises:
        RuntimeError: If model_id is known-incompatible with a visible GPU.
    """
    if model_id not in _A100_INCOMPATIBLE_MODELS:
        return []

    for gpu in get_gpu_info():
        if "A100" in gpu["name"]:
            raise RuntimeError(
                f"{model_id}: known AWQ marlin kernel degradation on "
                f"{gpu['name']} (device {gpu['index']}). Results would be "
                "silently wrong, not just slow or absent -- refusing to run. "
                "Use a different GPU type or a different model."
            )
    return []


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------


def write_run_metadata(output_dir: Path, *, start_time: float | None = None, **kwargs: Any) -> None:
    """Write run_metadata.json to output_dir.

    Automatically populates git_commit, git_dirty, run_timestamp, gpu_info,
    and (if start_time is given) runtime_seconds.  Any additional keyword
    arguments are merged in.

    Args:
        output_dir: Directory to write ``run_metadata.json``.
        start_time: ``time.time()`` value recorded at the start of the run.
            If provided, ``runtime_seconds`` is computed and included.
        **kwargs: Additional fields (dataset, model, model_id, seed, …).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        **get_git_info(),
        "gpu_info": get_gpu_info(),
        **kwargs,
    }
    if start_time is not None:
        metadata["runtime_seconds"] = round(time.time() - start_time, 1)

    path = output_dir / "run_metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Run metadata    : {path}")


# ---------------------------------------------------------------------------
# Run metadata loading (used by analysis/loaders.py)
# ---------------------------------------------------------------------------


def load_run_metadata(output_dir: Path) -> dict | None:
    """Load run_metadata.json from output_dir, or return None if absent."""
    path = output_dir / "run_metadata.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# Path helpers -- legacy (dataset, model, date) addressing
#
# Formerly experiments/paths.py, moved here verbatim (byte-identical
# behavior; only the import location changed). analysis/*.py and every
# committed calibration/ablation number depend on this exact addressing
# against data/experiments/ -- explicitly out of scope for this restructure,
# so nothing below this banner changes behavior.
#
# Directory schema
# ----------------
# data/experiments/
#   {dataset}/
#     extraction/{model}/{YYYY_mm_dd}/
#     ablations/ablation{N}/{model}/{YYYY_mm_dd}/
#     judge/{ext_model}/{ext_date}/{judge_model}/{judge_date}/
#     judge/{ext_model}/{ext_date}/combined/
#     jacobian_lens/{ext_model}/{ext_date}/{lens_model}/{lens_date}/
#     representation_lm/{model}/{date}/
#     attribution/{ext_model}/{ext_date}/{judge_model}/{method}/{date}/
#     attribution_synthetic/{judge_model}/{method}/{date}/
#     attribution_synthetic_test/{judge_model}/{method}/{date}/
#     synthetic_probe/{judge_model}/{judge_date}/
#     synthetic_probe/{judge_model}/trained_probe/
#     synthetic_probe_test/{judge_model}/{judge_date}/
#     analysis/
#     analysis/figures/
#   cross_dataset/
# ===========================================================================

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


def jacobian_lens(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    lens_model: str,
    date: str | None = None,
) -> Path:
    """data/experiments/{dataset}/jacobian_lens/{extraction_model}/{extraction_date}/{lens_model}/{date}/

    Deliberately a separate tree from judge()/judge_base(), not a subdirectory
    of judge/ — jacobian-lens output isn't a judge-combine participant and
    shouldn't be discoverable as if it were one.
    """
    return (
        EXPERIMENTS_ROOT
        / dataset / "jacobian_lens"
        / extraction_model / extraction_date / lens_model / (date or today())
    )


def representation_lm(dataset: str, model: str, date: str | None = None) -> Path:
    """data/experiments/{dataset}/representation_lm/{model}/{date}/

    Key-term representation collection (RepresentationLM). A separate tree,
    like jacobian_lens() — not keyed by an extraction run, since it reads raw
    OCR documents directly, not extraction output.
    """
    return EXPERIMENTS_ROOT / dataset / "representation_lm" / model / (date or today())


def attribution(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    judge_model: str,
    method: str,
    date: str | None = None,
) -> Path:
    """data/experiments/{dataset}/attribution/{extraction_model}/{extraction_date}/{judge_model}/{method}/{date}/

    Input-token attribution scores (``src/scholarlm/attribution.py``). A separate
    tree from judge()/jacobian_lens(): attribution output is not a
    judge-combine participant. Keyed by the extraction run *and* the judge model
    whose true/false judgement (or head probe) was attributed, plus the
    attribution method (``contrastive_gradient`` / ``probe``).
    """
    return (
        EXPERIMENTS_ROOT
        / dataset / "attribution"
        / extraction_model / extraction_date / judge_model / method / (date or today())
    )


def attribution_synthetic(
    dataset: str,
    judge_model: str,
    method: str,
    date: str | None = None,
    split: str = "train",
) -> Path:
    """data/experiments/{dataset}/attribution_synthetic[_test]/{judge_model}/{method}/{date}/

    Synthetic-probe-dataset counterpart of attribution(), mirroring the
    synthetic_probe / synthetic_probe_test split convention (no extraction run
    to key by).
    """
    if split not in {"train", "test"}:
        raise ValueError(f"Invalid split: {split} (expected 'train' or 'test')")
    subdir = "attribution_synthetic_test" if split == "test" else "attribution_synthetic"
    return EXPERIMENTS_ROOT / dataset / subdir / judge_model / method / (date or today())


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
    ablation: str | None = None,
) -> Path:
    """Return path to the most-recent attention_outputs.npz for the given extraction date.

    ``ablation`` mirrors ``judge()``/``judge_base()``: when set, reads from
    ``ablations/ablation{N}/{extraction_model}/{extraction_date}/judge/{judge_model}/``
    instead of ``judge/{extraction_model}/{extraction_date}/{judge_model}/``.

    Raises:
        FileNotFoundError: If no attention_outputs.npz exists.
    """
    judge_dir = judge_base(dataset, extraction_model, extraction_date, ablation) / judge_model
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
        f"judge='{judge_model}' ablation={ablation!r} under {judge_dir}"
    )


def find_judge_responses(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    judge_model: str,
    judge_date: str | None = None,
    ablation: str | None = None,
) -> tuple[Path, str]:
    """Return (path to responses.json, resolved judge_date) for an interp-judge run.

    Mirrors ``find_activations``: with ``judge_date=None`` it returns the
    most-recent judge run under
    ``judge/{extraction_model}/{extraction_date}/{judge_model}/`` (or the
    ablation-tree equivalent when ``ablation`` is set) that has a
    ``responses.json``; with ``judge_date`` set it pins that exact directory.

    Raises:
        FileNotFoundError: If the judge directory or a matching responses.json
            does not exist.
    """
    judge_dir = judge_base(dataset, extraction_model, extraction_date, ablation) / judge_model
    if not judge_dir.exists():
        raise FileNotFoundError(f"No judge directory: {judge_dir}")
    if judge_date is not None:
        candidate = judge_dir / judge_date / "responses.json"
        if candidate.exists():
            return candidate, judge_date
        raise FileNotFoundError(f"Judge responses not found: {candidate}")
    for date_dir in sorted(judge_dir.iterdir(), reverse=True):
        candidate = date_dir / "responses.json"
        if candidate.exists():
            return candidate, date_dir.name
    raise FileNotFoundError(
        f"No responses.json for dataset='{dataset}' extraction_model='{extraction_model}' "
        f"extraction_date='{extraction_date}' judge='{judge_model}' ablation={ablation!r} under {judge_dir}"
    )


def find_layer_outputs(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    judge_model: str,
    judge_date: str | None = None,
    ablation: str | None = None,
) -> Path:
    """Return path to the most-recent layer_outputs.npz for the given extraction date.

    ``ablation`` mirrors ``find_activations``.

    Raises:
        FileNotFoundError: If no layer_outputs.npz exists.
    """
    judge_dir = judge_base(dataset, extraction_model, extraction_date, ablation) / judge_model
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
        f"judge='{judge_model}' ablation={ablation!r} under {judge_dir}"
    )


_SYNTHETIC_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_]*$")


def _synthetic_subdir(split: str = "train", name: str | None = None) -> str:
    """Directory name under ``{dataset}/`` for a synthetic-probe judge run.

    ``name`` (the ``--synthetic-name`` arg) takes precedence over ``split`` and
    routes to a dedicated ``synthetic_probe_<name>`` tree — used by the
    augmentation pipeline, which produces three files (augmented train, primary
    test, diagnostic test) that don't fit the two-way ``train``/``test`` split.
    """
    if name is not None:
        if not _SYNTHETIC_NAME_RE.match(name):
            raise ValueError(
                f"--synthetic-name must match [a-z0-9][a-z0-9_]*, got {name!r}"
            )
        return f"synthetic_probe_{name}"
    if split == "test":
        return "synthetic_probe_test"
    if split == "train":
        return "synthetic_probe"
    raise ValueError(f"Invalid split: {split} (expected 'train' or 'test')")


def synthetic_probe(
    dataset: str,
    judge_model: str,
    judge_date: str | None = None,
    *,
    split: str = "train",
    name: str | None = None,
) -> Path:
    """data/experiments/{dataset}/synthetic_probe[_test|_<name>]/{judge_model}/{judge_date}/"""
    sub = _synthetic_subdir(split, name)
    return EXPERIMENTS_ROOT / dataset / sub / judge_model / (judge_date or today())


def synthetic_probe_test(
    dataset: str,
    judge_model: str,
    judge_date: str | None = None,
) -> Path:
    """data/experiments/{dataset}/synthetic_probe_test/{judge_model}/{judge_date}/"""
    return synthetic_probe(dataset, judge_model, judge_date, split="test")


def synthetic_probe_named(
    dataset: str,
    name: str,
    judge_model: str,
    judge_date: str | None = None,
) -> Path:
    """data/experiments/{dataset}/synthetic_probe_<name>/{judge_model}/{judge_date}/"""
    return synthetic_probe(dataset, judge_model, judge_date, name=name)


def trained_probe_dir(dataset: str, judge_model: str, source: str | None = None) -> Path:
    """data/experiments/{dataset}/synthetic_probe[_<source>]/{judge_model}/trained_probe/

    ``source`` names the synthetic training corpus the probe / NTP calibrator
    under this directory was fit on. ``None`` (default) is the original baseline
    path, byte-for-byte unchanged — the artifacts every committed calibration
    number depends on. A non-``None`` ``source`` (e.g. ``"v2"``) routes to the
    parallel ``synthetic_probe_<source>`` tree (same one the matching
    ``--synthetic-name`` judge run writes to), so a probe retrained on an
    augmented corpus never overwrites the baseline pickles.
    """
    if source is None:
        subdir = "synthetic_probe"
    else:
        if not _SYNTHETIC_NAME_RE.match(source):
            raise ValueError(
                f"trained_probe_dir source must match [a-z0-9][a-z0-9_]*, got {source!r}"
            )
        subdir = f"synthetic_probe_{source}"
    return EXPERIMENTS_ROOT / dataset / subdir / judge_model / "trained_probe"


def _find_synthetic(
    dataset: str,
    judge_model: str,
    filename: str,
    judge_date: str | None = None,
    split: str = "train",
    name: str | None = None,
) -> Path:
    """Locate ``filename`` in the most-recent (or date-pinned) synthetic-probe run."""
    judge_dir = EXPERIMENTS_ROOT / dataset / _synthetic_subdir(split, name) / judge_model
    if not judge_dir.exists():
        raise FileNotFoundError(f"No synthetic probe directory: {judge_dir}")
    if judge_date is not None:
        candidate = judge_dir / judge_date / filename
        if candidate.exists():
            return candidate
    else:
        for date_dir in sorted(judge_dir.iterdir(), reverse=True):
            candidate = date_dir / filename
            if candidate.exists():
                return candidate
    raise FileNotFoundError(
        f"No {filename} for dataset='{dataset}' judge='{judge_model}' under {judge_dir}"
    )


def find_synthetic_activations(
    dataset: str, judge_model: str, judge_date: str | None = None,
    split: str = "train", name: str | None = None,
) -> Path:
    """Return path to the most-recent attention_outputs.npz in a synthetic-probe run."""
    return _find_synthetic(dataset, judge_model, "attention_outputs.npz",
                           judge_date, split, name)


def find_synthetic_layer_outputs(
    dataset: str, judge_model: str, judge_date: str | None = None,
    split: str = "train", name: str | None = None,
) -> Path:
    """Return path to the most-recent layer_outputs.npz in a synthetic-probe run."""
    return _find_synthetic(dataset, judge_model, "layer_outputs.npz",
                           judge_date, split, name)


def find_synthetic_responses(
    dataset: str, judge_model: str, judge_date: str | None = None,
    split: str = "train", name: str | None = None,
) -> Path:
    """Return path to the most-recent responses.json in a synthetic-probe run."""
    return _find_synthetic(dataset, judge_model, "responses.json",
                           judge_date, split, name)


def find_human_responses(
    dataset: str,
    extraction_model: str,
    extraction_date: str | None = None,
    judge_date: str | None = None,
) -> tuple[Path, str]:
    """Return (path to human responses.json, resolved extraction_date).

    Searches for the most-recent (or date-pinned) human validation run.
    Historically produced by experiments/validation.py (a Streamlit app,
    since removed) -- this only reads responses.json files that already
    exist on disk.

    Raises:
        FileNotFoundError: If no matching responses.json exists.
    """
    base = EXPERIMENTS_ROOT / dataset / "judge" / extraction_model

    if not base.exists():
        raise FileNotFoundError(
            f"No judge directory for dataset='{dataset}' model='{extraction_model}': {base}"
        )

    if extraction_date is None:
        for ext_dir in sorted(base.iterdir(), reverse=True):
            human_dir = ext_dir / "human"
            if not human_dir.exists():
                continue
            for date_dir in sorted(human_dir.iterdir(), reverse=True):
                candidate = date_dir / "responses.json"
                if candidate.exists():
                    return candidate, ext_dir.name
        raise FileNotFoundError(
            f"No human responses.json for dataset='{dataset}' model='{extraction_model}' under {base}"
        )

    human_dir = base / extraction_date / "human"
    if not human_dir.exists():
        raise FileNotFoundError(f"No human judge directory: {human_dir}")

    if judge_date is not None:
        candidate = human_dir / judge_date / "responses.json"
        if candidate.exists():
            return candidate, extraction_date
        raise FileNotFoundError(f"Human responses not found: {candidate}")

    for date_dir in sorted(human_dir.iterdir(), reverse=True):
        candidate = date_dir / "responses.json"
        if candidate.exists():
            return candidate, extraction_date

    raise FileNotFoundError(
        f"No human responses.json for dataset='{dataset}' "
        f"model='{extraction_model}' extraction_date='{extraction_date}' under {human_dir}"
    )


def find_combined(
    dataset: str,
    extraction_model: str,
    extraction_date: str,
    ablation: str | None = None,
) -> Path:
    """Return path to combined.json for the given extraction date.

    Raises:
        FileNotFoundError: If combined.json does not exist.
    """
    path = judge_combined(dataset, extraction_model, extraction_date, ablation) / "combined.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Combined judge file not found: {path}. Run run_judge_combine.py first."
        )
    return path


# ===========================================================================
# Path helpers -- new experiment-id addressing
#
# Used by experiments/run_{type}.py runners and experiments/submit.sh.
# Replaces the "most recent date wins" ambiguity above with an explicit id:
# a judge config names the extraction experiment-id it judges, rather than a
# date it hopes is the right one.
# ===========================================================================

RESULTS_ROOT = _REPO_ROOT / "experiments" / "results"
EXPERIMENT_CONFIGS_ROOT = _REPO_ROOT / "experiments" / "experiment-configs"

_EXPERIMENT_ID_RE = re.compile(r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+-\d{2}$")


def result_dir(dataset: str, experiment_type: str, experiment_id: str) -> Path:
    """experiments/results/{dataset}/{experiment_type}/{experiment_id}/

    The one location a run_{type}.py runner writes its output to, and the
    one location callers who already know an experiment's (dataset, type)
    read it back from. For a caller that only has the id (e.g. a judge
    config referencing an upstream extraction run by id alone), use
    ``find_result_dir`` instead.
    """
    if not _EXPERIMENT_ID_RE.match(experiment_id):
        raise ValueError(
            f"experiment_id must match YYYY-MM-DD-slug-NN, got {experiment_id!r}"
        )
    return RESULTS_ROOT / dataset / experiment_type / experiment_id


def find_result_dir(experiment_id: str) -> Path:
    """Locate experiments/results/{dataset}/{experiment_type}/{experiment_id}/
    by id alone, without the caller needing to already know its dataset or
    experiment-type.

    Raises:
        FileNotFoundError: If no matching directory exists.
        ValueError: If more than one matches (an id collision -- experiment
            ids are meant to be globally unique; this should never happen
            for ids minted through the normal contract).
    """
    if not _EXPERIMENT_ID_RE.match(experiment_id):
        raise ValueError(
            f"experiment_id must match YYYY-MM-DD-slug-NN, got {experiment_id!r}"
        )
    matches = sorted(RESULTS_ROOT.glob(f"*/*/{experiment_id}"))
    if not matches:
        raise FileNotFoundError(
            f"No results directory found for experiment_id={experiment_id!r} "
            f"under {RESULTS_ROOT}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"experiment_id={experiment_id!r} matches more than one results "
            f"directory (should be globally unique): {matches}"
        )
    return matches[0]


def experiment_config_dir(dataset: str, experiment_type: str, experiment_id: str) -> Path:
    """experiments/experiment-configs/{dataset}/{experiment_type}/{experiment_id}/

    Holds {experiment_id}.yaml and out/ (SGE -o/-e stdout+stderr only).
    """
    if not _EXPERIMENT_ID_RE.match(experiment_id):
        raise ValueError(
            f"experiment_id must match YYYY-MM-DD-slug-NN, got {experiment_id!r}"
        )
    return EXPERIMENT_CONFIGS_ROOT / dataset / experiment_type / experiment_id


# ===========================================================================
# Model-config loading
#
# Replaces scripts/_resolve_model.py's eval'd-shell-variable approach: reads
# experiments/model-configs/{kind}/{model_name}.yaml directly and returns a
# plain dict, callable from Python (experiments/submit.sh) instead of a
# subprocess whose stdout gets eval'd.
# ===========================================================================

MODEL_CONFIGS_ROOT = _REPO_ROOT / "experiments" / "model-configs"

GpuNeed = Literal["vllm_server", "direct_gpu", "none"]


def load_model_config(kind: str, model_name: str) -> dict:
    """Load experiments/model-configs/{kind}/{model_name}.yaml.

    Args:
        kind: One of the model-configs subdirectories -- extraction,
            baseline, interp_judge, jacobian_lens, representation_lm,
            vllm_judge, ocr.
        model_name: The model's key (matches the yaml filename stem).

    Raises:
        FileNotFoundError: If no such model-config file exists.
    """
    path = MODEL_CONFIGS_ROOT / kind / f"{model_name}.yaml"
    if not path.exists():
        kind_dir = MODEL_CONFIGS_ROOT / kind
        available = sorted(p.stem for p in kind_dir.glob("*.yaml")) if kind_dir.exists() else []
        raise FileNotFoundError(
            f"No model-config at {path}. Available {kind} models: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def get_model_config(kind: str, model_name: str) -> ModelConfig:
    """Load a model-config and wrap it as a ``scholarlm.config.ModelConfig``.

    Every extraction-shaped runner (extraction, ablation, table_cleaning,
    the baseline runners) works in terms of this dataclass
    (``.model_id``, ``.sampling_params``, ``.api_base``, ...) rather than a
    raw dict -- this is the one place that bridges the two, so the internal
    pipeline code (MeasurementLM(...) and friends) never needed to change
    when model sourcing moved from model_registry.py's Python dict to
    experiments/model-configs/ yaml files.

    Args:
        kind: model-configs subdirectory (extraction, baseline, ...).
        model_name: The model's key (matches the yaml filename stem).

    Raises:
        FileNotFoundError: If no such model-config file exists.
    """
    d = load_model_config(kind, model_name)
    return ModelConfig(
        name=model_name,
        model_id=d["model_id"],
        hf_revision=d.get("hf_revision"),
        sampling_params=d.get("sampling_params", {}),
        api_base=d.get("api_base"),
    )


def classify_gpu_need(model_config: dict, *, source: str | Path | None = None) -> GpuNeed:
    """Classify what a model-config needs to run.

    - ``"vllm_server"``: a persistent vLLM server must be brought up first
      (model-config carries a ``serve:`` block).
    - ``"direct_gpu"``: no server, but a GPU allocation is still needed --
      NNsight-loaded models (interp_judge/jacobian_lens/representation_lm)
      and local-encoder baselines (GLiNER) load weights directly in-process.
    - ``"none"``: no GPU at all -- a frontier/API model (``api_base`` set).

    Deliberately does NOT default an ambiguous case to ``"none"``: a model
    with neither ``api_base`` nor ``serve`` nor ``resources`` needs a GPU
    (it isn't a frontier model) but the resource numbers to request it with
    are simply missing from its model-config -- silently treating that as
    "no GPU needed" would qsub the job onto a non-GPU node and let it fail
    confusingly (or worse, run degraded) far from the actual cause. Several
    model-configs are in exactly this state today (see the commit that
    split model_registry.py into model-configs/) because they were always
    run with hand-picked resources, never through a resolvable path.

    Args:
        model_config: A dict as returned by ``load_model_config``.
        source: Optional path/description of the model-config, included in
            the error message when resources are missing.

    Raises:
        ValueError: If the model needs a GPU but carries no ``resources:``
            block, or if it's ambiguous (no api_base, no serve, no resources).
    """
    where = f" ({source})" if source is not None else ""

    if model_config.get("api_base"):
        return "none"

    if "serve" in model_config:
        if "resources" not in model_config:
            raise ValueError(
                f"model-config{where} has a `serve:` block but no `resources:` "
                "block -- cannot resolve an SGE GPU request for it."
            )
        return "vllm_server"

    if "resources" in model_config:
        return "direct_gpu"

    raise ValueError(
        f"model-config{where} has no `api_base`, `serve`, or `resources` -- "
        "cannot determine whether this model needs a GPU. If it's a local "
        "NNsight or encoder model, add a `resources:` block "
        "(gpu_memory/gpu_capability/walltime/omp) before submitting it."
    )


# ===========================================================================
# Experiment-config loading + job resolution
#
# Used by experiments/submit.sh (via the thin experiments/_resolve_job.py CLI
# shim) and by experiments/run_{type}.py runners once they take a config path
# directly. Ties together dataset-configs/model-configs/experiment-configs
# addressing into one "given an experiment id, what do I run and how do I
# resource it" answer.
# ===========================================================================

# Tier-1 experiment types: each maps 1:1 to a run_{type}.py runner and gets
# full submit.sh automation. `model_kind` names the model-configs/
# subdirectory that type's `params.model` is looked up in; None means the
# runner takes no model-config at all (run_judge_combine.py is pure JSON
# voting, process_pdfs.py is pure PDF rendering -- neither loads a model).
#
# Composite/manual experiment types (probe_calibration, validation_set, ...)
# are NOT listed here -- they still get an experiment-configs/ directory for
# the reproducibility record, but no runner/submit.sh automation. See the
# restructure plan's two-tier taxonomy.
EXPERIMENT_TYPES: dict[str, dict[str, str | None]] = {
    "extraction":           {"runner": "run_extraction.py",          "model_kind": "extraction"},
    "ablation":              {"runner": "run_ablation.py",            "model_kind": "extraction"},
    "table_cleaning":        {"runner": "run_table_cleaning.py",       "model_kind": "extraction"},
    "baseline_chatextract":  {"runner": "run_baseline_chatextract.py", "model_kind": "extraction"},
    "probe_augment":         {"runner": "run_probe_augment.py",        "model_kind": "extraction"},
    "baseline_gliner":       {"runner": "run_baseline_gliner.py",      "model_kind": "baseline"},
    "baseline_nuextract":    {"runner": "run_baseline_nuextract.py",   "model_kind": "baseline"},
    "judge_interp":          {"runner": "run_judge_interp.py",         "model_kind": "interp_judge"},
    "attribution":           {"runner": "run_attribution.py",          "model_kind": "interp_judge"},
    "judge_local":           {"runner": "run_judge_local.py",          "model_kind": "vllm_judge"},
    "jacobian_lens":         {"runner": "run_jacobian_lens.py",        "model_kind": "jacobian_lens"},
    "representation_lm":     {"runner": "run_representation_lm.py",    "model_kind": "representation_lm"},
    "ocr":                   {"runner": "run_ocr.py",                  "model_kind": "ocr"},
    "judge_combine":         {"runner": "run_judge_combine.py",        "model_kind": None},
    "process_pdfs":          {"runner": "process_pdfs.py",             "model_kind": None},
}


def find_experiment_config(experiment_id: str) -> Path:
    """Locate experiments/experiment-configs/{dataset}/{type}/{id}/{id}.yaml
    by id alone.

    Raises:
        FileNotFoundError: If no matching config exists.
        ValueError: If more than one matches (an id collision).
    """
    if not _EXPERIMENT_ID_RE.match(experiment_id):
        raise ValueError(
            f"experiment_id must match YYYY-MM-DD-slug-NN, got {experiment_id!r}"
        )
    matches = sorted(
        EXPERIMENT_CONFIGS_ROOT.glob(f"*/*/{experiment_id}/{experiment_id}.yaml")
    )
    if not matches:
        raise FileNotFoundError(
            f"No experiment-config found for experiment_id={experiment_id!r} "
            f"under {EXPERIMENT_CONFIGS_ROOT}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"experiment_id={experiment_id!r} matches more than one experiment-config "
            f"(should be globally unique): {matches}"
        )
    return matches[0]


def load_experiment_config(path: Path) -> dict:
    """Load and validate an experiment-configs/.../{id}.yaml envelope.

    Enforces the harness contract's standardized envelope (see
    notes/hub/conventions.md): only ``id``, ``project``, ``description``,
    ``seed``, and ``params`` are required, ``params`` is a free-form mapping,
    and ``id`` must match the filename stem.

    Raises:
        ValueError: If the envelope is malformed.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in ("id", "project", "description", "seed", "params") if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required key(s): {missing}")
    if cfg["id"] != path.stem:
        raise ValueError(
            f"{path}: id {cfg['id']!r} does not match filename stem {path.stem!r}"
        )
    if not isinstance(cfg["params"], dict):
        raise ValueError(f"{path}: params must be a mapping")
    return cfg


def resolve_job(experiment_id: str) -> dict:
    """Resolve everything needed to submit + run one experiment.

    Locates the experiment's config by id, determines its dataset and
    experiment-type from the config's own path (not a redundant field inside
    the yaml -- the directory placement already encodes it), and -- for
    experiment-types that use a model -- loads and classifies that model's
    config too.

    Args:
        experiment_id: An experiment id (``YYYY-MM-DD-slug-NN``).

    Returns:
        Dict with keys: ``id``, ``dataset``, ``experiment_type``, ``runner``
        (script filename), ``config_path``, ``params`` (the config's own
        params mapping), ``gpu_need`` (``"vllm_server"`` | ``"direct_gpu"`` |
        ``"none"``), ``model`` (model name, or absent if ``gpu_need`` is
        ``"none"`` with no model at all), and ``model_config`` (the loaded
        model-config dict, or ``None``).

    Raises:
        FileNotFoundError: If no config exists for this id.
        ValueError: If the config, its experiment-type, or its model-config
            is malformed or missing required fields.
    """
    config_path = find_experiment_config(experiment_id)
    dataset, experiment_type = config_path.parts[-4], config_path.parts[-3]

    if experiment_type not in EXPERIMENT_TYPES:
        raise ValueError(
            f"{config_path}: experiment-type {experiment_type!r} (from its own "
            f"directory path) has no run_{{type}}.py automation -- known "
            f"automated types: {sorted(EXPERIMENT_TYPES)}. Composite/manual "
            "experiment-types are not runnable through submit.sh."
        )
    type_info = EXPERIMENT_TYPES[experiment_type]
    cfg = load_experiment_config(config_path)
    params = cfg["params"]

    result: dict[str, Any] = {
        "id": experiment_id,
        "dataset": dataset,
        "experiment_type": experiment_type,
        "runner": type_info["runner"],
        "config_path": config_path,
        "params": params,
    }

    model_kind = type_info["model_kind"]
    if model_kind is None:
        result["gpu_need"] = "none"
        result["model_config"] = None
        return result

    if "model" not in params:
        raise ValueError(
            f"{config_path}: params.model is required for experiment-type "
            f"{experiment_type!r}"
        )
    model_name = params["model"]
    model_config = load_model_config(model_kind, model_name)
    gpu_need = classify_gpu_need(model_config, source=f"{model_kind}/{model_name}.yaml")

    result["model"] = model_name
    result["model_config"] = model_config
    result["gpu_need"] = gpu_need
    return result


def require_params(params: dict, *keys: str, config_path: Path | str | None = None) -> None:
    """Raise a clear error if any of ``keys`` is missing from ``params``.

    Every run_{type}.py runner now takes a config path instead of individual
    CLI flags, so a missing required param no longer surfaces as argparse's
    own "the following arguments are required" -- this is that check's
    replacement, called at the top of each runner's main() before touching
    any of params' values.

    Args:
        params: The experiment config's ``params`` mapping.
        *keys: Required key names.
        config_path: Optional, included in the error message.

    Raises:
        ValueError: If any key in ``keys`` is missing from ``params``.
    """
    missing = [k for k in keys if k not in params]
    if missing:
        where = f"{config_path}: " if config_path is not None else ""
        raise ValueError(f"{where}missing required params key(s): {missing}")
