"""Shared utilities for ScholarlM experiment runners.

Provides seeding, git introspection, GPU detection, compatibility checks,
run-metadata persistence, and config loading.  Import from this module
rather than duplicating logic across runner scripts.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).parent.parent
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
    """Return warning strings for known GPU–model incompatibilities.

    Currently detects: Gemma 3 27B AWQ on A100 (AWQ marlin kernel issue).

    Args:
        model_id: HuggingFace model ID string.

    Returns:
        List of human-readable warning strings; empty if no issues detected.
    """
    if model_id not in _A100_INCOMPATIBLE_MODELS:
        return []

    gpu_info = get_gpu_info()
    issues = []
    for gpu in gpu_info:
        if "A100" in gpu["name"]:
            issues.append(
                f"{model_id}: known AWQ marlin kernel degradation on "
                f"{gpu['name']} (device {gpu['index']}). "
                "Results are unreliable and should be excluded from analysis."
            )
    for w in issues:
        warnings.warn(w, stacklevel=3)
    return issues


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
