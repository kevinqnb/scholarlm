"""
Central model registry for all experiment runners.

Defines three registries imported by the various runner scripts:

    MODEL_REGISTRY         — extraction models (ModelConfig); used by
                             run_extraction.py, run_ablation.py,
                             run_table_cleaning.py.

    INTERP_JUDGE_REGISTRY  — NNsight/JudgementLM judge models; used by
                             run_judge_interp.py.

    VLLM_JUDGE_REGISTRY    — vLLM judge models; used by
                             run_judge_local.py.
"""
from __future__ import annotations

from pathlib import Path
import sys

# Make scholarlm importable when this module is loaded directly or via import
# from a runner that has not yet set up sys.path.
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from scholarlm.config import ModelConfig

# torch is only needed for the interpretability judge registries; guard the
# import so that extraction-only runners don't require a torch installation.
try:
    import torch as _torch
    _bfloat16 = _torch.bfloat16
except ImportError:
    _bfloat16 = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Extraction model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    # --- vLLM models ---
    "llama-3.1-8b": ModelConfig(
        name="llama-3.1-8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        hf_revision=None,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
            "seed": 342,
            "enable_thinking": False,
        },
    ),
    "gemma-3-27b": ModelConfig(
        name="gemma-3-27b",
        model_id="gaunernst/gemma-3-27b-it-int4-awq",
        hf_revision=None,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
            "seed": 342,
        },
    ),
    "llama-3.3-70b": ModelConfig(
        name="llama-3.3-70b",
        model_id="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        hf_revision=None,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
            "seed": 342,
            "enable_thinking": False,
        },
    ),
    "qwen-2.5-72b": ModelConfig(
        name="qwen-2.5-72b",
        model_id="Qwen/Qwen2.5-72B-Instruct-AWQ",
        hf_revision=None,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
            "seed": 342,
            "enable_thinking": False,
        },
    ),
    "qwen-3.5-27b": ModelConfig(
        name="qwen-3.5-27b",
        model_id="Qwen/Qwen3.5-27B-FP8",
        hf_revision=None,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
            "seed": 342,
            "enable_thinking": False,
        },
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b",
        model_id="openai/gpt-oss-120b",
        hf_revision=None,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
            "seed": 342,
            "enable_thinking": False,
        },
    ),
}


# ---------------------------------------------------------------------------
# Interpretability / NNsight judge registry
#
# Used by run_judge_interp.py (as JUDGE_REGISTRY).
# ---------------------------------------------------------------------------

INTERP_JUDGE_REGISTRY: dict[str, dict] = {
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },
    "qwen-2.5-7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },

}


# ---------------------------------------------------------------------------
# vLLM judge registry
#
# Used by run_judge_local.py (as JUDGE_REGISTRY). 
# ---------------------------------------------------------------------------

VLLM_JUDGE_REGISTRY: dict[str, dict] = {
    "llama-3.3-70b": {
        "model_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    },
    "qwen-2.5-72b": {
        "model_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
    },
}


