"""
Central model registry for all experiment runners.

Defines registries imported by the various runner scripts:

    MODEL_REGISTRY          — extraction models (ModelConfig); used by
                              run_extraction.py, run_ablation.py,
                              run_table_cleaning.py.

    INTERP_JUDGE_REGISTRY   — NNsight/JudgementLM judge models; used by
                              run_judge_interp.py.

    VLLM_JUDGE_REGISTRY     — vLLM judge models; used by
                              run_judge_local.py.

    BASELINE_MODEL_REGISTRY — external comparison baselines (ModelConfig);
                              used by run_baseline_nuextract.py. Kept separate
                              from MODEL_REGISTRY because these models are not
                              compatible with the standard 7-step pipeline or
                              the ablation registry (different input modality,
                              different prompting convention).
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
    # --- Frontier models (api_base set; runners skip --api-base and vLLM extra_body) ---
    "gpt-5-mini": ModelConfig(
        name="gpt-5-mini",
        model_id="gpt-5-mini",
        api_base="https://api.openai.com/v1",
        sampling_params={"max_completion_tokens": 8192},
    ),
}


# ---------------------------------------------------------------------------
# Baseline model registry
#
# External, non-MeasurementLM comparison methods. Used by
# run_baseline_nuextract.py and run_baseline_gliner.py.
#
# GLiNER2 is a small local encoder model (Fastino AI) run directly via
# `GLiNER2.from_pretrained(model_id)` — it does NOT go through a vLLM /
# OpenAI-compatible server, so `sampling_params` is unused (its precision/recall
# knob is the `--threshold` CLI flag, not a generation temperature).
# ---------------------------------------------------------------------------

BASELINE_MODEL_REGISTRY: dict[str, ModelConfig] = {
    "nuextract-2.0-8b": ModelConfig(
        name="nuextract-2.0-8b",
        model_id="numind/NuExtract-2.0-8B",
        hf_revision=None,
        sampling_params={
            "temperature": 0.0,
            "max_tokens": 4096,
        },
    ),
    "gliner-large-v1": ModelConfig(
        name="gliner-large-v1",
        model_id="fastino/gliner2-large-v1",
        hf_revision=None,
        sampling_params={},
    ),
    "gliner-base-v1": ModelConfig(
        name="gliner-base-v1",
        model_id="fastino/gliner2-base-v1",
        hf_revision=None,
        sampling_params={},
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
    "qwen-2.5-7b-base": {
        "model_id": "Qwen/Qwen2.5-7B",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
        # Non-instruction-tuned model; its tokenizer inherits a ChatML template
        # from the Instruct sibling that it was never trained to follow, so
        # apply_chat_template is skipped (see JudgementLM.use_chat_template).
        "use_chat_template": False,
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


