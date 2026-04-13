"""
Central model registry for all experiment runners.

Defines four registries imported by the various runner scripts:

    MODEL_REGISTRY         — extraction models (ModelConfig); used by
                             run_extraction.py, run_ablation.py,
                             run_vllm_table_cleaning.py.

    INTERP_JUDGE_REGISTRY  — NNsight/JudgementLM judge models; used by
                             run_judge.py and run_judge_interp.py.

    VLLM_JUDGE_REGISTRY    — vLLM logprob judge models; used by
                             run_judge_local.py.

    FRONTIER_JUDGE_PROVIDERS — set of supported frontier batch-API
                               judge providers; used by run_judge.py.
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
    "gemma-3-27b": ModelConfig(
        name="gemma-3-27b",
        model_id="gaunernst/gemma-3-27b-it-int4-awq",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "gemma-4-31b": ModelConfig(
        name="gemma-4-31b",
        model_id="cyankiwi/gemma-4-31B-it-AWQ-4bit",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 16384,
        },
    ),
    "qwen-2.5-vl-72b": ModelConfig(
        name="qwen-2.5-vl-72b",
        model_id="Qwen/Qwen2.5-VL-72B-Instruct-AWQ",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "qwen-3-vl-30b": ModelConfig(
        name="qwen-3-vl-30b",
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        sampling_params={
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "llama-4-scout-109b": ModelConfig(
        name="llama-4-scout-109b",
        model_id="nvidia/Llama-4-Scout-17B-16E-Instruct-NVFP4",
        sampling_params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "glm-4.6v-106b": ModelConfig(
        name="glm-4.6v-106b",
        model_id="cyankiwi/GLM-4.6V-AWQ-4bit",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "intern-vl3-78b": ModelConfig(
        name="intern-vl3-78b",
        model_id="OpenGVLab/InternVL3-78B-AWQ",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "llama-3.3-70b": ModelConfig(
        name="llama-3.3-70b",
        model_id="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "qwen-2.5-72b": ModelConfig(
        name="qwen-2.5-72b",
        model_id="Qwen/Qwen2.5-72B-Instruct-AWQ",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "qwen-3.5-27b": ModelConfig(
        name="qwen-3.5-27b",
        model_id="Qwen/Qwen3.5-27B-FP8",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "glm-4.5-110b": ModelConfig(
        name="glm-4.5-110b",
        model_id="cyankiwi/GLM-4.5-Air-AWQ-4bit",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b",
        model_id="openai/gpt-oss-120b",
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 8192,
        },
    ),
    # --- Frontier models (api_base set; runners skip --api-base and vLLM extra_body) ---
    "gpt-4o-mini": ModelConfig(
        name="gpt-4o-mini",
        model_id="gpt-4o-mini",
        api_base="https://api.openai.com/v1",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
    "gpt-4.1-mini": ModelConfig(
        name="gpt-4.1-mini",
        model_id="gpt-4.1-mini",
        api_base="https://api.openai.com/v1",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
    "gpt-5-mini": ModelConfig(
        name="gpt-5-mini",
        model_id="gpt-5-mini",
        api_base="https://api.openai.com/v1",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
    "gpt-4o": ModelConfig(
        name="gpt-4o",
        model_id="gpt-4o",
        api_base="https://api.openai.com/v1",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
    "gemini-2-flash-lite": ModelConfig(
        name="gemini-2-flash-lite",
        model_id="gemini-2.0-flash-lite",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
    "gemini-2-flash": ModelConfig(
        name="gemini-2-flash",
        model_id="gemini-2.0-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
    "gemini-3-flash-lite": ModelConfig(
        name="gemini-3-flash-lite",
        model_id="gemini-3-flash-lite",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
    "gemini-1.5-flash": ModelConfig(
        name="gemini-1.5-flash",
        model_id="gemini-1.5-flash",
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
    ),
}


# ---------------------------------------------------------------------------
# Interpretability / NNsight judge registry
#
# Used by run_judge.py (as LOCAL_JUDGE_REGISTRY) and run_judge_interp.py
# (as JUDGE_REGISTRY).  Merged from both runners' prior inline definitions.
# ---------------------------------------------------------------------------

INTERP_JUDGE_REGISTRY: dict[str, dict] = {
    "llama-3.1-8b": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },
    "qwen-3-8b": {
        "model_id": "Qwen/Qwen3-8B",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },
    "gemma-3-12b": {
        "model_id": "google/gemma-3-12b-it",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2-9b-it",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },
    "mistral-7b": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "nnsight_kwargs": {"torch_dtype": _bfloat16},
        "sampling_params": {"do_sample": False, "max_new_tokens": 1},
    },
}


# ---------------------------------------------------------------------------
# vLLM logprob judge registry
#
# Used by run_judge_local.py (as JUDGE_REGISTRY).  Models are served via
# a vLLM OpenAI-compatible API; probability extraction uses logprobs.
# ---------------------------------------------------------------------------

VLLM_JUDGE_REGISTRY: dict[str, dict] = {
    "gemma-3-27b": {
        "model_id": "gaunernst/gemma-3-27b-it-int4-awq",
    },
    "qwen-3.5-27b": {
        "model_id": "Qwen/Qwen3.5-27B-FP8",
    },
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


# ---------------------------------------------------------------------------
# Frontier batch-API judge providers
#
# Used by run_judge.py (as FRONTIER_PROVIDERS).
# ---------------------------------------------------------------------------

FRONTIER_JUDGE_PROVIDERS: set[str] = {"openai", "anthropic", "gemini"}
