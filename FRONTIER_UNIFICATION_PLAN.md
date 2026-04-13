# Plan: Unify Frontier and Local Models Under One Runner

## Context

The codebase has two runner scripts (`run_extraction.py`, `run_ablation.py`) that currently only support local vLLM models. A previous session added separate `run_ablation_gpt.py` and `run_ablation_gemini.py` for frontier models. The goal is to consolidate everything: frontier models (GPT, Gemini) live in the same `MODEL_REGISTRY` and work with the existing runners. The Claude runner is dropped entirely.

**Files to delete:** `experiments/run_ablation_gpt.py`, `experiments/run_ablation_gemini.py`, `experiments/run_ablation_claude.py`

---

## Change 1 — `src/scholarlm/config.py`: extend `ModelConfig`

Add one optional field:

```python
api_base: str | None = None
```

When `None`, the model is vLLM — runners use their `--api-base` CLI arg. When set, runners use this value directly and ignore `--api-base`. This is how the registry distinguishes vLLM from frontier.

---

## Change 2 — `src/scholarlm/measurementlm.py`: suppress `extra_body` for frontier models

`MeasurementLM.__init__` currently merges vLLM-specific defaults (`top_k`, `repetition_penalty`, `enable_thinking`) into `self.sampling_params`, and `_acall` unconditionally pushes them into `extra_body`. OpenAI and Gemini reject unknown body params with a 400.

Add a constructor param:

```python
use_extra_body: bool = True
```

In `_acall`, guard the `extra_body` block:

```python
if self.use_extra_body:
    # existing top_k / repetition_penalty / enable_thinking logic
```

Runners set `use_extra_body=False` when they detect a frontier model (i.e. `model_config.api_base is not None`).

---

## Change 3 — `experiments/run_extraction.py`: add frontier entries to `MODEL_REGISTRY`

```python
"gpt-5-mini": ModelConfig(
    name="gpt-5-mini",
    model_id="gpt-5-mini",
    api_base="https://api.openai.com/v1",
    sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
),
"gemini-3-flash-lite": ModelConfig(
    name="gemini-3-flash-lite",
    model_id="gemini-3-flash-lite",
    api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
),
```

Update `run_pipeline` and `run_single_step` to:

- Use `model_config.api_base` when set, otherwise fall back to the `--api-base` CLI arg
- Pass `use_extra_body=False` to `MeasurementLM` when `model_config.api_base` is set
- Skip table cleaning when `model_config.api_base` is set (frontier models don't need it and it's expensive)
- Resolve API key: if `--api-key` is the default `"EMPTY"` and the model is frontier, check `OPENAI_API_KEY` or `GEMINI_API_KEY` env vars based on the `api_base` domain

---

## Change 4 — `experiments/run_ablation.py`: mirror Change 3

Apply the same updates to `run_ablation`. The `run_ablation` function already receives `model_config`, so the same `model_config.api_base` detection logic applies directly.

---

## What does NOT change

- `MeasurementLMAblation6` — works as-is once `use_extra_body` is plumbed through
- Dataset configs (`experiments/configs/pond.py`, `experiments/configs/nfix.py`) — untouched
- All judge runners — untouched
- The `--api-base` and `--api-key` CLI flags on both runners remain fully functional for vLLM use
