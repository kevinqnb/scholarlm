---
id: 2026-10-02-analysis-synthetic-probe-configs-01
kind: build
---

## Session 2026-10-02

### Prompts

- Update `analysis/synthetic_probe_train.py` to take a config file in `analysis/analysis-configs/` naming the experiment id of the interp judge run on the synthetic dataset and the dataset (pond, nfix, or supermat); add `analysis/results/synthetic_probe/` for figures and results under the config id; keep the main content of the code the same.
- Include seeds in the config and wire the script to use them; fix the argument-order bug.
- Draft configs for every training run: pond / nfix / supermat x qwen-2.5-7b / llama-3.1-8b / mistral-nemo-12b, on the latest synthetic judge-interp ids.
- Wire up a common bash script to submit analysis jobs, like `experiments/submit.sh`. (Kept local and uncommitted.)

### Implemented

`analysis/synthetic_probe_train.py` now takes a single analysis config (`params.dataset`, `params.judge_interp_id`, top-level `seed`) validated by the new `load_synthetic_probe_config` in `analysis/analysis_config.py`. The config seed replaces the hardcoded `random_state=42`, and figures plus `results.json` are written under `analysis/results/synthetic_probe/<config id>/`. Nine configs were added, `analysis/analysis-configs/2026-10-02-{pond,nfix,supermat}-synthetic-probe-{qwen-2.5-7b,llama-3.1-8b,mistral-nemo-12b}-v3-01.yaml`. A separate commit fixes the `compute_ece` argument order in the training-ECE printout, which invalidates any previously printed head-probe train ECE from this script. Only unit tests (35 passed) and a config-load check ran; no probe has been trained under the new path.

### Commits

77455bc synthetic_probe_train: fix compute_ece argument order in training-ECE printout
1711ae3 synthetic_probe_train: take an analysis config; add 9 v3 probe-training configs
