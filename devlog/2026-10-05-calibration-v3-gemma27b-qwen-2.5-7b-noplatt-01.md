---
id: 2026-10-05-calibration-v3-gemma27b-qwen-2.5-7b-noplatt-01
kind: build
config: analysis/analysis-configs/2026-10-05-calibration-v3-gemma27b-qwen-2.5-7b-noplatt-01.yaml
---

## Session 2026-10-05

### Prompts

This is a summary: the requests were spread across many messages.

- Update `analysis/synthetic_probe_train.py` to account for Platt scaling now happening inside `analysis/calibration_updated_v3.py`, so the training data no longer needs it. Keep it as an option, but default to not using it.
- Drop both the train-side Platt scaling and the 5-fold ensemble, so only v3 Platt-scales a single model's output ("option C"); keep 5-fold CV for head selection. Then update the v3 configs and rerun `synthetic_probe_train.py` for each dataset.
- Save cached probes and all results under `analysis/results/synthetic_probe` rather than in the experiments directory, and submit probe-training jobs through the analysis directory's wrapper.
- `analysis/calibration_ids.py` was deleted by accident; recover it from git history. Delete the stray probe pickles and fix the stale test.

### Implemented

`analysis/synthetic_probe_train.py` now defaults to `USE_PLATT_SCALING = False`: the head probe and NTP calibrator are a single fit rather than a Platt-calibrated 5-fold ensemble (the flag restores the old behavior), and probes are saved under `analysis/results/synthetic_probe/<config id>/trained_probe/`, which `analysis/calibration_ids.py` now resolves. `calibration_ids.py` was restored from history in its own commit. A new `probe_variant: noplatt` calibration config (`analysis/analysis-configs/2026-10-05-calibration-v3-gemma27b-qwen-2.5-7b-noplatt-01.yaml`) loads the new probes, and `tests/test_synthetic_probe_run_ids.py` was rewritten for the current script API. The nfix and supermat probes were retrained locally and checked against a prediction (head selection must not change); the pond retrain is submitted as a cluster job and the noplatt v3 calibration run has not been done yet.

### Commits

- `c7ed2cd` restore analysis/calibration_ids.py (deleted by mistake in 8c13783)
- `9446535` synthetic_probe_train: Platt scaling off by default; probes and results under analysis/results/synthetic_probe
