---
id: 2026-10-04-calibration-v3-gemma27b-qwen-2.5-7b-v3-01
kind: build
config: analysis/analysis-configs/2026-10-04-calibration-v3-gemma27b-qwen-2.5-7b-v3-01.yaml
---

## Session 2026-10-04

### Prompts

- Make `analysis/calibration_updated_v3.py`, very similar to v2, but replace prevalence-based adjustment with Platt scaling on a small set of new examples: 100 random measurements from the training splits of the real pond, nfix and supermat datasets, labelled by their judge-combined files (plus matchings when the use-matching flag is on). A probe trained on a synthetic dataset is Platt scaled separately for each real extraction dataset using that dataset's own sample. No Platt scaling for synthetic test settings.
- Don't run the full script in shell.

### Implemented

Added `analysis/calibration_updated_v3.py`, which fits one Platt scaler per (train probe, real test dataset, method) on a seeded sample of the test dataset's probe-training-document rows, applies it to real test rows outside that pool, and leaves synthetic cells unscaled. `fit_platt`/`apply_platt` are new in `src/scholarlm/utils/calibration.py`, and `analysis_config.py` gains `load_calibration_v3_config` (required `platt_n`, no `pi_te_estimate`), with unit tests for both. The draft config is `analysis/analysis-configs/2026-10-04-calibration-v3-gemma27b-qwen-2.5-7b-v3-01.yaml`. Rung 1 only: the script has not completed a run.

### Commits

- `3b1ab46` calibration_updated_v3: Platt scaling on a small real sample instead of prevalence rescaling
