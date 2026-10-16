---
id: 2026-10-05-platt-scaling-sweep-gemma27b-qwen-2.5-7b-v3-01
kind: build
config: analysis/analysis-configs/2026-10-05-platt-scaling-sweep-gemma27b-qwen-2.5-7b-v3-01.yaml
---

## Session 2026-10-06

### Prompts
(Summary of a multi-message session.)
- Draft `analysis/platt_scaling.py`: replicate the real-extraction path of `calibration_updated_v3.py`, but Platt-scale the cached probe and NTP models with 10/50/100/250/500/1000 nested training samples per evaluation dataset; plot smECE only, same titles/colors/line styles, with a config following the v3 platt-probe variant.
- Replace the bootstrap intervals with ~100 random trials, plotting the mean with intervals over trials.
- Run the smoke test as a small job through `analysis/submit.sh`, not through `experiments/`.
- Set supermat `use_matching_labels` to true in the new config as well.

### Implemented
New `analysis/platt_scaling.py` sweeps the Platt sample size for each (train probe, test dataset, method), real setting only: samples are nested prefixes of a per-trial permutation of the v3 Platt pool, and each scaler is scored by smECE on one fixed held-out test set. The curve is the mean over `n_trials` with a central `ci` interval over trials; single-class samples follow an explicit `single_class_policy`. `analysis/analysis_config.py` gains `load_platt_sweep_config`, `analysis/analysis-configs/2026-10-05-platt-scaling-sweep-gemma27b-qwen-2.5-7b-v3-01.yaml` is the full sweep config, and `2026-10-06-platt-scaling-smoke-01.yaml` is a small smoke config. Rung 1 tests are in `tests/test_platt_scaling.py`; the smoke job is submitted via `analysis/submit.sh platt_scaling` and its result is not yet reviewed.

### Commits
6a3c310 analysis: smECE vs Platt-sample-size sweep (platt_scaling.py)
