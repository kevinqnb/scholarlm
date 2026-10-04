---
id: 2026-10-04-calibration-v2-gemma27b-qwen-2.5-7b-v3-01
kind: build
config: analysis/analysis-configs/2026-10-04-calibration-v2-gemma27b-qwen-2.5-7b-v3-01.yaml
---

## Session 2026-10-04

### Prompts

- Make `analysis/calibration_updated_v2.py`, very similar to `calibration_updated.py`, with three changes: (1) a prevalence estimate per real test dataset (pond, nfix, supermat), wired from the config, and whenever a probe trained on X is tested on Y, adjust with Y's estimate; (2) replace the within/cross-domain groupings with one plot (or set of table entries) per train dataset showing its evaluation on each of pond, nfix and supermat, for synthetic and again for real data; (3) separate plots/tables for Probe and NTP.
- Don't do a full run in shell.

### Implemented

Added `analysis/calibration_updated_v2.py`, which rescales each real-data cell to the test dataset's configured prevalence, plots one figure per (method, train dataset) with a curve per test dataset, and writes separate Probe and NTP figures and metrics CSVs. `analysis_config.py` gains `load_calibration_v2_config` (a required per-dataset `pi_te_estimate`, shared validation with the v1 loader), covered by new tests in `tests/test_calibration_config.py`. The draft config `analysis/analysis-configs/2026-10-04-calibration-v2-gemma27b-qwen-2.5-7b-v3-01.yaml` carries placeholder prevalence values that must be replaced. Rung 1 only: the script has not completed a run.

### Commits

- `016a999` calibration_updated_v2: per-dataset prevalence, per-train-dataset plots, probe/NTP split
