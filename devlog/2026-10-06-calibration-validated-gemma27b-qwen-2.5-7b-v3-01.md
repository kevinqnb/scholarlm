---
id: 2026-10-06-calibration-validated-gemma27b-qwen-2.5-7b-v3-01
kind: build
config: analysis/analysis-configs/2026-10-06-calibration-validated-gemma27b-qwen-2.5-7b-v3-01.yaml
---

# Calibration scored on human-validated labels

## Session 2026-10-06

### Prompts

> Create `analysis/calibration_validated.py`, identical to `analysis/calibration_updated_v3.py` except that instead of LLM + matching labels, calibration is evaluated on a small set of human-validated labels. Create a new config file and wire it up to the `analysis/submit.sh` pipeline.

> Human validations exist only for pond and supermat, so compute only those, and only on the subset of test measurements that have validations.

> Platt scaling still uses n=100 training-split measurements, which stay labelled by LLM + matching since the training split has no validations.

> Do we really need a separate helper script? Can't we load this ad hoc in the new calibration script?

> Just submit the full rung 4 run now.

### Implemented

Added `analysis/calibration_validated.py`, a copy of v3 whose plotting and metric code is unchanged and whose real test cells are scored only on human-validated rows that are also test rows. Added `load_calibration_validated_config` to `analysis/analysis_config.py`, which restricts the datasets to pond and supermat and pins each validations file by sha256. Wired the script into `analysis/submit.sh` via `analysis/_resolve_job.py` (both git-ignored, so not in the commit) and added loader tests. The config is `analysis/analysis-configs/2026-10-06-calibration-validated-gemma27b-qwen-2.5-7b-v3-01.yaml`. The full run was submitted at the user's request without the smoke and tiny end-to-end rungs; afterwards its Platt fits and synthetic-cell metrics matched the v3 run exactly.

### Commits

- `75539f2` calibration_validated: score v3 calibration on human-validated labels
