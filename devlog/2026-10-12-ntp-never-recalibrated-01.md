---
id: 2026-10-12-ntp-never-recalibrated-01
kind: build
---

## Session 2026-10-12

### Prompts
- "Make a targeted change to analysis/calibration.py, calibration_validated.py, platt_scaling.py and any other analysis scripts (e.g. cluster, meta analysis) which load and use a recalibrated NTP model. NTP probabilities should NEVER be recalibrated; always taken directly as is. Begin by investigating and reporting exactly where this is used and what changes could fix it."
- "NTP can be dropped from platt scaling entirely. Lift the [prior_shift] guard. Leave nested bootstrap for now, it's harmless."
- "Did clustering and meta analysis ever actually use recalibrated NTP then?" (answered: yes, via the stored predictions) then "go right ahead with the plan."

### Implemented
NTP is now the judge's raw p(true) in every cell: `analysis/calibration.py` and `analysis/calibration_validated.py` fit a recalibration map for the probe only, and `analysis/platt_scaling.py` no longer includes NTP. The `prior_shift` guard is lifted for the probe, using its training prevalence. `analysis/common/prediction_store.py` now refuses a stored cell whose NTP differs from the raw value, so decision-threshold, meta and clustering cannot reuse old recalibrated predictions. `analysis/calibration_latex.py` and the test fixtures follow the probe-only layout. Earlier clustering and meta configs read intercept-shifted NTP, so every earlier NTP number is superseded. Unit tests and a smoke run on `analysis/analysis-configs/calibration/2026-10-10-calibration-gemma27b-qwen-2.5-7b-intercept-fit-smoke-01.yaml` passed; the `prior_shift` path has not been run end to end.

### Commits
- `7c182ac` calibration: NTP is never recalibrated; prior_shift allowed for the probe
