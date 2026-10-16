---
id: 2026-10-08-calibration-v4-01
kind: build
config: analysis/analysis-configs/2026-10-08-calibration-v4-gemma27b-qwen-2.5-7b-intercept-fit-sample-01.yaml
---

## Session 2026-10-08

### Prompts
- "Please help me create a new `analysis/calibration_updated_v4.py` script. This is going to be a much simpler version of @analysis/calibration_updated_v3.py . Here's what we'll do. Use only the prior shift recalibration method. Give user the option to (a) bootstrap sample documents and uniform sample training points within them -- all to estimate the prior estimate. OR (b) just input their own prior estimate. No averaging smECE scores or computing confidence intervals over training fits. Use this one prior estimate to do label shift and then compute results. Although we should still bootstrap on the evaluation set like we do for synthetic cases in @analysis/calibration_updated_v3.py . This should look closer to earlier versions of @analysis/calibration_updated_v3.py . Create a config file to run it. To test it initally, I want to use perfect prior estimates -- just to see what it can do under perfect conditions."
- "submit the full oracle run"
- "Please help me make the calibration_v4 procedure even simpler now. First of all, add an intercept-fit option for recalibration. Second, no resampling the recalibration training data at all -- for either prior estimation or intercept-fit training data. Take one random seed, compute smECE results and bootstrap over 2000 resamples at the DOCUMENT level in the evaluation set. Variations from the choice of the random seed for training will be another experiment."
- (on the helper module name) "just name this recalibration"
- "submit new runs (including one with intercept fit) then /devlog and commit."

### Implemented
New `analysis/calibration_updated_v4.py` recalibrates each real cell with a single slope-1 map, either `prior_shift` or `intercept_fit`. The map is fit once on rows chosen by `fit_source`: a uniform `fit_n`-row draw from the probe-training pool by `fit_seed` (no resampling), a manual prior, or a diagnostic oracle on the evaluated rows. Confidence intervals come only from test-document cluster resampling (`n_boot`) through the existing `analysis/nested_bootstrap.py`; no metric code changed. The helpers are in `analysis/recalibration.py`, the config loader is `load_calibration_v4_config` in `analysis/analysis_config.py`, and rung-1 tests are in `tests/test_calibration_v4.py`. The configs are `analysis/analysis-configs/2026-10-08-calibration-v4-gemma27b-qwen-2.5-7b-{oracle,prior-shift-sample,intercept-fit-sample,intercept-fit-oracle}-01.yaml` plus the smoke configs `2026-10-08-calibration-v4-{oracle,intercept-fit-sample}-smoke-01.yaml`.

### Commits
a153212 calibration v4: one prior-shift / intercept-fit map per cell, fit once, document-bootstrap CIs
