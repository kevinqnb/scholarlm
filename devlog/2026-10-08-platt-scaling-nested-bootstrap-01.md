---
id: 2026-10-08-platt-scaling-nested-bootstrap-01
kind: build
config: analysis/analysis-configs/2026-10-08-platt-scaling-intercept-fit-nested-tiny-01.yaml
---

## Session 2026-10-08

### Prompts
- "Now lets work on the platt scaling script. This should reuse @analysis/calibration_updated_v3.py 's recalibration methods, along with its training sampling, and testing sampling/bootstrapping procedures. It should to 100 x 100 for every n_training setting. Simply put, we just want this to be consistent with @analysis/calibration_updated_v3.py"
- Chose: single-class fit samples at small n are skipped and redrawn until there are n_fit_samples two-class samples.

### Implemented
`analysis/platt_scaling.py` now uses `calibration_updated_v3.py`'s recalibration methods, fit-sample draws and nested bootstrap at every n in `platt_ns`. That means `n_fit_samples` pool-resampled fit samples crossed with `n_doc_boot` test-document resamples, using `analysis/nested_bootstrap.py`. Fit sample r is v3's fit sample r, so at v3's `platt_n` the sweep reproduces v3's real cells, and single-class draws are skipped and redrawn. The sweep config (`analysis/analysis_config.py`) replaces `n_trials`/`ci`/`single_class_policy` with `recalibration`, `n_fit_samples` and `n_doc_boot`. Configs: `analysis/analysis-configs/2026-10-08-platt-scaling-nested-smoke-01.yaml`, `2026-10-08-platt-scaling-intercept-fit-nested-tiny-01.yaml` and `2026-10-08-platt-scaling-sweep-gemma27b-qwen-2.5-7b-{platt-fit,intercept-fit,prior-shift}-nested-01.yaml`. This is an evaluation change: it supersedes the earlier sweep results.

### Commits
984cf66 platt_scaling: nested-bootstrap sweep consistent with calibration v3
