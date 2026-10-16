---
id: 2026-10-03-calibration-gemma27b-qwen-2.5-7b-v3-01
kind: build
---

## Session 2026-10-03

### Prompts

> Update `analysis/calibration_updated.py` to take a config in `analysis/analysis-configs/` naming one real extraction id and one synthetic-probe training analysis config per dataset (pond, nfix, supermat); draft one on the latest judge-interp runs and the qwen-2.5-7b cached probes; write results under `analysis/results/calibration/<config id>/`; keep the main content the same.

> Add a per-dataset parameter for whether matchings feed the labels (`judgement_combined | has_matching_edge` vs `judgement_combined`).

> Reuse the match caches that `match_cache.py` already built. Never build new matchings in the calibration script.

> Calibration should always be computed on the `final.json` datapoints; matchings only affect labels for the subset they cover when the flag is on.

> Recovery should only be computed in `recovery_validity.py`. Remove recovery and the validity/recovery plots from calibration and move `plot_validity_recovery` over. Don't change the behavior of `recovery_validity.py`.

### Implemented

`analysis/calibration_updated.py` now takes one analysis config (`analysis/analysis-configs/2026-10-03-calibration-gemma27b-qwen-2.5-7b-v3-01.yaml`), validated by a new `load_calibration_config`, with every named run cross-checked against its own committed config in `analysis/calibration_ids.py`. Matchings are never built: with the per-dataset `use_matching_labels` flag on, labels use the prebuilt match cache (provenance-checked, dataset-config rules, edges mapped back onto the `final.json` rows); with it off, labels are the judge alone. Recovery and the validity/recovery plots are removed from calibration, and the curve functions are added to `analysis/recovery_validity.py` without changing it. Unit tests (144 passed) and a full run with matching labels off completed; the sanity controls have not been run, so the numbers are preliminary. This changes calibration's evaluation logic and invalidates earlier calibration numbers.

### Commits

- `5737d2f` recovery_validity: add validity/recovery operating-curve helpers (additive)
- `fccf31c` calibration_updated: take an analysis config; drop recovery; read cached matchings
