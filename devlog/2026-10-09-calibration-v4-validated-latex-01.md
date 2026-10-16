---
id: 2026-10-09-calibration-v4-validated-latex-01
kind: build
---

## Session 2026-10-09

### Prompts
- "`analysis/calibration_validated.py` and `analysis/calibration_latex.py` need to be updated to the v4 format -- i.e. to look like and work with `analysis/calibration.py`. Also, please double check that `calibration_validated.py` is not cutting off supermat entries anymore -- we've switched to v3s which means validations should have no overlap with the training data."

### Implemented
`analysis/calibration_validated.py` now follows `calibration.py`: one recalibration map fit once on LLM+matching-labelled pool rows, document-bootstrap intervals, and real cells scored on the human-validated rows. Those rows must lie outside the probe-training documents, which is asserted rather than filtered; a check against the training files found no overlap for pond or supermat. `fit_source: oracle` is rejected for validated configs. `analysis/calibration_latex.py` reads v4 output for both label sources, with captions that state the recalibration method and fit settings and CSV columns checked against the config. `analysis-configs/calibration-validated/2026-10-09-calibration-validated-gemma27b-qwen-2.5-7b-intercept-fit-01.yaml` is re-keyed to v4 and the latex configs are no longer marked invalid. This is an eval-logic change that invalidates earlier calibration-validated numbers. Rung 1 passed; the scripts have not been run on real data because the probes are untrained.

### Commits
- `7b796c8` synthetic-probe: move hardcoded training-paper exclusion into config (also carries this build's `config.py` v4 validated-loader change and the rewritten validated config tests)
- `27f64a4` calibration_validated: port to v4 recalibration and document bootstrap
- `296c5b7` calibration_latex: read v4 output for both label sources
