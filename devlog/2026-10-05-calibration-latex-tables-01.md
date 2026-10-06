---
id: 2026-10-05-calibration-latex-tables-01
kind: build
config: analysis/analysis-configs/2026-10-05-calibration-v3-latex-01.yaml
---

## Session 2026-10-05

### Prompts
(Summary of a multi-message session.)
- Create `analysis/calibration_latex.py` to turn `calibration_updated_v3.py` results into LaTeX tables, analogous to `recovery_validity_latex.py`; synthetic and real settings always in separate tables.
- Main table: an NTP block and a probe block (one midrule between), a row per dataset (PLW/NF/SM), columns PLW/NF/SM, cells smooth ECE with confidence interval.
- Separate, neat tables for all other saved metrics for each method and model.
- Plug-in ECE intervals that don't contain the point estimate are expected behaviour; leave them as stored.

### Implemented
New `analysis/calibration_latex.py` formats the existing v3 metrics CSVs into per-setting tables: a smooth-ECE main table plus classification and alternative-ECE tables, driven by `analysis/analysis-configs/2026-10-05-calibration-v3-latex-01.yaml`. It computes no metrics and validates CSV coverage, `Platt N`, judge model and finiteness before rendering. Plug-in ECE columns print their stored percentile interval as-is, and precision/F1 with no predicted positives print as `--` with a caption note. Rung 1 tests are in `tests/test_calibration_latex.py`, and the script was run on the committed v3 CSVs.

### Commits
1358fdb analysis: LaTeX tables for v3 calibration metrics (synthetic/real separate)
