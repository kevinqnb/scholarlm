---
id: 2026-10-05-recovery-validity-latex-table-01
kind: build
config: analysis/analysis-configs/2026-10-05-full-pipeline-vs-baselines-recovery-validity-latex-01.yaml
---

## Session 2026-10-05

### Prompts
(Summary of a multi-message session.)
- Add LaTeX output for recovery/validity pairs (point ± bootstrap interval); make it a separate script over existing results, one table for all datasets (PLW/NF/SM columns, a row per method, baselines and full extraction as separate blocks, ablation 1 as "Direct {model}").
- Single rule between blocks; \texttt model names; (recovery, validity) comma-separated pairs; bold the best recovery and validity per dataset column (not intervals).
- Save outputs of both scripts under analysis/results/recovery-validity/ and rerun.

### Implemented
New `analysis/recovery_validity_latex.py` formats existing recovery_validity CSVs into one LaTeX table, driven by `analysis/analysis-configs/2026-10-05-full-pipeline-vs-baselines-recovery-validity-latex-01.yaml`; it computes no metrics and hard-errors if a CSV is stale relative to its analysis config. `recovery_validity.py` changed only its ad-hoc default output path, and the three `2026-10-05-*-full-pipeline-vs-baselines-recovery-01` configs now write to `analysis/results/recovery-validity/`. Rung 1 tests added in `tests/test_recovery_validity_latex.py`.

### Commits
a2f8331 analysis: recovery/validity LaTeX table across pond/nfix/supermat
