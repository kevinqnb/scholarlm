---
id: 2026-10-07-fuzzy-threshold-curve-01
kind: build
---

## Session 2026-10-07

### Prompts
- "Currently @analysis/recovery_validity.py produces its main recovery score by observing if any GT has an edge to an extracted data point which passes the fuzzy threshold defined in the dataset's config file. Likewise validity is reported as the fraction of entries judged as valid -- OR which have a match to a GT point. The goal now is to also plot recovery (x-axis) and validity (y-axis) plots which show how these two measurements change as the fuzzy threhsold changes. We should do this only for a subset of the extraction ids given in a config -- for example, I really only want to do this with the full pipeline gemma-3-27b extractions. The curves should be grey lines with points colored on the cool warm scale (0 to 1 threshold). No titles on the plots, just axes saying "Recovery" and "Validity". A color bar showing the color map and labeled as "Fuzzy Threshold" should go into a separate plot. Both plots saved as pdf images under a `analysis/results/recovery-validity/{config}/figures/` directory."
- "One more thing: can we clearly highlight the selected fuzzy threshold? I.e. 0.577 should be made distinct in the plot. Make it a diamond marker that is just slightly larger than others."

### Implemented
`analysis/recovery_validity.py` gains an optional `params.recovery_validity.fuzzy_threshold_curve` block (`experiment_ids`, a subset of the config's ids, and an explicit `thresholds` list that must include the dataset's own fuzzy threshold). For each listed id it computes recovery and validity at every threshold with the same definitions as the headline numbers. It writes a per-id curve CSV, a recovery-vs-validity PDF (grey line, coolwarm points, dataset threshold as an outlined diamond) and a separate "Fuzzy Threshold" colour bar under `analysis/results/recovery-validity/<config id>/`. The point at the dataset threshold is asserted equal to the headline row. The block is enabled for the full-pipeline gemma-3-27b id in `analysis/analysis-configs/2026-10-05-{pond,nfix,supermat}-full-pipeline-vs-baselines-recovery-01.yaml`. A preceding refactor moved `compute_metrics_for_id`'s cache-loading/guard block and judge lookup into shared helpers; output was byte-identical before and after, so no earlier numbers change.

### Commits
8ab40f6 recovery_validity: factor load/guard + judge resolution out of compute_metrics_for_id (pure move)
1ed3f80 recovery_validity: fuzzy-threshold recovery/validity curves for a config-selected subset of ids
be18068 results: regenerate stale nfix/supermat recovery CSVs and latex table under current any-edge code
