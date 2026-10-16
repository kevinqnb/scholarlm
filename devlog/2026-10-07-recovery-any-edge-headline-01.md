---
id: 2026-10-07-recovery-any-edge-headline-01
kind: build
---

## Session 2026-10-07

### Prompts
- "Do we have an option in analysis/recovery_validity.py (or wherever the final recover matching is computed) for computing recovery by total ground truth entires covered by at least one threshold passing edge? I.e. not a strict matching."
- "Ok I want the recovery_any_edge to be the main "recovery" number and the matching score can be recorded as "recovery_max_weight_matching". Also, when an extracted data point gets labeled as valid because it "matched" to the ground truth, this should be because it has some fuzzy threshold surviving edge to ANY ground truth data point -- not because it got used in the max weight matching. This will affect the labels that then feed into analysis/calibration_updated_v3.py, analysis/meta_updated.py, analysis/platt_scaling.py, and (any others?). Please design a plan to cleanly and safely make this transition."
- Chose to drop the judge edge-filter mode rather than combine it with the fuzzy threshold.
- "commit these changes. Hold off on regeneration for right now."
- "look into why chatextract recovery is exactly zero"
- "For the latex table, I would honestly just leave entries as "--" blanks for pond and nfix instead of trying to claim 0 recovery when it can't extract the necessary fields." Chose to blank the whole cell rather than only the recovery half.

### Implemented
In `analysis/recovery_validity.py`, `recovery` and its paper-clustered CI now count ground truth rows with at least one edge at or above the fuzzy threshold, which matches `metrics.recovery_rate`. The max-weight 1-1 matching count is reported separately as `recovery_max_weight_matching`, with its own CI. The `edge_filter` option (`--edge-filter`, `params.recovery_validity.edge_filter`) and its judge mode are removed, and the key is gone from the nine `analysis/analysis-configs/*-recovery-01.yaml` configs. `analysis/recovery_validity_latex.py` now refuses CSVs that lack the new column. Validity and calibration labels already used any threshold-surviving edge and are unchanged; new tests pin that down. Unit tests, a smoke run and a pond end-to-end check passed (the latter exactly reproduced the predicted columns). This is an evaluation change: `recovery` values from this script since 2026-10-05 were matching counts and must be regenerated.

On pond and nfix, ChatExtract's recovery is zero by construction. It never extracts the non-name fuzzy fields (`ecosystem`; `ecosystem_type`/`substrate_type`), and a field that is null on only one side scores 0, so its best edge score falls below the dataset threshold. `analysis/analysis-configs/2026-10-05-full-pipeline-vs-baselines-recovery-validity-latex-01.yaml` now renders those two cells as `--`; supermat, which matches on name alone, is unchanged.

### Commits
9e2c7d7 recovery_validity: any-edge recovery as headline, matching as recovery_max_weight_matching; drop judge edge filter
d524de5 latex table: blank ChatExtract pond/nfix cells (recovery 0 by construction)
