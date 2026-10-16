---
id: 2026-10-05-recovery-edge-filter-matching-01
kind: build
config: analysis/analysis-configs/2026-10-05-pond-full-pipeline-vs-baselines-recovery-judge-filter-01.yaml
---

## Session 2026-10-05

### Prompts

- Update how the match cache is used to compute recovery in `analysis/recovery_validity.py`: let users filter the raw cached edges either by a fuzzy threshold on the weights (current behavior) or by the combined LLM judges' judgements (drop any edge whose extracted measurement is judged invalid). After either filter, compute a maximum-weight matching over only the surviving edges and their weights, and count every ground truth point in the matching as recovered, rather than counting any ground truth point with a surviving edge.
- Zero-weight edges: weight first, cardinality as a tie-break.
- Write the judge-filter configs as new files rather than editing the existing ones, dropping ids that have no judge_combine output yet; do the same for nfix and supermat.
- Align judgements to postprocessed rows by `(document_id, measurement_id)`; split rows inherit their parent's judgement.

### Implemented

`analysis/recovery_validity.py` gains a required `edge_filter` (`threshold` | `judge`; `params.recovery_validity.edge_filter` and `--edge-filter`, no default) and computes recovery as a max-weight matching over the surviving cached edges, with new guards (valid 1-1 matching, subset of any-edge recovery, edges stay within one paper) and new output columns. `load_validity_labels` now joins judgements on `(document_id, measurement_id)`, so postprocessed rows that were split from one judged row inherit that row's judgement. Added unit tests for all of it, and `edge_filter: threshold` to the three existing recovery configs. New configs: `analysis/analysis-configs/2026-10-05-{pond,nfix,supermat}-full-pipeline-vs-baselines-recovery-judge-filter-01.yaml`. Rungs 1 (unit tests) and 2 (one-id smoke) passed; the analysis runs were executed directly in the shell. This changes evaluation logic and invalidates earlier recovery numbers from this script.

### Commits

- `24aa486` recovery_validity: max-weight-matching recovery, edge_filter (threshold|judge), keyed judgement join
- `8209b32` analysis: pond/nfix/supermat judge-filter recovery configs and results
