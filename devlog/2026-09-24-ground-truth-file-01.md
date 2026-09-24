<!-- Public devlog entry: <repo>/devlog/<id>.md. Written by /devlog as a trimmed
version of the private build note at notes/<project>/builds/<id>.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-24-ground-truth-file-01
kind: build
---

## Session 2026-09-24

### Prompts

- The analysis configs for recovery and matching in `analysis/analysis-configs/`
  need an explicit field to define the ground truth file they are comparing
  against. Implement this in the configs, and in `analysis/match_cache.py` and
  `analysis/recovery_validity.py` themselves.
- Run the tiny end-to-end verification now, but submit it as an SGE job
  rather than running it directly in this shell.

### Implemented

Added a required `params.ground_truth_file` field to the analysis-config
envelope (`analysis/analysis_config.py`) so an analysis run's ground truth is
pinned to a specific file rather than read implicitly off the run's
dataset's own `DatasetConfig.ground_truth_file`. `analysis/match_cache.py`
and `analysis/recovery_validity.py` now take/require that path explicitly
(new `--ground-truth-file` CLI flag), and `match_cache.py` writes a
`match_cache.meta.json` sidecar (path/sha256/row count) that
`recovery_validity.py` verifies before trusting a cache — closing a gap
where a cache built against one ground truth file could otherwise be
silently scored against a different one. All three
`analysis/analysis-configs/*.yaml` (pond, nfix, supermat) were updated with
their `ground_truth_file`, and `analysis/match_cache.sh` was updated for the
new required flag. Verified via unit tests (758 passing) plus a real SGE job
that rebuilt one pond cache and reproduced its recovery number exactly.

### Commits
d9c3ab4 Require an explicit ground_truth_file on every analysis config
