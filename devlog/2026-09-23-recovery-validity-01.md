<!-- Public devlog entry: scholarlm/devlog/2026-09-23-recovery-validity-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-23-recovery-validity-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-23-recovery-validity-01
kind: build
---

## Session 2026-09-23

### Prompts

- Create `analysis/recovery_validity.py`: take a list of experiment ids, compute
  recovery and validity using pre-cached matchings from `analysis/match_cache.py`
  and judgements stored in experiment results (same/related id), following
  `analysis/ablation.py`/`baselines.py` for precedent but as a cleaner, centrally
  organized script, with bootstrapped confidence intervals for each metric.
- Bootstrap should resample whole papers (document_id clusters), not individual
  rows.
- Skip the tens-of-minutes real match-cache rebuild for now; ship on unit tests plus
  whatever can be checked read-only against real data.
- Follow-up: (a) matching configs should live in each dataset's own
  `experiments/dataset-configs/{dataset}.py`, read from there by both
  `match_cache.py` and this script, with `load_match_cache` calls explicitly passing
  the dataset's own fuzzy threshold (every cache is stored at threshold 0). (b) Add
  a way to skip validity computation entirely, for an id with no judge labels yet.

### Implemented

`analysis/recovery_validity.py` is the new centralized recovery/validity script:
given a list of experiment ids, it loads each one's pre-built `match_cache.pkl`
(never recomputes, and refuses a cache that's stale or was built under different
matching rules), resolves the one `judge_combine` experiment whose `judge_ids` all
trace back to that id (via each judge's own committed config, not its
`run_metadata.json`), joins judgements to the extraction by `measurement_id` with
alignment assertions, cross-checks its own recovered/matched masks against
`analysis.metrics.recovery_rate`/`validity_rate` on every call, and reports a
percentile bootstrap CI resampled over whole papers for both metrics.
`--skip-validity` reports recovery only, skipping judge resolution entirely, for an
id with no judge coverage yet. Matching config (strict/fuzzy column maps, fuzzy
threshold, numeric-coercion list) moved out of `match_cache.py`'s own dict into new
`DatasetConfig` fields (`strict_matching`/`fuzzy_matching`/`fuzzy_threshold`/
`numeric_coerce`, `src/scholarlm/config.py`), set for pond in
`experiments/dataset-configs/pond.py` and bridged by `match_cache.py`'s new
`get_matching_config`. Along the way, fixed a real crash in judge resolution (an
unrelated candidate whose judge hadn't finished running yet) and added a
matching-columns guard that catches a legacy match cache the mtime-freshness check
alone couldn't. 35 unit tests added (`tests/test_recovery_validity.py`,
`tests/test_match_cache.py`); full repo suite reran clean. Verified against real
pond data via a real `match_cache.py` run plus a known-answer cross-check on
`recovery_validity.py --skip-validity`.

### Commits

- f163b4b Add analysis/recovery_validity.py: centralized recovery/validity with paper-clustered bootstrap CIs
