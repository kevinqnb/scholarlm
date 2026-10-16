<!-- Public devlog entry: scholarlm/devlog/2026-10-06-deduplicate-cache-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-10-06-deduplicate-cache-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-10-06-deduplicate-cache-01
kind: build
config: analysis/analysis-configs/2026-10-06-pond-deduplicate-cache-01.yaml
---

## Session 2026-10-06

### Prompts
(Summary of a multi-message session.)
- Check whether the pairwise duplicate rule in `src/scholarlm/utils/deduplication.py` is consistent with the current `match_datasets`, especially null handling; then "go ahead with the fix".
- Discussed clustering the duplicate graph and picking a representative per cluster instead of the greedy pass; started with measuring greedy's order-dependence on pond.
- "Probably too large of a job to compute in-shell. Instead, let's go to analysis/ and write a `deduplicate_cache.py` script mirroring match_cache.py ... cache edges and weights. Write up configs for deduplicating multiple extraction ids ... Wire deduplication cache up to the analysis directory submit.sh program. Get this submitted as a job."
- Added the supermat and nfix gemma27b full extractions.

### Implemented
`deduplication.pair_score` now follows `match_datasets`' current fuzzy-null rules (both-null abstains, one-sided null scores 0.0, all-abstain scores 1.0), which restores the parity test and invalidates earlier `deduplicate_records` output. New `analysis/deduplicate_cache.py` caches every strict-equal within-extraction pair with its mean fuzzy score at threshold 0.0, keyed by config id and experiment id, with a sha256 sidecar and a per-threshold component summary; it chooses no deduplication technique. Configs are `analysis/analysis-configs/2026-10-06-{pond,supermat,nfix}-deduplicate-cache-01.yaml` plus a pond smoke config; the script is allowlisted in the analysis submit wrapper (local, untracked wiring). Rung 1 tests (`tests/test_deduplicate_cache.py`) and a tiny end-to-end check against an independent groupby count passed before the three full-run jobs were submitted; their results had not been read at the time of writing.

### Commits
04ec3eb dedup: follow match_datasets' 2026-10-03 fuzzy-null rules
37cd505 analysis: deduplicate_cache.py, cached within-extraction duplicate graph
