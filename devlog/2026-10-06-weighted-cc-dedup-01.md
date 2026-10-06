<!-- Public devlog entry: scholarlm/devlog/2026-10-06-weighted-cc-dedup-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-10-06-weighted-cc-dedup-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-10-06-weighted-cc-dedup-01
kind: build
config: analysis/analysis-configs/2026-10-06-pond-deduplication-01.yaml
---

## Session 2026-10-06

### Prompts
(Summary of a multi-message session; continues 2026-10-06-deduplicate-cache-01.)
- Compare, on the cached duplicate graphs at each dataset's own threshold, how many non-clique edges the components hold and how many clusters complete-linkage and correlation clustering produce; then whether edge weights can enter the correlation-clustering objective ("Yes, run the weighted version with option 1").
- "Let's start working on implementing correlation clustering into a `analysis/deduplication.py` script ... (a) weighted correlation clustering to find clusters, and then (b) from each keep its center -- the data point with maximum average edge weight to all other items within its cluster. The resulting artifact should be a deduplicated dataset that we can save to `analysis/results/deduplication`."
- Submit the three full jobs; fix the error one of them hit.

### Implemented
New `analysis/deduplication.py` clusters an extraction's rows by exact weighted correlation clustering over the cached duplicate graph (threshold-centred costs, tau from the dataset config, non-strict-equal pairs forbidden, every component proven optimal or a hard error), keeps each cluster's center (highest mean cached weight, ties to lowest measurement id then row), and merges provenance lists into it. It writes the deduplicated records, a per-row cluster audit and a metadata file under the gitignored `analysis/results/deduplication/`, after checking the extraction file still matches the cache. Configs are `analysis/analysis-configs/2026-10-06-{pond,supermat,nfix}-deduplication-01.yaml` plus a pond smoke config; the script is allowlisted in the analysis submit wrapper (local, untracked wiring). Rung 1 tests, a smoke run checked against an independent brute-force prediction, and a determinism check passed; the first full submission failed on an over-strict measurement-id uniqueness guard (list-expanded rows legitimately share ids), which was fixed and the three full runs completed. Which row is kept is decided by the tie-break for most multi-row clusters, an open question.

### Commits
86453a7 analysis: deduplication.py, weighted correlation-clustering dedup with center selection
