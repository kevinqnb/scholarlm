<!-- Public devlog entry: scholarlm/devlog/2026-09-20-extraction-field-simplification-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-20-extraction-field-simplification-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-20-extraction-field-simplification-01
kind: build
---

## Session 2026-09-20

### Prompts

- Improve nfix's `substrate_type` event-field description; then tighten it into a
  closed vocabulary (`"benthos"` / `"water column"` / `"other"` only).
- Per-dataset extraction field simplification across pond, nfix, supermat, and
  measeval: restrict `identifiers` to the real pipeline and its ablations (never
  baseline methods, never the judge); rename `additional_details` →
  `event_details` everywhere as an explicit catch-all for distinguishing
  measurements; drop or fold several redundant entity/event fields per dataset
  (pond's `location`; nfix's `site_type`→`ecosystem_type`, `location`,
  `nfix_method`, `sample_depth`; supermat's `sample_details`, `pressure`,
  `me_method`); un-filter nfix's `substrate_type` from the judge.
- Mid-session correction: keep pond's `location` after all — no, drop it, but leave
  the downstream matching code that references it unfixed for now (separate
  planned update).
- Run a smoke test on one nfix paper and one measeval paper.
- Clean up `experiments/dataset-configs/measeval.py`'s docstrings/comments to
  concise, current-state descriptions (no change history).

### Implemented

Rewrote the entity/event schemas, prompts, JSON output-format blocks, GLiNER
descriptions, and NuExtract few-shot examples in all four
`experiments/dataset-configs/*.py` files to match the simplified field set
above, and added a new `DatasetConfig.baseline_filter_fields` (mirroring
`judge_filter_fields`) wired through `src/scholarlm/measurementlm_nuextract.py`
and `measurementlm_nuextract3.py` so `identifiers` is extracted by the main
pipeline and its ablations only, never by the NuExtract-2.0/NuExtract3
baselines. measeval gained an `identifiers` field it previously lacked, for
parity with the other three datasets. Verified with the full `pytest` suite
(564 passed) plus targeted schema/filtering checks, and two rung-2 smoke runs
via `experiments/submit.sh`: measeval succeeded fully on a real paper;
nfix's existing tiny-e2e config turned out to target a paper excluded by
`_nfix_paper_filter` (pre-existing, unrelated to this change), so a
replacement smoke config
(`experiments/experiment-configs/nfix/extraction/2026-09-20-nfix-extraction-gemma27b-schema-smoke-01/2026-09-20-nfix-extraction-gemma27b-schema-smoke-01.yaml`)
confirmed the new schema round-trips cleanly through a real extraction call.
Left `analysis/ablation.py`'s matching rules (which still reference pond's
`location` and nfix's `site_type`) and a few other decoupled, now-stale
references unfixed by explicit scope decision.

### Commits
533d684 Simplify per-dataset extraction fields; restrict identifiers to the real pipeline
