<!-- Public devlog entry: scholarlm/devlog/2026-09-21-compute-time-recording-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-21-compute-time-recording-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-21-compute-time-recording-01
kind: build
---

## Session 2026-09-21

### Prompts

- Audit whether `experiments/` has good functionality for recording compute time of
  experiments in detail, across all MeasurementLM, ablation, and baseline methods.
- Go ahead and fix the 3 prioritized gaps: resume-safe per-step timing, token/call
  accounting, and `run_single_step` writing no metadata at all.
- Clarified what timing exists vs. is missing for ablations/direct-mode/baselines.
- Add per-step timing to ablations 2-6 specifically — ablation 1, direct mode, and
  the baselines (except ChatExtract) are single-call methods with no steps to break
  down.

### Implemented

`experiments/utils.py`'s `write_run_metadata` now merges `step_seconds` and
`token_usage` across invocations on the same output dir instead of the last write
overwriting the rest — the fix for the top finding: a checkpointed run resumed
across separate job submissions used to report only the last invocation's time, with
nothing on disk even flagging that. `MeasurementLM._acall` (the chokepoint shared by
the base pipeline, all 6 ablations, `MeasurementLMv2`, and the vLLM-backed baselines)
now sums prompt/completion tokens and call counts, not just a running max.
`run_extraction.py`'s checkpointed pipeline and `run_extraction_v2.py` now time each
step individually; `run_single_step` (previously wrote no metadata at all) now does.
GLiNER and langextract explicitly mark `token_accounting="n/a: ..."` rather than
silently omitting it, since both bypass `_acall` entirely. On follow-up, added
per-step timing to ablations 2-6: ablations 4-6 inherit it for free from the base
`fit()`; ablations 2 and 3 override `fit()` with their own merged steps, so each got
step names reflecting what it actually merges (e.g. ablation 3's
`pair_provenance_full_context` vs. ablation 2's `pair_provenance` — different
mechanisms, deliberately not sharing a name). `values_text`/`values_tables` are kept
as separate keys everywhere rather than one combined `values`, so ablation 5's
table-only change stays visible rather than blended with unchanged text extraction.
4 new test files plus additions to `tests/test_measurementlm.py`; full suite: 589
passed, 1 skipped. Rung 1 (unit tests, plus stubbed integration tests standing in for
a live-model smoke run — none was reachable in-session) only; rungs 2-3 against a
real model need a cluster shell.

### Commits

- bda4bee Add resume-safe per-step timing and token accounting to run_metadata.json
