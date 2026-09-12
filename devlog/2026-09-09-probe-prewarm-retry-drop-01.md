---
id: 2026-09-09-probe-prewarm-retry-drop-01
kind: build
---

# Probe-dataset generation: retry and drop failed gpt-oss prewarm calls

## Session 2026-09-09

### Prompts

> Our goal today is to fix the `create_probe_dataset.py` scripts for pond, nfix,
> and supermat. Currently, these are set to fail loud on one invalid case. In my
> previous run to generate a pond synthetic dataset, the script failed from a
> single bad case where the generation exceeded the allowed length. Instead, we
> should (1) retry failed cases, and (2) drop them with a warning if they fail
> again after retrying.

> can we actually do this on a new branch off of naacl? Let's call it
> 'diff-experiments'

> Please make sure naacl is committed, pushed and up to date before switching

> yes, start with the tests

### Implemented

`GptOssClient._run_batch` in `src/scholarlm/utils/probe_augment.py` (shared by
the three `data/{pond,nfix,supermat}/create_probe_dataset.py --augment` paths)
now resamples a truncated / empty / unparseable gpt-oss prewarm response
`prewarm_max_retries` times (default 2) before dropping it: the dropped key is
recorded in `_dropped_detail` and `dropped_prewarm.json`, kept out of the
response cache, and served a `feasible: false` decline on the real pass so the
row becomes a clean skip (tallied under a `dropped` bucket) instead of aborting
the run. A `prewarm_drop_ceiling` fraction (default 0.02) drives an in-batch
circuit breaker so a wedged server still fails fast. Each `create_probe_dataset.py`
gains `--augment-prewarm-max-retries` and `--augment-prewarm-drop-ceiling`
(defaults are the committed source of truth). Also fixed a pre-existing bare-`%`
in a supermat `--help` string. Rung 1 (12 new + 3 updated tests in
`tests/test_probe_augment.py`, full suite 295 passed / 1 skipped) and Rung 2
(in-process `run_and_write` end-to-end + a `--augment-stub` CLI smoke) pass; the
real `gpt-oss-120b` regeneration is a follow-up experiment session under
`configs/2026-09-03-probe-synthetic-augmentation-01.yaml`.

### Commits

- `58c8844` probe generation: retry then drop failed gpt-oss prewarm calls
