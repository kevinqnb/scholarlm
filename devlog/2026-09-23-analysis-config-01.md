<!-- Public devlog entry: scholarlm/devlog/2026-09-23-analysis-config-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-23-analysis-config-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-23-analysis-config-01
kind: build
---

## Session 2026-09-23

### Prompts

- Set up some sort of config in the analysis directory to be read by (at
  least) `analysis/match_cache.py` and `analysis/recovery_validity.py`, and
  potentially other analysis scripts as well: per-experiment, the ids that
  feed into it and the high-level parameters to run it with.
- Design decisions: keep the existing ad-hoc CLI alongside `--config`,
  mutually exclusive, rather than going config-only; a declared
  `judge_combine_ids` override should still be verified against its
  extraction id, not blindly trusted.
- Why does the analysis config's `seed` need to agree with the experiment
  configs' `defaults.seed`? Is that not a separate process? -- caught as a
  mistake, fixed.
- Write an example config with some of the latest extraction results from
  the main pipeline and other baselines (no judgements yet, many without
  match caches yet either). Then the same for nfix and supermat.

### Implemented

New `analysis/analysis_config.py`: a shared loader for
`analysis/analysis-configs/<id>.yaml`, reusing the harness's standard
`id`/`project`/`description`/`seed`/`params` envelope. `params.experiment_ids`
is a non-empty, duplicate-free id list shared by every consumer script; a
script's own parameters live in a namespaced `params.<script_name>` section
via `get_section()`, which rejects unknown keys and requires every key
explicit (no defaults). `seed` is required but, unlike an
`experiments/experiment-configs/` entry, deliberately **not** checked
against `experiments/config.yaml`'s `defaults.seed` -- it seeds
`recovery_validity.py`'s bootstrap resample, an unrelated RNG stream from
the one that seeds a model run.

`analysis/match_cache.py` and `analysis/recovery_validity.py` both gained
`--config <path>`, mutually exclusive with their existing ad-hoc CLI
(unchanged). `recovery_validity.py`'s section additionally supports an
optional `judge_combine_ids` id-to-id override map, validated as
string-to-string and checked against the id list; a declared override is
still verified against its extraction id the same way automatic resolution
verifies a scanned candidate (`verify_judge_combine_id`, sharing a
`find_judge_combine_id`-derived helper) -- it only skips the scan, never the
check. `compute_metrics_for_id` gained one new `judge_combine_id=None`
parameter to carry this through, its default preserving prior behavior
exactly. Every output row now carries `analysis_config_id` for traceability.

Wrote three example configs in `analysis/analysis-configs/`: the latest full
pond/nfix/supermat main-pipeline runs (gemma-3-27b, gpt-oss-120b,
llama-3.1-8b where available) against the latest full baseline runs
(langextract, gliner, nuextract3), `compute_validity: false` since none have
judge coverage yet. `baseline_chatextract` is excluded from all three (its
output schema has no `point_value`/`units` fields at all -- flagged, not
fixed); nfix and supermat additionally can't run yet since their
`DatasetConfig`s have no matching rules set -- documented in each config
rather than invented.

28 new unit tests (`tests/test_analysis_config.py` plus additions to the
`match_cache`/`recovery_validity` test files); full suite 737 passed, 1
skipped. Rung 3: ran `recovery_validity.py` once via the old CLI and once
via an equivalent `--config` against a real cached id -- byte-identical
output apart from the new column.

### Commits

- 83c4d92 Add --config support to match_cache.py and recovery_validity.py
