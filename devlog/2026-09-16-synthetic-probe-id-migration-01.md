<!-- Public devlog entry: scholarlm/devlog/2026-09-16-synthetic-probe-id-migration-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-16-synthetic-probe-id-migration-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-16-synthetic-probe-id-migration-01
kind: build
---

## Session 2026-09-16

### Prompts

- Update `analysis/synthetic_probe_train.py` so it saves trained probes to
  `experiments/results/{dataset}/judge_interp/{synthetic judge run ID}/`,
  adapting the script to the new experiment contract, then retroactively copy
  the latest cached probes to the new locations.
- Migration scope: mint real experiment ids for the synthetic judge runs
  themselves too, not just relabel the probe output path.
- Correction on backfill scope: migrate the `v2`/`v2_diag`/`v2_primary`
  sources (the current ones), and drop the ad hoc "v2" naming from the ids
  entirely — dataset + judge + date + split already disambiguate.
- Both the synthetic judge output and the trained-probe pkls should move to
  the id-addressed tree; `analysis/calibration_updated.py` (which still reads
  the old tree) is explicitly left for a follow-up session.
- Flagged and stopped an in-progress full production retrain that had been
  started directly on the login node as an oversized "smoke test" — redirected
  to a reviewable SGE job instead.
- Explicit go-ahead to submit the verification job.

### Implemented

Migrated `run_judge_interp.py`'s `synthetic_file` mode from the old
date-addressed `data/experiments/` tree onto the id-addressed
`experiments/results/{dataset}/judge_interp/{id}/` tree used by every other
Tier-1 experiment type; `params.synthetic_name` is retired since the
experiment id's own slug now carries that label (a config that still sets it
is a hard error). Backfilled three experiment-configs
(`2026-09-10-pond-qwen-2.5-7b-synthetic-judge-{train,test-primary,test-diag}-01`)
reconstructing the historical pond synthetic-probe judge runs, and copied
their output plus the cached head probe / NTP calibrator into the new tree.
`analysis/synthetic_probe_train.py` gained a `--judge-run-ids` mode that
resolves dataset/judge/activations directly from an experiment id, mutually
exclusive with the legacy `--datasets/--judges/--judge-date/--source` flags
(kept working unchanged for runs still on the old tree). Added 15 unit tests
covering the new resolution path, fail-loud checks, and a tiny-fixture
end-to-end train-and-save. Verified via a real SGE job that retraining from
the migrated data reproduces the original cached probe byte-identical.

### Commits

- `7478b7f` judge_interp: migrate synthetic_file mode to id-addressed output; add --judge-run-ids to synthetic_probe_train.py
