<!-- Public devlog entry: scholarlm/devlog/2026-09-17-calibration-cross-domain-restore-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-17-calibration-cross-domain-restore-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-17-calibration-cross-domain-restore-01
kind: build
---

## Session 2026-09-17

### Prompts

- Diagnose why supermat's rung-4 probe-augmentation job crashed partway
  through, and fix it: a quick fix now, plus a full fix explained (but not
  yet implemented) for the future.
- Diagnose why the calibration script's cross-domain synthetic figures were
  empty and why two datasets were missing from some settings' figures.
- Restore cross-domain synthetic calibration, believed to have been dropped
  from the script at some earlier point.

### Implemented

Diagnosed and quick-fixed a crash in supermat's probe-augmentation pipeline:
gpt-oss occasionally typos the `rewrite` schema's `"replace"` key as
`"replace:"`, which the prewarm admission gate doesn't schema-check (only
JSON-parses), so it gets cached as a success and crashes the strict real pass
hours later. Evicted the 12 affected cache entries. The durable fix (schema
validation in the gate itself) is scoped but not implemented, written up
separately.

Restored cross-domain synthetic calibration in
`analysis/calibration_updated.py`, dropped by the 2026-09-16 id-addressed-contract
migration when it collapsed a previously-uniform per-dataset loop down to
`[train_ds]` only (reasonable when pond was the only dataset with a migrated
synthetic probe, silently stale once nfix/supermat got their own). A trained
probe is now scored against every other dataset's own synthetic test set
again, not just its own. Verified with an isolated single-pair smoke test
before recomputing full calibration figures/metrics/predictions for all
three judged settings.

### Commits

- 168a648 calibration: restore cross-domain synthetic evaluation; fix supermat probe-augment cache corruption
