<!-- Public devlog entry: scholarlm/devlog/2026-09-17-prewarm-schema-gate-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-17-prewarm-schema-gate-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-17-prewarm-schema-gate-01
kind: build
---

## Session 2026-09-17

### Prompts

- Implement the fix scoped (but not implemented) by the prior session: the
  `GptOssClient` prewarm admission gate only checked that a `rewrite`
  response parsed as JSON, not that it matched the `rewrite` schema, so a
  schema-invalid response (a `"replace:"` trailing-colon typo in one `edits`
  entry, supermat rung-4 job 7600542, 12/16,524 cached entries) passed
  admission, got cached as a "success," and only crashed the strict real
  pass hours later. Proposed fix: factor the schema check into a shared,
  non-raising function used both by the admission gate (resample-then-drop,
  like a parse failure) and by `rewrite_context` (still raises); validate the
  current caches once as a diagnostic; add a unit test using a real captured
  bad-response shape as a fixture.

### Implemented

Added `_rewrite_schema_error` (shared, non-raising schema check) and
`_prewarm_admission_error` (composes it with JSON-parsing) to
`src/scholarlm/utils/probe_augment.py`; `GptOssClient._run_batch`'s admission
gate now calls the latter instead of the old JSON-object-only check, so a
schema violation is resampled and, if it still doesn't validate, dropped
through the existing `dropped_prewarm.json`/circuit-breaker path — the same
safety net already in place for parse failures. `rewrite_context` still
raises on a schema violation, now only reachable via a pre-existing bad cache
entry rather than a fresh call. Deleted the now-redundant `_looks_like_json_object`
and `_parse_edits`. Updated `tests/test_probe_augment.py` for the new
behavior and added a fixture reconstructing the real incident shape, with
tests covering the validator, the admission gate, `_run_batch`'s drop path,
and `rewrite_context`'s hard-error on a stale cached copy. Ran the new check
once against all three datasets' current `probe_augment_cache.json` files
(49,591 entries total) as a diagnostic — zero further violations found, so
the fix changes nothing about previously computed results. No config for
this session (pure code fix, no experiment run). Full test suite:
416 passed, 1 skipped.

### Commits

- ab3ebdf probe_augment: validate rewrite schema at prewarm admission, not just JSON-object-ness
