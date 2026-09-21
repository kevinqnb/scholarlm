<!-- Public devlog entry: scholarlm/devlog/2026-09-20-table-cleaning-separation-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-20-table-cleaning-separation-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-20-table-cleaning-separation-01
kind: build
---

## Session 2026-09-20

### Prompts

- Fully separate table cleaning from `MeasurementLM`: create a standalone
  `TableCleaner` class, remove the `clean_tables` and `has_tables` flags, and
  retroactively fix any experiments/configs in `experiments/` that depended on the
  old behavior.
- Clarified that `ocr_dir` stays a plain directory-path string — only the auto-decide
  logic around it was being removed, not its type or plumbing.
- Go ahead and implement.

### Implemented

Table cleaning is now a standalone `TableCleaner` class (`src/scholarlm/table_cleaner.py`),
not a mode of `MeasurementLM`. The shared batched-call plumbing both classes need
(`_acall`/`_call_batch` retry/concurrency handling, page/table tag helpers) was
pulled into a new `BatchLLMBase` that both inherit, rather than duplicated.
`MeasurementLM` lost `clean_tables`/`cleaned_ocr_output_dir`/`_clean_tables` entirely;
`DatasetConfig.has_tables` and the auto-decide branch it fed in `run_extraction.py`/
`run_ablation.py` are gone outright — extraction and ablation runners no longer clean
tables themselves. `run_table_cleaning.py` now builds a `TableCleaner`; its output
directory is what `params.ocr_dir` should point extraction/ablation configs at.
Added `tests/test_table_cleaner.py` (6 tests, hand-built fixtures, no network) and
updated the tests that referenced the removed flags. Retroactively pinned
`params.ocr_dir` on the 6 committed experiment configs (pond×4, nfix, supermat) that
relied on the deleted auto-clean branch, verifying the pond/supermat cleaned corpora
already existed on disk before pointing at them; nfix pinned to raw OCR explicitly
since no cleaned corpus exists for it. Full test suite: 577 passed, 1 skipped. Rung 1
(unit tests) only — rungs 2-3 need a cluster shell with a live vLLM server.

### Commits

- d40846c Separate table cleaning from MeasurementLM into a standalone TableCleaner
