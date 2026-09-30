<!-- Public devlog entry: scholarlm/devlog/2026-09-29-pond-probe-v3-rebuild-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-29-pond-probe-v3-rebuild-01
kind: build
config: experiments/experiment-configs/pond/probe_augment/2026-09-29-pond-probe-augmentation-v3-rung3-01/2026-09-29-pond-probe-augmentation-v3-rung3-01.yaml
---

## Session 2026-09-29

### Prompts

- Assess the pond probe-dataset generator: what fields each row has and whether they match the pond dataset config, how synthetic valid and invalid rows are made, and how realistic they are for real extractions.
- Make the v2 training set as faithful as possible to a real extraction setting: (1) only the fields the judge now sees; (2) reflect that `value` is now a string span; (3) use the table-cleaned OCR the pond judge configs use; (4) find and handle unit changes that leave the measurement valid.
- Use verbatim value spans only in a few unambiguous cases, and never in the primary test set.
- Keep the not-found GT rows, keep the full span, map ecosystem onto the extraction vocabulary, keep date injection; submit the rung-3 job.

### Implemented

Rebuilt the pond `--augment` path of `data/pond/create_probe_dataset.py` (shared helpers in `src/scholarlm/utils/probe_augment.py`, `params.ocr_dir` in `experiments/run_probe_augment.py`) so v3 rows carry only judge-visible fields, ecosystem in the extraction vocabulary, the paper's own name casing, and context from the table-cleaned OCR (`--ocr-dir` is required). Verbatim uncertainty spans replace `value` only in train and diagnostic files and only where every occurrence in the paper agrees; `bad_units` no longer draws numerically equivalent units and `bad_value` has a numeric-aware collision guard for pond. The default v1 path and nfix/supermat outputs were checked byte-identical. Unit tests are in `tests/test_pond_probe_v3.py` plus additions to `tests/test_probe_augment.py` and `tests/test_run_probe_augment.py`; rungs 1 and 2 passed and a rung-3 yield check (`2026-09-29-pond-probe-augmentation-v3-rung3-01.yaml`) was submitted via `experiments/submit.sh`. Probe numbers on the v2 pond files are not comparable to v3.

### Commits

- e3dc02c Pond probe v3: judge-visible columns, extraction-vocab ecosystem, cleaned-OCR context, equivalence-aware negatives
