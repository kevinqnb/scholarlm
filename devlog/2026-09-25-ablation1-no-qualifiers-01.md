<!-- Public devlog entry: scholarlm/devlog/2026-09-25-ablation1-no-qualifiers-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-25-ablation1-no-qualifiers-01
kind: build
---

## Session 2026-09-25

### Prompts

- Run an experiment removing the qualifier fields from Ablation 1's direct
  extraction (remove from all direct-extraction instructions, prompts, and
  schemas) -- nothing about the other fields should change. On pond, model
  gemma-3-27b. After extraction, run results through
  `analysis/postprocessing.py` to parse the qualifier/shape fields out of the
  raw `value` field, then compare recovery to the existing
  `2026-09-21-pond-ablation1-gemma27b-full-01` baseline. Hypothesis: recovery
  will be >25 percentage points greater.
- (Design call, via AskUserQuestion) Add an opt-in `params.include_qualifiers`
  flag (default true) rather than editing the shared ablation-1 code in
  place, which would have silently changed nfix/supermat's ablation 1 too and
  broken comparability with the existing baseline run.
- Also draft an experiment config to run this on the full pond dataset.

### Implemented

Added `DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS_NO_QUALIFIERS` to
`src/scholarlm/instruction_prompts.py` and `DirectExtractionItemSchemaNoQualifiers`/
`_DIRECT_EXTRACTION_PROMPT_NO_QUALIFIERS` to `experiments/dataset-configs/pond.py`
-- each identical to its original minus the 7 qualifier/shape fields, verified
byte-for-byte in tests. `src/scholarlm/config.py` gained two matching optional
`DatasetConfig` fields (nfix/supermat get `None`, untouched).
`MeasurementLMAblation1` now takes `direct_extraction_instructions` as a
constructor kwarg. `experiments/run_ablation.py` gained an opt-in, type-checked
`params.include_qualifiers` (default `True`, so every existing ablation-1
config is unaffected), which fails loud if set for a non-ablation-1 config or
a dataset with no no-qualifiers variant defined, and is recorded in
`run_metadata.json`.

Pre-flight diligence before drafting any config: confirmed via `git diff`
that the shared files gained additions only, and via `git log` that no
model-call-path code changed between the existing baseline's commit and
HEAD. The baseline's own recovery is currently 10.11%; simulating this
experiment's actual mechanism against the baseline's own output shows
postprocessing-based parsing agrees with the model's own qualifier
extraction 96.6% of the time it applies at all -- so a >25pp gain, if real,
is more likely driven by the simpler schema improving the model's core
field extraction than by better parsing. Drafted (not committed, not
submitted) a full-run config, same as the baseline except
`include_qualifiers: false`; verified it resolves and dry-run submits
cleanly. 893 tests pass (9 new, `tests/test_ablation1_no_qualifiers.py`).

Rung 3 (tiny end-to-end) is still outstanding before the drafted full-run
config can be submitted.

### Commits

- `0d86f44` Add opt-in include_qualifiers flag to Ablation 1's direct extraction
