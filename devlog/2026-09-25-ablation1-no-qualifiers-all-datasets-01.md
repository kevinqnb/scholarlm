<!-- Public devlog entry: scholarlm/devlog/2026-09-25-ablation1-no-qualifiers-all-datasets-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-25-ablation1-no-qualifiers-all-datasets-01
kind: build
---

## Session 2026-09-25

### Prompts

- Extend Ablation 1's no-qualifiers direct-extraction option
  (2026-09-25-ablation1-no-qualifiers-01) to NuExtract3, LangExtract, and
  GLiNER, and to every dataset (previously pond-only, Ablation-1-only) --
  same schema/prompt/example changes, then new experiment configs per
  method/dataset.
- Fix `run_baseline_gliner.py` silently ignoring `params.ocr_dir`.
- Remove the `identifiers` field from every direct-extraction method's
  schema and prompt, across all four datasets (including measeval, added
  after a follow-up question about why it had been left out).

### Implemented

`include_qualifiers` is now a symmetric option across all four
direct-extraction methods (Ablation 1, NuExtract3, LangExtract, GLiNER) and
all four dataset configs (pond, nfix, supermat, measeval): each dataset
config gained a `direct_extraction_schema_no_qualifiers`/
`_prompt_no_qualifiers`/`nuextract_examples_no_qualifiers` triple, and
`measurementlm_nuextract3.py`/`measurementlm_langextract.py` gained
constructor overrides for the shared instructions text (GLiNER needed no
schema change -- its quantity fields are hardcoded, gated by a new
`include_qualifiers` flag instead). `run_baseline_gliner.py` now accepts and
respects `params.ocr_dir`, matching the other baseline runners' convention.
`identifiers` was removed from every direct-extraction schema/prompt (the
real 7-step pipeline's own entity schema keeps it; that's its actual use).
12 new no-qualifiers experiment configs drafted across
ablation/baseline_nuextract3/baseline_langextract/baseline_gliner ×
pond/nfix/supermat. 940 tests passing (up from 893); rung 3 (tiny
end-to-end) not yet run for any of the new arms -- do not submit before it
passes.

### Commits

- `d614549` Extend include_qualifiers to NuExtract3/LangExtract/GLiNER; drop identifiers from direct extraction
