---
id: 2026-09-18-measurementlmv2-01
kind: build
config: experiments/experiment-configs/pond/extraction_v2/2026-09-18-pond-gemma3-27b-v2-smoke-01/2026-09-18-pond-gemma3-27b-v2-smoke-01.yaml
---

# MeasurementLMv2: quantity-first extraction pipeline

## Session 2026-09-18

### Prompts

> Let's implement `MeasurementLMv2` in `src/scholarlm/measurementlmv2.py`. The key
> idea is to flip the order of extraction: enumerate all (page, attribute) pairs,
> prompt per pair for a list of quantities ({value, unit, CI lower, CI upper, CI,
> type, quantifier} — type/quantifier/CI are genuinely new, supporting ranges,
> inequalities, and confidence intervals). Deduplicate the collected quantities in
> code, then for each deduplicated quantity prompt with the full paper to fill in
> the remaining entity/event fields — one quantity can expand into several final
> measurements if genuinely distinct. Units enforced to a predefined set as in v1;
> attribute names attached to quantities manually, not model-generated; new
> instructions where necessary; reproducible via seeded calls; keep it simple.

> Don't call the clean-tables method — assume table cleaning is a separate upstream
> process, `fit()` takes only the OCR document list. Don't store full page text or
> excerpts on quantity records or in the contextualize prompt — it overwhelms the
> model with context.

> No GPU in this shell — write an experiment config and run it as a real (small)
> cluster job, using vLLM with gemma-3-27b, matching the shape of a real
> MeasurementLM experiment config. Check that results are reproducible.

> [After finding a table_cleaning run pointed to the wrong model] All we're doing is
> pointing to where MeasurementLMv2 should look for OCR files — that's a separate
> choice from the extraction model, which should stay gemma-3-27b.

> Does the other collected measurements look okay? Should we try to fix the
> over-attribution bug found in the smoke test? [Then, after realizing quantity
> records don't track table/cell provenance:] Just do the instruction fix, skip the
> table-locator fix — that needs plumbing that doesn't exist yet.

> Is there anything more we can do for determinism, or is this an expected
> consequence of running LLMs?

### Implemented

Added `MeasurementLMv2(MeasurementLM)`: collect quantities per (page, attribute) →
standardize units/formatting → deduplicate in code (no LLM call) → attribute each
deduplicated quantity to entity/event using the full paper. A new
`validate_quantity_shape()` enforces the type/quantifier/CI invariants and feeds the
collection step's retry validator. Three new prompts added to
`instruction_prompts.py`; no new dataset-config fields needed. Wired up
`experiments/run_extraction_v2.py` and an `extraction_v2` experiment-type so this
runs through the standard harness. 31 unit tests added
(`tests/test_measurementlmv2.py`). Two live smoke-test rounds on the cluster
(gemma-3-27b via vLLM) surfaced a real bug (a table column misread, compounded by
the standardize step performing a unit conversion it was explicitly told not to)
and a systemic entity over-attribution pattern (a value genuinely shared by two
entities gets attached to every similar entity in the same table). Tightened the
contextualization instructions to address the over-attribution pattern; this
narrowed but did not fully resolve it — a follow-up fix would need per-quantity
table/cell provenance that doesn't exist yet. Seed-determinism improved
substantially after the fix (the deduplicated quantity list came out bit-identical
across two runs) but isn't exact, consistent with vLLM's continuous-batching
behavior.

### Commits

- `cbd6442` Add MeasurementLMv2: quantity-first extraction pipeline
