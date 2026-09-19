<!-- Public devlog entry: scholarlm/devlog/2026-09-19-chatextract-rework-01.md. Written by
/devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-19-chatextract-rework-01.md -->

---
id: 2026-09-19-chatextract-rework-01
kind: build
config: experiments/experiment-configs/pond/baseline_chatextract/2026-09-19-pond-chatextract-gemma27b-smoke-01/2026-09-19-pond-chatextract-gemma27b-smoke-01.yaml
---

# ChatExtract baseline rework

## Session 2026-09-19

### Prompts

> Evaluate measurementlm_chatextract.py for correctness and faithfulness to the
> ChatExtract paper. Specifically: when is _table_caption used and why? In
> general, we want as little engineering built on top of the original method as
> possible — is there anywhere the implementation doesn't meet that goal?

> [On the findings] Remove _table_caption entirely, don't guess a caption when
> raw OCR has none. Failures should be counted and documented, not swallowed.
> Fix the yes/no matching bug with anchored matching. The table cap can go to
> 64, or just be removed if not necessary. Table processing should be more
> robust — have the model hand back structured data directly instead of us
> parsing free text. Clean up the dead code. Remove _deduplicate() — that's
> MeasurementLM's own contribution, not something to credit a reimplemented
> reference method with.

> Why don't we just explicitly tell it something like "in the form of json..."

> Is attribute-type/units enforcement (exact match, drop-on-failure) done here
> the way it is for other baselines? Is anything ever dropped for failing it?

> Evaluate the prompts reaching ChatExtract from instruction_prompts.py and the
> four dataset configs for faithfulness, fairness/consistency with
> MeasurementLM, and completeness across all four datasets.

> [After finding supermat had no chatextract_property_names/entity_noun, so
> every prompt read "a value of tc"] Go ahead and make this change, yes.

> Write a unit test file for measurementlm_chatextract.py if we do not already
> have one. Then get ready to run this on full data. Write and submit smoke and
> tiny-e2e test configs for all datasets.

> Don't use gpt-5-mini or any API model. Use only gemma-3-27b and write/submit
> configs for real jobs when we need a gpu. [Hypothesis] No new hypothesis,
> really — just that the method runs cleanly and is consistent with results
> we've seen before for it.

### Implemented

Reworked `MeasurementLMChatExtract` (`src/scholarlm/measurementlm_chatextract.py`)
after a faithfulness/fairness review: removed the `_table_caption` heuristic
(raw-OCR tables now pass through unchanged rather than getting a guessed
caption), replaced substring yes/no matching with anchored `_is_yes`/`_is_no`
(the old check false-matched inside "know"/"not"/"none"), switched table
extraction from CSV/pipe-splitting to guided-JSON decoding
(`_TABLE_RESPONSE_FORMAT`, fixing a comma-in-value parsing bug), added tracked
failure counting (`self.failures`, surfaced into `run_metadata.json`) in place
of a bare `except Exception: return []`, removed the unjustified
`max_tables_per_document` cap, and removed the `_deduplicate()` call from
`fit()` (same precedent as the NuExtract3/LangExtract baselines — dedup is
MeasurementLM's own contribution, not a baseline's). Fixed
`experiments/dataset-configs/supermat.py`, which had no
`chatextract_property_names`/`chatextract_entity_noun` at all and so degraded
every prompt to the bare attribute key ("a value of tc"). Added 39 unit tests
(`tests/test_measurementlm_chatextract.py`), including a cross-dataset
regression locking in the supermat fix. Wrote and ran 8 new experiment configs
(`baseline_chatextract`, smoke + tiny-e2e per dataset) against `gemma-3-27b` as
real SGE GPU jobs — all 8 completed with zero failed work items, and every
tiny-e2e record count met or exceeded its pre-rework reference for the same
paper.

### Commits

- `ee4e7dd` Rework ChatExtract baseline for faithfulness and robustness
