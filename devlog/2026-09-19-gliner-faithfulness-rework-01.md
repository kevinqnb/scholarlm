<!-- Public devlog entry: scholarlm/devlog/2026-09-19-gliner-faithfulness-rework-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-19-gliner-faithfulness-rework-01.md -->

---
id: 2026-09-19-gliner-faithfulness-rework-01
kind: build
config: experiments/experiment-configs/pond/baseline_gliner/2026-09-19-pond-gliner-large-tinye2e-01/2026-09-19-pond-gliner-large-tinye2e-01.yaml
---

# GLiNER2 baseline faithfulness rework

## Session 2026-09-19

### Prompts

> Let's walk through the implementation of measurementlm_gliner.py and
> evaluate it for overall faithfulness of implementation... The deduplicate
> method is added on top of the base model... it is extra code from my own
> methods that got added on... it should be removed. We should be extracting
> everything that measurementlm.py and other baselines collect, for fair
> comparison. Take a look at the overall prompts given to the model... are
> these fair and consistent with other methods?

> (1) We don't need to extract the 'identifiers' field... drop it everywhere.
> (2) The gliner field descriptions should come directly from existing entity
> or event schemas. This should also be the case for the nfix/supermat/measeval
> datasets as well. (3) Please explain what the threshold parameter controls.

> What is this? Don't rewrite schemas used for MeasurementLM. Just FOR GLINER,
> use descriptions that are taken word for word from MeasurementLM
> prompts/schemas. For example, I would borrow directly from the direct
> extraction prompt/schema.

> Keep it at 0.5 everywhere.

> Now please write full rung 4 configs for running Gliner.

> I thought we had run those? Yes draft tiny-e2e scripts and submit them.
> Skip the smoke test.

### Implemented

Removed `MeasurementLMGliner`'s borrowed `_deduplicate()` call (matching the
ChatExtract/NuExtract3 precedent, not NuExtract v1's chunk-merge precedent),
and expanded its field scope from name+value+units(+date) to every
entity/event field named in a new GLiNER-only `DatasetConfig.gliner_field_descriptions`
mapping, populated per dataset with text copied verbatim from each dataset's
own `_DIRECT_EXTRACTION_PROMPT` bullets rather than the real pipeline's
pydantic schemas. `identifiers` and measeval's `quantity` are deliberately
excluded everywhere. Also fixed a silent `units[:4]` truncation, removed a
dead config fallback, and added central-value/range guidance to the value
field. Added a `resources:` block to `gliner-{base,large}-v1.yaml` (missing
entirely before, so no `baseline_gliner` job could resolve through
`submit.sh`), and wrote 8 new experiment configs (rung-3 tiny-e2e + rung-4
full-run, one pair per dataset). The 4 tiny-e2e configs were submitted for
real and checked against each paper's prior real GLiNER output; 3 of 4 landed
on-prediction, one (nfix) surfaced a real GLiNER false positive traced to an
unrelated reagent-purity value in the source text, not a code bug.

### Commits

132b96d Rework GLiNER2 baseline for faithfulness: drop borrowed dedup, expand field scope

