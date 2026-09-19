<!-- Public devlog entry: scholarlm/devlog/2026-09-18-nuextract3-baseline-01.md. Written by
/devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-18-nuextract3-baseline-01.md -->

---
id: 2026-09-18-nuextract3-baseline-01
kind: build
config: experiments/experiment-configs/pond/baseline_nuextract3/2026-09-18-pond-nuextract3-smoke-01/2026-09-18-pond-nuextract3-smoke-01.yaml
---

# NuExtract3 baseline

## Session 2026-09-18

### Prompts

> Write a new NuExtract3 adapter, modeled on the old NuExtract-2.0-8B baseline.
> Use each dataset's DIRECT_EXTRACTION_PROMPT as NuExtract3's instructions field
> (avoiding separate per-attribute calls); use Pydantic schemas as extraction
> input, with attribute/units as Literal types over predefined sets; extract from
> full document text rather than PDF images, for a fair comparison against the
> other baselines; review ISSUE-nuextract-baseline.md and preemptively fix
> whatever transfers.

> numind/NuExtract3 is the model card. We can download it to the HF_CACHE.

> [Design decisions given directly] Units enforcement should be flat Literal
> enums, not a per-attribute discriminated union. attribute_info_dict is the
> units source of truth.

> [Correction] We should do no deduplication at all — that's a genuine part of my
> method, not something to build onto a baseline. Likewise, no cleaning step, and
> we don't need processed_pdf_dirs as input. Also not sure the max token limit is
> large enough for the full-text docs.

> Continue with the runner and model-config now. Let's run a small smoke test with
> a real GPU. [mid-turn] We should also write a model-config for the nuextract
> model.

> [After the first resource draft] Is an A100 with 48G really necessary? Isn't
> this a 4b model?

> How come the smoke test job is still sitting in the queue, did we set the config
> correctly? ... That's okay, the job actually finished. Can you check on it.

> Can you walk me through the code? What do the input and output schemas look
> like. This is supposed to be a baseline comparison and should be as faithful to
> the intended use as possible, with as little extra engineering on top of it as
> possible. To what extent is that the case?

> The only thing I think we should get rid of is retry then drop. Let's change to
> 2 retries, and do not drop results that don't fit the unit conventions. Those
> will just be ignored by the matching process later anyways.

> Sorry, can we actually revert the units retry and drop change?

### Implemented

Added `MeasurementLMNuExtract3` (`src/scholarlm/measurementlm_nuextract3.py`): one
vLLM call per document via NuExtract3's native `template`/`instructions`/ICL-example
chat-template interface (its fixed chat-template bug from NuExtract-2.0 — one image
placeholder per message — is what makes single-call-per-document possible), reusing
each dataset's existing `direct_extraction_schema`/`direct_extraction_prompt`/
`nuextract_examples` unmodified. `attribute` and `units` are typed as `Literal`
enums (built from `attribute_info_dict`) for the guided-decoding `response_format`,
kept separate from the lenient schema used to parse responses so one
out-of-vocabulary item can't invalidate an entire document's response; both fields
get a client-side retry-then-drop backstop (`max_retries=2`) behind that constraint.
No `_deduplicate()`/`_standardize()`/table-cleaning — all three are treated as the
real MeasurementLM pipeline's own contribution, not something a single-shot
baseline should be credited with. Added `experiments/run_baseline_nuextract3.py`
and `experiments/model-configs/baseline/nuextract3.yaml` (`numind/NuExtract3` on
`vllm-openai_v0.24.0.sif`, `L40S`/32G after correcting an initial resource request
that was sized by vendor precedent rather than model size), plus 33 unit tests in
`tests/test_measurementlm_nuextract3.py`. A mid-session faithfulness audit weighed
NuExtract3's own documented/benchmarked usage (no guided decoding at all) against
consistency with the rest of this repo's extraction suite; landed on keeping
guided decoding and the attribute/units vocabulary backstop for consistency, with
`max_retries` reduced from 4 to 2. Smoke-tested end-to-end on the cluster
(`2026-09-18-pond-nuextract3-smoke-01`, job 7641800): 9 clean records from one
paper, zero validation failures, zero context-length overflows.

### Commits

- `d0681e4` Add NuExtract3 baseline: single-shot, full-document-text extraction
