<!-- Public devlog entry: <repo>/devlog/<id>.md. Written by /devlog as a trimmed
version of the private build note at notes/<project>/builds/<id>.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-18-langextract-baseline-01
kind: build
config: experiments/experiment-configs/pond/baseline_langextract/2026-09-18-pond-baseline-langextract-gemma27b-01/2026-09-18-pond-baseline-langextract-gemma27b-01.yaml
---

## Session 2026-09-18

### Prompts

> Let's adopt google's `langextract` method as a baseline extraction method in
> `src/scholarlm/measurementlm_langextract.py`. Their library's code (which we'll
> need to install) should handle most of the heavy lifting. The code we write
> should be a lightweight wrapper designed to translate between the standard
> MeasurementLM input/output conventions. The model should be given full text
> documents, along with the DIRECT_EXTRACTION_PROMPT, and some examples, and should
> output a structured set of measurements in the same form that MeasurementLM does.
> Importantly, we want to run this using models served locally with vLLM, OpenAI
> API style.

> [Two design questions] Real langextract chunking (not one giant chunk per doc);
> reuse the NuExtract baseline's existing synthetic examples rather than authoring
> new ones or drawing from a real paper.

> Use uv add.

> Can you explain to me why we need to strip page tags and why we need titles as
> input? [On explanation] Simplify and keep page tags. Drop titles.

> Let's use gemma-3-27b. Write a minimal experiment config and I will run it.

> [After a run finished with 0 records, root-caused as a prompt contradiction and
> fixed] ... [After a follow-up run] Re-run it and check the attribute names.

### Implemented

Added `MeasurementLMLangExtract` (`src/scholarlm/measurementlm_langextract.py`),
wrapping Google's langextract library as a baseline extraction method served
through a local vLLM OpenAI-compatible endpoint. Reuses each dataset's existing
`direct_extraction_schema`/`direct_extraction_prompt` (Ablation 1's flat schema)
and `nuextract_examples` (already-audited synthetic passages) rather than
authoring new per-dataset resources; langextract handles chunking, prompting, and
character-level grounding itself. Wired up via `experiments/run_baseline_
langextract.py` and a new `baseline_langextract` experiment type in
`experiments/utils.py`; added `langextract[openai]` as a dependency
(`uv add`). 13 unit tests added (`tests/test_measurementlm_langextract.py`), all
offline. A rung-2 smoke run (gemma-3-27b/vLLM, one pond paper,
`2026-09-18-pond-baseline-langextract-gemma27b-01.yaml`) initially completed
cleanly but with zero records — root-caused to each dataset's `direct_extraction_
prompt` prescribing an output envelope that conflicts with langextract's own
contract — fixed and reverified with a second run producing 5 plausible,
page-grounded records.

### Commits

- `0701668` Add langextract baseline: chunked, schema-grounded extraction via
  Google's langextract library

## Session 2026-09-18 (continued)

### Prompts

> [Asked how `extraction_class` works, how much of the module is langextract
> "as intended" vs. our own engineering, and the alternative design] ... I
> didn't realize we were doing deduplicate. If we are, we should not be. That
> is not part of LangExtract, that is part of the method this repo has
> designed. LangExtract is a baseline for comparison. We need to be as
> faithful as possible to the original implementation and nothing else.

> [Asked how `attribute` is constrained today] Yes, let's use an
> output_schema. Wire that up. And get rid of the code which throws items out
> for not satisfying the attribute. We should keep those, and just have them
> be invalidated by the matching later, if anything.

> Let's run this test again with schema constraints. Before we submit, please
> walk me through the rest of the run parameters for this model. How would
> you advise to set them in a real run?

> Go ahead with (1-3). Stick with the default 8192.

> Let's try vllm 0.24.0. Change the model config for gemma. ... Submit the
> smoke test again.

### Implemented

Removed `_deduplicate()` and the out-of-vocabulary-attribute filter from the
langextract baseline's `fit()` — both are this repo's own post-processing
(needed by ChatExtract/NuExtract's own calling conventions), not part of
langextract's method, so a faithful baseline must not apply them. Added a
real generation-time vocabulary constraint instead: `_build_output_schema`
passes an explicit JSON Schema to `lx.extract(output_schema=...)` that
enum-constrains `attribute`, since langextract's own example-inferred schema
only types by Python type, never by vocabulary value. Threaded the model's
own `max_tokens` through as `language_model_params.max_output_tokens`. Test
suite grew from 13 to 18. A live schema-constrained smoke run hung under vLLM
v0.10.1's guided decoding; bumped `gemma-3-27b.yaml`'s vLLM image to v0.24.0
(`experiments/model-configs/extraction/gemma-3-27b.yaml`), which resolved it
cleanly — the resubmitted run produced 14 records with zero out-of-vocabulary
attributes, confirming the new constraint. Prepared, not yet submitted, a
10-paper rung-3 config
(`2026-09-18-pond-baseline-langextract-gemma27b-10papers-01.yaml`).

### Commits

- `e31b34e` Faithfulness fixes for the langextract baseline: drop
  _deduplicate and the out-of-vocabulary attribute filter, constrain
  `attribute` via output_schema, bump gemma-3-27b's vLLM image to v0.24.0
