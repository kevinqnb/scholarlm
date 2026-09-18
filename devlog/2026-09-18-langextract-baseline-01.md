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
