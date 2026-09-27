<!-- Public devlog entry: <repo>/devlog/<id>.md. Written by /devlog as a trimmed
version of the private build note at notes/<project>/builds/<id>.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-25-langextract-repetition-penalty-01
kind: build
---

## Session 2026-09-25

### Prompts

- "I've found an issue with LangExtract where it will repeatedly generate blank
  tokens until erroring out. Can we update `src/scholarlm/measurementlm_langextract.py`
  to use a repetition penalty parameter? We'll need to make sure it actually
  reaches the model."
- "I don't know, this seems like a pretty complicated change. Was there anything
  simpler we could have done to reduce repetition?"
- "yes, do the refactor"

### Implemented

Added `repetition_penalty` support to the LangExtract baseline
(`src/scholarlm/measurementlm_langextract.py`). langextract's own OpenAI provider
silently drops `repetition_penalty` (it's not in either of its two internal
key-forwarding allowlists, and langextract never uses `extra_body`), so a plain
provider kwarg would never have reached vLLM. Fixed with a small
`OpenAILanguageModel` subclass (`_build_vllm_openai_model`) that injects it into
`extra_body`, built directly by `fit()` and passed to `lx.extract(model=...)` — a
first-class langextract parameter for a pre-configured model instance, avoiding an
earlier, more complex design that registered the subclass with langextract's
provider router. `experiments/run_baseline_langextract.py` gained a
`repetition_penalty_override` param mirroring the existing `temperature_override`
pattern, exposed via `params.repetition_penalty` and recorded in
`run_metadata.json`. No experiment config for this session (code build only).

### Commits

- `2cbede7` Forward repetition_penalty to vLLM in the LangExtract baseline
