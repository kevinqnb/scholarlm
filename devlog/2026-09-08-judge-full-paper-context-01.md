<!-- Public devlog entry. Trimmed from notes/scholarlm/builds/2026-09-08-judge-full-paper-context-01.md. -->

---
id: 2026-09-08-judge-full-paper-context-01
kind: build
---

## Session 2026-09-08

### Prompts

> Update the judgement process in this repository so that we always judge against
> full paper text instead of per-page text. This may include changes to
> `src/scholarlm/judgementlm.py`, as well as individual judgement experiment
> scripts. This should be applied to all judge models, including both external
> APIs, and models locally run with nnsight.

> Can we add fail loud precautions to `run_judge_local._judge_one`?

Scoping answers: judge context only (no metric/parser rewrite), commit
independently of the pending probe re-run, validate to rung 2 (unit test + smoke
+ model-free blast-radius estimate) and stop before any GPU run.

### Implemented

`experiments/judge_common.prepare_chat_entries` — the one prompt builder shared
by every judge backend (`run_judge_local.py`, `run_judge_interp.py`,
`run_jacobian_lens.py`, `run_attribution.py`) — now puts the whole OCR document
in `## CONTEXT` instead of slicing to the extracted `page_number` block(s); the
`context_override` path (augmentation pipeline) is unchanged. `JUDGE_INSTRUCTIONS`
wording updated to match, and the entry key `page_text` renamed `context_text`.
This is an eval-logic change: it invalidates every judge number computed with
page-level context (the old slice was a median ~10% of the paper; the context
changes for 100% of measurements across pond/nfix/supermat). Separately,
`run_judge_local._judge_one` now raises on an API error, an empty response, or a
response with no true/false verdict, and the runner aborts before writing a
partial `responses.json` — previously those became a silent `judgement: null`
that biased `run_judge_combine`'s majority vote. `src/scholarlm/judgementlm.py`
needed no change. No `configs/<id>.yaml` — this is a code build, not an
experiment. Rungs 1–2 pass (full suite green; all prompts fit the 90k served
window with headroom); rungs 3–4 (old-vs-new judgement diff, full re-judge) are a
GPU experiment for a later session.

### Commits

- `e7da363` judge: judge against the full paper, not the extracted page(s)
- `2c3f8ea` run_judge_local: fail loud on a dropped or unparseable judge response
- `30ae210` test_probe_augment: paired tests for b10c93b (feasible-but-no-edits skip)
