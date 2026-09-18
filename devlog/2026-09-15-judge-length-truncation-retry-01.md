---
id: 2026-09-15-judge-length-truncation-retry-01
kind: build
---

# Judge pipeline: bounded retry + flag for length-truncated verdicts

## Session 2026-09-15

### Prompts

> Running 2026-09-13-pond-gemma3-27b-extraction-gptoss120b-judge-local-01 and
> 2026-09-13-nfix-gemma3-27b-extraction-gptoss120b-judge-local-01 I found again 2
> cases where the model did not respond since it spent all tokens on reasoning. What
> I want to do is not fail the entire job because of these 2 entries. I understand
> the issue is there and I think 2 cases is a small enough portion that it should not
> affect results. Instead, let's just adjust the system to set those 2 cases, or any
> that fail because of the `judge response has no true/false verdict
> (finish_reason='length'): ''` error, to False. Please help me implement this and
> resubmit those jobs.

> Commit it, then submit both jobs.

### Implemented

`run_judge_local._judge_one` now retries only the narrow case of an empty response
with `finish_reason='length'` (the reasoning channel spending the whole token budget
before a verdict), up to a required per-judge `max_retries` from
`experiments/model-configs/vllm_judge/<judge>.yaml`; every other no-verdict case
still raises immediately, unretried. If the row is still empty after the retry
budget, it's recorded as `judgement: null, judgement_truncated: true` — not a
fabricated verdict — instead of aborting the whole run. `run_local_vllm_judge` writes
a `truncated_verdicts.json` manifest alongside `responses.json` when any row resolves
this way, and enforces a required `drop_ceiling` (an absolute row count) as a circuit
breaker so a systemically broken server still aborts the run. `gpt-oss-120b.yaml`
(the only judge with an observed, accepted rate of this failure) sets
`max_retries: 2, drop_ceiling: 10`; `llama-3.3-70b.yaml` / `qwen-2.5-72b.yaml` set
`max_retries: 0, drop_ceiling: 0`, reproducing their exact prior behavior. 19 unit
tests added/updated in `tests/test_run_judge_local.py`, all passing; full non-GPU
suite shows no regressions. Both originally-failed jobs resubmitted:
`2026-09-13-pond-gemma3-27b-extraction-gptoss120b-judge-local-01` (SGE 7580134) and
`2026-09-13-nfix-gemma3-27b-extraction-gptoss120b-judge-local-01` (SGE 7580135).

### Commits

- `bad923f` judge: bounded retry + flag for empty finish_reason=length verdicts
