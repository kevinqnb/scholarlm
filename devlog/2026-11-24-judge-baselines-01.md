---
id: 2026-11-24-judge-baselines-01
kind: build
---

# Judge baselines: verbalized, sampling and next-token confidence

## Session 2026-11-24

### Prompts

> Please help me set up a `experiments/run_judge_baselines.py` script. We will use this to instantiate a few different baseline methods for computing confidence scores. The setup should remain as close as possible to `run_judge_local.py` with a few small adjustments. **Baseline 1:** the model is prompted to respond with a confidence score between 0 and 100; we divide by 100. **Baseline 2:** the model responds True or False, but is repeatedly sampled (default temperature 1, 10 samples); the score is the percentage of responses that evaluated to True. **Baseline 3:** the model responds True or False, and we record the relative next-token probability for True, identical to `run_judge_interp.py` but without activation collection. Begin with the script and general implementation, then check in with me.

> Confidence prompt wording is good; leave the reasoning-model issue alone. Unusable answers should not abort the whole run: allow a retry budget on every sample or query (default 2), then abort if it still doesn't pass. Use llama-8b against the existing interp scores. Parameters should live in the experiment configs rather than being hardcoded. Docstring is too long and wordy; function descriptions must detail each input and output parameter; descriptions should not refer to another script or to code history.

> Move B1/B2 to the B3 layout. Use 0.0 and 1.0 as suggested (B1 temperature / retry temperature). The full run should not abort or error out; failed rows should be marked as failed and given empty output. Then pause to reflect before the smoke run.

> Yes to: request errors abort the run while unusable answers are marked failed; a `row_indices` param; keep B2 as is; add `judgement_failed` to `run_judge_combine` as a separate commit. Run smoke tests; if these require GPU, submit them as throwaway jobs.

> Set up configs for the tiny and full runs. Full: all three baselines over the gemma-3-27b extractions `2026-09-22-pond-extraction-gemma27b-parseqty-standardization-valueonly-01`, `2026-09-21-nfix-extraction-gemma27b-full-01` and `2026-09-21-supermat-extraction-gemma27b-full-01`; baselines 1 and 2 with `llama-3.1-8b`, baseline 3 with `llama-3.3-70b`; baselines 1 and 3 at temperature 0. Tiny: a minimal set needed to confirm working order for these settings.

### Implemented

`experiments/run_judge_baselines.py` scores extracted measurements with a vLLM-served judge using one of three methods: a stated 0-100 confidence, the fraction of sampled true/false answers that are true, or the next-token probability of true relative to false. All three send the same single-message prompt, set every sampling field explicitly, and keep their parameters in the experiment config. Unusable answers are retried and then mark the row failed; any other error aborts the run. Added the `judge_baselines` experiment type, a confidence prompt, `experiments/model-configs/vllm_judge/llama-3.1-8b.yaml`, and 15 configs under `experiments/experiment-configs/{pond,nfix,supermat}/judge_baselines/` (9 full runs, 6 pond tiny end-to-end runs including a seed-determinism pair and a wrong-paper control). Unit tests and a one-row smoke run passed; `run_judge_combine.py` now keeps `judgement_failed` per judge.

### Commits

- `6f7a39a` run_judge_combine: namespace judgement_failed per judge
- `d85605e` Add run_judge_baselines: verbalized, sampling and next-token confidence baselines
- `58434c6` Add judge_baselines configs: 9 full runs and 6 pond tiny end-to-end runs
