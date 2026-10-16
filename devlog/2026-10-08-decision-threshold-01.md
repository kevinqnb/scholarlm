---
id: 2026-10-08-decision-threshold-01
kind: build
config: analysis/analysis-configs/2026-10-08-decision-threshold-gemma27b-qwen-2.5-7b-intercept-fit-sample-01.yaml
---

## Session 2026-10-08

### Prompts

> Please help me create a new `analysis/decision_threshold.py` script to create recovery-validity curves with changing decision thresholds on the probe or NTP predicted probabilites. Heres how it should work:
>
> 1. It should use a config file which specifies where to read prediction probabilities from: e.g. `2026-10-08-calibration-v4-gemma27b-qwen-2.5-7b-intercept-fit-sample-01` will be our starting default. Include any other parameters or files necessary to define a run of this analysis.
> 2. For probe and NTP models separately, iterate through a range of decision thresholds on those predicted probabilities. Above the threshold predict valid, and below the threshold predict invalid.
> 3. For a specific threshold setting, compute recovery and validity scores for the extraction set that survives the threshold filter. For example, if the threshold is 0.5, then only extractions with probability >= 0.5 can contribute edges to the matching which feeds the recovery score, and can be counted in the set with LLM judge + matching labels which is used to compute validity. For more information on these scores, see @analysis/recovery_validity.py and aim to recreate its scores exactly.
> 4. Plot recovery validity curves using the same styling we used for fuzzy threshold decision curves in @analysis/recovery_validity
> 5. Wire this up to run through `analysis/submit.sh` and save to `analysis/results/decision_threshold`

Design choices: scoring is restricted to test documents only (the documents that have predictions), thresholds apply to the recalibrated probabilities, cells are the three within-dataset pairs with supermat flagged invalid, and the highlighted threshold is 0.5.

> we don't need max weight matching recovery here.

> go ahead and submit it

### Implemented

New `analysis/decision_threshold.py` keeps the extractions whose recalibrated probe or NTP probability from a v4 calibration run's `predictions.pkl` is `>= t`. It then scores any-edge recovery and judged-or-matched validity on the kept set, using `analysis/recovery_validity.py`'s own loaders, cache checks and `analysis.metrics`. Scoring is restricted to the test documents. At t=0 the curve is asserted to equal `recovery_validity`'s scores on those documents. A seeded permutation control is plotted alongside. Configs: `analysis/analysis-configs/2026-10-08-decision-threshold-gemma27b-qwen-2.5-7b-intercept-fit-sample-01.yaml` (full run) and `analysis/analysis-configs/2026-10-08-decision-threshold-smoke-01.yaml` (smoke). Tests are in `tests/test_decision_threshold.py`. It runs as `bash analysis/submit.sh decision_threshold <id>`.

### Commits
0a8da01 decision_threshold: recovery/validity curves over probe/NTP decision thresholds
