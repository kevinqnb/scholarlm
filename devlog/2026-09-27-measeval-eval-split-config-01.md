---
id: 2026-09-27-measeval-eval-split-config-01
kind: build
config: analysis/analysis-configs/2026-09-27-measeval-full-pipeline-vs-baselines-official-eval-01.yaml
---

## Session 2026-09-27

### Prompts

- Draft an `analysis/analysis-configs/` config for measeval, in the existing style,
  feeding `analysis/measeval_evaluation.py` rather than the recovery/validity path.
- Explain why `evaluate(dev=False)` couldn't score any real run, and whether
  eval-split filtering could be wired into `measeval_evaluation.py` via that config.
- Implement both the eval-split filter and the `--config` loop, as two separate
  commits.

### Implemented

`analysis/measeval_evaluation.py`'s `evaluate(dev=False)` used to require every
document in a run to already be in the official `eval` split, so it had never scored
a real (train+trial+eval-spanning) full run. It now filters non-eval documents out of
the submission before scoring instead of raising, reporting what was excluded rather
than dropping it silently — necessary because the official scorer would otherwise
count every non-eval-split prediction as a guaranteed false positive. `main()` gained
`--config` support (mirroring `match_cache.py`/`recovery_validity.py`), reading
`params.experiment_ids` and a new `params.measeval_evaluation` section; the shared
`analysis/analysis_config.py` loader now recognizes that section. Added the first real
config, `configs/2026-09-27-measeval-full-pipeline-vs-baselines-official-eval-01.yaml`
(13 ids: 3 extraction models, their ablation1-noqualifiers/ablation7 variants, and 4
baselines), and ran it end-to-end to produce the first leaderboard-comparable
Quantity/Unit/MeasuredEntity numbers for these runs.

### Commits

- b074945 Filter measeval_evaluation.evaluate(dev=False) to eval-split docs before scoring
- cd9516e Wire up analysis-config --config loop for measeval_evaluation.py
