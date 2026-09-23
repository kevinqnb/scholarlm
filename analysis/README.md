# analysis/

Experiment analysis code and notebooks. Lives outside `src/scholarlm/` so it can freely
import both the core library and `experiments/paths.py` without making the library depend
on experiment scaffolding.

## Structure

```
analysis/
  __init__.py
  loaders.py          — load experiment outputs by (dataset, model, date)
  metrics.py          — recovery rate, hallucination rate, per-paper summaries
  ablation.py              — computes recovery and validity rates for extracted data (over ablations)
  validity_evaluation.py   — evaluate validity assessment methods (synthetic + human arms)
  synthetic_probe_train.py — trains and pre-calibrates the synthetic probe models
  calibration.py      - evaluates the trained probe / NTP models on test data
  clustering.py       - trains and evaluates a downstream clustering model
  match_cache.py       — computes and caches extraction<->ground-truth matches, by experiment id
  recovery_validity.py — recovery/validity with paper-clustered bootstrap CIs, from match caches + judge_combine
  analysis_config.py   — shared loader for analysis-configs/<id>.yaml (see below)
  analysis-configs/     — committed configs consumed by match_cache.py/recovery_validity.py via --config
```

## Analysis configs

`analysis-configs/<id>.yaml` groups the experiment ids that feed one analysis
report together with the high-level parameters to run it, following the
harness's standard `id`/`project`/`description`/`seed`/`params` envelope (see
`notes/hub/conventions.md`) rather than a bespoke shape. `params.experiment_ids`
is shared by every consumer; a script that needs its own parameters reads them
from its own `params.<script_name>` section (e.g. `params.recovery_validity`)
via `analysis_config.get_section`, which requires every key in that section
explicit -- no defaults, no unrecognized key silently ignored.

```yaml
id: 2026-09-23-pond-recovery-report-01
project: scholarlm
description: Recovery report for a pond extraction variant (no judge coverage yet -- recovery only).
seed: 342                          # recovery_validity.py's bootstrap RNG seed -- unrelated to
                                    # experiments/config.yaml's defaults.seed, never checked against it
params:
  experiment_ids:
    - 2026-09-22-pond-extraction-gemma27b-parseqty-valueonly-01
  recovery_validity:
    n_resamples: 2000
    alpha: 0.05
    compute_validity: false
    output: analysis/results/2026-09-23-pond-recovery-report-01.csv
```

The optional `judge_combine_ids` key (only meaningful with `compute_validity: true`) overrides
auto-resolution for specific ids, e.g. `{2026-05-05-pond-gemma-3-27b-extraction-01:
2026-09-13-pond-gemma3-27b-extraction-judge-combine-01}` -- as of this writing no
`pond` id has both a fresh (`point_value`-column) schema and judge_combine coverage
at once, so this path is currently exercised by `tests/test_recovery_validity.py`'s
fixtures rather than real data; treat it as fixture-verified, not
production-verified, until a real id needs it.

Usage:

```bash
python analysis/match_cache.py --config analysis/analysis-configs/<id>.yaml
python analysis/recovery_validity.py --config analysis/analysis-configs/<id>.yaml
```

`--config` is mutually exclusive with passing experiment ids and flags
directly -- the ad-hoc CLI (`python analysis/recovery_validity.py <id> ...
--n-resamples ... --seed ...`) still works unchanged for one-off analysis.
Every `recovery_validity.py` output row carries an `analysis_config_id`
column (`None` in ad-hoc CLI mode) so a number can be traced back to the
config that produced it. A declared `judge_combine_ids` override is still
verified against its extraction id (`verify_judge_combine_id`) the same way
automatic resolution (`find_judge_combine_id`) verifies a candidate found by
scanning -- it only skips the scan, never the check.