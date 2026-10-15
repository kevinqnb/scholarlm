# analysis/

Experiment analysis code, outside `src/scholarlm/` so it can import both the core library
and `experiments/utils.py`. Every analysis is one script run on one committed config.

## How to run

```bash
python analysis/<script>.py analysis/analysis-configs/<type>/<id>.yaml           # most scripts
python analysis/<script>.py --config analysis/analysis-configs/<type>/<id>.yaml  # see table below
```

## Scripts

Config type = directory under `analysis-configs/` and `results/`. Flag = how the config path is passed.

| script | config type | flag | what it does |
|---|---|---|---|
| `postprocessing.py` | recovery-validity | `--config` | Qualifier fill / unit standardization → `postprocessed.json` |
| `match_cache.py` | recovery-validity | `--config` | Ground-truth ↔ extraction match cache per experiment id |
| `deduplicate_cache.py` | recovery-validity | `--config` | Pairwise duplicate-candidate graph per extraction |
| `deduplication.py` | recovery-validity | `--config` | Exact clustering of that graph → `deduplicated.json` |
| `recovery_validity.py` | recovery-validity | `--config` | Recovery / validity with paper-clustered bootstrap CIs |
| `recovery_validity_latex.py` | recovery-validity | `--config` | LaTeX table (methods × datasets) from the CSVs above |
| `runtime.py` | recovery-validity | `--config` | LaTeX runtime/compute table: extraction methods (`params.runtime`) or interp judges over one extraction (`params.judge_runtime`) |
| `measeval_evaluation.py` | measeval | `--config` | MeasEval official-scorer cross-check (Quantity, Unit, MeasuredEntity) |
| `measeval_latex.py` | measeval | `--config` | LaTeX table of the MeasEval F-measures |
| `synthetic_probe_train.py` | synthetic-probe | positional | Train the validity probe on a synthetic judge run |
| `calibration.py` | calibration | positional | Probe / NTP calibration on synthetic + real test sets (LLM+matching labels) |
| `calibration_validated.py` | calibration-validated | positional | Same cells, real rows scored on human-validated labels |
| `calibration_latex.py` | calibration, calibration-validated | `--config` | LaTeX tables from either calibration CSV (`labels: llm_matching \| human_validated`) |
| `platt_scaling.py` | platt-scaling | positional | smECE vs recalibration sample size n |
| `decision_threshold.py` | decision-threshold | positional | Recovery / validity vs confidence threshold |
| `pond_meta_analysis.py` | meta | positional | Pond: do confidence-filtered extractions reproduce GT distributions? |
| `pond_clustering.py` | clustering | positional | Pond: does confidence weighting recover GT clusters under KMeans? |

`postprocessing.py` and `match_cache.py` also accept experiment ids directly on the command line.

## Typical flows

- **Recovery / validity table:** on one recovery-validity config run `postprocessing` →
  `match_cache` → `deduplicate_cache` → `deduplication` → `recovery_validity` →
  `recovery_validity_latex`. The config's `deduplicate_cache` / `deduplication` sections build the
  dedup artefacts that meta and clustering configs name via `deduplication_config_id`.
- **Calibration:** `synthetic_probe_train` → `calibration` (and/or `calibration_validated`) →
  `calibration_latex`. `platt_scaling`, `decision_threshold`, `pond_meta_analysis` and
  `pond_clustering` read a finished calibration run (its stored predictions) named in their config.
- **MeasEval:** `measeval_evaluation` → `measeval_latex`.

## Layout

```
analysis/
  *.py                      entry points (table above); import from common/, never from each other
  analysis-configs/<type>/  committed configs, one directory per type
  results/<type>/           outputs by type, then config id; also match-cache/<experiment id>/,
                            deduplicate-cache/<config id>/, deduplication/<config id>/
  common/                   shared code (below)
```

`common/`:

| module | contents |
|---|---|
| `config.py` | config envelope, `ANALYSIS_TYPES`, `analysis_config_path`, every loader except dedup's |
| `dedup.py` | dedup paths and both dedup config loaders |
| `matching.py` | match-cache paths/loading, matching rules, cache guards |
| `recovery.py` | judge_combine resolution, validity labels, recovery/matched masks |
| `metrics.py` | `recovery_rate` / `validity_rate` |
| `loaders.py` | ground-truth file, trained probe / NTP artifacts, legacy `load_extraction` |
| `calibration_ids.py` | cross-checks every run a calibration config names |
| `recalibration.py` | `prior_shift` / `intercept_fit` / `platt_fit` maps |
| `doc_bootstrap.py` | document-level bootstrap (calibration, calibration_validated, platt_scaling) |
| `nested_bootstrap.py` | legacy fit-sample × document bootstrap; only tests import it now |
| `prediction_store.py` | row identity/provenance of stored real-cell predictions (`predictions.pkl`) |
| `head_activations.py` | cached head-activation loading for probe scoring |
| `calibration_plot_utils.py` | reliability curves, support mask |
| `meta_inputs.py`, `pond_meta.py`, `outlier_weight.py` | pond meta/clustering inputs, cell universe + unit bounds, robust-z confidence factor |
| `provenance.py` | `sha256_file`, `repo_relative` |

## Configs

`analysis-configs/<type>/<id>.yaml` uses the harness envelope (`id`, `project`, `description`,
`seed`, `params`; see `notes/hub/conventions.md`). `params.experiment_ids` is shared by all
consumers; a script's own settings live in `params.<section>` (read via `common.config.get_section`).
Every key is required, there are no defaults, and unknown keys are errors. The `seed` here is the
analysis RNG seed, unrelated to `experiments/config.yaml`'s `defaults.seed`.

```yaml
id: 2026-09-23-pond-recovery-report-01
project: scholarlm
description: Recovery report for a pond extraction variant.
seed: 0
params:
  experiment_ids: [2026-09-22-pond-extraction-gemma27b-parseqty-valueonly-01]
  recovery_validity:
    n_resamples: 2000
    alpha: 0.05
    compute_validity: false
    output: analysis/results/recovery-validity/2026-09-23-pond-recovery-report-01.csv
```

- Section names are fixed by the schema, not the script name: `pond_meta_analysis.py` reads
  `params.meta_v2`, `pond_clustering.py` reads `params.clustering`.
- Each type directory is checked at load time, and cross-config references
  (`synthetic_probe_config`, `calibration_config_id`, …) resolve through
  `analysis_config_path(type, id)`, so a misfiled config is a hard error.
- Every `recovery_validity.py` row carries `analysis_config_id` (`None` in ad-hoc CLI mode).
- Optional `judge_combine_ids` (with `compute_validity: true`) overrides judge_combine
  auto-resolution per id; it is verified against the extraction id. Fixture-tested only
  (`tests/test_recovery_validity.py`), not yet exercised on real data.
