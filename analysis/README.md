# analysis/

Experiment analysis code. It lives outside `src/scholarlm/` so it can import both the
core library and `experiments/utils.py` without making the library depend on
experiment scaffolding.

## Layout

Top-level `analysis/*.py` files are **entry points**: each one reads one
`analysis-configs/<type>/<id>.yaml` and writes under `analysis/results/<type>/`. Shared code lives in
`analysis/common/`. Entry points import from `common/` and never from each other.
`common/` modules never import an entry point. The one exception is
`_resolve_job.py`, the submit-time validator. It imports each script's own config
loader so that a bad config fails before `qsub`.

```
analysis/
  match_cache.py             build ground-truth <-> extraction match caches, by experiment id
  postprocessing.py          qualifier fill / unit standardization -> postprocessed.json
  deduplicate_cache.py       pairwise duplicate-candidate graph per extraction
  deduplication.py           exact clustering of that graph -> deduplicated.json
  recovery_validity.py       recovery/validity with paper-clustered bootstrap CIs
  recovery_validity_latex.py LaTeX tables from recovery_validity CSVs
  measeval_evaluation.py     MeasEval official-scorer cross-check
  synthetic_probe_train.py   train + pre-calibrate the synthetic probe / NTP calibrator
  calibration.py             calibration of the trained probe / NTP on real + synthetic test data (v4)
  calibration_validated.py   calibration scored on human-validated labels (pond, supermat)
  calibration_latex.py       LaTeX tables from calibration CSVs
  platt_scaling.py           recalibration-sample-size sweep over training resamples
  decision_threshold.py      recovery/validity vs confidence-threshold curves
  pond_meta_analysis.py      pond meta-analysis: confidence-thresholded vs GT distributions
  pond_clustering.py         pond downstream clustering, confidence-weighted
  submit.sh, _submit_job.sh, _resolve_job.py   SGE submission (see below)

  common/
    config.py                analysis-config envelope + every config loader except dedup's
    provenance.py            sha256_file, repo_relative
    matching.py              match-cache paths/loading, matching rules, frame loading + cache guards,
                             load_cached_matching, edges_to_judged_rows
    recovery.py              judge_combine resolution, validity labels, recovery/matched masks +
                             their re-verification, load_checked_inputs
    metrics.py               recovery_rate / validity_rate (and *_from_labels)
    loaders.py               ground-truth file, trained probe / NTP artifacts, legacy load_extraction
    calibration_ids.py       resolve_calibration_inputs: cross-check every run a calibration config names
    recalibration.py         prior_shift / intercept_fit / platt_fit maps (calibration, platt_scaling)
    doc_bootstrap.py         document-level bootstrap (calibration, platt_scaling)
    nested_bootstrap.py      fit-sample x document nested bootstrap (calibration_validated only)
    prediction_store.py      provenance of stored real-cell predictions (predictions.pkl)
    head_activations.py      cached head-activation loading for probe scoring
    calibration_plot_utils.py reliability-curve drawing, support mask
    dedup.py                 dedup cache / output paths and both dedup config loaders
    meta_inputs.py           meta config loader, input resolution, the score join
    pond_meta.py             pond cell universe, unit conversion + bounds, load_data
    outlier_weight.py        non-outlier confidence factor (pond_meta_analysis, pond_clustering)
  analysis-configs/<type>/   committed configs, one directory per analysis type
  results/<type>/            outputs, by analysis type then config id
```

## Analysis types

Configs and results share one directory name per analysis type
(`common.config.ANALYSIS_TYPES`). Every loader checks that its config sits in its own
type directory, and every cross-config reference (a calibration config's
`synthetic_probe_config`, a meta config's `calibration_config_id`, ...) resolves through
`common.config.analysis_config_path(type, id)`, so a misfiled config is a hard error.

| type | scripts | results |
|---|---|---|
| `recovery-validity` | setup: `postprocessing`, `match_cache`, `deduplicate_cache`, `deduplication`; then `recovery_validity`, `recovery_validity_latex` | `recovery-validity/`, plus `match-cache/<experiment id>/`, `deduplicate-cache/<config id>/`, `deduplication/<config id>/` |
| `measeval` | `measeval_evaluation` | `measeval/` |
| `synthetic-probe` | `synthetic_probe_train` | `synthetic-probe/` |
| `calibration` | `calibration`, `calibration_latex` (`labels: llm_matching`) | `calibration/` |
| `calibration-validated` | `calibration_validated`, `calibration_latex` (`labels: human_validated`) | `calibration-validated/` |
| `platt-scaling` | `platt_scaling` | `platt-scaling/` |
| `decision-threshold` | `decision_threshold` | `decision-threshold/` |
| `meta` | `pond_meta_analysis` | `meta/` |
| `clustering` | `pond_clustering` | `clustering/` |

A recovery-validity config is the one setup config for its experiment ids: run
postprocessing, match_cache, deduplicate_cache and deduplication on it, in that order,
before recovery_validity. Its `deduplicate_cache` and `deduplication` sections build the
dedup artefacts that meta / clustering configs name by `deduplication_config_id`.

## Analysis configs

`analysis-configs/<type>/<id>.yaml` groups the experiment ids that feed one analysis report
together with the parameters to run it. It follows the harness's standard
`id`/`project`/`description`/`seed`/`params` envelope (see `notes/hub/conventions.md`).
`params.experiment_ids` is shared by every consumer. A script that needs its own
parameters reads them from its own `params.<section>` via `common.config.get_section`.
Every key in that section is required: there are no defaults, and an unrecognized key
is an error.

```yaml
id: 2026-09-23-pond-recovery-report-01
project: scholarlm
description: Recovery report for a pond extraction variant (no judge coverage yet -- recovery only).
seed: 0                            # recovery_validity.py's bootstrap RNG seed -- unrelated to
                                    # experiments/config.yaml's defaults.seed, never checked against it
params:
  experiment_ids:
    - 2026-09-22-pond-extraction-gemma27b-parseqty-valueonly-01
  recovery_validity:
    n_resamples: 2000
    alpha: 0.05
    compute_validity: false
    output: analysis/results/recovery-validity/2026-09-23-pond-recovery-report-01.csv
```

Section names are part of the config schema and did not change when scripts were
renamed. For example, `pond_meta_analysis.py` still reads `params.meta_v2` and
`pond_clustering.py` reads `params.clustering`.

Every `recovery_validity.py` output row carries an `analysis_config_id` column (`None`
in ad-hoc CLI mode), so a number can be traced back to the config that produced it.
A declared `judge_combine_ids` override is verified against its extraction id
(`common.recovery.verify_judge_combine_id`), the same way automatic resolution
(`find_judge_combine_id`) verifies a candidate it finds by scanning.

The optional `judge_combine_ids` key (only meaningful with `compute_validity: true`)
overrides auto-resolution for specific ids, e.g. `{2026-05-05-pond-gemma-3-27b-extraction-01:
2026-09-13-pond-gemma3-27b-extraction-judge-combine-01}`. At the time it was written, no
`pond` id had both a fresh (`point_value`-column) schema and judge_combine coverage, so
this path is exercised by `tests/test_recovery_validity.py`'s fixtures rather than real
data. Treat it as fixture-verified, not production-verified, until a real id needs it.

## Running

```bash
python analysis/<script>.py [--config] analysis/analysis-configs/<type>/<id>.yaml
bash analysis/submit.sh <key> <id> --walltime HH:MM:SS --omp N [--dry-run]
```

`postprocessing`, `match_cache`, `deduplicate_cache`, `deduplication`, `recovery_validity`
and `measeval_evaluation` take `--config`; the rest take the config path positionally.
`submit.sh` keys are the script names in `_resolve_job.SCRIPTS` (`calibration`,
`platt_scaling`, `pond_meta_analysis`, `pond_clustering`, …); each key fixes the type
directory its `<id>` is looked up in. `--walltime` and `--omp`
are required, because cost varies by script and config. Job logs go to
`analysis/out/<id>.<key>.log`.

## Known issues

- **`calibration_latex.py` cannot format `calibration.py` (v4) output.** It accepts only
  v3-schema (`labels: llm_matching`) and `calibration_validated` (`labels:
  human_validated`) configs, and v3-schema results now come only from older runs, since
  `calibration_updated_v3.py` was removed. Retargeting it to v4 changes what the tables
  report: diff the v3 and v4 CSV columns first.
- **`calibration_validated.py` still uses the v3 machinery** (`common/nested_bootstrap.py`,
  v3-schema config). Porting it to `recalibration.py` / `doc_bootstrap.py` is an eval
  change that invalidates its numbers.
  Until then the nine `calibration/*-latex-01` configs fail to load (they name v4
  calibration configs), and `calibration-validated/*-latex-01` fails at `load_metrics`
  (its captions are written for `platt_fit`; the validated config is `intercept_fit`).
  The validated config itself is in the current v3 schema and gets re-keyed by the port.
