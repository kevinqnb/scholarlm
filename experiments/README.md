# Experiments

All commands run from the **repository root**.

## Layout

| Path | Contents |
|---|---|
| `config.yaml` | Global seed only (`defaults.seed`). Every experiment config's `seed` must match it. |
| `dataset-configs/` | One `DatasetConfig` per dataset: `pond`, `nfix`, `supermat`, `measeval`, plus subset variants `pond_ten`, `nfix_ten`. |
| `model-configs/{kind}/<model>.yaml` | One YAML per model: sampling defaults, `serve:` (vLLM params), `resources:` (SGE GPU request). Kinds: `extraction`, `baseline`, `vllm_judge`, `interp_judge`, `jacobian_lens`, `representation_lm`, `ocr`. |
| `experiment-configs/{dataset}/{type}/<id>/<id>.yaml` | One committed config per experiment. `cross_dataset/` holds configs that span datasets. |
| `results/{dataset}/{type}/<id>/` | Run output (gitignored). `results/out/` has SGE stdout/stderr. |
| `utils.py` | Path helpers, config loading, `EXPERIMENT_TYPES` (type → runner → model kind). Never build paths by hand. |

`<id>` is `YYYY-MM-DD-slug-NN`. It is the same string as the config's `id:` field, its directory name, and the results directory name.

## Runners

Each takes one positional argument, the experiment config path, and no override flags.

| Type (config dir) | Runner | Purpose |
|---|---|---|
| `extraction` | `run_extraction.py` | 7-step MeasurementLM pipeline |
| `extraction_v2` | `run_extraction_v2.py` | Quantity-first pipeline (needs already-cleaned OCR) |
| `ablation` | `run_ablation.py` | Ablations 1–7 (`params.ablation`) |
| `table_cleaning` | `run_table_cleaning.py` | Clean OCR tables with a vLLM model |
| `ocr` | `run_ocr.py` | OCR PDFs (needs `tesseract` on `PATH`; see `CLAUDE.local.md`) |
| `process_pdfs` | `process_pdfs.py` | Render PDF pages to `.b64` images (separate env) |
| `judge_local` | `run_judge_local.py` | LLM judge via vLLM |
| `judge_interp` | `run_judge_interp.py` | NNsight judge; collects activations for the probe |
| `judge_combine` | `run_judge_combine.py` | Majority vote over listed `params.judge_ids` → `combined.json` |
| `attribution` | `run_attribution.py` | Attribution (NNsight) |
| `jacobian_lens` | `run_jacobian_lens.py` | Jacobian-lens j-scores |
| `representation_lm` | `run_representation_lm.py` | Key-term last-layer representations |
| `probe_augment` | `run_probe_augment.py` | Augment the probe dataset (writes to `data/{dataset}/`, manifest only here) |
| `baseline_gliner` / `baseline_nuextract` / `baseline_nuextract3` / `baseline_chatextract` / `baseline_langextract` | `run_baseline_*.py` | Baseline extractors. `nuextract` is the old 2.0-8B; prefer `nuextract3`. |

Manual runners, not dispatched by `submit.sh`; run directly:

| Script | Purpose |
|---|---|
| `run_drop_references.py` | Strip reference/appendix sections from existing OCR output into a new id |
| `run_schema_diag.py` | gpt-oss-120b `_parse_quantities()` failure diagnostic |
| `combine_ablation_parts.py` | Merge an ablation split across jobs by `paper_subset` (config lists `params.parts`) |

## Helper scripts

| Script | Purpose |
|---|---|
| `submit.sh` | `bash experiments/submit.sh <id> [--dry-run]`: qsub one experiment |
| `_resolve_job.py` | Resolves runner, GPU need, resources, and serve params for an id (called by `submit.sh`) |
| `_submit_job.sh` | SGE job body: starts or reuses a vLLM server, waits for `/health`, calls the runner |
| `submit_judges.py` | Submit a named list of `judge_local` ids after pre-checks |
| `submit_judge_combines.py` | Submit `judge_combine` ids only once all their judges have finished |
| `judge_batch_status.py` | Read-only status report for the 2026-09-27 judge batch |
| `yarn_needle_test.py` | One-off YaRN long-context check for `qwen-2.5-7b` |

`train_synthetic_probe_nfix_supermat.sh`, `verify_probe_migration_pond_train.sh`, and `yarn_needle_test.sh` are gitignored one-off job scripts. `{pond,nfix,measeval}/EXPERIMENTS.md` are pre-restructure notes and are stale (they reference `experiments/configs/` and the old CLI flags).

## Config shape

```yaml
id: 2026-09-13-pond-ablation1-gptoss120b-judge-local-01
project: scholarlm
description: ...            # purpose and predicted result
seed: 342                   # must equal config.yaml defaults.seed
params:
  dataset: pond             # a dataset-configs/ name
  model: gemma-3-27b        # a model-configs/{kind}/ name (`judge:` for judge/attribution types)
  walltime: "01:00:00"      # required for every GPU job; no model-config fallback
  # type-specific keys, e.g. ocr_dir, paper_subset, ablation, extraction_id, judge_ids
```

Required and optional `params` keys for each type are in the docstring at the top of its runner; run `python experiments/run_<type>.py --help`. Missing keys are hard errors.

## Reproducing an experiment

```bash
# 1. Run directly (use this for CPU-only types, or when a vLLM server is already up)
python experiments/run_extraction.py \
    experiments/experiment-configs/pond/extraction/<id>/<id>.yaml

# 2. Or submit to SGE; this starts the vLLM server in-job when the model needs one
bash experiments/submit.sh <id> --dry-run   # inspect the qsub request
bash experiments/submit.sh <id>
```

Output goes to `experiments/results/{dataset}/{type}/<id>/`, with `run_metadata.json` recording params and seed. Stages depend on each other by id, so reproduce them in order:

1. `ocr` → `table_cleaning` → `extraction` / `ablation` / `baseline_*`. Configs point at the cleaned text via `params.ocr_dir`.
2. `judge_local` and `judge_interp` take `params.extraction_id`, the id of the extraction or ablation run they judge.
3. `judge_combine` takes `params.judge_ids`, the ids of the judge runs to vote over.

Anything under `data/{dataset}/` that a config names (OCR output, ground truth, probe datasets) must already exist; see `data/*/README.md` for how to build it.

Before any full run, work up the staged gates in `CLAUDE.md`: unit tests, smoke run, tiny end-to-end with a predicted result, then the full run.

## Adding things

- **Dataset:** add `dataset-configs/<name>.py` exporting `CONFIG`; see `CLAUDE.md`.
- **Model:** add `model-configs/{kind}/<model>.yaml`. vLLM-served and direct-GPU models need a `resources:` block; `walltime` goes in the experiment config, not here.
- **Experiment:** add `experiment-configs/{dataset}/{type}/<id>/<id>.yaml`; the type directory name must be a key in `utils.EXPERIMENT_TYPES` for `submit.sh` to run it.
