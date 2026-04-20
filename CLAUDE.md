# ScholarlM — LLM Navigation Index

A research library for extracting entity-attribute-value triplets from scientific PDFs using local LLMs (vLLM) and frontier models (OpenAI / Anthropic / Gemini).

## Repo layout

```
src/scholarlm/          Core library — pipeline, config, probe/calibration utilities
experiments/            Runner scripts, model/dataset registries, path helpers
experiments/configs/    One DatasetConfig per dataset (pond.py, nfix.py, …)
analysis/               Experiment analysis code and notebooks
examples/               Jupyter notebooks
data/experiments/       All outputs — never committed to git
docs/                   Plans and documentation
```

## Key concepts

**Extraction pipeline** (`MeasurementLM` in `src/scholarlm/measurementlm.py`)  
Seven sequential steps: entities → attributes → entity_prov → attribute_prov → events → values → final. Each step has a JSON checkpoint; `--resume` skips steps whose output already exists.

**Ablations** (`src/scholarlm/measurementlm_ablation{1–6}.py`)  
Each ablation subclass overrides one or more pipeline steps. Run via `experiments/run_ablation.py`.

**DatasetConfig / ModelConfig** (`src/scholarlm/config.py`)  
Single source of truth for dataset- and model-specific values. Config files live in `experiments/configs/`.

**Path helpers** (`experiments/paths.py`)  
Every path in the output tree is constructed here. Never build paths by hand in scripts.

**Judge pipeline**  
- `run_judge_interp.py` — NNsight (local, collects attention activations)
- `run_judge_local.py` — vLLM (local, fast)
- `run_judge_frontier_v2.py` — async direct API (current standard for frontier)
- `run_judge_frontier.py` — batch API (cost-effective for large runs; currently needs debugging)
- `run_judge_regex.py` — regex heuristic (non-voting, for diagnostics)
- `run_judge_combine.py` — majority-vote combination of frontier judges → `combined.json`

**Analysis utilities** (`src/scholarlm/utils/`)  
- `probe.py` — logistic-regression probe on attention activations
- `calibration.py` — ECE and reliability diagram
- `unit_conversion.py` — `apply_unit_conversion(df, unit_conversion_table)` converts extracted values to standard units before ground-truth matching

**Experiment analysis** (`analysis/`)  
- `loaders.py` — load experiment outputs by (dataset, model, date)
- `metrics.py` — recovery rate, hallucination rate, per-paper summaries
- `plots.py` — standard plot functions (return `matplotlib.Figure`)
- `cross_dataset.py` — cross-dataset probe transferability

## Output directory schema

```
data/experiments/
  {dataset}/
    extraction/{model}/{YYYY_mm_dd}/       → 7 JSON checkpoints + final.json
    ablations/ablation{N}/{model}/{date}/   → final.json (+ judge/ subdir)
    judge/{ext_model}/{ext_date}/{judge_model}/{judge_date}/
    judge/{ext_model}/{ext_date}/combined/ → combined.json
    analysis/                              → CSV / NPZ outputs
    analysis/figures/                      → PDF / PNG plots
  cross_dataset/                           → cross-dataset probe CSV
```

## Common commands

```bash
# Extraction
python experiments/run_extraction.py --dataset pond --model gemma-3-27b

# Ablation
python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 2

# Judging (frontier, async)
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date 2026_04_01

# Combine judge results
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b --extraction-date 2026_04_01

# Analysis
python experiments/run_analysis.py probe-heatmap \
    --dataset pond \
    --extraction-models gemma-3-27b \
    --extraction-dates 2026_04_01 \
    --judge-models llama-3.1-8b
```

## Adding a new dataset

1. Create `experiments/configs/{name}.py` exporting `CONFIG: DatasetConfig`.
2. Create `data/{name}/preprocessing.py` to generate `ground_truth.csv` (and `ground_truth_ten.csv` if a subset exists). Use `data/pond/preprocessing.py` or `data/nfix/preprocessing.py` as a template.
3. Set `ground_truth_file` in the config. If units vary across papers, populate `unit_conversion_table` with per-attribute `{unit: multiplier}` entries.
4. Run `python data/{name}/preprocessing.py` to generate the ground truth CSVs.
5. Run `run_extraction.py --dataset {name}` to verify the pipeline end-to-end.

## Adding a new model

Add an entry to `MODEL_REGISTRY` in `experiments/model_registry.py`.
