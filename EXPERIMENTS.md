# Experiments Guide

All experiment scripts live in `experiments/`. They share path conventions
defined in `experiments/paths.py` and dataset/model registries in
`experiments/model_registry.py`.

## Runner scripts

| Script | Purpose |
|---|---|
| `run_extraction.py` | Full 7-step MeasurementLM pipeline |
| `run_ablation.py` | Ablation variants (1–6) of the pipeline |
| `run_judge_interp.py` | NNsight judge — collects attention activations |
| `run_judge_local.py` | vLLM judge — fast local inference |
| `run_judge_frontier_v2.py` | Frontier judge — async direct API (current standard) |
| `run_judge_frontier.py` | Frontier judge — batch API (cost-effective; needs debugging) |
| `run_judge_regex.py` | Regex judge — heuristic, non-voting, for diagnostics |
| `run_judge_combine.py` | Majority-vote combination of frontier judges |
| `run_analysis.py` | Probe heatmap, calibration, cross-dataset analysis |

## Output schema

```
data/experiments/
  {dataset}/
    extraction/{model}/{YYYY_mm_dd}/
      entities.json
      attributes.json
      entity_prov.json
      attribute_prov.json
      events.json
      values.json
      final.json
    ablations/ablation{N}/{model}/{YYYY_mm_dd}/
      final.json
      judge/{judge_model}/{YYYY_mm_dd}/responses.json
      judge/combined/combined.json
    judge/{ext_model}/{ext_date}/{judge_model}/{judge_date}/
      responses.json
      attention_outputs.npz    (interp judges only)
    judge/{ext_model}/{ext_date}/combined/
      combined.json
    analysis/
      probe_heatmap.csv
      calibration_{model}.csv
      calibration_{model}_{judge}.npz
      figures/
        *.pdf / *.png
  cross_dataset/
    probe_matrix_{judge}_{model}.csv
    cross_dataset_results_{judge}_{model}.json
```

## Model and dataset registries

**`experiments/model_registry.py`** defines four registries:
- `MODEL_REGISTRY` — extraction models (vLLM / frontier)
- `INTERP_JUDGE_REGISTRY` — NNsight judge configs
- `VLLM_JUDGE_REGISTRY` — vLLM judge configs
- `FRONTIER_JUDGE_PROVIDERS` — frontier provider keys

**`experiments/configs/{name}.py`** — one file per dataset, exports `CONFIG: DatasetConfig`.

## Running experiments end-to-end

```bash
# 1. Extract
python experiments/run_extraction.py --dataset pond --model gemma-3-27b

# 2. Judge (frontier, async)
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini --extraction-date 2026_04_01
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge anthropic --frontier-model claude-haiku-4-5-20251001 --extraction-date 2026_04_01
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge gemini --frontier-model gemini-2.5-flash-lite --extraction-date 2026_04_01

# 3. Combine
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b --extraction-date 2026_04_01

# 4. Interp judge (for probe analysis)
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date 2026_04_01

# 5. Analysis
python experiments/run_analysis.py probe-heatmap \
    --dataset pond \
    --extraction-models gemma-3-27b \
    --extraction-dates 2026_04_01 \
    --judge-models llama-3.1-8b
```

## Running ablations end-to-end

```bash
# Run ablation N (1–6)
python experiments/run_ablation.py \
    --dataset pond --model gemma-3-27b --ablation 2

# Judge the ablation outputs
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date 2026_04_01 --ablation 2
```

## Regex judge

`run_judge_regex.py` uses heuristic regex matching rather than an LLM. Its outputs
are excluded from the majority-vote combination intentionally — use it for
diagnostic analysis of extraction quality, not as a ground-truth label.
