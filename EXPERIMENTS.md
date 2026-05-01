# Experiments Guide

All scripts are run from the **repository root**. Dataset and model registries are in
`experiments/model_registry.py`; per-dataset configs in `experiments/configs/`.

## End-to-end workflow

```bash
# 1. Extract
python experiments/run_extraction.py --dataset pond --model gemma-3-27b

# 2. Judge (frontier, async — submit all three providers)
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini --extraction-date 2026_04_01
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge anthropic --frontier-model claude-haiku-4-5-20251001 --extraction-date 2026_04_01
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge gemini --frontier-model gemini-2.5-flash-lite --extraction-date 2026_04_01

# 3. Combine frontier judges into ground-truth labels
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b --extraction-date 2026_04_01

# 4. Interpretability judge (required for probe analysis)
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

## Ablations

```bash
# Run ablation N (1–6)
python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 2

# Judge ablation outputs (same flags as standard judging, add --ablation N)
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date 2026_04_01 --ablation 2
```

## Notes

- `run_judge_regex.py` uses heuristic regex matching rather than an LLM. Its output is
  excluded from majority-vote combination — use it for diagnostic analysis only.
- `run_judge_frontier.py` (batch API) is cost-effective for large runs but currently
  needs debugging; prefer `run_judge_frontier_v2.py` for standard use.
