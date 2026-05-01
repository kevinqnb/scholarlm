# Pond Dataset — Original Experiment

All commands are run from the **repository root**.

The original experiment used `gemma-3-27b` for extraction with tables pre-cleaned by
`gpt-5-mini`, and three frontier judges (OpenAI, Anthropic, Gemini) plus one
interpretability judge (LLaMA 3.1 8B).

## Step 1 — OCR

```bash
python experiments/run_ocr.py --dataset pond
# Output: data/pond/ocr_output_raw/
```

## Step 2 — PDF pre-processing *(preprocessing environment)*

```bash
python experiments/process_pdfs.py --dataset pond
# Output: data/pond/processed_pdfs/
```

## Step 3 — Table cleaning (gpt-5-mini)

```bash
python experiments/run_table_cleaning.py --dataset pond --model gpt-5-mini
# Output: data/pond/ocr_output_cleaned_openai_gpt_5_mini/
```

## Step 4 — Start the vLLM server

```bash
python experiments/gen_serve_script.py
qsub experiments/serve_gemma-3-27b.sh
```

## Step 5 — Extraction

```bash
python experiments/run_extraction.py \
    --dataset pond --model gemma-3-27b \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini
# Output: data/experiments/pond/extraction/gemma-3-27b/YYYY_mm_dd/
```

> Six ablation variants are available via `run_ablation.py --ablation N` (N=1–6).

## Step 6 — Judge

```bash
# Interpretability judge (LLaMA 3.1 8B, NNsight):
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

# Frontier judges:
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-5-mini \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge anthropic --frontier-model claude-haiku-4-5 \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge gemini --frontier-model gemini-2.5-flash-lite \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

# Combine:
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b
# Output: data/experiments/pond/judge/gemma-3-27b/combined/combined.json
```
