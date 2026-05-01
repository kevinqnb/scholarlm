# Nfix Dataset — Original Experiment

All commands are run from the **repository root**.

The original experiment ran extraction with three models (`gemma-3-27b`, `qwen-3.5-27b`,
`gpt-oss-120b`) using integrated vLLM table cleaning. No judge step was run for nfix.

> **Note:** `experiments/configs/nfix.py` defaults to a 10-paper development subset.
> To replicate the full-corpus run, pass `--paper-subset` with all paper codes or
> remove the default from the config.

## Step 1 — OCR

```bash
python experiments/run_ocr.py --dataset nfix
# Output: data/nfix/ocr_output_raw/
```

## Step 2 — PDF pre-processing *(preprocessing environment)*

```bash
python experiments/process_pdfs.py --dataset nfix
# Output: data/nfix/processed_pdfs/
```

## Step 3 — Start the vLLM server

```bash
python experiments/gen_serve_script.py
qsub experiments/serve_gemma-3-27b.sh
```

## Step 4 — Extraction (with integrated table cleaning)

```bash
python experiments/run_extraction.py --dataset nfix --model gemma-3-27b
python experiments/run_extraction.py --dataset nfix --model qwen-3.5-27b
python experiments/run_extraction.py --dataset nfix --model gpt-oss-120b
# Cleaned texts: data/nfix/ocr_output_cleaned_{model}/
# Output: data/experiments/nfix/extraction/{model}/YYYY_mm_dd/
```

> **Note on gpt-oss-120b:** Verify the `model_id` in `model_registry.py` matches the
> HuggingFace ID used originally before running.
