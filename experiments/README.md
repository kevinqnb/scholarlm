# Experiments

All scripts are run from the **repository root**. Per-dataset configs live in
`experiments/configs/`; model keys are defined in `experiments/model_registry.py`.

## Scripts

| Script | Purpose |
|---|---|
| `run_extraction.py` | Run the 7-step extraction pipeline |
| `run_ablation.py` | Run an ablation variant (1–6) |
| `run_table_cleaning.py` | Pre-clean OCR tables with a vLLM model |
| `run_judge_interp.py` | Interpretability judge (NNsight, collects activations) |
| `run_judge_local.py` | Local judge via vLLM server |
| `run_judge_combine.py` | Majority-vote combination of judge outputs → `combined.json` |
| `run_ocr.py` | Run OLMo-OCR on PDFs |
| `process_pdfs.py` | Pre-process PDFs (requires separate environment) |
| `validation.py` | Streamlit app for human validation of extraction results |

## Serving a model

`run_extraction.py`, `run_table_cleaning.py`, and `run_judge_local.py` require a
running vLLM server. All three default to the endpoint `http://localhost:8081/v1`; pass
`--api-base` to use a different endpoint. `gen_serve_script.py` can generate
model-specific serve scripts for your environment.

## Basic workflow

```bash
# Extract
python experiments/run_extraction.py --dataset pond --model gemma-3-27b
# If the server is not at the default endpoint:
python experiments/run_extraction.py --dataset pond --model gemma-3-27b \
    --api-base http://<host>:8081/v1

# Judge (run one or more; use --extraction-date to pin a run)
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date 2026_04_01
python experiments/run_judge_local.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.3-70b --extraction-date 2026_04_01

# Combine judge outputs
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b --extraction-date 2026_04_01
```

## Ablations

```bash
python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 2

# Judge an ablation run (add --ablation N to any judge command)
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date 2026_04_01 --ablation 2
```
