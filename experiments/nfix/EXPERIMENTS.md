# Nfix Dataset — Experiment Commands

All commands are run from the **repository root**.

---

## Part 1 — Recreating the original experiments

The original experiments ran extraction with three models: `gemma-3-27b`,
`qwen-3.5-35b`, and `gpt-oss-120b`.  Table cleaning used the local vLLM
backend (gemma-3-27b).  No judge step was run for nfix.

> **Note on dev subset:** `experiments/configs/nfix.py` has a default
> `paper_subset` set to a 10-paper development set.  The original extraction
> scripts ran on the full corpus.  To replicate that, either pass
> `--paper-subset` with all paper codes or remove the default from the config.

> **Note on gpt-oss-120b:** The original `extract_gpt.py` used the HuggingFace
> model ID `openai/gpt-oss-120b`.  Verify the current registry entry matches
> before running.

### Step 1 — OCR

```bash
python experiments/run_ocr.py --dataset nfix
# Output: data/nfix/ocr_output_raw/
```

### Step 1b — PDF image pre-processing (preprocessing environment)

```bash
python experiments/process_pdfs.py --dataset nfix
# Output: data/nfix/processed_pdfs/
```

### Step 2 — Start the vLLM server

Each extraction run requires a vLLM server already running and serving the
chosen model.  Start it before running the extraction script.

```bash
# Example: serve gemma-3-27b on a single GPU
vllm serve gaunernst/gemma-3-27b-it-qat-autoawq \
    --tensor-parallel-size 1 \
    --port 8000
```

### Step 3 — Extraction (with integrated table cleaning)

The extraction model now performs table cleaning automatically as Step 0.
To replicate the original experiments (vLLM gemma-3-27b table cleaning +
extraction), simply run:

```bash
# gemma-3-27b (equivalent to original extract.py with vLLM table cleaning):
python experiments/run_extraction.py --dataset nfix --model gemma-3-27b
# Cleaned texts saved to: data/nfix/ocr_output_cleaned_gemma-3-27b/
# Output: data/experiments/nfix/extraction/gemma-3-27b/YYYY_mm_dd/

# qwen-3.5-35b: first serve that model with vLLM, then:
python experiments/run_extraction.py --dataset nfix --model qwen-3-vl-30b
```

---

## Part 2 — New experiments with different models

### Step 1 — OCR

OCR uses a single model (olmOCR); no new variants here.

```bash
python experiments/run_ocr.py --dataset nfix
python experiments/run_ocr.py --dataset nfix --resume  # resume partial run
```

### Step 1b — PDF image pre-processing (preprocessing environment)

Run once per dataset; output is reused across all extraction runs.

```bash
python experiments/process_pdfs.py --dataset nfix
python experiments/process_pdfs.py --dataset nfix --resume
# Output: data/nfix/processed_pdfs/
```

### Step 2 — Start the vLLM server

Each extraction run requires a vLLM server already running and serving the
chosen model via an OpenAI-compatible API (default: `http://localhost:8000/v1`).

```bash
# Example: serve gemma-3-27b on a single GPU
vllm serve gaunernst/gemma-3-27b-it-qat-autoawq \
    --tensor-parallel-size 1 \
    --port 8000
```

### Step 3 — Extraction (with integrated table cleaning)

Table cleaning is performed automatically by the extraction model as Step 0,
using raw OCR from `data/nfix/ocr_output_raw/`.  Cleaned texts are saved to
`data/nfix/ocr_output_cleaned_{model}/` for reuse.

```bash
python experiments/run_extraction.py --dataset nfix --model gemma-3-27b
python experiments/run_extraction.py --dataset nfix --model gemma-4-31b
python experiments/run_extraction.py --dataset nfix --model qwen-2.5-vl-72b
python experiments/run_extraction.py --dataset nfix --model qwen-3-vl-30b
python experiments/run_extraction.py --dataset nfix --model llama-4-scout-109b
python experiments/run_extraction.py --dataset nfix --model glm-4.6v-106b
python experiments/run_extraction.py --dataset nfix --model intern-vl3-78b

# Useful flags:
#   --ocr-dir DIR       skip table cleaning, load texts from DIR instead
#   --api-base URL      vLLM server base URL (default: http://localhost:8000/v1)
#   --api-key KEY       API key for the server (default: EMPTY)
#   --resume            resume from last completed step
#   --final-only        save only final.json, discard intermediates
#   --step <name>       run a single step (entities, attributes, entity_prov,
#                         attribute_prov, values, final)
#   --date YYYY_mm_dd   set a specific output date tag
#   --paper-subset p1 p2 ...  process specific papers (overrides config default)
```

To use OpenAI API table cleaning instead of the extraction model:

```bash
python experiments/run_table_cleaning.py \
    --dataset nfix --model gpt-4o-mini
# Output: data/nfix/ocr_output_cleaned_openai_gpt_4o_mini/

python experiments/run_extraction.py \
    --dataset nfix --model gemma-3-27b \
    --ocr-dir data/nfix/ocr_output_cleaned_openai_gpt_4o_mini
```

### Step 3 — Judge

No judge experiments were run on nfix originally.  Commands below follow the
same pattern as pond.  Pass `--ocr-dir` matching what was used during extraction.

```bash
# Local judges:
python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge qwen-3-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge gemma-3-12b --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_<model>

# Frontier judges:
python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge anthropic --frontier-model claude-haiku-4-5 \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge gemini --frontier-model gemini-2.5-flash-lite \
    --dest-gcs gs://my-bucket/judge-output/ \
    --gcp-project my-gcp-project \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_<model>

# Step-by-step (for large batches):
python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_<model> submit

python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    poll --state .batch_state_openai.json

python experiments/run_judge.py \
    --dataset nfix --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    process --state .batch_state_openai.json

# Combine all judges:
python experiments/run_judge_combine.py \
    --dataset nfix --extraction-model <model>
```
