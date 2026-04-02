# Pond Dataset — Experiment Commands

All commands are run from the **repository root**.

---

## Part 1 — Recreating the original experiments

The original experiments used `gemma-3-27b` for extraction with tables
pre-cleaned by `gpt-5-mini` (OpenAI API), and three frontier judges
(OpenAI, Anthropic, Gemini) plus one local judge (LLaMA 3.1 8B).

### Step 1 — OCR

```bash
python experiments/run_ocr.py --dataset pond
# Output: data/pond/ocr_output_raw/
```

### Step 2 — Table cleaning (OpenAI gpt-5-mini, legacy)

```bash
python experiments/run_table_cleaning.py \
    --dataset pond --model gpt-5-mini
# Output: data/pond/ocr_output_cleaned_openai_gpt_5_mini/
```

### Step 3 — Extraction (gemma-3-27b, with pre-cleaned texts)

```bash
python experiments/run_extraction.py \
    --dataset pond --model gemma-3-27b \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini
# Output: data/experiments/pond/extraction/gemma-3-27b/YYYY_mm_dd/
```

> **Note:** The original experiments also included 8 ablation variants
> (`extract_ablation1.py` through `extract_ablation9.py`) that tested specific
> prompt configurations not replicated in the unified framework.  To re-run
> those, use the original scripts directly.

### Step 4 — Judge

```bash
# Local judge (LLaMA 3.1 8B):
python experiments/run_judge.py \
    --dataset pond \
    --extraction-model gemma-3-27b \
    --judge llama-3.1-8b \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

# Frontier judges (submit all three, then combine):
python experiments/run_judge.py \
    --dataset pond \
    --extraction-model gemma-3-27b \
    --judge openai \
    --frontier-model gpt-5-mini \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

python experiments/run_judge.py \
    --dataset pond \
    --extraction-model gemma-3-27b \
    --judge anthropic \
    --frontier-model claude-haiku-4-5 \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

python experiments/run_judge.py \
    --dataset pond \
    --extraction-model gemma-3-27b \
    --judge gemini \
    --frontier-model gemini-2.5-flash-lite \
    --dest-gcs gs://my-bucket/judge-output/ \
    --gcp-project my-gcp-project \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

# Combine all judges into a single ground-truth file:
python experiments/run_judge_combine.py \
    --dataset pond \
    --extraction-model gemma-3-27b
# Output: data/experiments/pond/judge/gemma-3-27b/combined/combined.json
```

---

## Part 2 — New experiments with different models

### Step 1 — OCR

OCR uses a single model (olmOCR); no new variants here.

```bash
python experiments/run_ocr.py --dataset pond
python experiments/run_ocr.py --dataset pond --resume  # resume partial run
```

### Step 2 — Extraction (with integrated table cleaning)

Table cleaning is now performed automatically by the extraction model as
Step 0, using raw OCR from `data/pond/ocr_output_raw/`.  Cleaned texts are
saved to `data/pond/ocr_output_cleaned_{model}/` for reuse.

```bash
python experiments/run_extraction.py --dataset pond --model gemma-3-27b
python experiments/run_extraction.py --dataset pond --model qwen-2.5-72b
python experiments/run_extraction.py --dataset pond --model llama-3.3-70b
python experiments/run_extraction.py --dataset pond --model qwen-3.5-35b
python experiments/run_extraction.py --dataset pond --model gpt-oss-120b

# Useful flags:
#   --ocr-dir DIR      skip table cleaning, load texts from DIR instead
#   --resume           resume from last completed step
#   --final-only       save only final.json, discard intermediates
#   --step <name>      run a single step (entities, attributes, entity_prov,
#                        attribute_prov, values, final)
#   --date YYYY_mm_dd  set a specific output date tag
#   --paper-subset p1 p2 ...  process only specific papers
```

To use OpenAI API table cleaning instead of the extraction model:

```bash
python experiments/run_table_cleaning.py \
    --dataset pond --model gpt-4o-mini
# Output: data/pond/ocr_output_cleaned_openai_gpt_4o_mini/

python experiments/run_extraction.py \
    --dataset pond --model gemma-3-27b \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_4o_mini
```

### Step 3 — Judge

Pass `--ocr-dir` matching what was used during extraction.

```bash
# Local judges:
python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge qwen-3-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge gemma-3-12b --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

# Frontier judges:
python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge anthropic --frontier-model claude-haiku-4-5 \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge gemini --frontier-model gemini-2.5-flash-lite \
    --dest-gcs gs://my-bucket/judge-output/ \
    --gcp-project my-gcp-project \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

# Step-by-step (for large batches):
python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model> submit

python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    poll --state .batch_state_openai.json

python experiments/run_judge.py \
    --dataset pond --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    process --state .batch_state_openai.json

# Combine all judges:
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model <model>

# Explicitly specify which judges to include:
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model <model> \
    --judges openai anthropic gemini llama-3.1-8b
```
