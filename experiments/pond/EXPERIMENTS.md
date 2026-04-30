# Pond Dataset — Experiment Commands

All commands are run from the **repository root**.

---

## Part 1 — Recreating the original experiments

The original experiments used `gemma-3-27b` for extraction with tables
pre-cleaned by `gpt-5-mini` (OpenAI API), and three frontier judges
(OpenAI, Anthropic, Gemini) plus one interpretability judge (LLaMA 3.1 8B).

### Step 1 — OCR

```bash
python experiments/run_ocr.py --dataset pond
# Output: data/pond/ocr_output_raw/
```

### Step 1b — PDF image pre-processing (preprocessing environment)

```bash
python experiments/process_pdfs.py --dataset pond
# Output: data/pond/processed_pdfs/
```

### Step 2 — Table cleaning (OpenAI gpt-5-mini)

```bash
python experiments/run_table_cleaning.py \
    --dataset pond --model gpt-5-mini
# Output: data/pond/ocr_output_cleaned_openai_gpt_5_mini/
```

### Step 3 — Start the vLLM server

Generate the serve script from `experiments/config.yaml` and submit it:

```bash
python experiments/gen_serve_script.py
qsub experiments/serve_gemma-3-27b.sh
```

### Step 4 — Extraction (gemma-3-27b, with pre-cleaned texts)

```bash
python experiments/run_extraction.py \
    --dataset pond --model gemma-3-27b \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini
# Output: data/experiments/pond/extraction/gemma-3-27b/YYYY_mm_dd/
```

> **Ablation variants:** Six ablation configurations are available via
> `run_ablation.py --ablation N` (N=1–6).

### Step 5 — Judge

```bash
# Interpretability judge (LLaMA 3.1 8B, NNsight):
python experiments/run_judge_interp.py \
    --dataset pond \
    --extraction-model gemma-3-27b \
    --judge llama-3.1-8b \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_5_mini

# Frontier judges (submit all three, then combine):
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

### Step 1b — PDF image pre-processing (preprocessing environment)

Run once per dataset; output is reused across all extraction runs.

```bash
python experiments/process_pdfs.py --dataset pond
python experiments/process_pdfs.py --dataset pond --resume  # skip already-done
# Output: data/pond/processed_pdfs/
```

### Step 2 — Start the vLLM server

Generate serve scripts from `experiments/config.yaml` and submit the one for
your chosen model:

```bash
python experiments/gen_serve_script.py
qsub experiments/serve_<model>.sh
```

### Step 3 — Extraction (with integrated table cleaning)

Table cleaning is performed automatically by the extraction model as Step 0,
using raw OCR from `data/pond/ocr_output_raw/`. Cleaned texts are saved to
`data/pond/ocr_output_cleaned_{model}/` for reuse.

```bash
python experiments/run_extraction.py --dataset pond --model llama-3.1-8b
python experiments/run_extraction.py --dataset pond --model gemma-3-27b
python experiments/run_extraction.py --dataset pond --model llama-3.3-70b
python experiments/run_extraction.py --dataset pond --model qwen-2.5-72b
python experiments/run_extraction.py --dataset pond --model qwen-3.5-27b
python experiments/run_extraction.py --dataset pond --model gpt-oss-120b

# Useful flags:
#   --ocr-dir DIR       skip table cleaning, load texts from DIR instead
#   --api-base URL      vLLM server base URL (default: http://localhost:8000/v1)
#   --api-key KEY       API key for the server (default: EMPTY)
#   --resume            resume from last completed step
#   --final-only        save only final.json, discard intermediates
#   --step <name>       run a single step (entities, attributes, entity_prov,
#                         attribute_prov, values, final)
#   --date YYYY_mm_dd   set a specific output date tag
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

### Step 4 — Judge

Pass `--ocr-dir` matching what was used during extraction.

```bash
# vLLM local judge (logprob scoring):
python experiments/run_judge_local.py \
    --dataset pond --extraction-model <model> \
    --judge gemma-3-27b --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

# Interpretability judge (NNsight, attention activations):
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model <model> \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

# Frontier judges (direct async API):
python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model <model> \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model <model> \
    --judge anthropic --frontier-model claude-haiku-4-5 \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

python experiments/run_judge_frontier_v2.py \
    --dataset pond --extraction-model <model> \
    --judge gemini --frontier-model gemini-2.5-flash-lite \
    --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_<model>

# Combine all judges:
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model <model>

# Explicitly specify which judges to include:
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model <model> \
    --judges openai anthropic gemini gemma-3-27b
```
