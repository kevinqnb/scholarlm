# Pond Dataset — Experiment Steps

All commands are run from the **repository root**.

For any case using an LLM, first serve it (default endpoint: `http://localhost:8081/v1`), then run.

## Dataset configuration

`experiments/configs/pond.py` is the single source of truth for all pond-specific
extraction settings. The key schemas are:

**Entity schema** — one record per distinct aquatic ecosystem:
| Field | Description |
|---|---|
| `name` | Primary name of the ecosystem |
| `identifiers` | Alternate short-form references (site codes, abbreviations) |
| `location` | General geographic location |
| `ecosystem` | Type: pond, lake, wetland, or other |

**Attribute Schema** — measurement types extracted for each entity:
| Attribute | Description |
|---|---|
| `surface_area` | Surface area of the water body |
| `max_depth` | Maximum physical water depth |
| `vegetation_cover` | Fraction of surface covered by aquatic vegetation |
| `ph` | Water pH |
| `tn` | Total nitrogen concentration |
| `tp` | Total phosphorus concentration |
| `chla` | Chlorophyll-a concentration |

**Event schema** — fields distinguishing individual measurements within an entity:
| Field | Description |
|---|---|
| `date` | Date of measurement |
| `additional_details` | Any other distinguishing context (treatment, sub-site, etc.) |

Other notable config fields: `paper_exclude` (papers omitted from extraction and
ground truth), `paper_subset` (set to a list of paper codes to restrict a run),
`judge_filter_fields` (fields omitted from judge prompts — currently `location`).

## Step 1 — OCR

```bash
python experiments/run_ocr.py --dataset pond
# Output: data/pond/ocr_output_raw/
```

## Step 2 — PDF pre-processing

```bash
python experiments/process_pdfs.py --dataset pond
# Output: data/pond/processed_pdfs/
```

## Step 3 — Table cleaning

```bash
python experiments/run_table_cleaning.py --dataset pond --model qwen-3.5-27b
# Pass --api-base if the server is not at the default endpoint.
# Output: data/pond/ocr_output_cleaned_qwen-3.5-27b/
```

## Step 4 — Extraction

```bash
python experiments/run_extraction.py \
    --dataset pond --model gemma-3-27b \
    --ocr-dir data/pond/ocr_output_cleaned_qwen-3.5-27b
# Pass --api-base if the server is not at the default endpoint.
# Output: data/experiments/pond/extraction/qwen-3.5-27b/YYYY_mm_dd/
```

> Six ablation variants are available via `run_ablation.py --ablation N` (N=1–6).

## Step 5 — Judge

```bash
# Interpretability judge (NNsight — loads model locally, no server needed):
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_gemma-3-27b

# Local vLLM judge (serve a model first, default endpoint: http://localhost:8081/v1):
python experiments/run_judge_local.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge gpt-oss-120bb --extraction-date YYYY_mm_dd \
    --ocr-dir data/pond/ocr_output_cleaned_gemma-3-27b
# Pass --api-base if the server is not at the default endpoint.

# Combine judge outputs:
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b --extraction-date YYYY_mm_dd
# Output: data/experiments/pond/judge/gemma-3-27b/YYYY_mm_dd/combined/combined.json
```
