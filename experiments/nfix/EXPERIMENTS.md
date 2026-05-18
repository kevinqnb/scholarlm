# Nfix Dataset — Experiment Steps

All commands are run from the **repository root**.

For any case using an LLM, first serve it (default endpoint: `http://localhost:8081/v1`), then run.

## Dataset configuration

`experiments/configs/nfix.py` is the single source of truth for all nfix-specific
extraction settings. The key schemas are:

**Entity schema** — one record per distinct dinitrogen fixation measurement site:
| Field | Description |
|---|---|
| `name` | Primary name of the site |
| `identifiers` | Alternate short-form references (site codes, abbreviations) |
| `site_type` | Ecosystem type (e.g. lake, estuary, freshwater wetland, soil) |
| `location` | General geographic location |

**Attribute Schema** — measurement types extracted for each entity:
| Attribute | Description |
|---|---|
| `nfix_rate_mass` | Dinitrogen fixation rate per unit mass |
| `nfix_rate_areal` | Dinitrogen fixation rate per unit area |
| `nfix_rate_volumetric` | Dinitrogen fixation rate per unit volume |

**Event schema** — fields distinguishing individual measurements within an entity:
| Field | Description |
|---|---|
| `date` | Date of measurement |
| `nfix_method` | Method used (e.g. acetylene reduction assay, ¹⁵N₂ incorporation) |
| `substrate_type` | Substrate measured (e.g. water column, benthos) |
| `sample_depth` | Depth at which sample was collected |
| `additional_details` | Any other distinguishing context |

Other notable config fields: `paper_subset` (defaults to a 10-paper development
subset — remove or override to run on the full corpus), `paper_filter` (restricts
to papers with detectable fixation data).

## Step 1 — OCR

```bash
python experiments/run_ocr.py --dataset nfix
# Output: data/nfix/ocr_output_raw/
```

## Step 2 — PDF pre-processing

```bash
python experiments/process_pdfs.py --dataset nfix
# Output: data/nfix/processed_pdfs/
```

## Step 3 — Table cleaning

```bash
python experiments/run_table_cleaning.py --dataset nfix --model qwen-3.5-27b
# Pass --api-base if the server is not at the default endpoint.
# Output: data/nfix/ocr_output_cleaned_qwen-3.5-27b/
```

## Step 4 — Extraction

```bash
python experiments/run_extraction.py \
    --dataset nfix --model gemma-3-27b \
    --ocr-dir data/nfix/ocr_output_cleaned_qwen-3.5-27b
# Pass --api-base if the server is not at the default endpoint.
# Output: data/experiments/nfix/extraction/{model}/YYYY_mm_dd/
```

> Six ablation variants are available via `run_ablation.py --ablation N` (N=1–6).

## Step 5 — Judge

The judge commands are the same for each extraction model; `gemma-3-27b` is shown
as an example.

```bash
# Interpretability judge (NNsight — loads model locally, no server needed):
python experiments/run_judge_interp.py \
    --dataset nfix --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_qwen-3.5-27b

# Local vLLM judge:
python experiments/run_judge_local.py \
    --dataset nfix --extraction-model gemma-3-27b \
    --judge llama-3.3-70b --extraction-date YYYY_mm_dd \
    --ocr-dir data/nfix/ocr_output_cleaned_qwen-3.5-27b
# Pass --api-base if the server is not at the default endpoint.

# Combine judge outputs:
python experiments/run_judge_combine.py \
    --dataset nfix --extraction-model gemma-3-27b --extraction-date YYYY_mm_dd
# Output: data/experiments/nfix/judge/gemma-3-27b/YYYY_mm_dd/combined/combined.json
```
