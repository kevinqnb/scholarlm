# ScholarlM Experiment Framework

This document describes the unified experiment framework under `experiments/`.
The framework replaced an earlier design where all experiment code lived in
dataset-specific directories (`experiments/pond/`, `experiments/nfix/`) with
duplicated logic and hardcoded paths. Those scripts are left untouched as
historical records.

---

## Overview

The library (`src/scholarlm/`) extracts structured scientific measurements from
PDFs. Two datasets are currently in use:

- **pond** — aquatic ecosystem observations
- **nfix** — dinitrogen fixation measurements

The framework provides a single set of runner scripts that work for any dataset,
driven by per-dataset config objects rather than hardcoded values.

---

## Key design pattern: `DatasetConfig` and `experiments/configs/`

`src/scholarlm/config.py` defines two dataclasses:

**`DatasetConfig`** — all dataset-specific values:

| Field | Description |
|---|---|
| `name` | Short identifier used in output paths and CLI (`"pond"`, `"nfix"`) |
| `data_dir` | Root directory for the dataset (`"data/pond"`).  Raw OCR lives at `{data_dir}/ocr_output_raw/`, PDFs at `{data_dir}/pdfs/` |
| `metadata_file` | Path to `directory.json` mapping paper codes to metadata |
| `entity_schema` | Pydantic `BaseModel` subclass defining the entity representation |
| `entity_identification_prompt` | System prompt for the entity extraction step |
| `entity_type_description` | One-sentence description of what an entity is |
| `attribute_info_dict` | `{attr_name: {description, units}}` passed to `MeasurementLM` |
| `paper_subset` | Optional explicit list of paper codes to process |
| `paper_filter` | Optional callable `(metadata: dict) -> bool` applied before `paper_subset` |

**`ModelConfig`** — extraction model configuration:

| Field | Description |
|---|---|
| `name` | Short identifier used in output paths and CLI |
| `model_id` | HuggingFace model ID passed to vLLM |
| `tensor_parallel_size` | Number of GPUs for tensor parallelism |
| `sampling_params` | vLLM `SamplingParams` kwargs |

Each dataset has a config file at `experiments/configs/{name}.py` that exports
a module-level `CONFIG: DatasetConfig`. Runner scripts load these dynamically
via `importlib` — no runner imports any dataset-specific code directly.

---

## Runner scripts

All scripts are run from the **repository root** and share the conventions
`--dataset` (required), `--date` (optional, defaults to today's date as
`YYYY_mm_dd`), and `--paper-subset` (optional, overrides the config's default
subset).

### `experiments/run_ocr.py`

Runs olmOCR (`allenai/olmOCR-2-7B-1025-FP8`) on all PDFs for a dataset.

```
Output: data/{dataset}/ocr_output_raw/
```

```bash
python experiments/run_ocr.py --dataset pond
python experiments/run_ocr.py --dataset pond --resume  # skip already-processed PDFs
```

### `experiments/run_table_cleaning.py`

**Legacy script** for API-based table cleaning (OpenAI only). For local
model table cleaning, use `run_extraction.py` — the extraction model
cleans tables automatically as the first step.

```
Output: data/{dataset}/ocr_output_cleaned_openai_{model_tag}/
```

```bash
python experiments/run_table_cleaning.py --dataset pond --model gpt-4o-mini
```

The output directory can then be passed to `run_extraction.py --ocr-dir` to
skip the integrated cleaning step.

Flags: `--input-dir`, `--output-dir` (path overrides), `--rate-limit` (RPM,
default 100), `--paper-subset`, `--resume`.

### `experiments/run_extraction.py`

Runs the full `MeasurementLM` extraction pipeline. Table cleaning is
integrated as **Step 0**: when `--ocr-dir` is not supplied, the extraction
model cleans tables from raw OCR before extraction begins.  Cleaned texts
are saved to `{data_dir}/ocr_output_cleaned_{model_name}/`.

The pipeline then runs 6 extraction steps written sequentially to the output
directory:

```
entities.json → attributes.json → entity_prov.json →
attribute_prov.json → values.json → final.json
```

```
Output: data/experiments/{dataset}/extraction/{model}/{YYYY_mm_dd}/
```

```bash
# Standard run (table cleaning + extraction):
python experiments/run_extraction.py --dataset pond --model gemma-3-27b

# Skip table cleaning by supplying pre-cleaned texts:
python experiments/run_extraction.py --dataset pond --model gemma-3-27b \
    --ocr-dir data/pond/ocr_output_cleaned_openai_gpt_4o_mini
```

**Available models** (keys of `MODEL_REGISTRY` in the script):

| Key | HuggingFace ID | GPUs |
|---|---|---|
| `gemma-3-27b` | `gaunernst/gemma-3-27b-it-qat-autoawq` | 1 |
| `qwen-2.5-72b` | `Qwen/Qwen2.5-72B-Instruct` | 2 |
| `llama-3.3-70b` | `meta-llama/Llama-3.3-70B-Instruct` | 2 |
| `qwen-3.5-35b` | `Qwen/Qwen3.5-35B-A3B-FP8` | 1 |
| `gpt-oss-120b` | `openai/gpt-oss-120b` | 2 |

**Additional flags:**

| Flag | Effect |
|---|---|
| `--ocr-dir DIR` | Load pre-cleaned texts from DIR; skip integrated table cleaning |
| `--resume` | Skip steps whose output file already exists |
| `--final-only` | Run all steps in a temp dir; copy only `final.json` to output |
| `--step <name>` | Run a single named step (mutually exclusive with `--final-only`) |
| `--paper-subset p1 p2` | Override the config's default paper subset |
| `--date YYYY_mm_dd` | Pin the output date tag |

Step names: `entities`, `attributes`, `entity_prov`, `attribute_prov`,
`values`, `final`.

### `experiments/run_judge.py`

Runs validation for a given (dataset, extraction model, judge model) triple.
Use `--ocr-dir` to supply the same OCR texts used during extraction (defaults
to `{data_dir}/ocr_output_raw/`).

```
Output: data/experiments/{dataset}/judge/{extraction_model}/{judge_model}/{YYYY_mm_dd}/
```

**Local judges** (NNsight / JudgementLM — produce `responses.json` +
`attention_outputs.npz`):

```bash
python experiments/run_judge.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd
```

Available local judge keys: `llama-3.1-8b`, `qwen-3-8b`, `gemma-3-12b`.

**Frontier judges** (batch API — produce `responses.json`):

```bash
python experiments/run_judge.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd
```

Frontier provider keys: `openai`, `anthropic`, `gemini` (Gemini additionally
requires `--dest-gcs` and `--gcp-project`).

Frontier runs also support a three-step mode for large batches:

```bash
# Submit, poll, and process separately:
python experiments/run_judge.py ... submit
python experiments/run_judge.py ... poll   --state .batch_state_openai.json
python experiments/run_judge.py ... process --state .batch_state_openai.json
```

### `experiments/run_judge_combine.py`

Merges per-judge `responses.json` files, computes a majority-vote ground-truth
label from the frontier judges, and writes a combined output file.

```
Output: data/experiments/{dataset}/judge/{extraction_model}/combined/combined.json
```

```bash
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b

# Specify judges explicitly (useful when multiple date dirs exist):
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judges openai anthropic gemini llama-3.1-8b
```

Only frontier judge votes (`openai`, `anthropic`, `gemini`) count toward the
ground-truth majority; local judge results are merged in as additional fields.

---

## `experiments/batch/` — generic batch API infrastructure

Provider-specific batch logic originally lived inside
`experiments/pond/judge/batch/` and was duplicated per dataset. It now lives in
`experiments/batch/` as a proper Python package importable by any runner.

| Module | Purpose |
|---|---|
| `common.py` | Prompt building, `prepare_chat_entries(data, documents, dataset_config=None)`, load/merge utilities |
| `openai_batch.py` | OpenAI Batch API: chunks by bytes (200 MB) and tokens (40 M enqueued), token-budget polling |
| `anthropic_batch.py` | Anthropic Batch API with prompt caching (`cache_control: ephemeral` on system + document blocks) |
| `gemini_batch.py` | Gemini Batch API via Vertex AI + GCS |
| `openai_batch_errors.py` | Standalone diagnostic: `python openai_batch_errors.py <batch_id>` |
| `delete_batch.py` | Gemini batch cleanup utility |

`experiments/pond/judge/batch/` now contains only `__init__.py` and `run.py`
(the CLI entry point for the original pond judge). `run.py` imports from
`batch.*` via `sys.path`.

### `sys.path` convention

Every runner adds `str(_REPO_ROOT / "src")` so `scholarlm` is importable.
`run_judge.py` additionally adds `str(_EXPERIMENTS_DIR)` so
`from batch import ...` resolves to `experiments/batch/`. `run.py` (four levels
deep in `pond/judge/batch/`) does the same by walking up four parent levels with
`Path(__file__).parent.parent.parent.parent`.

---

## Directory structure

```
experiments/
├── configs/
│   ├── pond.py              # DatasetConfig for the pond dataset
│   └── nfix.py              # DatasetConfig for the nfix dataset
├── batch/                   # Generic batch API infrastructure
│   ├── common.py
│   ├── openai_batch.py
│   ├── anthropic_batch.py
│   ├── gemini_batch.py
│   ├── openai_batch_errors.py
│   └── delete_batch.py
├── pond/
│   ├── EXPERIMENTS.md       # Commands for pond experiments (old + new)
│   ├── ocr/                 # Original OCR scripts (historical)
│   ├── extract/             # Original extraction scripts (historical)
│   ├── judge/               # Original judge scripts (historical)
│   │   └── batch/run.py     # CLI entry point; imports from experiments/batch/
│   ├── preprocessing.py
│   └── validation.py
├── nfix/
│   ├── EXPERIMENTS.md       # Commands for nfix experiments (old + new)
│   ├── ocr/                 # Original OCR scripts (historical)
│   ├── extract/             # Original extraction scripts (historical)
│   └── preprocessing.py
├── run_ocr.py
├── run_table_cleaning.py
├── run_extraction.py
├── run_judge.py
├── run_judge_combine.py
└── run_analysis.py
```

---

## Experiment documentation

`experiments/pond/EXPERIMENTS.md` and `experiments/nfix/EXPERIMENTS.md` each
document two things:

1. **Recreating the original experiments** using the new unified scripts.
   - Pond: `gemma-3-27b` extraction, `gpt-5-mini` table cleaning, four judge
     models (LLaMA 3.1 8B local + OpenAI/Anthropic/Gemini frontier). The eight
     ablation variants in `pond/extract/` have no equivalent in the unified
     framework and must be re-run with the original scripts.
   - Nfix: `gemma-3-27b`, `qwen-3.5-35b`, and `gpt-oss-120b` extraction,
     vLLM `gemma-3-27b` table cleaning. No judge step was run originally.

2. **New experiments** with all available model combinations.

Note: `ocr_dir` was removed from `DatasetConfig` in favour of the
`--ocr-dir` CLI argument on `run_extraction.py` and `run_judge.py`.
If no `--ocr-dir` is passed, `run_extraction.py` performs integrated
table cleaning and saves the cleaned texts automatically.
