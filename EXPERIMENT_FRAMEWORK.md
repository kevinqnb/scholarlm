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
| `measurement_event_schema` | Optional Pydantic `BaseModel` defining a measurement event; enables event-resolution step |
| `measurement_event_prompt` | Dataset-specific instructions for the event-resolution step |
| `direct_extraction_schema` | Optional Pydantic `BaseModel` for Ablation 1 (direct triple extraction) |
| `direct_extraction_prompt` | Dataset-specific prompt for Ablation 1 |
| `ablation3_entity_schema` | Optional Pydantic `BaseModel` for Ablation 3 (combined entity-attribute extraction); must include entity fields + `attribute (str)` + `attribute_terms (list[str])` |
| `ablation3_entity_identification_prompt` | Dataset-specific prompt for Ablation 3 that instructs the model to emit one item per (entity, attribute) pair |

**`ModelConfig`** — extraction model configuration:

| Field | Description |
|---|---|
| `name` | Short identifier used in output paths and CLI |
| `model_id` | HuggingFace model ID (vLLM) or API model name (frontier); used as the `model` field in API requests |
| `sampling_params` | Generation parameters forwarded to the API (`temperature`, `top_p`, `top_k`, `max_tokens`, `repetition_penalty`) |
| `api_base` | API base URL for frontier models (e.g. `"https://api.openai.com/v1"`).  When `None` the model is vLLM and runners use `--api-base` |

Each dataset has a config file at `experiments/configs/{name}.py` that exports
a module-level `CONFIG: DatasetConfig`. Runner scripts load these dynamically
via `importlib` — no runner imports any dataset-specific code directly.

---

## Centralized model registry: `experiments/model_registry.py`

All model registries live in a single file imported by every runner.  Adding or
modifying a model in one place immediately takes effect across all scripts.

| Registry | Type | Used by |
|---|---|---|
| `MODEL_REGISTRY` | `dict[str, ModelConfig]` | `run_extraction.py`, `run_ablation.py`, `run_vllm_table_cleaning.py` |
| `INTERP_JUDGE_REGISTRY` | `dict[str, dict]` | `run_judge.py` (as `LOCAL_JUDGE_REGISTRY`), `run_judge_interp.py` (as `JUDGE_REGISTRY`) |
| `VLLM_JUDGE_REGISTRY` | `dict[str, dict]` | `run_judge_local.py` (as `JUDGE_REGISTRY`) |
| `FRONTIER_JUDGE_PROVIDERS` | `set[str]` | `run_judge.py` (as `FRONTIER_PROVIDERS`) |

### `MODEL_REGISTRY` — extraction models

`ModelConfig.api_base` is the key field that distinguishes vLLM from frontier:

- **`api_base = None`** (vLLM models): runners use the `--api-base` CLI flag; vLLM-specific
  sampling params (`top_k`, `repetition_penalty`, `enable_thinking`) are forwarded in
  `extra_body`; table cleaning runs before extraction.
- **`api_base` set** (frontier models): runners use the registered URL directly, skip
  `--api-base`; `extra_body` is suppressed; table cleaning is skipped; the API key is
  resolved from environment variables (`OPENAI_API_KEY` or `GEMINI_API_KEY`) unless
  `--api-key` is passed explicitly.

Current vLLM entries: `gemma-3-27b`, `gemma-4-31b`, `qwen-2.5-vl-72b`, `qwen-3-vl-30b`,
`llama-4-scout-109b`, `glm-4.6v-106b`, `intern-vl3-78b`, `llama-3.3-70b`, `qwen-2.5-72b`,
`qwen-3.5-27b`, `glm-4.5-110b`, `gpt-oss-120b`.

Current frontier entries: `gpt-4o-mini`, `gpt-4.1-mini`, `gpt-5-mini`, `gpt-4o`,
`gemini-2-flash-lite`, `gemini-2-flash`, `gemini-3-flash-lite`, `gemini-1.5-flash`.

### `INTERP_JUDGE_REGISTRY` — NNsight judge models

Five models; merged from what was previously split across `run_judge.py` and
`run_judge_interp.py`: `llama-3.1-8b`, `qwen-3-8b`, `gemma-3-12b`, `gemma-2-9b`,
`mistral-7b`.

### `VLLM_JUDGE_REGISTRY` — vLLM logprob judge models

Five models served via vLLM's OpenAI-compatible API: `gemma-3-27b`, `qwen-3.5-27b`,
`llama-3.3-70b`, `qwen-2.5-72b`, `gpt-oss-120b`.

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

### `experiments/process_pdfs.py`

Pre-renders all PDF pages to base64-encoded PNGs in a preprocessing environment
(Pillow / pypdf / pdfinfo).  This must be run **before** `run_extraction.py`
whenever integrated table cleaning is used, because the rendering libraries are
not available in the vLLM environment.

```
Output: data/{dataset}/processed_pdfs/{paper_code}/{page_index}.b64
```

```bash
python experiments/process_pdfs.py --dataset pond
python experiments/process_pdfs.py --dataset nfix --resume
```

Flags: `--paper-subset`, `--target-longest-dim` (default: 1536), `--resume`.

### `experiments/run_vllm_table_cleaning.py`

Runs only the table-cleaning step using a vLLM server, without running
the full extraction pipeline.  Useful when you want to clean tables once
and then run extraction (or re-run extraction) with ``--ocr-dir``.

**Prerequisite:** `process_pdfs.py` must be run first to produce
`data/{dataset}/processed_pdfs/`, and a vLLM server must be running.

```
Output: data/{dataset}/ocr_output_cleaned_{model_name}/
```

```bash
# Start the vLLM server first, then:
python experiments/run_vllm_table_cleaning.py --dataset pond --model gemma-3-27b

# Resume a partial run:
python experiments/run_vllm_table_cleaning.py --dataset pond --model gemma-3-27b --resume
```

Flags: `--ocr-dir`, `--output-dir` (path overrides), `--api-base`, `--api-key`,
`--paper-subset`, `--resume`.

### `experiments/run_table_cleaning.py`

**Legacy script** for API-based table cleaning (OpenAI only).

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

Runs the full `MeasurementLM` extraction pipeline.  Supports both local vLLM
models (requires a separately-started vLLM server) and frontier API models
(OpenAI, Gemini) from `MODEL_REGISTRY`.

For **vLLM models**, table cleaning is integrated as **Step 0**: when `--ocr-dir`
is not supplied, the extraction model cleans tables from raw OCR before
extraction begins.  Cleaned texts are saved to
`{data_dir}/ocr_output_cleaned_{model_name}/`.  **Prerequisite:** `process_pdfs.py`
must be run first (in the preprocessing environment) to produce
`data/{dataset}/processed_pdfs/`.

For **frontier models**, table cleaning is skipped and no vLLM server is needed.
The API key is read from `OPENAI_API_KEY` or `GEMINI_API_KEY` environment
variables (or pass `--api-key` explicitly).

The pipeline runs 6 extraction steps written sequentially to the output directory:

```
entities.json → attributes.json → entity_prov.json →
attribute_prov.json → events.json → values.json → final.json
```

```
Output: data/experiments/{dataset}/extraction/{model}/{YYYY_mm_dd}/
```

#### Starting a vLLM server (vLLM models only)

```bash
# Single-GPU example (quantized model):
vllm serve gaunernst/gemma-3-27b-it-int4-awq \
    --tensor-parallel-size 1 \
    --port 8000

# Multi-GPU example (large model):
vllm serve Qwen/Qwen2.5-VL-72B-Instruct-AWQ \
    --tensor-parallel-size 4 \
    --port 8000
```

The server is ready when it prints `Application startup complete`.

#### Running extraction

```bash
# vLLM model (table cleaning + extraction):
python experiments/run_extraction.py --dataset pond --model gemma-3-27b

# vLLM model, skip table cleaning:
python experiments/run_extraction.py --dataset pond --model gemma-3-27b \
    --ocr-dir data/pond/ocr_output_cleaned_gemma-3-27b

# vLLM model on a non-default server URL:
python experiments/run_extraction.py --dataset pond --model gemma-3-27b \
    --api-base http://gpu-node-01:8000/v1

# Frontier model (no server needed; reads OPENAI_API_KEY from env):
python experiments/run_extraction.py --dataset pond --model gpt-4o-mini

# Frontier model with explicit API key:
python experiments/run_extraction.py --dataset nfix --model gemini-2-flash \
    --api-key $GEMINI_API_KEY
```

**Available models** are defined in `experiments/model_registry.py` — see the
`MODEL_REGISTRY` section above for the current list.  Pass any registry key
as `--model`.

**Additional flags:**

| Flag | Effect |
|---|---|
| `--ocr-dir DIR` | Load pre-cleaned texts from DIR; skip integrated table cleaning (vLLM only) |
| `--api-base URL` | vLLM server base URL (default: `http://localhost:8081/v1`); ignored for frontier models |
| `--api-key KEY` | API key (default: `EMPTY` for vLLM; auto-resolved from env for frontier) |
| `--resume` | Skip steps whose output file already exists |
| `--final-only` | Run all steps in a temp dir; copy only `final.json` to output |
| `--step <name>` | Run a single named step (mutually exclusive with `--final-only`) |
| `--paper-subset p1 p2` | Override the config's default paper subset |
| `--date YYYY_mm_dd` | Pin the output date tag |

Step names: `entities`, `attributes`, `entity_prov`, `attribute_prov`, `events`,
`values`, `final`.

### `experiments/run_ablation.py`

Runs a single ablation variant of the `MeasurementLM` pipeline.  Supports both
vLLM and frontier models from `MODEL_REGISTRY` — the same `api_base` detection
logic as `run_extraction.py` applies.

```
Output: data/experiments/{dataset}/ablations/ablation{N}/{model}/{YYYY_mm_dd}/
```

Writes a single `final.json` file (standardized, deduplicated records with paper metadata
merged in, same schema as `run_extraction.py`'s `final.json`).  Intermediate step files
are not written because ablations run as a single `fit()` call.

**Available ablations:**

| N | Description |
|---|---|
| `1` | **Direct triple extraction** — the entire pipeline is replaced by a single LLM call per document that extracts all (entity, attribute, value) triples at once.  Requires the dataset config to define `direct_extraction_schema` and `direct_extraction_prompt`. |
| `2` | **Direct table value extraction** — the model returns the value directly from the table instead of first identifying row/column indices for programmatic lookup. |
| `3` | **Combined entity-attribute extraction** — entity detection and attribute detection merged into one step; provenance is also combined. Requires the dataset config to define `ablation3_entity_schema` and `ablation3_entity_identification_prompt`. |
| `4` | **Full-document context for value extraction and event resolution** — the full document (not just the relevant page/table) is sent to the value extractor and event resolver at both text and table extraction steps. |
| `5` | **No chain-of-thought explanations** — the `explanation` field is removed from all structured JSON response schemas, so the model does not produce reasoning traces.  Event resolution is unchanged (event schemas contain no explanation fields). |
| `6` | **Full-document pair provenance** — both provenance steps (entity + attribute) are replaced by a single full-document query per (entity, attribute) pair that returns a list of provenance locations. |

```bash
# vLLM model, run ablation 1 on the pond dataset (direct triple extraction):
python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 1

# vLLM model, skip table cleaning:
python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 5 \
    --ocr-dir data/pond/ocr_output_cleaned_gemma-3-27b

# Frontier model, ablation 3 (combined entity-attribute extraction):
python experiments/run_ablation.py --dataset nfix --model gpt-4o-mini --ablation 3

# Run on a specific paper subset:
python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 2 \
    --paper-subset physical_and_chemical_limnological prairie_wetland
```

**Flags:**

| Flag | Effect |
|---|---|
| `--ablation N` | Ablation to run (required, 1–6) |
| `--ocr-dir DIR` | Load pre-cleaned texts from DIR; skip integrated table cleaning (vLLM only) |
| `--api-base URL` | vLLM server base URL (default: `http://localhost:8081/v1`); ignored for frontier models |
| `--api-key KEY` | API key (default: `EMPTY` for vLLM; auto-resolved from env for frontier) |
| `--paper-subset p1 p2` | Override the config's default paper subset |
| `--date YYYY_mm_dd` | Pin the output date tag |

**Note on ablation 3:** The dataset config must define `ablation3_entity_schema` (entity
fields plus `attribute: str` and `attribute_terms: list[str]`) and
`ablation3_entity_identification_prompt` (instructs the model to emit one item per
(entity, attribute) pair).  The script raises a clear error if either is missing.
Both are defined in `experiments/configs/pond.py` and `experiments/configs/nfix.py`.

---

### Judge scripts

Three judge runners share the same output path convention and produce
`responses.json` files that are compatible with `run_judge_combine.py`.
Use `--ocr-dir` on any runner to supply the same OCR texts used during
extraction (defaults to `{data_dir}/ocr_output_raw/`).

```
Output: data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_model}/{judge_date}/
```

#### `experiments/run_judge_interp.py` — NNsight / mechanistic interpretability

Loads a local model through NNsight (JudgementLM) and collects per-layer,
per-head attention output activations alongside binary judgement probabilities.

Saves: `responses.json` + `attention_outputs.npz`

Output fields: `judgement`, `judgement_prob`, `judgement_p_true`,
`judgement_p_false`, `judgement_logit_p_true`, `judgement_logit_p_false`,
`judgement_model`.

```bash
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date YYYY_mm_dd
```

Available judge keys (`INTERP_JUDGE_REGISTRY` in `model_registry.py`):
`llama-3.1-8b`, `qwen-3-8b`, `gemma-3-12b`, `gemma-2-9b`, `mistral-7b`.

#### `experiments/run_judge_local.py` — local model via vLLM

Sends async requests to a running vLLM server with `max_tokens=1` and
`logprobs=True`. Extracts P(true)/P(false) from the next-token log-probability
distribution by looking for `"true"`, `"True"`, `"false"`, `"False"` in the
top-N returned tokens (default N=20, matching vLLM's `--max-logprobs` default).
If those tokens are absent from the top-N list, probability fields are `null`
and `judgement` falls back to the generated token text.

Saves: `responses.json`

Output fields: `judgement`, `judgement_prob`, `judgement_p_true`,
`judgement_p_false`, `judgement_logit_p_true`, `judgement_logit_p_false`,
`judgement_model`.

```bash
# Start a vLLM server first, e.g.:
#   vllm serve gaunernst/gemma-3-27b-it-int4-awq --port 8081

python experiments/run_judge_local.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge gemma-3-27b --extraction-date YYYY_mm_dd

# Raise the logprob ceiling (requires --max-logprobs N on the vLLM server):
python experiments/run_judge_local.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge gemma-3-27b --top-logprobs 50
```

Available judge keys (`VLLM_JUDGE_REGISTRY` in `model_registry.py`):
`gemma-3-27b`, `qwen-3.5-27b`, `llama-3.3-70b`, `qwen-2.5-72b`, `gpt-oss-120b`.

Flags: `--api-base URL` (default: `http://localhost:8081/v1`), `--api-key KEY`,
`--max-concurrent N` (default: 64), `--top-logprobs N` (default: 20).

#### `experiments/run_judge_frontier.py` — frontier batch API

Submits extraction results to a frontier provider's batch API and saves
judgements. Supports a one-shot run (submit → poll → process) or a three-step
mode for large batches.

Saves: `responses.json`

Output fields: `judgement`, `judgement_prob` (always `null` for frontier),
`judgement_raw_text`, `judgement_model`.

```bash
# One-shot run:
python experiments/run_judge_frontier.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini \
    --extraction-date YYYY_mm_dd

# Three-step mode (useful for large batches or to resume after interruption):
python experiments/run_judge_frontier.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge anthropic --frontier-model claude-haiku-4-5 \
    submit
python experiments/run_judge_frontier.py ... poll    --state .batch_state_anthropic.json
python experiments/run_judge_frontier.py ... process --state .batch_state_anthropic.json
```

Frontier provider keys (`FRONTIER_JUDGE_PROVIDERS` in `model_registry.py`):
`openai`, `anthropic`, `gemini` (Gemini additionally requires `--dest-gcs` and
`--gcp-project`).

#### `experiments/run_judge.py` (legacy)

The original unified judge runner that combined both local-NNsight and frontier
paths. Superseded by the three scripts above but left in place as a historical
reference.

### `experiments/run_judge_combine.py`

Merges per-judge `responses.json` files, computes a majority-vote ground-truth
label from the frontier judges, and writes a combined output file.

```
Output: data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/combined/combined.json
```

```bash
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b --extraction-date 2026_04_01

# Specify judges explicitly (useful when multiple judge date dirs exist):
python experiments/run_judge_combine.py \
    --dataset pond --extraction-model gemma-3-27b --extraction-date 2026_04_01 \
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

`model_registry.py` also adds `str(_REPO_ROOT / "src")` to `sys.path` so it
can import `scholarlm.config.ModelConfig` without relying on the importing
runner to have set up the path first.

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
├── model_registry.py        # All model registries (MODEL_REGISTRY, INTERP_JUDGE_REGISTRY,
│                            #   VLLM_JUDGE_REGISTRY, FRONTIER_JUDGE_PROVIDERS)
├── run_ocr.py
├── run_vllm_table_cleaning.py
├── run_table_cleaning.py
├── run_extraction.py
├── run_ablation.py
├── run_judge_interp.py      # NNsight / mechanistic interpretability judge
├── run_judge_local.py       # Local model via vLLM (logprob extraction)
├── run_judge_frontier.py    # Frontier batch API judge (OpenAI/Anthropic/Gemini)
├── run_judge.py             # Legacy unified judge (superseded by the three above)
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
   - Nfix: `gemma-3-27b`, `qwen-3.5-27b`, and `gpt-oss-120b` extraction,
     vLLM `gemma-3-27b` table cleaning. No judge step was run originally.

2. **New experiments** with all available model combinations.

Note: `ocr_dir` was removed from `DatasetConfig` in favour of the
`--ocr-dir` CLI argument on `run_extraction.py` and `run_judge.py`.
If no `--ocr-dir` is passed, `run_extraction.py` performs integrated
table cleaning and saves the cleaned texts automatically (vLLM models only).
