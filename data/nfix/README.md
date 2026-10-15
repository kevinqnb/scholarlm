# data/nfix — Dinitrogen Fixation Dataset

Papers reporting N₂ fixation rates measured in marine, estuarine, and freshwater environments.
Original paper with data available [here](https://aslopubs.onlinelibrary.wiley.com/doi/10.1002/lol2.10459).
All commands below run from the repo root, in the `uv sync` environment.

Units use LLM-friendly notation applied during preprocessing and used throughout our experiments.

## Tracked files

| File | What it is / who uses it |
|---|---|
| `directory.json` | Paper registry (title, author, year). `DatasetConfig.metadata_file`. |
| `ground_truth.json` | Raw ground truth built from `raw_data/`. Input to `generate_review_csv.py` / `apply_review.py`. |
| `ground_truth_ten.json` | Top-10-paper development subset of the above. Only used by `apply_review.py`. |
| `page_review.csv` | Manual corrections and exclusions (hand-edited). Input to `apply_review.py`. |
| `ground_truth_review.json` | **The ground truth used everywhere** (`DatasetConfig.ground_truth_file`, all analysis configs): `ground_truth.json` + `page_review.csv`. |
| `ground_truth_ten_review.json` | Same corrections applied to the ten-paper subset. Used by `nfix_ten.py`. |
| `split_document_ids.json` | Train (49) / test (45) paper-id split, taken from the v3 probe datasets. Read by the pilot validation-set builder so it doesn't depend on the large untracked v3 files. |
| `probe_augment_cache_v3.json` | gpt-oss-120b response cache for the v3 probe dataset. It makes the rebuild reproducible; pinned by the v3 probe-augment configs. |
| `prevalence_validation_set_<id>.json` | Prevalence-estimation sample for extraction run `<id>` (human-validation site). |
| `matching_validation_set_<id>.json` | 100 GT↔extraction candidate pairs for run `<id>` (human-validation site). |
| `preprocessing.py`, `create_probe_dataset.py` | Builders, see below. |

Untracked but live (gitignored): `probe_dataset_v3.json` (train), `probe_dataset_test_v3.json` (primary test), `probe_dataset_test_v3_diag.json` (diagnostic test), `validation_set_pilot-gemma27b-full.json`, and the data not shared for licensing: `pdfs/`, `processed_pdfs/` (base64 page images), `raw_data/` (source CSVs), `ocr_output_raw/`.

## Reproducing the files

**Ground truth** (`ground_truth.json`, `ground_truth_ten.json`). Reads `raw_data/aquatic_N2fix_rates.csv`, no arguments. Page-row alignment with `page_review.csv` is positional, so diff against the committed file before trusting a rebuild (see the docstring).

```bash
python data/nfix/preprocessing.py
```

**Review file** (`page_review.csv`, `ground_truth_review.json`, `ground_truth_ten_review.json`):

```bash
python data/generate_review_csv.py --dataset nfix   # (re)creates page_review.csv; keeps existing corrections
# ... edit the corrected_* / excluded columns in data/nfix/page_review.csv ...
python data/apply_review.py --dataset nfix          # writes the *_review.json files
```

**Probe dataset v3** (`probe_dataset_v3.json`, `probe_dataset_test_v3.json`, `probe_dataset_test_v3_diag.json`). Needs a served gpt-oss-120b, so run it as an SGE job. The config pins seed 42, `--reviewed`, the table-cleaned OCR dir, and the cache above.

```bash
bash experiments/submit.sh 2026-09-30-nfix-probe-augmentation-v3-01 [--dry-run]
# runner directly (needs the vLLM server up):
python experiments/run_probe_augment.py experiments/experiment-configs/nfix/probe_augment/2026-09-30-nfix-probe-augmentation-v3-01/2026-09-30-nfix-probe-augmentation-v3-01.yaml
```

**Validation sets** (`prevalence_*`, `matching_*`, `validation_set_pilot-*`). Builders are in `data/validation/`, with their inputs, ordering and rebuild checks documented in `data/validation/README.md`. Sets that are already live must be rebuilt only to verify, never "refreshed".

```bash
python data/validation/build_prevalence_validation_sets.py nfix
python data/validation/build_matching_validation_sets.py nfix
python data/validation/build_pilot_validation_sets.py nfix   # splits from split_document_ids.json
```
