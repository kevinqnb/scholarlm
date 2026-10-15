# data/supermat — Superconductivity Dataset

Adapted from SuperMat (Foppiano et al. 2021, `reference.pdf`): a linked annotated corpus of 142
superconductivity papers with `<material>`, `<class>`, `<tc>`, `<tcValue>`, `<pressure>`, and
`<me_method>` annotations. Papers report superconducting critical temperature (Tc) measurements
for specific materials, along with the applied pressure and measurement method.

The pipeline maps SuperMat's tags onto the entity/attribute/event model: the **material** is the
entity, **Tc** is the (only) attribute, and **pressure**/**measurement method** are the event
fields distinguishing different Tc measurements of the same material.

All commands below run from the repo root, in the `uv sync` environment.

## Tracked files

| File | What it is / who uses it |
|---|---|
| `raw_data.csv` | Source CSV export (material, tcValue, pressure, me_method, filename). Input to `preprocessing.py`. |
| `reference.pdf` | SuperMat paper describing the annotation scheme. |
| `directory.json` | Paper registry (title, author, year; None where unavailable). `DatasetConfig.metadata_file`. |
| `ground_truth_qualifiers.json` | **The ground truth used everywhere** (`DatasetConfig.ground_truth_file`, all analysis configs): raw, undropped `tcValue` text parsed into qualifier/shape fields. |
| `ground_truth.json`, `ground_truth_ten.json` | Earlier ground truth (ranges/approximations/bounds dropped, see below) and its ten-paper subset. Not read by current configs. |
| `probe_split_v3s.json` | `{document_id: "train"\|"test"}` for all 142 papers. Test = the 72 papers the validation sets are drawn from. Read by the pilot and main validation-set builders and by `realign_probe_split.py`. |
| `probe_augment_cache_v3.json` | gpt-oss-120b response cache for the v3 probe dataset (makes the rebuild reproducible; pinned by the v3 probe-augment configs). |
| `prevalence_validation_set_<id>.json` | Prevalence-estimation sample for extraction run `<id>` (human-validation site). |
| `matching_validation_set_<id>.json` | 100 GT↔extraction candidate pairs for run `<id>` (human-validation site). |
| `preprocessing.py`, `build_directory.py`, `create_probe_dataset.py`, `realign_probe_split.py` | Builders, see below. |

Untracked but live (gitignored): `probe_dataset_v3s.json` (train), `probe_dataset_test_v3s.json` (primary test), `probe_dataset_test_v3s_diag.json` (diagnostic test), `probe_dataset_v3s_realign_log.json`, `validation_set_{pilot,main}-gemma27b-full.json`, and the data not shared for licensing: `pdfs/`, `processed_pdfs/`, `ocr_output_raw/`, `ocr_output_cleaned_gemma-3-27b/` (one file, pinned by a smoke config).

## Document reconciliation note

`raw_data.csv` references 147 filename codes, but only 142 PDFs are available locally. Of the 5 codes with no local PDF, 3 (`JPS0731655-CC`, `L095167004-CC`, `SSC1310125-CC`) are duplicate bibliographic registrations of a paper already present under a different code (verified by matching DOIs in SuperMat's biblio metadata and the PDF content), so their rows are merged into the real-PDF document's `document_id`. Only 2 codes (`PHC2640145-CC`, `yamaguchi2014ac`) are genuinely missing and their rows are dropped. See `preprocessing.py`'s docstring.

## Ground truth value policy

`ground_truth.json` values are single, unambiguous reported numbers (matching pond/nfix): rows whose `tcValue` is a range (`"7-8 K"`), approximation (`"~30 K"`), or bound (`"< 10 K"`, `"up to 33 K"`) are dropped, so a few papers have zero rows. `ground_truth_qualifiers.json` keeps those rows and parses the qualifier into separate fields.

## Reproducing the files

**Paper registry** (`directory.json`). `build_directory.py` reads SuperMat biblio files from a hardcoded path (`BIBLIO_ROOT`, a macOS location); edit it to point at your SuperMat checkout.

```bash
python data/supermat/build_directory.py
```

**Ground truth.** Reads `raw_data.csv`. Page attribution (page_number/page_score/page_confidence) needs OCR text in `ocr_output_raw/` first; there is no committed OCR experiment config for supermat.

```bash
python data/supermat/preprocessing.py               # ground_truth.json, ground_truth_ten.json
python data/supermat/preprocessing.py --qualifiers  # ground_truth_qualifiers.json only
```

**Probe dataset v3** (the source of v3s). Needs a served gpt-oss-120b, so run it as an SGE job. The config pins seed 42, `--qualifiers`, the cache above, and the table-cleaned OCR dir `experiments/results/supermat/drop_references/2026-09-18-supermat-table-cleaning-drop-references-01` (produced by that experiment, so it must already exist). The outputs are `probe_dataset_v3.json`, `probe_dataset_test_v3.json` and `probe_dataset_test_v3_diag.json`.

```bash
bash experiments/submit.sh 2026-09-30-supermat-probe-augmentation-v3-01 [--dry-run]
# runner directly (needs the vLLM server up):
python experiments/run_probe_augment.py experiments/experiment-configs/supermat/probe_augment/2026-09-30-supermat-probe-augmentation-v3-01/2026-09-30-supermat-probe-augmentation-v3-01.yaml
```

**Probe dataset v3s** (use v3s, not v3). v3s is the v3 rows rearranged onto the fixed paper split in `probe_split_v3s.json`, so the test papers are exactly the papers the validation sets are drawn from. Nothing is regenerated. The script never overwrites, so output paths must be new.

1. Rearrange and trim the v3 rows (seed 342) to get `probe_dataset_v3s.json`, `probe_dataset_test_v3s.json`, `probe_dataset_test_v3s_diag.json` and `probe_dataset_v3s_realign_log.json`. `--in-dir` is where the three v3 files from the step above are:

   ```bash
   python data/supermat/realign_probe_split.py realign --split-file data/supermat/probe_split_v3s.json --seed 342 --in-dir data/supermat --out-dir <new dir>
   ```

2. Move the judge outputs the same way, once per judge model `<m>` in `llama-3.1-8b`, `mistral-nemo-12b`, `qwen-2.5-7b`. Needs the three v3 judge_interp runs for `<m>` under `experiments/results/supermat/judge_interp/` (`2026-09-30-supermat-v3-<m>-synthetic-judge-{train,test-primary,test-diag}-01`), and the step-1 files in `data/supermat/`. It writes the `2026-10-08-supermat-v3s-<m>-synthetic-judge-*-01` runs (responses, activations, run_metadata).

   ```bash
   J=experiments/results/supermat/judge_interp
   python data/supermat/realign_probe_split.py judge --log data/supermat/probe_dataset_v3s_realign_log.json --data-dir data/supermat \
     --train-run $J/2026-09-30-supermat-v3-<m>-synthetic-judge-train-01 \
     --primary-run $J/2026-09-30-supermat-v3-<m>-synthetic-judge-test-primary-01 \
     --diag-run $J/2026-09-30-supermat-v3-<m>-synthetic-judge-test-diag-01 \
     --out-train $J/2026-10-08-supermat-v3s-<m>-synthetic-judge-train-01 \
     --out-primary $J/2026-10-08-supermat-v3s-<m>-synthetic-judge-test-primary-01 \
     --out-diag $J/2026-10-08-supermat-v3s-<m>-synthetic-judge-test-diag-01
   ```

A fresh `create_probe_dataset.py --qualifiers --augment --split-file data/supermat/probe_split_v3s.json ...` is a new random draw and does not byte-match the v3s files.

**Validation sets** (`prevalence_*`, `matching_*`, `validation_set_*`). Builders are in `data/validation/`, with inputs, ordering and rebuild checks in `data/validation/README.md`. Live sets must be rebuilt only to verify, never "refreshed".

```bash
python data/validation/build_prevalence_validation_sets.py supermat
python data/validation/build_matching_validation_sets.py supermat
python data/validation/build_pilot_validation_sets.py supermat   # splits from probe_split_v3s.json
python data/validation/build_main_validation_sets.py supermat    # run after pilot
```
