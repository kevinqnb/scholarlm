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
| `probe_split_v3s.json` | `{document_id: "train"\|"test"}` for all 142 papers. Test = exactly the 72 v1 test papers the validation sets were drawn from; train adds the 6 qualifier-only papers. Read by the pilot and main validation-set builders and by `realign_probe_split.py`. |
| `probe_augment_cache_v3.json` | gpt-oss-120b response cache for the v3 probe dataset (makes the rebuild reproducible; pinned by the v3 probe-augment configs). |
| `prevalence_validation_set_<id>.json` | Prevalence-estimation sample for extraction run `<id>` (human-validation site). |
| `matching_validation_set_<id>.json` | 100 GT↔extraction candidate pairs for run `<id>` (human-validation site). |
| `preprocessing.py`, `build_directory.py`, `create_probe_dataset.py`, `realign_probe_split.py` | Builders, see below. |

Untracked but live (gitignored): `probe_dataset_v3s.json` (train), `probe_dataset_test_v3s.json` (primary test), `probe_dataset_test_v3s_diag.json` (diagnostic test), `probe_dataset_v3s_realign_log.json`, `validation_set_{pilot,main}-gemma27b-full.json`, and the data not shared for licensing: `pdfs/`, `processed_pdfs/`, `ocr_output_raw/`, `ocr_output_cleaned_gemma-3-27b/` (one file, pinned by a smoke config).

`older/` (gitignored) holds retired files, including the **v3 probe files that `realign_probe_split.py` needs as input** (`probe_dataset_v3.json`, `probe_dataset_test_v3.json`, `probe_dataset_test_v3_diag.json`); `older/MANIFEST.txt` lists everything.

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

**Probe dataset v3** (the source of v3s). Needs a served gpt-oss-120b, so run it as an SGE job. The config pins seed 42, `--qualifiers`, the table-cleaned OCR dir, and the cache above. The outputs (`probe_dataset_v3.json`, `probe_dataset_test_v3.json`, `probe_dataset_test_v3_diag.json`) are what `older/` holds.

```bash
bash experiments/submit.sh 2026-09-30-supermat-probe-augmentation-v3-01 [--dry-run]
# runner directly (needs the vLLM server up):
python experiments/run_probe_augment.py experiments/experiment-configs/supermat/probe_augment/2026-09-30-supermat-probe-augmentation-v3-01/2026-09-30-supermat-probe-augmentation-v3-01.yaml
```

**Probe dataset v3s** (use v3s, not v3). The split is a seeded shuffle over the GT's paper list, so v3 (built from the 142-paper qualifiers GT) got a different split from v1/v2 (136 papers): 38 v1-train papers landed in v3 test and 26 v1-test papers in v3 train. The validation sets were drawn from the v1 test split, so v3s rearranges the v3 rows onto `probe_split_v3s.json` with no regeneration. Re-running on the retired v3 files reproduces the live v3s files byte-for-byte (verified 2026-10-15):

```bash
python data/supermat/realign_probe_split.py split --gt-file data/supermat/ground_truth_qualifiers.json --out <path>   # recreates probe_split_v3s.json
python data/supermat/realign_probe_split.py realign --split-file data/supermat/probe_split_v3s.json --seed 342 --in-dir data/supermat/older --out-dir <dir>
```

The matching judge-interp runs are `2026-10-08-supermat-v3s-*-synthetic-judge-*-01`, assembled from the v3 runs by `realign_probe_split.py judge`. A fresh `create_probe_dataset.py --qualifiers --augment --split-file data/supermat/probe_split_v3s.json ...` is a new random draw and does not byte-match the v3s files.

**Validation sets** (`prevalence_*`, `matching_*`, `validation_set_*`). Builders are in `data/validation/`, with inputs, ordering and rebuild checks in `data/validation/README.md`. Live sets must be rebuilt only to verify, never "refreshed".

```bash
python data/validation/build_prevalence_validation_sets.py supermat
python data/validation/build_matching_validation_sets.py supermat
python data/validation/build_pilot_validation_sets.py supermat   # splits from probe_split_v3s.json
python data/validation/build_main_validation_sets.py supermat    # run after pilot
```
