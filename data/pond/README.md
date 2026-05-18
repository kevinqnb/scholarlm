# data/pond — Aquatic Ecosystem Dataset

Ground truth and probe dataset for the pond (aquatic ecosystem) dataset.
Papers report physical and chemical measurements for ponds, lakes, and wetlands.

## Directory structure

```
data/pond/
  ocr_output_raw/           — plain-text OCR files (one .txt per paper -- not shared)
  ocr_output_cleaned_{model}/ - plain-text OCR files after table cleaning (not shared)
  pdfs/                     - directory containing all PDF documents (not shared due to licensing)
  processed_pdfs/           - 64 bit representations for all pages of PDF images (not shared)
  raw_data/                 — source CSV files (not shared)
  directory.json            — paper registry: title, author, year
  preprocessing.py          — builds ground_truth.csv and ground_truth_ten.csv
  create_probe_dataset.py   — builds probe_dataset.json for judge calibration
  ground_truth.csv          — ground truth dataset
  ground_truth_ten.csv      — top-10 paper data subset
  page_review.csv           - CSV documenting manual corrections, exclusions made to the dataset
  ground_truth_review.csv   — ground truth dataset with manual corrections applied
  ground_truth_ten.csv      — top-10 paper data subset with manual corrections applied
  probe_dataset.json        — synthetic valid/invalid records (train split)
  probe_dataset_test.json   — synthetic valid/invalid records (test split)
```

---

## Preprocess ground truth data

```bash
python data/pond/preprocessing.py
```
