# data/nfix — Dinitrogen Fixation Dataset

Ground truth and probe dataset for the nfix (aquatic dinitrogen fixation) dataset.
Papers report N₂ fixation rates measured in marine, estuarine, and freshwater environments.

## Directory structure

```
data/nfix/
  directory.json            — paper registry: title, author, year
  preprocessing.py          — builds ground_truth.csv and ground_truth_ten.csv
  create_probe_dataset.py   — builds probe_dataset.json for judge calibration
  ground_truth.csv          — all registered papers (long format)
  ground_truth_ten.csv      — top-10 paper development subset
  probe_dataset.json        — synthetic valid/invalid records (train split)
  probe_dataset_test.json   — synthetic valid/invalid records (test split)
  raw_data/                 — source CSV files
  ocr_output_raw/           — plain-text OCR files (one .txt per paper)
  pdfs/                     — source PDFs
```

---

## Ground truth

```bash
python data/nfix/preprocessing.py
```

**Output schema:**
`document_id, name, identifiers, location, site_type, date, nfix_method, substrate_type, sample_depth, additional_details, attribute, value, units, page_number, page_score, page_confidence`

| Column | Description |
|---|---|
| `page_number` | JSON list of candidate 1-indexed page numbers within the score margin; `NaN` if OCR file missing |
| `page_score` | Weighted similarity score in [0, 1] for the top attributed page |
| `page_confidence` | `"table-anchored"`, `"high"`, `"medium"`, or `"ambiguous"` |

`"table-anchored"` means the page was inferred from `extraction_location_details` in
`directory.json` rather than scored across all pages.

**Attributes:** all records use the generic attribute `nfix_rate`. The sub-type is inferred
from the units string at probe-dataset creation time:

| Sub-type | Units pattern | Example |
|---|---|---|
| `nfix_rate_mass` | contains `g⁻¹` or `kg⁻¹` | nmol g⁻¹ h⁻¹ |
| `nfix_rate_areal` | contains `m⁻²` or `cm⁻²` | µmol m⁻² d⁻¹ |
| `nfix_rate_volumetric` | contains `L⁻¹`, `m⁻³`, `mL⁻¹`, or `cm⁻³` | nmol L⁻¹ h⁻¹ |
| `nfix_rate` | no match | fallback |

---

## Probe dataset

```bash
python data/nfix/create_probe_dataset.py [--seed N]
```

Same structure and generation logic as `data/pond` — 2:1 invalid:valid ratio, paper-level
train/test split, three invalid subsets. Nfix-specific differences:

- `change_attribute` is not applicable (all records share the same generic attribute type)
- Entity fields are `name` and `site_type` (instead of pond's `name` and `ecosystem`)
- Fabricated entity names are site names (e.g. "Seahaven Bay") rather than pond names

**Output schema:** same columns as `ground_truth.csv`, plus `label` (`"valid"` / `"invalid"`),
`modification_type`, `gt_row_index`, `donor_gt_row_index`, `measurement_id`.
