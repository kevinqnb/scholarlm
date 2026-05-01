# data/pond — Aquatic Ecosystem Dataset

Ground truth and probe dataset for the pond (aquatic ecosystem) dataset.
Papers report physical and chemical measurements for ponds, lakes, and wetlands.

## Directory structure

```
data/pond/
  directory.json            — paper registry: title, author, year, per-attribute units
  preprocessing.py          — builds ground_truth.csv and ground_truth_ten.csv
  create_probe_dataset.py   — builds probe_dataset.json for judge calibration
  ground_truth.csv          — all registered, non-excluded papers (long format)
  ground_truth_ten.csv      — top-10 paper development subset
  probe_dataset.json        — synthetic valid/invalid records (train split)
  probe_dataset_test.json   — synthetic valid/invalid records (test split)
  raw_data/                 — pond_data_corrected.csv (long) and pond_data_corrected_.csv (wide)
  ocr_output_raw/           — plain-text OCR files (one .txt per paper)
  pdfs/                     — source PDFs
```

---

## Ground truth

```bash
python data/pond/preprocessing.py           # standard run
python data/pond/preprocessing.py --build-corrected  # rebuild wide-format source first
```

**Pipeline:**
```
raw_data/pond_data_corrected_.csv   (wide, standard units)
    → filter to registered + non-excluded papers
    → melt to long format, map attribute column names to canonical names
    → assign document_id from directory.json
    → apply per-paper unit conversions (values stored in paper-original units)
    → page attribution (score all pages per document via OCR)
ground_truth.csv / ground_truth_ten.csv
```

**Output schema:**
`document_id, name, identifiers, location, ecosystem, date, additional_details, attribute, value, units, page_number, page_score, page_confidence`

| Column | Description |
|---|---|
| `page_number` | JSON list of candidate 1-indexed page numbers within the score margin; `NaN` if OCR file missing |
| `page_score` | Weighted similarity score in [0, 1] for the top attributed page |
| `page_confidence` | `"high"`, `"medium"`, or `"ambiguous"` |

**Attributes and standard units:**

| Attribute | Standard unit | Notes |
|---|---|---|
| `max_depth` | m | Maximum physical water depth |
| `surface_area` | m² | Water body surface area |
| `vegetation_cover` | percent | Fraction of surface covered by macrophytes |
| `ph` | — (dimensionless) | Water pH |
| `tn` | µg/L | Total nitrogen |
| `tp` | µg/L | Total phosphorus |
| `chla` | µg/L | Chlorophyll-a |

**Unit conversion:** papers report values in their own units; `directory.json` records the original unit per paper per attribute. `preprocessing.py` converts standard values back to paper-original units so `ground_truth.csv` reflects what the paper actually reports. The `unit_conversion_table` in `experiments/configs/pond.py` inverts these back to standard units at analysis time.

**`directory.json` entry structure:**
```json
"classification_trees": {
    "title": "classification trees as a tool for predicting cyanobacterial blooms",
    "author": "peretyatko et al.",
    "year": 2011,
    "units": { "chla": "µg/L", "max_depth": "m", "ph": null, "tp": "mg/L" }
}
```
`ph` is always `null` (dimensionless). Papers without target-attribute data have no `"units"` field.

**Excluded papers:** two categories are excluded from `ground_truth.csv` and from extraction (listed in `_EXCLUDED_PAPERS` in `experiments/configs/pond.py`):

- *No target attributes in tables:* `application_and_transferability`, `comprehensive_approach`,
  `improved_method`, `livin'_on_the_edge`, `predictions_of_climate_change`, `satellite_radar`,
  `spatio-temporal_surface`, `species_numbers`, `temporary_wetlands`, `the_effects_of_ambient`,
  `the_influence_of_an_in-network`
- *Data quality:* `bacterioplankton` (values from figures, not tables), `summer_assessment`
  (data only in supplemental text)

---

## Probe dataset

```bash
python data/pond/create_probe_dataset.py [--seed N]
```

Built from `ground_truth.csv` and OCR files. ~50% of papers (whole papers at a time) are
labelled valid; two invalid counterparts are generated per valid record (**2:1 invalid:valid
ratio**). The paper-level split ensures no paper appears in both train and test files.

**Subset 1 — Swap invalids** (one per valid record): one field replaced with a value from
a different ground-truth record.

| Type | What changes |
|---|---|
| `change_value` | Value from a record with a different entity or attribute |
| `change_attribute` | Attribute name; value and units unchanged |
| `change_entity` | All entity fields (name, ecosystem) from a record with a different entity key |
| `change_units` | Units replaced with a different canonical unit for the same attribute; value unchanged |

**Subset 2 — Noise invalids** (~half the valid set): one modification chosen uniformly.

| Type | What changes |
|---|---|
| `noise_value` | Gaussian noise (σ = 30% of \|value\|); formatted to original decimal precision |
| `noise_entity` | Entity name replaced with a fabricated place-name |

**Subset 3 — OCR-table invalids** (~other half of the valid set): value replaced with a
numeric cell from an HTML table in the paper's own OCR file (plausible-looking numbers
from the same document).

**Output schema:** same columns as `ground_truth.csv`, plus `label` (`"valid"` / `"invalid"`),
`modification_type`, `gt_row_index`, `donor_gt_row_index`, `measurement_id`.
