# data/pond — Aquatic Ecosystem Dataset

Ground truth, probe dataset, and preprocessing for the pond (aquatic ecosystem) dataset.
Papers report physical and chemical measurements for ponds, lakes, and wetlands.

## Directory structure

```
data/pond/
  directory.json            — paper registry: title, author, year, per-attribute units
  preprocessing.py          — builds ground_truth.csv and ground_truth_ten.csv
  create_probe_dataset.py   — builds probe_dataset.json for judge calibration
  ground_truth.csv          — all registered, non-excluded papers (long format)
  ground_truth_ten.csv      — top-10 paper development subset
  probe_dataset.json        — synthetic valid/invalid records for probe analysis
  raw_data/                 — pond_data_corrected.csv (long) and pond_data_corrected_.csv (wide)
  ocr_output_raw/           — plain-text OCR files (one .txt per paper)
  pdfs/                     — source PDFs
```

---

## Document IDs

Every paper is identified by a single string called its **document ID** (or paper code).
This same identifier is used consistently across all files:

| Location | Role |
|---|---|
| Key in `directory.json` | Primary definition of the paper |
| Stem of `.txt` file in `ocr_output_raw/` | Must match the directory.json key exactly |
| `document_id` column in `ground_truth.csv` and `ground_truth_ten.csv` | Links measurements to papers |
| `document_id` field in `probe_dataset.json` | Identifies which paper a probe record comes from |
| Values in `paper_subset`, `paper_exclude` in `DatasetConfig` | Control which papers are processed |

Example: `"classification_trees"` is the document ID for the paper by Peretyatko et al. (2011).
Its OCR text lives at `ocr_output_raw/classification_trees.txt` and all its ground-truth rows
have `document_id == "classification_trees"`.

---

## Ground truth preprocessing

Run from the repo root:

```bash
python data/pond/preprocessing.py
```

### Pipeline

```
raw_data/pond_data_corrected_.csv   (wide format, standard units)
    ↓  filter to registered + non-excluded papers
    ↓  melt to long format
    ↓  map attribute column names → canonical attribute names
    ↓  assign document_id from directory.json
    ↓  apply per-paper unit conversions from directory.json
    ↓  page attribution (score all pages per document via OCR)
ground_truth.csv                    (all papers, paper-original units)
ground_truth_ten.csv                (top-10 development subset)
```

If `pond_data_corrected_.csv` does not exist (or you need to rebuild it from the hand-corrected long-format source):

```bash
python data/pond/preprocessing.py --build-corrected
```

### Output schema

`document_id, name, identifiers, location, ecosystem, date, additional_details, attribute, value, units, page_number, page_score, page_confidence`

| Column | Description |
|---|---|
| `page_number` | JSON list of candidate 1-indexed page numbers within the score margin (e.g. `[3]` or `[3, 4]`); `NaN` if OCR file is missing |
| `page_score` | Weighted similarity score in [0, 1] for the top attributed page |
| `page_confidence` | `"high"`, `"medium"`, or `"ambiguous"` |

Attribution is implemented in `src/scholarlm/utils/page_attribution.py`.
Pond attribution scores all pages (no table pre-filtering).

### Attributes and standard units

| Attribute | Standard unit | Notes |
|---|---|---|
| `max_depth` | m | Maximum physical water depth |
| `surface_area` | m² | Water body surface area |
| `vegetation_cover` | percent | Fraction of surface covered by macrophytes |
| `ph` | — (dimensionless) | Water pH |
| `tn` | µg/L | Total nitrogen |
| `tp` | µg/L | Total phosphorus |
| `chla` | µg/L | Chlorophyll-a |

### Unit conversion

Papers report measurements in their own units (e.g. ha, km², mg/L). `directory.json`
records the original units per paper per attribute under a `"units"` key. `preprocessing.py`
reads these and applies multiplicative conversions so that each value in `ground_truth.csv`
is in the paper's original units.

Conversion formula: `paper_value = standard_value × factor`

Supported units and factors (defined in `_UNIT_CONVERSION` in `preprocessing.py`):

| Unit | Attribute | Factor from standard |
|---|---|---|
| `ha` | surface_area | 1×10⁻⁴ |
| `km²` | surface_area | 1×10⁻⁶ |
| `acres` | surface_area | 1/4046.856 |
| `x10^-2 km²` / `10^-2 x km²` | surface_area | 1×10⁻⁴ (= ha) |
| `x10^-6 m²` | surface_area | 1×10⁻⁶ (= km²; paper column labelled ×10⁶ m²) |
| `cm` | max_depth | 100 |
| `ft` | max_depth | 3.28084 |
| `mg/L` | tn, tp, chla | 1×10⁻³ |
| `mg/m³` | tn, tp, chla | 1.0 (1 µg/L ≡ 1 mg/m³) |
| `µg/cm²` | chla | 1.0 (area-based unit; no volume conversion) |
| `fraction` | vegetation_cover | 0.01 |

For attributes where the paper's unit matches the standard, no conversion is applied.

### directory.json structure

Each entry maps a paper code (document ID) to its metadata:

```json
"classification_trees": {
    "title": "classification trees as a tool for predicting cyanobacterial blooms",
    "author": "peretyatko et al.",
    "year": 2011,
    "units": {
        "chla": "µg/L",
        "max_depth": "m",
        "ph": null,
        "tp": "mg/L"
    }
}
```

The `"units"` field is present only for papers that have ground-truth data.
`ph` is always `null` (dimensionless). Papers without any target attributes in tables
have no `"units"` field.

### Excluded papers

Two categories of papers are excluded from `ground_truth.csv` (and from extraction):

**No target attributes in tables** — these papers are in the registry but were confirmed to have
none of the seven target attributes reported in tabular form:

`application_and_transferability`, `comprehensive_approach`, `improved_method`,
`livin'_on_the_edge`, `predictions_of_climate_change`, `satellite_radar`,
`spatio-temporal_surface`, `species_numbers`, `temporary_wetlands`,
`the_effects_of_ambient`, `the_influence_of_an_in-network`

**Data quality exclusions** — present in the registry but excluded due to data source issues:

| Paper | Reason |
|---|---|
| `bacterioplankton` | Values digitised from figures, not tables |
| `summer_assessment` | Data only in supplemental text |

All 13 excluded papers are listed in `_EXCLUDED_PAPERS` in `experiments/configs/pond.py`
and are automatically skipped by the extraction pipeline via `DatasetConfig.paper_exclude`.

---

## Probe dataset

Run from the repo root:

```bash
python data/pond/create_probe_dataset.py [--seed N]
```

The probe dataset is used to calibrate and evaluate the judge model's ability to
distinguish valid from invalid extraction records. It is built from `ground_truth.csv`
and the OCR text files in `ocr_output_raw/`.

Two files are produced: `probe_dataset.json` (train, ~50% of papers) and
`probe_dataset_test.json` (test, remaining ~50% of papers). Both use identical
generation logic; the split is at the paper level so no paper appears in both.

### Structure

Approximately 50% of ground-truth records (sampled whole papers at a time) are
labelled `"valid"`. Two invalid counterparts are generated per valid record, giving
a **2:1 invalid:valid ratio**. Invalid records are spread across three subsets.

### Subset 1 — Swap invalids (one per valid record)

One field is replaced with a value drawn from a different ground-truth record.
Same-paper candidates are preferred; falls back to cross-paper if none exist.
One modification type is chosen at random from whichever are applicable:

| Type | What changes | Distinctness guarantee |
|---|---|---|
| `change_value` | Value replaced from a record with a different entity or attribute | New value ≠ original value |
| `change_attribute` | Attribute name replaced; value and units unchanged | New attribute ≠ original attribute |
| `change_entity` | All `judge_entity_fields` (name, ecosystem) replaced from a record with a different entity key | Entity key guaranteed different |
| `change_units` | Units replaced with a different canonical unit for the same attribute (from the attribute catalogue); value unchanged | New unit ≠ original unit (case-insensitive) |

`ph` records produce no `change_units` candidates (the attribute has no canonical units).

### Subset 2 — Noise invalids (~half the valid set)

One modification type is chosen uniformly when both are applicable:

| Type | What changes | Distinctness guarantee |
|---|---|---|
| `noise_value` | Gaussian noise (σ = 30% of \|value\|) added; formatted to original decimal precision | Formatted string guaranteed ≠ original |
| `noise_entity` | Entity name replaced with a fabricated place-name (e.g. "Bluebell Pond") | Fabricated name guaranteed ≠ original name |

Only `noise_entity` is available when the value is non-numeric.

### Subset 3 — OCR-table invalids (~other half of the valid set)

| Type | What changes | Fallback |
|---|---|---|
| `table_value` | Value replaced with a numeric cell from an HTML table in the paper's own OCR file | Text numerics if no tables; noise (Subset 2) if no numerics found |

These values are plausible-looking numbers drawn from the same document — harder
for the judge to reject on distributional grounds alone.

### Output schema

Same columns as `ground_truth.csv`, plus:

| Field | Description |
|---|---|
| `page_number` | Inherited from the parent ground-truth row (JSON list of candidate page numbers) |
| `label` | `"valid"` or `"invalid"` |
| `modification_type` | One of the types above, or `null` for valid records |
| `gt_row_index` | Index into `ground_truth.csv` for the base record |
| `donor_gt_row_index` | Index of the donor record for swap invalids; `null` otherwise |
| `measurement_id` | Sequential integer ID across all records |
