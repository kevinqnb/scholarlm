# data/nfix — Dinitrogen Fixation Dataset

Ground truth, probe dataset, and preprocessing for the nfix (aquatic dinitrogen fixation) dataset.
Papers report N₂ fixation rates measured in marine, estuarine, and freshwater environments.

## Directory structure

```
data/nfix/
  directory.json            — paper registry: title, author, year
  preprocessing.py          — builds ground_truth.csv and ground_truth_ten.csv
  create_probe_dataset.py   — builds probe_dataset.json for judge calibration
  ground_truth.csv          — all registered papers (long format)
  ground_truth_ten.csv      — top-10 paper development subset
  probe_dataset.json        — synthetic valid/invalid records for probe analysis
  raw_data/                 — source CSV files
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

---

## Ground truth preprocessing

Run from the repo root:

```bash
python data/nfix/preprocessing.py
```

### Output schema

`document_id, name, identifiers, location, site_type, date, nfix_method, substrate_type, sample_depth, additional_details, attribute, value, units`

### Attributes

All records share the generic attribute name `nfix_rate`. The specific sub-type is
inferred from the units string at probe-dataset creation time:

| Sub-type | Units pattern | Example units |
|---|---|---|
| `nfix_rate_mass` | contains `g⁻¹` or `kg⁻¹` | nmol g⁻¹ h⁻¹ |
| `nfix_rate_areal` | contains `m⁻²` or `cm⁻²` | µmol m⁻² d⁻¹ |
| `nfix_rate_volumetric` | contains `L⁻¹`, `m⁻³`, `mL⁻¹`, or `cm⁻³` | nmol L⁻¹ h⁻¹ |
| `nfix_rate` | no match | fallback |

---

## Probe dataset

Run from the repo root:

```bash
python data/nfix/create_probe_dataset.py [--seed N]
```

The probe dataset calibrates and evaluates the judge model's ability to distinguish
valid from invalid extraction records. It is built from `ground_truth.csv` and the
OCR text files in `ocr_output_raw/`.

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
| `change_entity` | All `judge_entity_fields` (name, site_type) replaced from a record with a different entity key | Entity key guaranteed different |
| `change_units` | Units replaced with a different canonical unit for the same attribute (from the attribute catalogue); value unchanged | New unit ≠ original unit (case-insensitive) |

`change_attribute` is not applicable for nfix because all records share the same
generic attribute type (`nfix_rate`).

### Subset 2 — Noise invalids (~half the valid set)

One modification type is chosen uniformly when both are applicable:

| Type | What changes | Distinctness guarantee |
|---|---|---|
| `noise_value` | Gaussian noise (σ = 30% of \|value\|) added; formatted to original decimal precision | Formatted string guaranteed ≠ original |
| `noise_entity` | Entity name replaced with a fabricated site name (e.g. "Seahaven Bay") | Fabricated name guaranteed ≠ original name |

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
| `label` | `"valid"` or `"invalid"` |
| `modification_type` | One of the types above, or `null` for valid records |
| `gt_row_index` | Index into `ground_truth.csv` for the base record |
| `donor_gt_row_index` | Index of the donor record for swap invalids; `null` otherwise |
| `measurement_id` | Sequential integer ID across all records |
