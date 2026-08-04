# data/supermat — Superconductivity Dataset

Ground truth and probe dataset for the supermat (superconductivity) dataset, adapted from
SuperMat (Foppiano et al. 2021, `reference.pdf`): a linked annotated corpus of 142
superconductivity papers with `<material>`, `<class>`, `<tc>`, `<tcValue>`, `<pressure>`, and
`<me_method>` annotations. Papers report superconducting critical temperature (Tc) measurements
for specific materials, along with the applied pressure and measurement method used.

The extraction pipeline maps SuperMat's tags onto this library's entity/attribute/event model:
the **material** is the entity, **Tc** is the (only) attribute, and **pressure**/**measurement
method** are the event fields distinguishing different Tc measurements of the same material.

## Directory structure

```
data/supermat/
  pdfs/                     - directory containing all PDF documents (not shared due to licensing)
  ocr_output_raw/           - plain-text OCR files (one .txt per paper -- not shared)
  raw_data.csv              - source CSV export (material, tcValue, pressure, me_method, filename)
  reference.pdf             - SuperMat reference paper describing the annotation scheme
  build_directory.py        - builds directory.json from SuperMat's own biblio metadata
  directory.json            - paper registry: title, author, year (None where unavailable)
  preprocessing.py          - builds ground_truth.json and ground_truth_ten.json
  create_probe_dataset.py   - builds probe_dataset.json for judge calibration
  ground_truth.json         - ground truth dataset
  ground_truth_ten.json     - top-10 paper (by row count) development subset
  probe_dataset.json        - synthetic valid/invalid records (train split)
  probe_dataset_test.json   - synthetic valid/invalid records (test split)
```

## Document reconciliation note

`raw_data.csv` references 147 filename codes, but only 142 PDFs are available locally. Of the 5
codes with no local PDF, 3 (`JPS0731655-CC`, `L095167004-CC`, `SSC1310125-CC`) turned out to be
duplicate bibliographic registrations of a paper already present under a different code — verified
by matching DOIs in SuperMat's own biblio metadata and confirming the PDF content — so their rows
are merged into the real-PDF document's `document_id` during preprocessing. Only 2 codes
(`PHC2640145-CC`, `yamaguchi2014ac`) are genuinely missing and their rows are dropped. See the
docstring in `preprocessing.py` for details.

## Ground truth value policy

Ground truth `value` must be a single, unambiguous reported number, matching pond/nfix's
convention and the judge's own rule against accepting a value that's merely an inferred range
endpoint. Rows whose `tcValue` was reported as a range (`"7-8 K"`), an approximation (`"~30 K"`),
or a bound (`"< 10 K"`, `"> 30 K"`, `"up to 33 K"`) are dropped during preprocessing rather than
collapsed to a synthesized midpoint or endpoint. A handful of papers whose only reported Tc values
were qualified in this way end up with zero ground-truth rows.

---

## Build the paper registry

```bash
python data/supermat/build_directory.py
```

## Preprocess ground truth data

```bash
python data/supermat/preprocessing.py
```

Page attribution (page_number/page_score/page_confidence) requires OCR text first:

```bash
python experiments/run_ocr.py --dataset supermat
python data/supermat/preprocessing.py
```

## Build the probe dataset

Also requires OCR text (`ocr_output_raw/`) to be present:

```bash
python data/supermat/create_probe_dataset.py
```
