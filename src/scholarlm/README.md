# scholarlm

Python package for extracting entity-attribute-value measurements from scientific PDFs, plus judge / interpretability tooling to evaluate them. Runners live in `experiments/`; this directory is the library they import.

## Layout

### Preprocessing

| File | Purpose |
|---|---|
| `documentlm.py` | `DocumentLM`: PDF → page/table-tagged text via a vLLM-served OCR VLM (olmOCR-2, chandra-ocr-2) |
| `table_cleaner.py` | `TableCleaner`: standalone table-cleaning pass over OCR text (run before extraction; needs a VLM) |

### Extraction

| File | Purpose |
|---|---|
| `measurementlm.py` | `MeasurementLM`: the core 7-step pipeline (below); also holds the shared async API plumbing and response schemas |
| `measurementlmv2.py` | `MeasurementLMv2`: quantity-first variant — collect and dedup quantities per (page, attribute), then attribute them to entities/events |
| `measurementlm_ablation{1–7}.py` | Ablations, each a `MeasurementLM` subclass: 1 single-pass direct extraction, 2 combined entity+attribute step, 3 full-document pair provenance, 4 full-document value extraction, 5 direct table values, 6 no explanations in responses, 7 no standardize/parse/dedup |
| `measurementlm_{chatextract,gliner,langextract,nuextract,nuextract3}.py` | Baselines reimplemented as `MeasurementLM` subclasses: ChatExtract, GLiNER2, LangExtract, NuExtract-2.0-8B, NuExtract3 |
| `instruction_prompts.py` | All LLM instruction prompts for extraction |
| `config.py` | `DatasetConfig` and `ModelConfig` dataclasses |

### Judging and interpretability

| File | Purpose |
|---|---|
| `judgementlm.py` | `JudgementLM`: NNsight-based true/false judge that collects attention activations |
| `jacobianlenslm.py` | `JacobianLensLM`: j-scores from a pretrained per-layer Jacobian lens |
| `representationlm.py` | `RepresentationLM`: key-term hidden states from a base LLM at chosen layers |
| `attribution.py` | Input-token attribution for judge decisions (contrastive gradient, probe-based) |

### `utils/`

| File | Purpose |
|---|---|
| `data.py` | `match_datasets`: strict/fuzzy matching of extracted rows to ground truth |
| `normalization.py`, `parsing.py`, `unit_conversion.py` | Value/unit canonicalisation, quantity-shape parsing, unit conversion to ground-truth units |
| `deduplication.py` | Strict + fuzzy dedup of extracted records (same pairwise rule as `match_datasets`) |
| `pdf.py`, `chandra_format.py`, `references.py`, `tables.py`, `sentences.py`, `page_attribution.py` | PDF rendering and OCR post-processing (chandra formatting, reference stripping, table row names), sentence splitting (ChatExtract), page lookup for ground-truth rows |
| `judge_prompts.py` | Judge prompt builders and document loading shared by all judge runners |
| `probe.py`, `calibration.py` | Grouped train/calibration/test splits for probes; Platt / prior-shift / intercept recalibration, ECE, reliability data |
| `probe_augment.py` | Synthetic probe-dataset augmentation (used by `data/*/create_probe_dataset.py`) |
| `fs.py` | Directory listing helper |

## Extraction pipeline

`MeasurementLM.fit()` runs these steps in order:

1. **Entities** — identify named entities (e.g. ponds, nitrogen fixation sites)
2. **Attributes** — detect which attributes (e.g. pH, temperature) appear per document
3. **Entity provenance** — which pages/tables each entity appears on
4. **Attribute provenance** — which pages/tables each attribute appears on
5. **Events** — for each (entity, attribute, page) intersection, enumerate distinct measurement events
6. **Values** — extract value and units per (entity, attribute, event), from text or tables
7. **Final** — standardize units, parse quantity qualifiers, deduplicate, assign measurement IDs

Each step writes a JSON checkpoint; `run_extraction.py --resume` skips steps whose output already exists. Table cleaning is not part of `fit()`; run `TableCleaner` first and point extraction's `params.ocr_dir` at its output.
