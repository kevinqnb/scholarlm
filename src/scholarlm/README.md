# scholarlm

Python package for extracting measurements from scientific PDFs.

## Modules

| Module | Purpose |
|---|---|
| `measurementlm.py` | Core 7-step extraction pipeline |
| `measurementlm_ablation{1–6}.py` | Ablation variants; each overrides one pipeline step |
| `config.py` | `DatasetConfig` and `ModelConfig` dataclasses |
| `judgementlm.py` | NNsight-based judge (collects attention activations) |
| `instruction_prompts.py` | All LLM system prompts |
| `utils/` | Shared utilities: data matching, file I/O, probe, calibration, unit conversion |

## Extraction pipeline

`MeasurementLM.fit()` runs these steps in order:

0. **Table Cleaning** - Optional, cleans tables by creating an index column and melting hierarchical columns. Note: requires a VLM model to run!
1. **Entity extraction** — identify named entities (e.g. ponds, nitrogen fixation sites)
2. **Attribute extraction** — identify which attributes (e.g. pH, temperature) appear per document
3. **Entity provenance** — locate which pages/tables each entity appears on
4. **Attribute provenance** — locate which pages/tables each attribute appears on
5. **Event resolution** — for each (entity, attribute, page) intersection, enumerate distinct measurement events
6. **Value extraction** — extract the numeric value and units for each (entity, attribute, event) triple
7. **Finalization** — standardize, deduplicate, assign measurement IDs

Each step writes a JSON checkpoint; `run_extraction.py --resume` skips steps whose output already exists.
