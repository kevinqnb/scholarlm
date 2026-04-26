# scholarlm

Python package for extracting entity-attribute-value triplets from scientific PDFs.

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

1. **Entity extraction** — identify named entities (e.g. ponds, nitrogen fixation sites)
2. **Attribute extraction** — identify which attributes (e.g. pH, temperature) appear per document
3. **Entity provenance** — locate which pages/tables each entity appears on
4. **Attribute provenance** — locate which pages/tables each attribute appears on
5. **Event resolution** — for each (entity, attribute, page) intersection, enumerate distinct measurement events
6. **Value extraction** — extract the numeric value and units for each (entity, attribute, event) triple
7. **Finalization** — standardize, deduplicate, assign measurement IDs

Each step writes a JSON checkpoint; `run_extraction.py --resume` skips steps whose output already exists.

## Configuration

`DatasetConfig` fields (all required unless noted):

| Field | Type | Notes |
|---|---|---|
| `name` | `str` | Short identifier used in output paths |
| `data_dir` | `str` | Root data directory |
| `metadata_file` | `str` | Paper directory JSON |
| `entity_schema` | `type[BaseModel]` | Pydantic schema for entity extraction |
| `entity_identification_prompt` | `str` | System prompt for entity step |
| `entity_type_description` | `str` | One-line description for judge prompts |
| `attribute_info_dict` | `dict` | `{attr: {description, units}}` |
| `paper_subset` | `list[str] \| None` | Restrict to these paper codes |
| `paper_filter` | `Callable \| None` | Predicate on paper metadata |
| `paper_exclude` | `list[str] \| None` | Paper codes to unconditionally skip (applied after `paper_filter`, before `paper_subset`) |
| `measurement_event_schema` | `type[BaseModel] \| None` | Enables event resolution step |
| `measurement_event_prompt` | `str \| None` | Instructions for event resolution |
| `direct_extraction_schema` | `type[BaseModel] \| None` | Enables Ablation 1 |
| `direct_extraction_prompt` | `str \| None` | Prompt for Ablation 1 |
| `ablation3_entity_schema` | `type[BaseModel] \| None` | Enables Ablation 3 |
| `ablation3_entity_identification_prompt` | `str \| None` | Prompt for Ablation 3 |
| `ground_truth_file` | `str \| None` | Path to manual ground-truth CSV/JSON |
| `unit_conversion_table` | `dict[str, dict[str, float]]` | Per-attribute `{unit: multiplier}` map; see `scholarlm.utils.unit_conversion` |
