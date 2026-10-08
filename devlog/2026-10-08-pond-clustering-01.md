---
id: 2026-10-08-pond-clustering-01
kind: build
config: analysis/analysis-configs/2026-10-08-pond-clustering-01.yaml
---

## Session 2026-10-08

### Prompts
- "Please help me update the `analysis/clustering.py` script to fit the newly adopted conventions of the `analysis/` directory. Specifically, it should: 1. Load from a config file specifying the extraction id to pull from, as well as any other artifacts or parameters we might need: e.g. judge files, THRESHOLD, N_RUNS, N_RANDOM_SAMPLES? 2. Read from that config file and wire up to be submitted as a job through `analysis/submit.sh` 3. Save results and figures to `analysis/results/clustering/`"
- (Design answers, summarized) Use the stored recalibrated calibration predictions rather than recomputing probe scores; make the probe's training dataset a config key set to pond; drop the never-plotted synthetic arm; use the deduplicated rows, with a cell's value the median and its confidence the mean of its rows.
- "Add a shuffled control"
- "Does this script agree with the structure of the pond dataset extractions as defined in @experiments/dataset-configs/pond.py ? For example, we currently group entities using location, but location is no longer being extracted for entities."
- "Let's /devlog and get the new clustering work commited, and the runs for it submitted."

### Implemented
`analysis/clustering.py` now reads one analysis config (`params.clustering`, every key required) instead of hard-coded dates and constants. It takes rows and probe/NTP confidences from `meta_updated.load_data` and a calibration config's stored predictions, cross-checking the declared extraction / judge-combine / judge ids against it. Ground-truth entities are keyed on the pond `EntitySchema` fields (no `location`). Besides the ntp / probe / random arms it runs a shuffled-confidence permutation control, asserts known answers (GT self-refit distance 0; all arms identical at gamma 0), and writes CSV/NPZ/JSON and figures under `analysis/results/clustering/<id>/`. Configs: `analysis/analysis-configs/2026-10-08-pond-clustering-01.yaml` and `analysis/analysis-configs/2026-10-08-pond-clustering-smoke-01.yaml`; tests in `tests/test_clustering.py`.

### Commits
`a3cf434` clustering: config-driven run on stored recalibrated scores, with shuffled control
