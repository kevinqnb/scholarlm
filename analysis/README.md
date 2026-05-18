# analysis/

Experiment analysis code and notebooks. Lives outside `src/scholarlm/` so it can freely
import both the core library and `experiments/paths.py` without making the library depend
on experiment scaffolding.

## Structure

```
analysis/
  __init__.py
  loaders.py          — load experiment outputs by (dataset, model, date)
  metrics.py          — recovery rate, hallucination rate, per-paper summaries
  ablation.py         - computes recovery and validity rates for extracted data (over ablations)
  synthetic_probe_train.py - trains and pre-calibrates the synthetic probe models
  calibration.py      - evaluates the trained probe / NTP models on test data
  clustering.py       - trains and evaluates a downstream clustering model
```