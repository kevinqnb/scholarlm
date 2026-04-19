# analysis/

Experiment-specific analysis code and notebooks. Lives outside `src/scholarlm/` so
it can freely import both the core library and `experiments/paths.py` without
making the library depend on experiment scaffolding.

## Structure

```
analysis/
  __init__.py
  loaders.py          — load experiment outputs by (dataset, model, date)
  metrics.py          — recovery rate, hallucination rate, per-paper summaries
  plots.py            — standard plot functions (return matplotlib.Figure)
  cross_dataset.py    — cross-dataset probe transferability

  extraction_analysis.ipynb    — recovery + hallucination for one extraction run
  ablation_comparison.ipynb    — compare baseline vs. ablation variants
  probe_analysis.ipynb         — probe accuracy, calibration, greedy head selection
  cross_dataset_probe.ipynb    — cross-dataset probe accuracy matrix
```

## Usage in notebooks

All notebooks start with the same path setup block:

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent          # analysis/../ = repo root
sys.path.insert(0, str(REPO_ROOT / 'src'))          # scholarlm importable
sys.path.insert(0, str(REPO_ROOT / 'experiments'))  # paths importable
sys.path.insert(0, str(REPO_ROOT))                  # analysis package importable
```

Then import from `analysis.*` and `scholarlm.*`:

```python
from analysis.loaders import load_extraction, load_combined_judgements
from analysis.metrics import recovery_rate, hallucination_rate
from analysis.plots import recovery_bar
from scholarlm.utils.probe import build_feature_matrix, train_probe, eval_probe
from scholarlm.utils.calibration import reliability_diagram_data
```

## Cross-dataset probe

```python
from analysis.cross_dataset import cross_dataset_probe_matrix

df = cross_dataset_probe_matrix(
    judge_model="llama-3.1-8b",
    datasets=["pond", "nfix"],
    extraction_model="gemma-3-27b",
    extraction_dates=["2026_04_01", "2026_04_01"],
)
```

## Match caching

`cached_match` wraps `match_datasets` with a disk cache keyed on the cache path
you provide. Invalidation is automatic if you overwrite the output and pass a
fresh `cache_path` that encodes the mtime.

## Probe and calibration utilities

`probe.py` and `calibration.py` are path-independent NumPy/sklearn utilities
that live in `src/scholarlm/utils/` as part of the installable library.
