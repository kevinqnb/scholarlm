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
  plots.py            — standard plot functions (return matplotlib.Figure)
  cross_dataset.py    — cross-dataset probe transferability

  extraction_analysis.ipynb    — recovery + hallucination for one extraction run
  ablation_comparison.ipynb    — compare baseline vs. ablation variants
  probe_analysis.ipynb         — probe accuracy, calibration, greedy head selection
  cross_dataset_probe.ipynb    — cross-dataset probe accuracy matrix
```

## Notebook path setup

```python
import sys
from pathlib import Path

REPO_ROOT = Path.cwd().parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))
```

## Key imports

```python
from analysis.loaders import load_extraction, load_combined_judgements, load_ground_truth
from analysis.metrics import recovery_rate, hallucination_rate
from analysis.plots import recovery_bar
from scholarlm.utils.probe import build_feature_matrix, train_probe, eval_probe
from scholarlm.utils.calibration import reliability_diagram_data
from scholarlm.utils.unit_conversion import apply_unit_conversion
```

## Notes

- For datasets with mixed units (e.g. pond), apply `apply_unit_conversion` before calling
  `recovery_rate` to convert extracted values to a standard unit before ground-truth matching.
- For datasets with a single unit per attribute (e.g. nfix), match on `"value"` directly.
- `probe.py` and `calibration.py` live in `src/scholarlm/utils/` as part of the installable
  library; they have no dependency on experiment paths.
