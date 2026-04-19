# Examples

Jupyter notebooks for exploring and analyzing ScholarlM experiment results.

## Notebooks

| Notebook | Purpose |
|---|---|
| `pond_extraction_analysis.ipynb` | Current — probe analysis, greedy head selection, calibration for the pond dataset |
| `pond_extraction.ipynb` | Historical — uses pre-framework output paths (many cells error silently) |
| `nfix_extraction.ipynb` | Nitrogen fixation dataset exploration |
| `pond_clustering.ipynb` | Entity clustering experiments |
| `pond_elicit.ipynb` | Elicitation experiments |
| `ocr.ipynb` | OCR output inspection |

## Usage

New analysis notebooks live in `analysis/`. The `analysis` package provides experiment-specific loaders, metrics, and plots:

```python
from analysis.loaders import load_extraction, load_combined_judgements
from analysis.metrics import recovery_rate, hallucination_rate
from analysis.plots import recovery_bar

records = load_extraction("pond", "gemma-3-27b", "2026_04_01")
judgements = load_combined_judgements("pond", "gemma-3-27b", "2026_04_01")
```

## Notes on historical notebooks

`pond_extraction.ipynb` was written before the current runner framework and
uses hard-coded paths like `../data/experiments/2026_03_04/pond_entities.json`.
These cells error silently on a fresh checkout. Use `pond_extraction_analysis.ipynb`
for current analysis instead.
