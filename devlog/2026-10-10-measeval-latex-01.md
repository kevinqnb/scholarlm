---
id: 2026-10-10-measeval-latex-01
kind: build
config: analysis/analysis-configs/measeval/2026-10-09-measeval-official-eval-latex-01.yaml
---

## Session 2026-10-10

### Prompts
- "Is there any code available for turning the saved measeval CSV into latex?"
- "yes, write it with F only per type"

### Implemented
New `analysis/measeval_latex.py` formats the `measeval_evaluation.py` CSV as a LaTeX table of the official scorer's F-measure per annotation type (Quantity, Unit, MeasuredEntity), bolding the best value per column. It mirrors `recovery_validity_latex.py`: rows, labels and caption come from `params.measeval_latex`, and the CSV must be current, non-dev and complete for the named evaluation config. The spec `analysis/analysis-configs/measeval/2026-10-09-measeval-official-eval-latex-01.yaml` reads `2026-10-09-measeval-official-eval-01`. Rung-1 tests are in `tests/test_measeval_latex.py`.

### Commits
- `38d6d1d` measeval_latex: format the official-eval CSV as a LaTeX table
