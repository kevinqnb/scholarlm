---
id: 2026-10-07-platt-doc-balanced-sampling-01
kind: build
---

## Session 2026-10-07

### Prompts
- Targeted update to `calibration_updated_v3.py` and `platt_scaling.py`: instead of choosing Platt training measurements uniformly at random, spread them evenly across documents — random document ordering, cycle through it adding one random not-yet-chosen point per visit, stop at the desired number of samples.
- "apply the same change to calibration_validated.py"

### Implemented
Added `document_balanced_order` to `analysis/calibration_ids.py`: it cycles through a random document order, taking one random unchosen row per visit and skipping exhausted documents. `calibration_updated_v3.py`, `calibration_validated.py` and `platt_scaling.py` now draw their Platt samples from its prefixes, so sweep samples still nest across n. `tests/test_platt_scaling.py` covers nesting, determinism and document spread on a small fixture. No configs changed. Only unit tests have been run, and prior Platt-scaled numbers from these scripts are invalidated.

### Commits
7903ea7 Platt sample: spread evenly across documents instead of uniform over rows
