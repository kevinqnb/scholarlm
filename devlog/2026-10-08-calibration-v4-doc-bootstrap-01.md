---
id: 2026-10-08-calibration-v4-doc-bootstrap-01
kind: build
config: analysis/analysis-configs/2026-10-08-calibration-v4-intercept-fit-sample-smoke-01.yaml
---

## Session 2026-10-08

### Prompts
- "Looking at the calibraiton diagrams for the results (now finished) I notice fairly wide confidence intervals. What is feeding into these? How many bootstrap resamples did we do?"
- "Where can i reference where this is computed?"
- "Why do we still rely on nested bootstrap if we're no longer doing a nested bootstrap? Is there a simpler way to do a document level bootstrap of the eval set? Preferably with library functions to the relplot (smoothECE library)? There should no longer be stuff like bandwidth estimation. Can we try to separate entirely from what was previously done for v3, and just make this as simple and robust as possible?"
- Chose: keep the binned ECEs alongside SmECE; "Keep our drawing, keep support maksing."
- "Good, yes submit the jobs"

### Implemented
`analysis/calibration_updated_v4.py` no longer uses `analysis/nested_bootstrap.py`. The new `analysis/doc_bootstrap.py` draws document-level resamples of the evaluation set and, for the full set and each resample, calls `relplot.prepare_rel_diagram` for the SmECE and the reliability curve, with relplot's own row-level bootstraps switched off. The binned ECEs are computed on the same resamples, and intervals and bands are percentiles over resamples. Our code chooses no bandwidth. Point estimates and metric intervals reproduce the previous v4 output exactly; only the curve bands change, since each resample now uses its own relplot bandwidth. That is an evaluation change, so the earlier v4 bands are invalidated. Tests are in `tests/test_doc_bootstrap.py`; the smoke config is `analysis/analysis-configs/2026-10-08-calibration-v4-intercept-fit-sample-smoke-01.yaml`, and the four `2026-10-08-calibration-v4-gemma27b-qwen-2.5-7b-*-01.yaml` runs were resubmitted.

### Commits
cdfb40e calibration v4: document bootstrap via relplot, drop nested_bootstrap
