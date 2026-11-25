---
id: 2026-11-25-calibration-baseline-01
kind: build
config: analysis/analysis-configs/calibration-baseline/2026-11-25-calibration-baseline-gemma27b-01.yaml
---

## Session 2026-11-25

### Prompts

> Please help me set up a `analysis/calibration_baseline.py` script for analyzing recent baseline methods for assigning confidence probabilities to extracted data (`experiments/run_judge_baselines.py`). This should largely follow `analysis/calibration.py` to compute calibration error (with diagrams), as well as BSS and AUROC scores to evaluate predictive performance. [Nine baseline runs listed by experiment config: verbalized and sampling with llama-8b and ntp with llama-70b, over the gemma27b extractions of pond, nfix and supermat.] In addition, consider one more baseline for each dataset: probability scores according to a Laplace distribution with the location set to the median and the scale fit as MAD / ln 2.

> (On the Laplace baseline, when asked how to handle mixed units and rows with no numeric value) That's a good point. I forgot about this. Drop this baseline entirely.

> Submit the full run.

### Implemented

`analysis/calibration_baseline.py` scores the verbalized, sampling and ntp judge-baseline runs on the stored real test rows and labels of a finished calibration run, so every method is evaluated on the same rows as the probe. It writes a metrics CSV (calibration errors, Brier skill score, AUROC, document-bootstrap CIs) and one reliability diagram per method; scores are not recalibrated. Added the `calibration-baseline` analysis type with its config loader and run resolver, and `analysis/analysis-configs/calibration-baseline/2026-11-25-calibration-baseline-gemma27b-01.yaml`. Unit tests, a two-resample smoke run, a shuffled-label control, a determinism check and a known-answer check (stored NTP scores reproduce the calibration run's metrics and CIs) passed; shared metric code is unchanged. The Laplace baseline was dropped before implementation. The full run was submitted and had not started when this was written.

### Commits

- `66e0e7b` Add calibration_baseline: calibration, BSS and AUROC of judge confidence baselines
