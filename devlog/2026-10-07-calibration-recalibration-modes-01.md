---
id: 2026-10-07-calibration-recalibration-modes-01
kind: build
---

## Session 2026-10-07

### Prompts
- Targeted update to `calibration_updated_v3.py` and `calibration_validated.py`: add an option to replace Platt scaling with a simpler intercept adjustment, i.e. only adjust for the prevalence of the small training sample rather than also fitting a slope.
- "Is this the right way to do it? Please reference @analysis/calibration_updated.py in which we used to do something like this. Let's dig into the math before writing the code."
- "Both, A and B as separate modes. So in total we have (a) prior-shift, (b) intercept-fit, (c) platt-fit options."
- "What takes so long from this script?" (the full script exceeds the shell's compute limit)
- "Go ahead and work on the loading fix. Double check that the fix you make loads activations identically on a small set of examples"
- Generate three new configs following `2026-10-04-calibration-v3-gemma27b-qwen-2.5-7b-v3-01`, one per option, and submit them as analysis jobs.

### Implemented
Calibration v3 and validated configs now require a `recalibration` key: `prior_shift` (label-shift correction from the scorer's synthetic training prevalence to the sample label rate), `intercept_fit` (slope fixed at 1, intercept by maximum likelihood) or `platt_fit` (unchanged). The fitters live in `scholarlm.utils.calibration` (`fit_intercept`, `fit_prior_shift`, `fit_recalibration`), and the method is recorded in predictions, `platt_fits.csv` and the metrics CSVs; `calibration_latex.py` refuses non-Platt runs. The existing configs `2026-10-04-calibration-v3-gemma27b-qwen-2.5-7b-v3-01`, `2026-10-05-calibration-v3-gemma27b-qwen-2.5-7b-noplatt-01` and `2026-10-06-calibration-validated-gemma27b-qwen-2.5-7b-v3-01` gained an explicit `recalibration: platt_fit`, which leaves their behaviour unchanged. A new `analysis/head_activations.py` reads each activation row once per run instead of once per head per probe; its output was checked bitwise-identical to the old reader in unit tests and on real rows. New analysis configs `2026-10-07-calibration-v3-gemma27b-qwen-2.5-7b-{platt-fit,intercept-fit,prior-shift}-01` were submitted; unit tests pass, and smoke / tiny end-to-end rungs were skipped because the script exceeds the shell's compute limit.

### Commits
04299da calibration v3/validated: recalibration modes + single-read head activations
0ead0b1 configs: v3 calibration runs for platt_fit / intercept_fit / prior_shift
