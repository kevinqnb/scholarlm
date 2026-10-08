---
id: 2026-10-08-platt-scaling-v2-01
kind: build
config: analysis/analysis-configs/2026-10-08-platt-scaling-v2-gemma27b-qwen-2.5-7b-intercept-fit-01.yaml
---

## Session 2026-10-08

### Prompts
- "Please help me simplify @analysis/platt_scaling.py and create an `analysis/platt_scaling_v2.py` script. The idea for this is that instead of trying to do nested confidence interval-ing / bootstrapping, we just want to measure the effect of what happens when the training dataset is resampled. To do that: bootstrap the training documents and uniformly select n_fit data points to fit with -- depending on the specific setting being worked. For a single sample, compute smoothECE scores and record. The final artifact should be a curve with varying n_fit, and where smoothECE scores have been captured by the mean and 95% confidence intervals over training samples. Do 2000 training resamples."
- "Before doing so I just want to check, what other analysis scripts are you relying on for things like document level bootstrapping or computation of smooth ece?"
- "Can we stop relying on the nested bootstrap and v1 platt scaling scripts? There's a new doc bootstrap script, would it satisfy what we need here?"
- "doc_bootstrap.py is now committed. Reuse pool resampling and scoring from it. Where is the row sampling and single class insurance for v2 currently implemented?"
- "Helpers for row sampling and single-class handling should live in doc_bootstrap.py. The relplot call is good."
- "How are matching included into the labels? Where does the fuzzy threshold get used?"
- "go ahead and submit the three full runs"

### Implemented
`analysis/platt_scaling_v2.py` replaces `analysis/platt_scaling.py`'s nested bootstrap with a sweep over training resamples. For each n in `platt_ns` it draws `n_train_resamples` fit samples. Each sample resamples the pool documents, then draws n rows, skipping single-class draws. Each sample's recalibration map is scored by smECE on the fixed real test rows. The output is the mean and a 95% percentile band over resamples, so the band covers training-resample variability only. The sampling helpers (`fit_resample_rng`, `resampled_fit_sample`, `two_class_fit_samples`) are new in `analysis/doc_bootstrap.py`, and smECE comes from doc_bootstrap's relplot call. v2 no longer imports `nested_bootstrap.py` or `platt_scaling.py`, and both are unchanged. The configs are `analysis/analysis-configs/2026-10-08-platt-scaling-v2-{smoke,intercept-fit-tiny}-01.yaml` and the three full `2026-10-08-platt-scaling-v2-gemma27b-qwen-2.5-7b-{intercept-fit,platt-fit,prior-shift}-01.yaml`. The loader is `load_platt_sweep_v2_config`. Tests are in `tests/test_platt_scaling_v2.py` and `tests/test_doc_bootstrap.py`.

### Commits
e57f35c platt_scaling_v2: Platt-n sweep over training resamples, on doc_bootstrap

## Session 2026-10-08 (follow-up: full-run failure)

### Prompts
- "All jobs failed"
- "Skip and redraw."
- "Commit and add this as a note to the previous devlog. Then resubmit."

### Implemented
All three full runs failed while drawing fit samples. For supermat at n=1000, a document resample that missed the large documents held fewer than n rows. That happened in 7 of the first 2000 draws, and the 100-draw tiny run never hit it. `analysis/doc_bootstrap.py`'s `resampled_fit_sample` now returns None for such a draw. `two_class_fit_samples` skips it, as it skips single-class draws, and reports per-kind skip counts. `analysis/platt_scaling_v2.py` writes those counts as `Short-pool skips` / `Single-class skips`. Draws with no skips are unchanged (the tiny run's per-resample output is byte-identical). The three full configs were resubmitted unchanged.

### Commits
4f94fa4 doc_bootstrap: skip fit draws whose document resample has fewer than n rows

## Session 2026-10-08 (follow-up: band width, n = 0 baseline)

### Prompts
- "The confidence intervals in these plots are quite large. What happened? Would more samples help?"
- "Ok here is what I want to do. Let's retry this, but with the number of training examples as 0,50,100,250,500,100. Note the 0: we need a baseline case." (the trailing 100 was read as 1000)
- "Commit it and add a follow-up to the devlog. Submit the full jobs."

### Implemented
The band width was diagnosed with no code change. More resamples would not narrow it: it is a percentile range, already estimated to about 0.003 at 2000 resamples. At large n it is set by document resampling of a pool with only about 10 effective documents. `analysis/platt_scaling_v2.py` now accepts n = 0 in `platt_ns` as a no-recalibration baseline: the raw probe / NTP-calibrator scores on the same test rows, one value per cell with a zero-width interval, drawn on a symlog x axis. `load_platt_sweep_v2_config` now accepts non-negative `platt_ns`. The new configs are `analysis/analysis-configs/2026-10-08-platt-scaling-v2-gemma27b-qwen-2.5-7b-{intercept-fit,platt-fit,prior-shift}-02.yaml` (platt_ns [0, 50, 100, 250, 500, 1000]), plus `...-baseline-smoke-01.yaml` and `...-intercept-fit-tiny-02.yaml`. In the tiny run, the n > 0 per-resample output is byte-identical to the earlier tiny run. The three `-02` runs were submitted.

### Commits
1e65e7e platt_scaling_v2: n = 0 no-recalibration baseline; -02 sweep configs
