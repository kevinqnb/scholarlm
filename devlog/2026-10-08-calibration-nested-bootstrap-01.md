---
id: 2026-10-08-calibration-nested-bootstrap-01
kind: build
config: analysis/analysis-configs/2026-10-07-calibration-v3-gemma27b-qwen-2.5-7b-intercept-fit-nested-tiny-01.yaml
---

## Session 2026-10-08

### Prompts
- "Please help me audit the intercept fit method in @analysis/calibration_updated_v3.py . How does it work? What dependencies does it use to compute the new intercept? Is this mathematically sound? Is it simply implemented and verifiable?"
- "Let's focus on (3) for now. I agree that we need to account for the intercept fitting training dataset as well in the confidence intervals. In practice this should look something like: do 100 random intercept training datasets, within each compute smECE and bootstrap over *documents* included in the test set -- maybe with 100 resamples. Then we can compute the mean and 95% confidence intervals across all 10,000 bootstrapped settings. This can immediately feed into the table results. However, I'm wondering, does it fit cleanly into how we are making the calibration diagrams?"
- "(2) Please tell me more about the bandwidth parameter. What does it control? How is it usually set? ... I just want to be very clear about the document level bootstrapping for the test set. Say there are X documents, then for each resample, we sample with replacement X times from this set and take all data points from each document -- however many times that document appeared. Does that make sense?"
- "regarding synthetic settings, there should be no intercept or platt fitting at all there, so there is no need to perform averaging over training sets. Synthetic does, however, have document ids and these can be document-level bootstrapped like the real data."
- Chose: binned ECE variants on the same nested scheme; threshold metrics averaged over fit samples; apply to both calibration scripts; new config keys required everywhere.
- "go ahead with rung 3, and switch latex to [lo, hi]"
- "100 x 100 is fine, let's go with that."
- "Would it be better if we did a uniform random sample for the training sets?" → "I want to do uniform with pool resampling" → "Let's do it yes."

### Implemented
New `analysis/nested_bootstrap.py` computes every calibration-error interval (smECE, ECE, adaptive ECE, debiased RMSCE) and every reliability-curve band in `analysis/calibration_updated_v3.py` and `analysis/calibration_validated.py`. Real cells cross `n_fit_samples` recalibration refits with `n_doc_boot` test-document cluster resamples, and synthetic cells use `n_syn_boot` document resamples. The point is the un-resampled value averaged over fit samples, and the interval is the percentile range over all replicates. Each Platt/intercept fit sample is now drawn uniformly from a document-resampled Platt pool, replacing the document-balanced draw. Its label rate therefore targets the per-row rate that the test metrics weight by, and the interval covers which documents formed the pool. The three counts are required keys in v3 and validated analysis configs. `analysis/calibration_latex.py` drops `ci_format` and prints every interval as `[lo, hi]`. Rung configs: `analysis/analysis-configs/2026-10-07-calibration-v3-gemma27b-qwen-2.5-7b-intercept-fit-nested-{smoke,tiny}-01.yaml` and `2026-10-07-calibration-validated-gemma27b-qwen-2.5-7b-nested-smoke-01.yaml`. This is an evaluation change: it invalidates all previously computed v3/validated calibration intervals, bands and point estimates.

### Commits
42b6952 calibration: nested-bootstrap CIs over recalibration fit samples x test documents
