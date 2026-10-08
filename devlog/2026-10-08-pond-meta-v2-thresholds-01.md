<!-- Public devlog entry: scholarlm/devlog/2026-10-08-pond-meta-v2-thresholds-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-10-08-pond-meta-v2-thresholds-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-10-08-pond-meta-v2-thresholds-01
kind: build
config: analysis/analysis-configs/2026-10-08-pond-meta-v2-thresholds-coarse-01.yaml
---

## Session 2026-10-08

### Prompts
- "Please help me implement a new `meta_updated_v2.py` script stemming from the current @analysis/meta_updated.py script, but made much simpler. The key changes we want to make to this new script are: 1. Filter at confidence probability thresholds rather than smoothly changing temperature... 2. Drop weighted quantiles and QQ plots. Once data has been filtered, compute quantiles and generate QQ plots in a standard fashion. 3. Keep only very limited data filtering before doing this analysis -- only remove data points which exceed obvious limits (i.e. the PHYSICAL_BOUNDS that we have...)"
- (Summarized design answers) An unweighted distance per threshold with bootstrap CI; threshold on the v4 intercept-fit-sample calibration predictions.
- "Wherever possible, use library functions for stuff like quantiles and wasserstein distances" (then: use scipy W1 rather than adding a W2 dependency)
- "Don't use the requirement that rows are restricted to documents shared by GT and extraction. But we do still need to make sure that we are only doing this analysis on documents in the test set (weren't used to train the probe or NTP models)"
- "I want ground truth on the x-axis, and then we just need to make sure that the quantiles for the 'valid' set are included as a visually distinct line in the plot."
- "Get rid of the judge filtered lines -- isn't that kind of the same thing as valid just without the matching?"
- "we should only have threshold lines for 0 (unfiltered), 0.25, 0.5, and 0.75. Likewise, those are the settings that should feed into the wasserstein table."
- "The unfiltered (0 threshold line) should be dashed and dark blue. The line for valid should be dashed and dark red. Everything else should be solid and should fall in between on a blue to red scale."

### Implemented
New `analysis/meta_updated_v2.py` replaces v1's confidence-weighted temperature sweep with hard thresholds (keep rows with confidence >= t; the grid starts at t = 0, the unfiltered set), and computes everything with library functions: `np.quantile` for quantiles and Q-Q lines, `scipy.stats.wasserstein_distance` (W1) and `scipy.stats.bootstrap` for CIs, plus a shuffled-confidence permutation control. It keeps only unit-conversion and `PHYSICAL_BOUNDS` filtering, and drops v1's shared-document restriction while still excluding every probe/NTP training document; the shared loaders in `analysis/meta_updated.py` / `analysis/meta_inputs.py` gained explicit `restrict_to_shared_docs` and `calibration_version` (v3 | v4) switches, plus asserts that the NTP calibrator and recalibration-fit rows stay inside the training documents (v1 behavior unchanged). Configs: `analysis/analysis-configs/2026-10-08-pond-meta-v2-thresholds-coarse-01.yaml` (t in {0, 0.25, 0.5, 0.75}) and `2026-10-08-pond-meta-v2-thresholds-01.yaml` (fine grid), each with a `-smoke-01`. Unit tests in `tests/test_meta_updated_v2.py`; smoke, predicted-outcome, and full runs all passed.

### Commits
a533e0a meta loaders: explicit calibration_version (v3|v4), restrict_to_shared_docs, training-doc asserts
ad06ec9 meta_updated_v2: hard confidence thresholds, np.quantile Q-Q, scipy W1

## Session 2026-10-08 (W1-vs-threshold curves)

### Prompts
- "Please add in plots that show how wasserstein distances change with increasing threhsold. In a single plot show this for (a) NTP probabilities, (b) probe probabilities, and (c) randomly shuffled probabilities (averaged and with CIs around the line). Take plotting tips from analogous plots in @analysis/clustering.py"
- (Answers) Shuffle both methods, each its own dotted line; the shuffled band is the 95% percentile range over permutations.
- "Good, add this to the previous devlog and commit. Then resubmit all the jobs."

### Implemented
The W1 permutation control in `analysis/meta_updated_v2.py` now averages over `n_shuffle_samples` permutations of each method's confidences (a new required config key), reporting the mean and 2.5/97.5 percentile range and the documents the shuffled subsets span; this is an evaluation change committed on its own, and it supersedes the single-permutation shuffled column of every earlier v2 run (all other columns unchanged). New figures `figures/w1_vs_threshold_{ecosystem}.pdf` plot W1 to ground truth against the confidence threshold for NTP and probe (with bootstrap CI bands) and their shuffled controls (with percentile bands), styled after `analysis/clustering.py`. New config `analysis/analysis-configs/2026-10-08-pond-meta-v2-w1-curves-01.yaml` (plus `-smoke-01`); the four earlier v2 configs gained `n_shuffle_samples`. Unit tests, the smoke run and the predicted-outcome checks passed.

### Commits
94d91f6 meta_updated_v2: shuffled-confidence control over n_shuffle_samples permutations
db6b02d meta_updated_v2: W1-vs-threshold figures (NTP, probe, shuffled controls)
