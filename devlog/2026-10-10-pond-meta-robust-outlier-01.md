---
id: 2026-10-10-pond-meta-robust-outlier-01
kind: build
config: analysis/analysis-configs/meta/2026-10-10-pond-meta-gemma27b-qwen-2.5-7b-01.yaml
---

## Session 2026-10-10

### Prompts
- "Please help me audit and review @analysis/pond_meta_analysis.py . Provide a high level outline and point to specific lines of code I can review. Is there any room for improvement -- both from the implementation side and from the analysis side?"
- (Follow-up, summarized) Asked what METHOD_PROB_COL holds, where log-scale W1 is used, how the W1-plot normalization relates to log scaling, and how `qq_levels` picks quantile levels.
- "Here are the fixes to the analysis that I would like to make: 1. We should compute ecosystem-level statistics when determining the outlier factors. 2. Use a robust z-score to determine outlier factors, i.e. the new factors should be of the form $e^{-z^2/2}$, where $z = \frac{x - median}{1.4846 \cdot MAD}$ and MAD is the median of the absolute deviations from the median. This can go as a fix straight into @analysis/common/outlier_weight.py 3. Change up the set of w1 curves being plotted: one for outlier factors only, one for confidence probabilities * outlier factors, and one in which confidence probabilities * outlier factors are shuffled (keep shuffling as is). 4. Make plotting log w1 curves a configurable option for the script. In this case, plot log w1 scores for all attributes (i.e. not just the log-scale attributes). 5. Drop the normalization factor from the w1 curve plots -- the only thing that matters is that we have relative comparisons to randomly shuffled cases."
- (Decisions, summarized) Use the Gaussian-consistent MAD constant 1.4826; allow the factor to underflow to 0 and count it; apply the shared change to `pond_clustering.py` as well.
- "A few more plotting changes: (a) make sure the y-axis labels were adusted to only say W_1. (b) x-axis labels should be 'Filter Level'. (c) The factor only curve (there should only be one per-plot) should be the pink tab10 color, with some white mixed in."
- "the legend entry for 'Shuffled: ... over permutations' should just read 'Shuffled: 95% CI'"
- "the legend in the QQ plots also needs to change. The color bar should simply read: 'Filter Level' instead of 'Bottom fraction of confidence...'"

### Implemented
`analysis/common/outlier_weight.py` now computes the non-outlier factor as exp(−z²/2) with a robust z-score (median and 1.4826·MAD per ecosystem and attribute). It fails loud on a zero MAD and counts factors that underflow to 0; `analysis/pond_clustering.py` inherits it. `analysis/pond_meta_analysis.py` adds a factor-only threshold setting as the value-filter baseline and computes log W1 for every attribute. Its W1 curves are now unnormalized and plotted on raw or log scale via a new required `w1_curve_scale` key (validated in `analysis/common/meta_inputs.py`), with simplified axis and legend labels. New config: `analysis/analysis-configs/meta/2026-10-10-pond-meta-gemma27b-qwen-2.5-7b-01.yaml`. This eval-logic change invalidates earlier pond meta numbers and clustering numbers computed with outlier adjustment. Unit tests, a smoke run and a predicted small run passed; the full run has not been done.

### Commits
- `668c8f6` meta: robust per-ecosystem outlier factor, factor-only W1 baseline, log W1 curves
