<!-- Public devlog entry: scholarlm/devlog/2026-10-08-outlier-adjusted-confidence-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-10-08-outlier-adjusted-confidence-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-10-08-outlier-adjusted-confidence-01
kind: build
---

## Session 2026-10-09

### Prompts
- "I want to make targeted changes to both analysis/clustering.py, as well as analysis/meta_updated_v2.py. In particular, I want to change the probabilites that they use to weight or filter data points. Right now, the probability is only based on validity and not on the actual numerical value observed. To adjust for that, we should multiply all probabilities by the heuristic non-outlier probability... Here mu and sigma are estimated by the extracted data distribution on a particular attribute type."
- (Answers) Gaussian form exp(-(x-mu)^2/(2 sigma^2)); mu/sigma per attribute on the raw scale; an explicit boolean config flag.
- "I don't think we should shuffle the full adjusted probabilites. Rather we should shuffle the probe or NTP probabilities and then apply the adjustment on top. That way we are assessing the contribution of the probe and NTP probabilities directly."
- "Remove the random arm from clustering. Go ahead with the proposed behaviors for these scenarios."
- "...rather than filtering by literal values in meta_updated_v2.py, let's now filter by percentiles. For example if we get a list of thresholds 0,0.1,0.2, then first filter nothing, then filter out the bottom 10 percent of adusted probability scores, then the bottom 20 percent, and so on."
- "Yes, go ahead with the three-commit split."

### Implemented
New `analysis/outlier_weight.py` multiplies the probe/NTP confidences by a per-row non-outlier factor exp(-(x-mu)^2/(2 sigma^2)), with mu and sigma per attribute over the extracted rows; it is switched on by a new required `outlier_adjust` bool in the `meta_v2` and `clustering` config sections. The shuffled controls in `analysis/meta_updated_v2.py` and `analysis/clustering.py` now permute the raw confidences and keep each row's own factor (a row-level `RowShuffler` in clustering), and the clustering random arm is removed. `meta_updated_v2.py` also gains a required `threshold_mode` (`value` | `percentile`); in percentile mode a threshold drops the bottom fraction of each cell's rows by adjusted confidence. No new config was written: the seven `meta_v2` and four `clustering` configs under `analysis/analysis-configs/` were updated with the new keys at their previous behavior (`outlier_adjust: false`, `threshold_mode: value`). Only unit tests have been run; no smoke or full run with the new modes yet.

### Commits
fc1020b analysis: non-outlier confidence adjustment (outlier_adjust); drop clustering random arm
844778d meta_updated_v2: percentile threshold mode (threshold_mode)
