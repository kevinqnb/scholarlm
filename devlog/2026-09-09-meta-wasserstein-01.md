<!-- Public devlog entry. Trimmed version of notes/scholarlm/builds/2026-09-09-meta-wasserstein-01.md. -->

---
id: 2026-09-09-meta-wasserstein-01
kind: build
---

## Session 2026-09-09

### Prompts

> Our goal is to make a targeted addition to analysis/meta_updated.py. In addition
> to producing Q-Q plots, we should report 2-wasserstein scores between the
> extracted and ground truth distributions for all ecosystem types, attributes,
> and methods (probe or NTP). We should approximate the wasserstein distance using
> differences between quantiles, since the distributions are not necessarily
> presumed to be normal.

> Can we also bootstrap the wasserstein scores?

> instead of doing shuffled weights, why don't we just compare against the
> unweighted extraction distribution? [... after discussion:] let's keep the
> shuffle.

> this is good to document and commit.

### Implemented

`analysis/meta_updated.py` gains `wasserstein2_quantile` (W_2 ≈
`sqrt(mean_i (q_ext(u_i) - q_gt(u_i))^2)` over a fixed midpoint quantile grid on
`[0.025, 0.975]`; weighted-Hazen on the confidence-weighted extracted side,
unweighted Hazen on GT; NaN + a skip reason when a sample can't span the grid
without clamping) and `build_wasserstein_table`, which writes a new
`results/wasserstein_{dataset}_{model}_{date}.csv` — one row per (ecosystem ×
attribute × setting) over `extracted`, `judge_filtered`, `ntp_weighted`,
`probe_weighted`, in raw units and (for the log-scale attributes) log10. Each
score carries a two-sample percentile bootstrap CI (`_bootstrap_w2_ci`, both
samples resampled, `N_BOOT` replicates via an order-independent per-cell
`_boot_rng`; the CI is NaN below 99% replicate survival rather than dropping
non-random wide-tail replicates) plus the existing `w2_shuffled` permutation
control. `main()` writes/prints the table after the stats CSV; `--seed` /
`--n-boot` drive it. No config file — the parameters match the module-constant
style already in the file. 12 new tests in `tests/test_meta_weighted_stats.py`
(38 total); full analysis run and a same-seed byte-identical determinism check
pass. No eval/metric code touched.

### Commits

- `e69373a` Add quantile-approximation 2-Wasserstein scoring to the meta-analysis
