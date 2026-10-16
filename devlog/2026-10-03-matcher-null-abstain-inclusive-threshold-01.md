<!-- Public devlog entry: scholarlm/devlog/2026-10-03-matcher-null-abstain-inclusive-threshold-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-03-matcher-null-abstain-inclusive-threshold-01
kind: build
---

## Session 2026-10-03

### Prompts

- Describe how `match_datasets` and `analysis/match_cache.py` handle nulls for strict and for fuzzy fields.
- Fix the fuzzy rule: a both-null field should not contribute to the mean at all; a one-sided null should score 0 and be included. Update all comments and docstrings to match, and use `>=` for the threshold everywhere.
- If every fuzzy field is null, the edge weight is 1.
- Leave `deduplication.py` alone for now.

### Implemented

`match_datasets` now lets a fuzzy field that is null on both sides abstain from the mean, scores a one-sided null as 0.0 (kept in the mean, so the threshold decides instead of an outright rejection), and gives an edge weight of 1.0 when every fuzzy field abstains; nulls are judged after `fuzzy_normalizers`. The threshold compare is now inclusive (`>=`) in `match_datasets`, `analysis/match_cache.py`, `metrics.py`, `probe_pca.py`, `validity_evaluation.py` and `calibration*.py`. This is an eval-logic change that invalidates all earlier match caches and recovery/validity numbers. Docstrings, the `numeric_coerce` comment, and stale measeval notes were updated, and the null unit tests were rewritten with hand-checkable values. Rung 1 only; `utils/deduplication.py` is intentionally unchanged, so its parity test against `match_datasets` fails until it is aligned or retired.

### Commits

- 4dc0e99 Matcher: both-null fuzzy field abstains, one-sided null scores 0; threshold inclusive everywhere
- c7675f2 Docs: update stale measeval null-matching comments to current matcher rules
