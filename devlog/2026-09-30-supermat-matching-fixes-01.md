<!-- Public devlog entry: scholarlm/devlog/2026-09-30-supermat-matching-fixes-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-30-supermat-matching-fixes-01
kind: build
---

## Session 2026-09-30

### Prompts

- Supermat recovery results look strange next to the other datasets; assess whether the matching algorithm and the fields it matches on are responsible.
- Fix supermat's `point_value` string/number mismatch, and make name comparison robust to chemical-formula notation (abbreviations are out of scope).
- Null rule: null == null matches on fuzzy fields; nulls never match non-nulls on fuzzy or strict fields. Keep recovery edge-based for now.
- Update the unused `deduplication.py` utility to the same null rule.

### Implemented

`match_datasets` now scores fuzzy null/null as agreement and rejects any one-sided null, replacing the earlier "no fuzzy evidence" default; this applies to every dataset and invalidates previously computed recovery/validity numbers and match caches. Supermat's dataset config gains `numeric_coerce=["point_value"]` and a `fuzzy_normalizers` entry using the new `canonical_formula_name` (`src/scholarlm/utils/normalization.py`), threaded through `DatasetConfig`, `match_datasets`, and `analysis/match_cache.py`. `src/scholarlm/utils/deduplication.py` was brought in line with the matcher. New and updated unit tests cover the null semantics and a hand-built formula-name fixture; this session reached rung 1 only, and match caches have not been rebuilt.

### Commits

- 4417672 Matcher: fuzzy null==null agrees, nulls never match non-nulls
- 53cfedb Supermat: coerce point_value, add formula-name normalizer for fuzzy name
