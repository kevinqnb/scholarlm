<!-- Public devlog entry: scholarlm/devlog/2026-09-24-supermat-qualifiers-parsing-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-24-supermat-qualifiers-parsing-01
kind: build
---

## Session 2026-09-24

### Prompts

- We need to update the ground truth files in `data/supermat`. Currently, the
  qualifiers file fills only the "value" field with raw strings from the original
  text based dataset. We need to parse this to fill in the qualifier fields like
  point_value, upper, lower, or tolerance. For the most part, this should be easy:
  just convert the string into a float directly and place it in the point value
  field. Otherwise, we'll need to do some parsing of characters like "-" for ranges
  or "<" signs to detect inequalities or ± symbols for tolerance.
- How to handle 5 rows written as a descending numeric pair (e.g. "60-58"), where
  hand-checking the source text found some are genuine ranges and others are trend
  fragments describing two different measurements: chose to leave all 5 unparsed
  rather than special-case the ones confirmed as genuine ranges.

### Implemented

Added `_parse_qualifiers` to `data/supermat/preprocessing.py`, parsing
`ground_truth_qualifiers.json`'s raw `value` text into the 7 qualifier/shape
fields (`point_value`, `lower`, `upper`, `tolerance`, `list_values`, `qualifiers`,
`standard_deviation`) instead of leaving them all null. Handles plain floats
(with an OCR colon-as-decimal fix and an orphan-punctuation fallback), ranges
(dash/tilde/word/inequality, ascending order only -- descending pairs are left
unparsed as genuinely ambiguous), approximations, tolerance (± symbol and compact
parenthetical uncertainty notation, with a physical-plausibility check that
reinterprets an oversized "tolerance" as an OCR-corrupted range instead), and
comma/`and` lists. `point_value`/`lower`/`upper` are floats, matching
pond/nfix's convention; `tolerance`/`list_values` stay strings. Of 1284 rows:
1082 plain, 185 tagged, 17 left unparsed; no non-qualifier field changed. New
test file `tests/test_supermat_qualifiers.py` (37 hand-built-fixture cases). No
dedicated experiment config -- this is ground-truth data prep run directly via
`python data/supermat/preprocessing.py --qualifiers`, not an experiment.

### Commits

- `affd93f` Parse supermat qualifiers GT value strings into point_value/lower/upper/tolerance/etc.
