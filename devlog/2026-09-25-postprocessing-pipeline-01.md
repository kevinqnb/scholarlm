<!-- Public devlog entry: scholarlm/devlog/2026-09-25-postprocessing-pipeline-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-25-postprocessing-pipeline-01
kind: build
---

## Session 2026-09-25

### Prompts

- Build `analysis/postprocessing.py` to postprocess the JSON data files from
  ID'd experiments in `analysis/analysis-configs`. For every data point: (1)
  if the model failed to fill in the qualifier fields (baselines struggle
  with this), parse the `value` field and attempt to fill them in -- look to
  `data/supermat/preprocessing.py` for inspiration, and share code via
  `src/scholarlm/utils/parsing.py` where it makes sense; (2) if the model
  failed to put units in a standardized form, convert them via a list of
  common notational variants per unit across pond/nfix/supermat; (3)
  deduplication is out of scope, a separate session. Output goes to
  `postprocessed.json` next to `final.json`; `match_cache.py` and
  `recovery_validity.py` should prefer it, falling back to `final.json`
  (and warning) if it's not found.
- Redesign `strip_embedded_unit_suffix` to also recover `units` from a
  trailing suffix in `value` when `units` itself is blank, not just strip a
  known `units` suffix.
- Convert point_value/lower/upper to float in `postprocessing.py` itself,
  rather than relying on `match_cache.py`'s numeric-coercion patch to do it
  later at match time.
- One more step: expand any list-value row into one row per list entry,
  `point_value` set to the individual value, everything else copied.
- Exclude supermat's "implausible tolerance -> OCR-corrupted range"
  heuristic from the shared parser's default (it's wrong for nfix).

### Implemented

Added `src/scholarlm/utils/parsing.py`: the quantity-shape parser refactored
out of `data/supermat/preprocessing.py` (verified byte-identical ground-truth
output), plus new unit-notation standardization utilities
(`normalize_unit_notation`, `standardize_units`, `split_value_and_unit_suffix`).
Added `analysis/postprocessing.py`, writing `postprocessed.json`: best-effort
qualifier-field fill (only when a row's shape fields are genuinely blank --
never overwrites a partially-filled row), unit standardization against each
attribute's real ground-truth unit vocabulary (never attribute_info_dict,
which can drift from it), and list-value row expansion. Updated
`analysis/match_cache.py`/`analysis/recovery_validity.py` to prefer
`postprocessed.json` over `final.json` (fallback with a warning), with a new
`extraction_file`/`extraction_sha256` sidecar guard mirroring the existing
ground-truth one. A stronger-model review plus a read-only audit over all 26
real `analysis-configs` ids caught and fixed several real bugs (a crash on
scientific-notation list values, `nan`/`inf` accepted as valid values, a
glued unit suffix not stripped, an overly guessy dimensionless-unit alias).
884 tests pass (66 new this session, across `tests/test_parsing.py` and
`tests/test_postprocessing.py`, plus extensions to the match-cache/
recovery-validity test files). No `configs/<id>.yaml` -- this is
analysis-layer tooling, not an experiment.

Every `match_cache.pkl` on disk predates `extraction_file` tracking and now
needs rebuilding before `recovery_validity.py` will accept it; the
`analysis/results/2026-09-23-*.csv` numbers predate this build and aren't
comparable to anything computed after it.

### Commits

- `1b29379` Refactor supermat's qualifier parser into a shared parsing library
- `36c3b62` Add analysis/postprocessing.py: qualifier fill, unit standardization, list-value expansion
- `7dd852d` Prefer postprocessed.json over final.json in match_cache/recovery_validity
