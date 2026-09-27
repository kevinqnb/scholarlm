<!-- Public devlog entry: scholarlm/devlog/2026-09-22-qualifier-ground-truth-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-22-qualifier-ground-truth-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-22-qualifier-ground-truth-01
kind: build
---

## Session 2026-09-22

### Prompts

- Add MeasEval-style qualifier/shape fields (`qualifiers`, `point_value`, `lower`,
  `upper`, `list_values`, `tolerance`, `standard_deviation`) to ground truth for
  pond, nfix, supermat, and measeval: null throughout except `point_value` for
  pond/nfix (copy of the existing value); for supermat/measeval, stop dropping
  rows for being a range/approximate/list/unparseable, carry the raw reported text
  into `value` verbatim, and keep all 7 fields null.
- Pond/nfix: put the field logic in each dataset's own `preprocessing.py`, not a
  standalone script — it needs to flow through the real review pipeline
  (`preprocessing.py` → `apply_review.py`) since that's what production eval
  actually reads.
- Rejected a patch-in-place approach: wanted a documented path through the real
  pipeline, with preprocessing made deterministic if it wasn't, and everything
  regenerated for real.
- After learning that fixing pond's row-order determinism would require remapping
  `page_review.csv`'s corrections (which are keyed by row position): decided row
  reordering itself doesn't need fixing, but the review corrections must still land
  on the correct rows without touching `page_review.csv`. Confirmed nfix's parallel
  fix (a units-normalization gap) should proceed.
- Asked for the docstrings added during this work to be trimmed to a high-level
  description of how the pipeline works, not a history of what changed — that
  belongs in the commit and this entry.

### Implemented

Added qualifier/shape fields across all four datasets' ground truth: pond and nfix
via `preprocessing.py` (fields injected unconditionally) feeding `apply_review.py`
unchanged; supermat and measeval via a new `--qualifiers` flag on their
`preprocessing.py`, writing a separate `ground_truth_qualifiers.json` since that
path also stops dropping rows for range/approximate/list/unparseable values.
Root-caused two pre-existing reproducibility gaps this surfaced rather than
papering over them: pond's `ground_truth.json` row order changed after an
unrelated pipeline rewrite postdating the last review, resolved by a one-time
reorder of a fresh build to match the already-committed file's row positions
(verified position-for-position identical, zero semantic change, `page_review.csv`
untouched); nfix's review file had never picked up an earlier units-normalization
fix, now propagated, plus a related gap in `apply_review.py` (manual unit
corrections weren't being normalized) fixed and verified. `uv run --extra dev
pytest`: 601 passed, 1 skipped throughout.

### Commits

- fafb16c Add qualifier/shape fields to ground truth for pond, nfix, supermat, measeval
