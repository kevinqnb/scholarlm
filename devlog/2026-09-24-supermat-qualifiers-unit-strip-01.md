<!-- Public devlog entry: <repo>/devlog/<id>.md. Written by /devlog as a trimmed
version of the private build note at notes/<project>/builds/<id>.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-24-supermat-qualifiers-unit-strip-01
kind: build
---

## Session 2026-09-24

### Prompts

- Update `data/supermat` ground truth so the `value` field has units removed
  (almost always a stray "K" or " K"). Update `data/supermat/preprocessing.py`
  to do this, regenerate (with qualifiers), and confirm everything else is
  identical. Keep other parts of the value field (inequality/range symbols)
  intact — only remove the units.
- Leave the four orphan-punctuation rows as-is; commit.

### Implemented

Added `_strip_value_units` to `data/supermat/preprocessing.py`, a regex that
removes only `K`/`mK`/`kelvin(s)` unit tokens from the qualifiers ground
truth's `value` field, leaving inequality symbols, range dashes, and
qualifier words (`up to`, `from ... to`, `below`, `above`, ...) untouched.
The strip runs *after* page attribution in `build_ground_truth`, not inside
`_raw_tcvalue`, because page attribution's table-match pass float()-parses
`value` and behaves differently once units are gone — stripping first would
have silently changed `page_number`/`page_score`/`page_confidence` for
hundreds of rows. Regenerated `data/supermat/ground_truth_qualifiers.json`
via `python data/supermat/preprocessing.py --qualifiers`; verified every
field except `value` is byte-identical to the prior committed file.
`ground_truth.json`/`ground_truth_ten.json` untouched (not affected — their
`value` was already numeric). Four rows keep orphan punctuation from
pre-existing raw-data artifacts (e.g. `"54.6-K"` -> `"54.6-"`), left as-is
per review.

### Commits

a4e7bba Strip Kelvin units from supermat qualifiers ground truth value field
