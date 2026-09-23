<!-- Public devlog entry: scholarlm/devlog/2026-09-23-extraction-dedup-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-23-extraction-dedup-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-23-extraction-dedup-01
kind: build
---

## Session 2026-09-23

### Prompts

- "Let's develop a new utility function in `src/scholarlm/utils/deduplication.py` for
  deduplicating extractions. Right now MeasurementLM has its own deduplication step,
  but that's mostly tied to its own internal entity IDs and still produces a lot of
  duplicates from similar entities with different IDs. This should work a lot like how
  the current matching process in `src/scholarlm/utils/data.py` works. We should take
  as input a set of fuzzy fields, a set of strict fields, and a threshold parameter. If
  for two points the strict fields match and the fuzzy fields pass the threshold, then
  those are duplicates and we can remove one of them (e.g. heuristically choose the
  first). ... Be very careful, this will be an important function with downstream
  experiment implications. Test it on real extraction data from
  2026-09-21-pond-extraction-gemma27b-full-01"
- Design choices: merge dropped rows' provenance lists into the kept row; treat a pair
  with no comparable fuzzy field as a duplicate when strict fields match.
- "How does it perform on 2026-09-19-pond-langextract-gemma27b-full-01?"
- "Here are the fuzzy fields we should definitely have: name, date. Then units and all
  the new qualifier fields should be strict equal. Try this on the earlier pond
  extractions."
- "Let's /devlog and commit this. There are still some issues here, and we should make
  note of that, but this is not immediately being used within analysis yet and so it's
  fine to just keep this as a first draft."

### Implemented

`scholarlm.utils.deduplication.deduplicate_records` drops duplicate extraction rows
using the same pairwise strict/fuzzy rule as `match_datasets` (reimplemented, not
refactored, so eval code is untouched), with one documented divergence for pairs that
have no comparable fuzzy field. Rows are compared only against already-kept rows (no
transitive chaining), dropped rows' provenance lists are merged into the kept row, and
an audit frame records every dropped→kept assignment and score.
`tests/test_deduplication.py` covers hand-verified fixtures, pairwise parity with
`match_datasets`, invariants, and fail-loud validation. This is a **first draft**:
it is not used by the pipeline or any analysis, and sweeps on pond extraction runs
showed that fuzzy name/date matching below an exact threshold merges distinct numbered
sites and sampling dates, so no configuration is validated yet. No experiment config.

### Commits

- 97cdbb0 Add deduplicate_records: strict+fuzzy dedup of extraction rows (first draft)
