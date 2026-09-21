<!-- Public devlog entry: scholarlm/devlog/2026-09-18-references-resume-heading-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-18-references-resume-heading-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-18-references-resume-heading-01
kind: build
config: experiments/experiment-configs/{pond,nfix,supermat}/drop_references/<id>/<id>.yaml
---

## Session 2026-09-18

### Prompts

- Improve the olmOCR reference-dropping mechanism: instead of dropping
  everything after the reference header is detected, drop everything until
  a bare appendix heading is detected (if there is one), then keep
  everything from the appendix onward.
- "There could also be a header like 'figures' or 'tables'. I think we need
  to be broader in scope." — broadened the resume vocabulary.
- "Why would you want supplementary content dropped? ... We want to cover
  broadly anything that could be an appendix." — chose expanding the
  enumerated word list over a generic heading-shape heuristic, given the
  false-positive risk on real reference-list text.
- "Please test to see if after removing we always ensure that page tags are
  not broken: every starting tag has a closing tag and there are no hanging
  tags."
- "Can we retroactively apply this to the ocr and table cleaned OCR that
  currently lives under `experiments/results`. ... create new IDs for the
  reference dropped versions, and store them separately as to not overwrite
  previous results. ... write up the configs for these, for completeness."
- "Don't push yet. Please note the 58% missed references for supermat in
  the devlog notes."

### Implemented

`drop_references_section` (`src/scholarlm/utils/references.py`) now
resumes at the first bare appendix/figures/tables/supplementary-material
heading found after the references cut, instead of dropping straight to
end-of-document — scoped to the olmOCR bare-line cut shape only (a chandra
tag/div cut still truncates as before, to avoid corrupting div structure).
Bare `figure(s)`/`table(s)` are matched case-sensitive-uppercase-only, since
real reference-list prose ("Tables 2-4 in Smith et al. (2019)...") would
otherwise false-trigger; "data availability"/"additional file"/"online
resource" are deliberately excluded from the vocabulary for the same
reason. Page-tag bookkeeping (closing/reopening `<page number="N">` at the
splice, under the original page number) is checked by a structural
open/close walk, not a count. Added `describe_drop_references` (same
detection pass, diagnostics only) alongside the existing function.

Added `experiments/run_drop_references.py` — a composite/manual-type runner
(no SGE automation; this is seconds of CPU work) that retroactively
re-processes an existing `ocr`/`table_cleaning` result directory into a new
`experiments/results/{dataset}/drop_references/{id}/`, writing the
reference-dropped text plus a per-paper diagnostics report, without
touching the source or re-running any model. Six new experiment-configs
(one per dataset x source-type) drive this against the existing pond/nfix/
supermat OCR and table-cleaning results.

**Resume rate on real data**: ~4-8% of papers per dataset had real
appendix/figures/tables/supplementary content rescued (pond: 12/128 ocr,
92,484 chars; nfix: 8/265 ocr, 59,691 chars; supermat: 9/142 ocr, 78,058
chars; similar for table_cleaning).

**Flagging for whoever reads the supermat drop_references output next**:
only 59/142 (42%) of supermat papers have a detectable reference heading at
all — the other 83/142 (58%) have no "REFERENCES"/"BIBLIOGRAPHY" line
anywhere in the OCR text (numbered citation lists starting with no section
marker), a pre-existing gap in the heading detector, not something this
session fixed. Both supermat `drop_references` runs are therefore ~58%
pass-through (byte-identical to source), which will understate any
downstream effect of reference-dropping on supermat relative to pond/nfix
(miss rate 3/128 and 6/265 respectively) — not because references matter
less for supermat, but because most of the corpus's headings weren't found.

Not done: `DocumentLM`'s `drop_references` flag is still not wired through
`run_ocr.py`/any OCR config — a fresh OCR run can't turn this on yet. The
supermat detection gap is also unaddressed.

### Commits

a3e9ab1 references: resume past appendix/figures/tables/supplementary headings; add retroactive drop_references runner
