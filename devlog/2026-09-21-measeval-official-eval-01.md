<!-- Public devlog entry: scholarlm/devlog/2026-09-21-measeval-official-eval-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-21-measeval-official-eval-01.md -- same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-21-measeval-official-eval-01
kind: build
---

## Session 2026-09-21

### Prompts

- Investigated whether the current measeval setup can compute standard MeasEval F1
  metrics comparable to published results, and what's missing if not.
- Scoped the work to only the Quantity, Unit, and MeasuredEntity components (no
  MeasuredProperty, no Qualifier spans, no relations), with an explicit terminology
  check first (MeasEval's "Modifiers" vs. our `qualifiers` tag list vs. MeasEval's
  unrelated "Qualifier" annotation type), then asked for a dedicated
  `analysis/measeval_evaluation.py` with the span-identification heuristics
  documented transparently, and to flag anything discovered missing along the way.
- Corrected the test plan to evaluate the real existing tinye2e extraction run
  directly rather than fabricated/synthetic records.
- Chose, when the official scorer turned out to crash on this scope (see below),
  to inject placeholder rows to keep the unmodified script running rather than
  reimplement the scoring logic ourselves.
- Asked why the reported Unit F1 was so low; investigated, no code changed.

### Implemented

`analysis/measeval_evaluation.py` exports a measeval extraction run's Quantity/Unit/
MeasuredEntity predictions into MeasEval's official TSV format and scores them with
the real, unmodified `data/measeval/raw/eval/measeval-eval.py`. Offsets are recovered
heuristically (the pipeline never computes one) by searching the pipeline's `value`/
`units`/`name` fields verbatim in the source text, with disambiguation by proximity
when a span recurs -- documented in full in the module docstring, including its
measured error rate against real ground truth. Two blockers in the official script
itself were discovered and fixed: an `annotId` format mismatch, and a crash
(`ValueError`) triggered by submitting zero MeasuredProperty/Qualifier rows (a latent
bug in the 2021 script, not a pandas-version issue) -- worked around with a small
self-matching "calibration" document whose known contribution is subtracted back out
of the reported Quantity/Unit numbers before they're shown. Added `pandasql`/
`vladiate` as a new `measeval-eval` extra in `pyproject.toml` (the official script's
own declared dependencies). `tests/test_measeval_evaluation.py` (10 tests, rung 1)
covers the span heuristic, TSV export, and the calibration-correction arithmetic
against a known-answer case from `data/measeval/README.md`'s own worked example.
Rungs 2-3 were run directly against the one real extraction result that exists
(`2026-09-20-measeval-extraction-gemma27b-qualifiers-tinye2e-01`, single `train`-split
document, via `--dev` mode); the results were hand-verified down to the individual
span-overlap and annotSet-resolution arithmetic, surfacing a real interaction between
the span-reconstruction heuristic and the official script's own annotSet matching as
a concrete, disclosed cause of degraded Unit scoring. No real `eval`-split extraction
run exists yet, so the leaderboard-comparable path (`evaluate(dev=False)`) has not
been exercised end to end.

### Commits

c95f0dd Add official MeasEval scoring for Quantity/Unit/MeasuredEntity
