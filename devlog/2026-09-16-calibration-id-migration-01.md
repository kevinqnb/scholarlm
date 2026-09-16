<!-- Public devlog entry: scholarlm/devlog/2026-09-16-calibration-id-migration-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-16-calibration-id-migration-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-16-calibration-id-migration-01
kind: build
---

## Session 2026-09-16

### Prompts

- Update `analysis/calibration_updated.py` to support the new experiment
  contract, in which synthetic judgement results and trained probes are
  stored in the experiments directory according to their config ID. Set up
  the script to analyze and produce figures for the latest probe models, and
  for the judged extractions from gemma-3-27b, gpt-oss-120b ablation 1, and
  the nuextract baseline (see devlog
  `2026-09-16-synthetic-probe-id-migration-01` for the new contract).
- Confirmed scope: only (pond, qwen-2.5-7b) has an id-addressed trained
  probe; llama-3.1-8b's old-tree probe predates the full-paper-judge
  rewrite and isn't comparable — scoped to qwen-2.5-7b only rather than
  mixing in a stale probe.
- "Can you just run the full analysis for all three settings mentioned:
  gemma-3-27b, gpt-oss-120b ablation 1, and nuextract? Run this by
  submitting a handwritten qsub job." — redirected the full run from a
  local shell loop to qsub, submitted one job at a time.

### Implemented

Split id-resolution logic into a new side-effect-free module,
`analysis/calibration_ids.py` — a registry of the three requested
pipeline-variants (gemma-3-27b-extraction, gpt-oss-120b-ablation1,
baseline-nuextract) keyed to pinned per-dataset extraction/judge_interp/
judge_combine experiment ids, plus boundary-check helpers
(`resolve_run`/`pinned_run_dir` against a run's own committed config,
`pinned_extraction_dir` against `run_metadata.json` for the pre-restructure
extraction/ablation/baseline runs that never got a committed config).
Rewrote `analysis/calibration_updated.py` to read every extraction/judge/
probe run through these pinned ids instead of "most recent date"
addressing; retired `--extraction-model`/`--syn-source`/`--syn-judge-date`
in favor of `--setting`/`--syn-split`; moved the ground-truth match cache
into the id-addressed extraction dir. Added `tests/test_calibration_ids.py`
(18 tests). Wrote `analysis/run_calibration_full.sh` (gitignored qsub
script, matching `experiments/verify_probe_migration_pond_train.sh`'s
conventions) and ran the full analysis for all three settings across all
three datasets via qsub (jobs 7598225/7598226/7598227), all successful —
verified metrics tables and figure PDFs for each.

### Commits

- `7f3492a` calibration: migrate calibration_updated.py to id-addressed experiment contract

## Session 2026-09-16 (2)

### Prompts

- Asked what's needed to get nfix/supermat-trained probes into the figures.
- "Yes why don't you do the code-side generalization now and then add it
  retroactively to the devlog. This is the way the script was intended to
  be." — generalize ahead of nfix/supermat having their own trained probes.

### Implemented

`analysis/calibration_ids.py`'s single-train-dataset scalars
(`TRAIN_DATASET`/`SYN_TRAIN_ID`/`SYN_TEST_IDS`, pond-only) became
dataset-keyed collections (`TRAIN_DATASETS`/`SYN_TRAIN_IDS`/`SYN_TEST_IDS`),
plus a `SYN_SPLITS` constant. `calibration_updated.py`'s probe/NTP-calibrator
loading and `compute_predictions` now loop over `TRAIN_DATASETS` instead of
assuming exactly one; the plotting/metrics functions needed no changes —
they were already generic over multiple train datasets. Adding nfix/supermat
once their own synthetic judge_interp runs exist is now a registry edit
only. Extended `tests/test_calibration_ids.py` to 19 tests and re-verified
the known-answer check reproduces identical numbers.

### Commits

- `3aca8ba` calibration: generalize calibration_ids/calibration_updated to multiple train datasets
