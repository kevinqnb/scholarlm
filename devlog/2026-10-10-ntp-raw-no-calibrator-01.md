---
id: 2026-10-10-ntp-raw-no-calibrator-01
kind: build
---

## Session 2026-10-10

### Prompts
- "Please help me make a targeted update to @analysis/synthetic_probe_train.py and its downstream usage. In particular I don't think it makes much sense to run NTP output through the CalibratedClassifierCV and then cache that as the 'trained' NTP model. Especially considering that we're going to do recalibration for NTP later in @analysis/calibration anyways. Please remove this entirely for NTP, and then make sure that all downstream loading for raw (non-recalibrated) NTP values is properly set up. I.e. look for these in @analysis/calibration.py and @analysis/platt_scaling.py ."
- "Hold on, why are you adjusting pond_meta.py? Don't all loaded probabilites there come out of the calibration results? I.e. they are coming from recalibrated models right?" → (after explanation) "Go ahead with removal, yes."
- "Say more about Exact 0 and 1 values" → "Is this an issue?"

### Implemented
`analysis/synthetic_probe_train.py` no longer fits or saves an NTP calibrator; `use_platt_scaling` now affects only the head probe. `analysis/calibration.py`, `analysis/calibration_validated.py` and `analysis/platt_scaling.py` use the judge's raw `judgement_p_true` as the unrecalibrated NTP score, and `recalibration: prior_shift` now fails loud because raw NTP has no training prevalence (all committed configs use `intercept_fit`). `analysis/common/pond_meta.py` drops its NTP-calibrator held-out assertion, and `loaders.load_trained_ntp_calibrator` is removed. No configs changed. This is an eval-logic change: every previously computed NTP calibration, platt-scaling and meta number is invalidated; probe numbers are unaffected. Unit tests (including a raw-NTP known-answer test) pass; smoke and end-to-end runs are pending.

### Commits
- `cc3036f` calibration: drop the synthetic-trained NTP calibrator; use raw judge p(true)
