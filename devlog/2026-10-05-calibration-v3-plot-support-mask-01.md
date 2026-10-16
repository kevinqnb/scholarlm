---
id: 2026-10-05-calibration-v3-plot-support-mask-01
kind: build
---

## Session 2026-10-05

### Prompts

(Summary of a multi-message session.)

> Update the v3 calibration plots: the only title is the method name ('NTP' or 'Probe'); only NTP plots keep the y-axis label.

> The nfix smooth calibration curves suddenly drop to observed frequency 0. I suspect a plotting artifact from the smoother being forced across 0-1 with no data at high predicted probability. Is that the right diagnostic, and how do we fix it?

### Implemented

`analysis/calibration_updated_v3.py` now titles each figure with the method name and keeps the y-label on NTP figures only. The smoothed reliability curve and its bootstrap band are drawn only within the data support (min/max prediction ± relplot's bandwidth), because relplot's smoother returns 0/eps where no predictions lie nearby; the helper is `analysis/calibration_plot_utils.py`. Plot-only change: smECE and all metrics are untouched. Rung 1 tests in `tests/test_calibration_plot_utils.py`; a render from saved predictions confirmed the curves stop at the data. The full script was not re-run, and no experiment config changed.

### Commits

- `de46bba` calibration v3 plots: method titles, NTP-only y-label, mask curve outside data support
