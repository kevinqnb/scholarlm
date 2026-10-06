<!-- Public devlog entry: scholarlm/devlog/2026-10-06-meta-dedup-wiring-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-10-06-meta-dedup-wiring-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-10-06-meta-dedup-wiring-01
kind: build
config: analysis/analysis-configs/2026-10-06-pond-meta-dedup-smoke-01.yaml
---

## Session 2026-10-06

### Prompts
(Summary of a multi-message session.)
- "We just implemented a full deduplication process in `analysis/deduplication.py` and have run it through configs for the pond, nfix, and supermat datasets. The goal is to now wire this up to `analysis/meta_updated.py`. These experiments should read and use the deduplicated dataset."
- Switch meta to the new run, configurable through a new analysis config; pond only; results under `analysis/results/meta/`.
- "Why are we loading in attention head vectors? Are we recomputing probe predictions? If we are we shouldn't. It should just be using stored predictions on the test set computed in `analysis/calibration_updated_v3.py` ... they get platt scaled as well."
- Confirm exactly which stored cell is used (pond-trained probe plus pond Platt coefficients, not a cross-domain cell).

### Implemented
`analysis/meta_updated.py` now takes one analysis config (`params.meta`) and reads the id-addressed pond run, with rows either the full `final.json` or the deduplicated records, writing to `analysis/results/meta/<config id>/`. Probe and NTP confidences are not recomputed: they are the Platt-scaled pond-to-pond predictions stored by `analysis/calibration_updated_v3.py`, mapped back to rows and joined by `measurement_id` (many-to-one, since list-expanded rows share an id) with assertions on every step. The new `analysis/meta_inputs.py` holds config validation, input resolution through the calibration config, and the join; the metric, weighting, bootstrap and plotting code is unchanged. Configs are `analysis/analysis-configs/2026-10-06-pond-meta-{final,dedup}-smoke-01.yaml`. Unit tests and a final-vs-deduplicated smoke comparison passed; the full run is still to do.

### Commits
9b9afe8 meta_updated: config-driven, stored Platt-scaled predictions, optional deduplicated rows
