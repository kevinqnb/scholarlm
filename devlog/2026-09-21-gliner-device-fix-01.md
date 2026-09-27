<!-- Public devlog entry: scholarlm/devlog/2026-09-21-gliner-device-fix-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-21-gliner-device-fix-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-21-gliner-device-fix-01
kind: build
---

## Session 2026-09-21

### Prompts

- The pond GLiNER walltime-calibration job ran up against its 2h budget for only 8
  papers; hypothesized the new qualifier fields were causing drop-and-retry stalls in
  parsing. Investigate and diagnose.
- Yes, go ahead and make this fix.

### Implemented

The GLiNER baseline (`measurementlm_gliner.py`) has been running on CPU on every GPU
node it's ever been submitted to, not stalling on retries (GLiNER2's `batch_extract`
is a single discriminative forward pass — there's no retry logic in this code at
all). `gliner2`'s `from_pretrained` only moves the model off its CPU default when
`map_location` is passed, and nothing upstream ever passed one. `qacct` on the
2026-09-21 pond/nfix walltime-calibration jobs shows `cpu ≈ ru_wallclock` (single-core
saturation) rather than GPU compute, and the same fingerprint appears on July 2026
runs — this predates the qualifier-field work entirely. Made `device` a required,
non-defaulted constructor argument in `MeasurementLMGliner` (was `str | None = None`
with a silent CPU fallback); added it as a fixed per-model value in
`experiments/model-configs/baseline/{gliner-large-v1,gliner-base-v1}.yaml`
(`device: cuda`), added a `device` field to `ModelConfig`
(`src/scholarlm/config.py`), and `experiments/run_baseline_gliner.py` now reads it
from the model config and raises `ValueError` if it's missing, instead of the old
silent `params.get("device")` path. Rung 1 (unit tests) only this session — 589
passed, 1 skipped; GPU verification (does the model now land on `cuda`, what's the
real per-pass throughput) deferred to the user on a cluster GPU node.

### Commits

5092cf0 Fix GLiNER baseline silently running on CPU instead of GPU
