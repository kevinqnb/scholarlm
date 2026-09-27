<!-- Public devlog entry: scholarlm/devlog/2026-09-21-seeddet-fixes-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-09-21-seeddet-fixes-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-21-seeddet-fixes-01
kind: build
---

## Session 2026-09-21

### Prompts

- Investigate the root causes of non-determinism found in the
  seed-determinism sanity control (`2026-09-21-qualifiers-seeddet-01`) and
  fix what can be fixed.
- Approved fixing LangExtract's dead temperature/seed kwargs and the shared
  base `_acall`'s missing seed forward; declined adding fail-loud error
  handling for LangExtract's dropped chunks (report the existing full-run
  logs' failure rate instead); set gemma-3-27b's temperature to 0.2.
  Expected GLiNER to stay unresolved and documented rather than fixed.
- Requested `extraction_passes: 2` for LangExtract, then declined it once it
  turned out to reprocess whole documents per pass rather than retry failed
  chunks specifically.
- Caught an oversight (LangExtract's own fixes hadn't been re-validated) and
  asked for more depth on the residual MeasurementLM v1 divergence and the
  LangExtract raw model output.
- Asked to test a higher LangExtract temperature, then to keep it at 0.2 for
  consistency with the shared model config.
- Close out with `/devlog`, then update the prior `/explog`.

### Implemented

Fixed two real, previously-silent bugs: `measurementlm_langextract.py`'s
`fit()` was passing `temperature=` to `lx.extract(config=config, ...)`, a
kwarg langextract's `config=` code path never reads, so every prior
LangExtract run sampled at vLLM's server-default temperature (~1.0) instead
of the configured value, and `seed` was never forwarded at all — both now go
into `ModelConfig(provider_kwargs=...)`. The shared base `MeasurementLM._acall`
never forwarded `seed` from `sampling_params`, silently dropping it for
extraction v1, ablations 1-6, NuExtract v1, and ChatExtract — now fixed at
the base-class level. Lowered `experiments/model-configs/extraction/gemma-3-27b.yaml`'s
temperature 0.6 → 0.2 (repo-wide for every `kind: extraction` consumer;
ChatExtract is unaffected, it hardcodes temperature 0.0). Added an optional
`params.temperature` override to `run_baseline_langextract.py` so LangExtract
can be tested independently of that shared value — used to test 1.0, then
left at 0.2 per the final decision, with the override mechanism kept in
place unused. Confirmed via new seed-determinism reruns (`extraction/2026-09-21-pond-extraction-gemma27b-qualifiers-seeddet-temp02-{01,02}`,
`extraction_v2/2026-09-21-pond-extractionv2-gemma27b-qualifiers-seeddet-temp02-{01,02}`,
several `baseline_langextract/2026-09-21-pond-langextract-gemma27b-qualifiers-seeddet-{temp02,temp10}-*`
configs) that MeasurementLMv2 is now fully deterministic (21/21 exact match)
and v1 is substantially improved; LangExtract's remaining failures turned
out to be a reproducible repetition-collapse generation bug, not a
temperature/seed problem, diagnosed via a temporary (reverted, uncommitted)
debug-logging patch rather than a code change. Full `pytest` suite green
throughout (599 → 601 passed / 1 skipped).

### Commits
6dec7bb Fix LangExtract's dead temperature/seed kwargs and base _acall's missing seed forward
