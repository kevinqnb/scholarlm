<!-- Public devlog entry: <repo>/devlog/<id>.md. Written by /devlog as a trimmed
version of the private build note at notes/<project>/builds/<id>.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-09-24-parsequantity-float-widening-01
kind: build
---

## Session 2026-09-24

### Prompts

- Implement the fix specified in the `2026-09-24-gptoss120b-parsequantity-schema-diag-02` experiment note: widen `point_value`/`lower`/`upper`/`tolerance`/`standard_deviation` from `str | None` to `str | float | None` everywhere that shape is defined, to fix a gpt-oss-120b guided-decoding stall.
- Scope question raised mid-session: the dataset configs' `DirectExtractionItemSchema` also feeds the NuExtract/NuExtract3/LangExtract baseline schema builders, not just Ablation 1. Decision: widen it too, accepting that NuExtract/NuExtract3's decoding grammar becomes slightly more permissive for future baseline runs than the completed 2026-09-19/22 runs.

### Implemented

Widened `point_value`/`lower`/`upper`/`tolerance`/`standard_deviation` to `str | float | None` in the 3 places that independently define this shape: `MeasurementLM.ParseQuantityResponse` (`src/scholarlm/measurementlm.py`), `MeasurementLMv2.QuantityItem` (`src/scholarlm/measurementlmv2.py`), and each of `pond.py`/`nfix.py`/`supermat.py`/`measeval.py`'s `DirectExtractionItemSchema` under `experiments/dataset-configs/`. No experiment config involved -- this is a library/dataset-config change, not a run. Added unit tests at all 3 sites (schema-shape checks against the validated diagnostic schema, bare-float and quoted-string round-trip validation, dedup-key float/string equivalence) plus a new `tests/test_dataset_config_quantity_fields.py`; full suite green with no regressions. Audited every downstream consumer of these fields (matching/dedup code in `analysis/match_cache.py`, `src/scholarlm/utils/data.py`, `MeasurementLM._deduplicate`, `measurementlmv2._norm_scalar`) for mixed str/float risk -- none found; no eval/metric code was touched. Smoke run and tiny end-to-end verification against a live gpt-oss-120b server are left for a follow-up experiment session.

### Commits

732f584 Widen parse-quantity scalar fields to str | float | None
