<!-- Public devlog entry: scholarlm/devlog/2026-09-20-measurement-qualifiers-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-20-measurement-qualifiers-01.md -->

---
id: 2026-09-20-measurement-qualifiers-01
kind: build
---

## Session 2026-09-20

### Prompts

> The original MeasEval dataset gives every quantity a "qualifier" (IsCount,
> IsApproximate, IsList, IsRange, IsMean, IsMedian, HasTolerance, plus compound
> forms)... update all methods to support this level of detail. For
> MeasurementLM, parsing the quantity should be its own step, called by every
> ablation except Ablation 1. For Ablation 1, LangExtract, NuExtract, and
> GLiNER, build the parsed fields directly into the extraction schemas.
> Nothing to do for ChatExtract.

> Don't drop or error on qualifier/field consistency mismatches — log and
> document them.

> The standardize step should ONLY do unit formatting now.

> Field name for the plain numeric value: `point_value`.

> Reconcile MeasurementLMv2's existing quantity shape onto the same design.

> Scope: all datasets, schema-only. Smoke tests deferred to a future session
> with formal configs and GPU jobs — finish the code for the baseline methods
> phase now.

### Implemented

Added `qualifiers` (`list[str]`, 8 base tags), `point_value`, `lower`, `upper`,
`list_values`, `tolerance`, and `standard_deviation` across the extraction
stack in three phases/commits. MeasurementLM (v1) gained a new
`_parse_quantities()` pipeline step and a non-fatal `check_quantity_consistency`
check; `_standardize()` narrowed to units-only. MeasurementLMv2's existing
quantity shape was reconciled onto the same fields and dedup key.
Schema-built baselines — Ablation 1, LangExtract, NuExtract, NuExtract3, and
GLiNER — got the same fields added directly to their extraction schemas rather
than a separate call: LangExtract needed a type-introspection fix
(`_json_type_for_field`) to emit real JSON arrays for list fields; GLiNER's
`dtype="str"`-only field builder can't represent lists, so `qualifiers`/
`list_values` are comma-joined strings there, a documented exception; NuExtract
and NuExtract3 needed no code changes since both already build schemas
generically from Pydantic field annotations. All four dataset configs
(`experiments/dataset-configs/{pond,nfix,supermat,measeval}.py`) got matching
schema/prompt/few-shot-example updates. ChatExtract untouched — no mechanism to
extend it. A follow-up pass then expanded each dataset's `_NUEXTRACT_EXAMPLES`
to a third example so every dataset demonstrates all 7 qualifier/shape fields
(point_value, lower/upper, list_values, tolerance, standard_deviation) at
least once — NuExtract's few-shot examples are its only real instruction
channel — and fixed a fidelity bug in three existing range examples that
reformatted the raw value text instead of preserving it verbatim. 573 passed,
1 skipped (rung 1 of the staged-gate ladder only; rung 2 deferred).

### Commits

- `6cabd7c` Add qualifier-tagged quantity parsing to MeasurementLM
- `93a7436` Reconcile MeasurementLMv2's quantity schema onto the qualifier design
- `5836377` Add qualifier-tagged quantity fields to the schema-built baselines
- `e6d0546` Expand NuExtract few-shot examples to cover all quantity shapes
