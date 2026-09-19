<!-- Public devlog entry: scholarlm/devlog/2026-09-19-measurementlmv2-groupctx-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-19-measurementlmv2-groupctx-01.md -->

---
id: 2026-09-19-measurementlmv2-groupctx-01
kind: build
config: experiments/experiment-configs/pond/extraction_v2/2026-09-19-pond-gemma3-27b-v2-groupctx-smoke-02/2026-09-19-pond-gemma3-27b-v2-groupctx-smoke-02.yaml
---

# MeasurementLMv2 grouped contextualization

## Session 2026-09-19

### Prompts

> Let's make an efficiency update to measurementlmv2.py: once quantities are
> collected, just do one pass with the full paper and ask the model to fill in
> all entity/event information for the whole list of quantities at once.

> I'm not sure quantity index is the right approach — I wouldn't rely on
> indexing abilities. Have the model copy back the quantity it's filling in so
> we can check it didn't mess up any originals. Also do groups rather than the
> whole list: all prose quantities in one call, then each table's quantities
> in a separate call. We still have table numbers, right?

> [On context scope and mixed-source dedup] Table groups still get the full
> paper as context, only the quantity batch narrows. A quantity seen in both
> prose and a table should be grouped with the table.

> Go ahead and do that now. [smoke test]

> [After the first smoke run dropped 0/19 quantities] Go ahead with this fix.

> Let's run rung 3.

> Write configs for the full rung 4 experiments, and then stop here and
> /devlog. [Corrected the proposed ids to drop the extra descriptive suffix —
> the date already distinguishes this build's configs from the prior ones.]

### Implemented

Reworked `MeasurementLMv2._contextualize_quantities`
(`src/scholarlm/measurementlmv2.py`) from one LLM call per deduplicated quantity
to one call per `(document, source-location)` group — a document's prose
quantities in one call, each table's quantities in another — with each response
item required to echo the quantity's identity fields back and get matched to its
source by content (`_quantity_match_key`) rather than an index. Added
`table_number` tracking through collection and dedup to support the grouping. A
pond/gemma-3-27b smoke test (`...-groupctx-smoke-01`) surfaced two real bugs in
the copy-back contract — the prompt rendered an absent field as the literal word
"None", and the model was observed re-deriving `quantifier` from table context
rather than echoing it — both fixed and reverified in a second smoke run
(`...-groupctx-smoke-02`, 19/19 matched) plus a hand check against the paper's
own source text. Added 6 new unit tests (37 total) and wrote, dry-run-validated,
but did not submit three rung-4 full-run configs (pond, nfix, supermat)
superseding the pre-rework `2026-09-18-*-gemma3-27b-v2-extraction-01` configs.

### Commits

- 9ffdcce Rework MeasurementLMv2 contextualization to batch by document x prose/table group with copy-back matching instead of per-quantity calls with an index
- 2c0d95b Add rung-4 extraction_v2 configs superseding the pre-groupctx-rework pond/nfix/supermat full runs
