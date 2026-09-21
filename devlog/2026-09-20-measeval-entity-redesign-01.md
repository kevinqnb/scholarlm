<!-- Public devlog entry: scholarlm/devlog/2026-09-20-measeval-entity-redesign-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-20-measeval-entity-redesign-01.md -->

---
id: 2026-09-20-measeval-entity-redesign-01
kind: build
---

## Session 2026-09-20

### Prompts

> Let's evaluate the state of implementation for the measeval dataset by
> comparing experiments/dataset-configs/measeval.py and
> experiments/dataset-configs/pond.py. Do we have measeval prompts, schemas,
> and parameters for all the methods referenced in the pond dataset config?
> Is coverage complete? Is it a well defined extraction task, or potentially
> confusing to certain models? What are the target entities, attributes,
> events, and quantities?

> Two points I want to make about coverage: (a) NuExtract3 (the main one we
> are using now) supports text and so we need to set up the schemas/prompts
> to run measeval with it. (b) Ablation2 *combines* the entity and attribute
> detection steps. It doesn't remove the attribute completely. So I think
> this method should also be applicable with measeval. Regarding the task
> definition: what if we get rid of entities for this task entirely? ...
> MeasurementLM currently has no path for a set of empty entity fields. How
> feasible would it be to add a path like this? What prompts would this
> break?

> Ok here is what we should do (1) include support for NuExtract and
> ablation 2 by writing examples and schemas for them, and (2) change the
> extraction strategy so that a general purpose 'subject' is the main entity
> field -- any subject of a measured quantity. The attribute should remain
> as 'quantity'. This way the extraction task becomes more natural, we are
> still getting the quantity itself in the value/units fields where they
> belong. Ensure that this change is properly implemented across all
> methods.

### Implemented

Reworked `experiments/dataset-configs/measeval.py`'s entity/event schema from
the prior quantity-first design (entity = the reported quantity span, subject
and property resolved afterwards as event fields) to a subject-first design
(entity = `name`, a general-purpose measurement subject; event = `property` +
`additional_details`), while keeping the property-deferral that avoided the
original (subject, property)-pair enumeration bottleneck the quantity-first
design was built to fix. Added measeval's first NuExtract3 few-shot examples
and Ablation 2 support (`ablation2_entity_schema` + identification prompt),
both previously missing. Simplified `src/scholarlm/measurementlm_gliner.py`'s
`_entity_name_field()` to fail loud on a missing `name` field instead of
carrying a fallback chain that the redesign made unreachable. No changes were
needed in eval code, ground truth, or any runner -- verified, not assumed.
Full test suite passes (580 passed, 1 skipped); no experiment run this
session.

### Commits
7899052 Rework measeval to a subject-first entity design with NuExtract3 and Ablation 2 support
