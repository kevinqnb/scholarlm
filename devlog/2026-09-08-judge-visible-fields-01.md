<!-- Public devlog entry. Trimmed version of the private build note at
notes/scholarlm/builds/2026-09-08-judge-visible-fields-01.md. -->

---
id: 2026-09-08-judge-visible-fields-01
kind: build
config: configs/2026-09-03-probe-synthetic-augmentation-01.yaml
---

# Restrict the judge prompt and the probe generator to judge-visible fields

## Session 2026-09-08

### Prompts

> I want to make a change to this repository, simplifying the set of extracted
> fields that a judge will see and judge an extraction with. For the pond
> dataset, this should only be the fields: `name, ecosystem type, date,
> additional details, attribute, value, units`. For the nfix dataset this should
> be: `name, date, additional details, attribute, value, units`. For the supermat
> dataset this should be: `name, additional details, attribute, value, units`. I
> believe we can do this just by adjusting the config files, but let me know if
> it needs changes elsewhere.

> Now I want to make a corresponding adjustment to how the synthetic probe
> datasets are created. When we are making synthetic changes (positive or
> negative) we should ONLY be using these fields, which get passed to the judges.
> In addition, the 'additional details' field is kind of like a catch-all, and
> since it does not appear anywhere in the ground truth, we should not use it in
> the probe set at all. NOTE that this gets rid of all event information for
> Supermat. That's ok, we just need to produce more synthetic changes elsewhere
> to get to the dataset size floor.

> keep additional_details as inert column, re-run rung 3 for all three

### Implemented

Narrowed `judge_filter_fields` in `experiments/configs/{pond,pond_ten,nfix,nfix_ten,supermat}.py`
so the LLM-judge prompt shows only `name` (+ `ecosystem` for pond) plus `date` /
`additional_details` (supermat: `additional_details` only) alongside the constant
attribute / value / units — a judge-prompt-only change (extraction output
unchanged) that invalidates prior judge numbers. Then restricted the synthetic
probe generator to that same set minus the always-null `additional_details`:
`_JUDGE_ENTITY_FIELDS` in the three `data/*/create_probe_dataset.py` shrinks to
`name` (+ `ecosystem` for pond), and `DatasetAugmentRules.event_field` in
`src/scholarlm/utils/probe_augment.py` becomes `str | None` with `__post_init__`
and every event code path guarding on it — supermat sets it to `None`, so it now
runs the `pos_entity` + `pos_value` positive axes and the `entity` / `value` /
`units` error types only (no `pos_event`, `bad_event`, or `bad_attribute`).
`configs/2026-09-03-probe-synthetic-augmentation-01.yaml` gains a per-dataset
`augment.pos_axes` map and marks `prompt_budget_multiple` stale pending a
rung-3 re-run; `additional_details` stays in `_GT_COLS` as an inert column
(it is the dedup key). Rung 1: full suite 257 passed / 1 skipped (7 new/updated
tests). Rung 2: `--augment-stub` end-to-end on all three datasets — synthetic
rows only ever touch judge-visible fields, supermat runs 2 axes / 3 error types,
outputs byte-identical across two runs.

### Commits

- `7746b8b` judge: narrow judge_filter_fields to a minimal entity/event field set
- `9817659` probe augmentation: restrict synthesis to judge-visible fields
