---
id: 2026-09-03-probe-synthetic-augmentation-01
kind: build
config: configs/2026-09-03-probe-synthetic-augmentation-01.yaml
---

# Synthetic augmentation of the judge probe-training data

<!--
Earlier sessions on this build (df99b12, 31483d1, 2b9697d, 8492be4, 7aa6cf6,
12a3dd1, 9e5bdc6, 81deefe, 272816f, cab983c, 4698751, 11081f2, f08910f, 9f35b3f)
predate the dev-log convention (5d05ca0) and are recorded in the private build
note + their Claude-Session commit trailers. This file starts at the first
session after the convention landed.
-->

## Session 2026-09-06

### Prompts

> Implement the changes outlined in `probe_augment_rewrite_protocol_change.md` (now described below).
> Use careful unit tests to ensure correctness. Remove any dead code this creates
> to keep things clean. Remove any unused probe augmentation output to give us a
> clean slate, and get things ready to run under the new changes.

### Implemented

Replaced the `rewrite` op's "re-emit the whole edited page as one JSON string"
protocol with a diff protocol: `GptOssClient.rewrite_context`
(`src/scholarlm/utils/probe_augment.py`) now asks gpt-oss only for a list of
`{"find","replace"}` edits (each `find` a verbatim page substring), applies them
itself via `apply_verified_edit` (`count=1` per edit, checked against the
running patched text), and verifies against the `expected` measurement spans; a
missing `find` span is a skip, a `feasible:true` with no usable `edits` is a hard
error. The rewrite cache payload gains `protocol: 2` so old full-page entries
become permanent misses while `event_fill` entries keep hitting; `_one` also now
raises on a `length` finish with non-empty content. Before coding, measured
`expected`-span multiplicity in the pond `--reviewed` train contexts (entity/
attribute names are a verbatim page substring <2% of the time) to confirm the
strict sanity check is not a yield killer under the new protocol. No entry
points or configs changed; `configs/2026-09-03-probe-synthetic-augmentation-01.yaml`
is unaffected (filenames and per-dataset invocations are the same). Rung 1: full
suite 228 passed / 1 skipped (`test_probe_augment.py` 69, was 52). Rung 2
(model-free): pond default path still byte-identical to on-disk
`probe_dataset{,_test}.json`; `--augment --augment-stub` byte-identical across two
runs. The real gpt-oss regeneration (Rungs 2–4, all three `gen_augment_*.sh`
jobs) is the user's and invalidates every `_v2` output plus the `rewrite` half of
the three `probe_augment_cache.json` files; the stale `_v2` outputs and the
resolved-incident `bad_responses/` dumps were deleted.

### Commits

- `c9dd3cc` probe augmentation: rewrite via a diff protocol, not a full-page echo

## Session 2026-09-07

### Prompts

> Clean up and log the recent probe-dataset work. In the log, summarize the
> creation of the probe datasets — dataset sizes, and how synthetic entries are
> created for each dataset. Commit the final probe datasets and any artifacts
> (the cache?) needed for reproducibility; gitignore or remove the rest. There
> are also commits building code that is no longer used (probe dataset creation
> was simplified) — remove them from the timeline if possible, but not if that
> would destroy other useful code or mess up the git history elsewhere.

### Implemented

Ran the Rung-4 real `gpt-oss-120b` generation for all three datasets
(`data/{pond,nfix,supermat}/create_probe_dataset.py --augment …`, per-dataset
invocations recorded in `configs/2026-09-03-probe-synthetic-augmentation-01.yaml`)
and committed the outputs.

One code fix, pond only: a Rung-4 diff-report spot-check found ~11 % of pond
`pos_entity` synthetic positives (the `wetland` ecosystem slice) getting a
fabricated name of a slightly different waterbody type, because `_MADE_UP_NAMES`
— a list originally written for the default path's `noise_entity` negatives,
where type is irrelevant — has no wetland-suffixed entries. Added a separate
`_WETLAND_NAMES` pool (kept out of `_MADE_UP_NAMES` so the default path's RNG
draw is unchanged and the byte-identical baseline gate still holds), extended
the name-suffix→type map with `marsh` / `wetland` / `fen` / `bog` / `mire`, and
routed both pools into the augmentation rules. `data/pond/create_probe_dataset.py`
only; new `tests/test_pond_probe_pools.py` (6 tests). Pond's `_v2` outputs and
the `rewrite` half of its cache were regenerated after the fix; nfix / supermat
are unchanged from the 2026-09-06 run.

Each generation produces three additional files per dataset, all balanced to
exactly 50 % `valid` prevalence; the byte-unchanged baseline
`probe_dataset{,_test}.json` stays in place (nfix's baseline was the authorised
re-pin from `272816f`):

| dataset | baseline train / test | `_v2` train | `_v2` primary test | `_v2` diagnostic test |
|---|---|---|---|---|
| pond | 5379 / 4620 | 6758 | 3080 | 2578 |
| nfix | 1656 / 1104 | 1620 | 736 | 492 |
| supermat | 1755 / 1743 | 1232 | 1162 | 54 |

Synthetic rows come from three mechanisms. **Axis-2 positives** take a
ground-truth-valid `(page context, measurement)` pair and ask `gpt-oss-120b` for
a minimal `{find, replace}` diff to the page that swaps one
equivalence-preserving property of the measurand, applied and verified against
the expected spans, with the matching measurement field updated and the edited
page inlined on the row as `context_override`. Live axes differ by dataset:
pond gets entity-name swaps plus shared-unit attribute swaps (tn/tp/chla); nfix
gets entity-name swaps only (355 axis-2 rows in the train file: 258 positives +
97 matched hard negatives — attribute swap is inert on its disjoint unit
families); supermat swaps only `sample_details` (44 train rows: 31 + 13 — the
formula name is never touched, and `pos_event` is disabled for lack of a date
field). Per-file counts are in the composition tables below.
`pos_value` / `pos_units` / `pos_event` ship off by default (label-noise risk).
**Event-fill** (pond only) has `gpt-oss-120b` populate `date` /
`additional_details` on GT-valid rows whose page states an event; it is a hard
error for nfix / supermat. **Hard negatives** are rule-based from GT valids (and
from axis-2 edited contexts, for leakage symmetry): `hard_value` (small
perturbation), `hard_units` (plausible alternate unit), `hard_entity`
(fabricated same-type name, skipped on null entity fields). The primary test
file carries only held-out GT valids + GT-derived hard negatives — no synthetic
positives, no event-fill.

#### `_v2` dataset composition

All nine `_v2` files are balanced to exactly 50 % `valid`, so the `valid` and
`invalid` blocks below are equal per file. Every GT-valid row is a `valid` in
exactly one split (train / test partitioned by paper, ~50 % each): pond 1793 +
1540 of 3333, nfix 552 + 368 of 920, supermat 585 + 581 of 1166. All figures are
row counts on the committed output (pond post-`_WETLAND_NAMES`-fix regen; nfix /
supermat from the 2026-09-06 run).

**Valid rows.** A `valid` is either a GT-valid row carried through (train only,
optionally with `gpt-oss` event-fill applied in place) or an axis-2 synthetic
positive (a GT-valid whose page `gpt-oss` rewrote, edited page inlined as
`context_override`):

| file | valid | GT-valid carried | — of which event-filled | axis-2 `pos_entity` | axis-2 `pos_attribute` |
|---|--:|--:|--:|--:|--:|
| pond `_v2` (train) | 3379 | 1793 | 1642 | 1437 | 149 |
| pond `_test_v2` (primary) | 1540 | 1540 | 0 | 0 | 0 |
| pond `_test_v2_diag` | 1289 | 0 | 0 | 1245 | 44 |
| nfix `_v2` (train) | 810 | 552 | 0 | 258 | 0 |
| nfix `_test_v2` (primary) | 368 | 368 | 0 | 0 | 0 |
| nfix `_test_v2_diag` | 246 | 0 | 0 | 246 | 0 |
| supermat `_v2` (train) | 616 | 585 | 0 | 31 | 0 |
| supermat `_test_v2` (primary) | 581 | 581 | 0 | 0 | 0 |
| supermat `_test_v2_diag` | 27 | 0 | 0 | 27 | 0 |

Event-fill (pond only, `--augment-events`) touched 1642 / 1793 = **92 %** of the
pond train GT-valids. `date` and `additional_details` are entirely null in
`ground_truth_review.json`, so every non-null `date` (273 rows) or
`additional_details` (1639 rows) on a pond train `valid` is `gpt-oss`-synthesised;
the remaining 151 GT-valids got neither field (model reported no event). nfix and
supermat get no event-fill — their carried GT-valids are verbatim. supermat's
axis-2 positives swap `sample_details` only (tagged `pos_entity` internally);
nfix's swap the entity `name` only. `pos_attribute` fires on pond alone (149 rows
train / 44 diagnostic — the tn/tp/chla shared-unit slice); it is inert on nfix
(disjoint unit families) and supermat (formula name never touched), and neither
runs `pos_event`.

**Invalid rows.** Every `invalid` is a rule-derived hard negative — no `gpt-oss`
call: `hard_value` (small numeric perturbation), `hard_units` (plausible wrong
unit for the attribute), `hard_entity` (fabricated same-type name). They are
derived from GT-valids ("GT ctx") and, for leakage symmetry, from the axis-2
edited pages ("edited ctx" — these carry the positive's `context_override` and
`augment_axis` tag):

| file | invalid | on GT ctx | on edited ctx | `hard_value` | `hard_units` | `hard_entity` |
|---|--:|--:|--:|--:|--:|--:|
| pond `_v2` (train) | 3379 | 2531 | 848 | 1218 | 935 | 1226 |
| pond `_test_v2` (primary) | 1540 | 1540 | 0 | 528 | 472 | 540 |
| pond `_test_v2_diag` | 1289 | 0 | 1289 | 456 | 363 | 470 |
| nfix `_v2` (train) | 810 | 713 | 97 | 250 | 307 | 253 |
| nfix `_test_v2` (primary) | 368 | 368 | 0 | 103 | 141 | 124 |
| nfix `_test_v2_diag` | 246 | 0 | 246 | 85 | 71 | 90 |
| supermat `_v2` (train) | 616 | 603 | 13 | 274 | 281 | 61 |
| supermat `_test_v2` (primary) | 581 | 581 | 0 | 255 | 280 | 46 |
| supermat `_test_v2_diag` | 27 | 0 | 27 | 9 | 10 | 8 |

Reading the three splits by construction: **primary test** = held-out GT-valids +
GT-derived hard negatives only, zero `context_override` rows, no `gpt-oss` content
at all; **diagnostic test** = fully synthetic, every `valid` an axis-2 positive
and every `invalid` a hard negative on one of those same edited pages; **train** =
carried (event-filled) GT-valids + axis-2 positives + hard negatives off both,
then downsampled to 50/50 and capped at `augment_max_derived_per_source=4` derived
rows per originating GT row (`augment_target_rows=10000` / `floor=5000` were not
reached for any dataset — material-limited, floor waived for nfix).

For contrast, the untouched baseline `probe_dataset{,_test}.json` is rule-only at
1 : 2 valid : invalid across seven mechanisms (`change_{value,attribute,entity,units}`,
`noise_{value,entity}`, `table_value`) — e.g. pond train 1793 valid + 3586 invalid
(`table_value` 897, `change_value` 473, `noise_value` 468, `change_entity` 467,
`change_attribute` 463, `noise_entity` 428, `change_units` 390); see the module
docstring in `create_probe_dataset.py` for the per-mechanism definitions.

#### Why two test files (`_test_v2` vs. `_test_v2_diag`)

They measure different things and are reported separately — the diagnostic
number is never merged into the primary metric. `_test_v2` is the **headline
eval**: real held-out reviewed GT valids as positives (original fields, no
`gpt-oss` step anywhere near them) + the improved GT-derived hard negatives. No
fallible inference touches the positive class, so `gpt-oss` label noise cannot
corrupt the reported figure; it supersedes the baseline `probe_dataset_test.json`
only by upgrading the negatives. `_test_v2_diag` is a **deliberately synthetic
probe of generalization** to the cases the augmentation targets: every positive
is an axis-2 rewrite (page edited so the measurand's surface form changed but
page and measurement still agree), every negative a matched hard negative on that
same edited page. Because positives and negatives share the edited contexts, a
probe keying on "this page was edited" scores at chance here — the leakage check
falls out for free. Event-fill runs in this file too (already synthetic, already
non-headline, so the fallible fill costs nothing and makes the slice more
realistic).

#### Why datasets have limited size capacity

The binding constraint is the **positive class**. `balance_and_cap` trims to
`min(#positives, #negatives)` and negatives are always plentiful, so the file
size is `2 × #positives`, and `#positives = held-out GT valids + accepted axis-2
rewrites`. nfix's train split has 552 reviewed GT valids, supermat's 585 (vs.
pond's 1793) — these datasets simply carry ~3× fewer annotated measurements.
Synthetic positives cannot close the gap: nfix has only one live axis
(`pos_entity`; `pos_attribute` is inert on its disjoint unit families) and
supermat only the `sample_details` swap, and per-attempt yield is low (nfix
258/552 ≈ 47 %, supermat 31/585 ≈ 5 % — the surface form is often not a clean
verbatim page span, or `gpt-oss` declines / fails span verification). The
generator also makes exactly one axis-2 attempt per (source, axis) by design:
extra positives from one source with only the entity name re-drawn are
near-duplicate contexts — same page, same measurement, same label, one token
different — so they inflate row count without adding independent signal, let a
few high-yield sources dominate, and make any eval on that data optimistic
(`augment_max_derived_per_source=4` caps this further). `pos_value` / `pos_units`
/ `pos_event` stay off (label-noise risk) and event-fill adds no rows (it mutates
in place). Net: nfix caps at 552 + 258 = 810 positives → 1620 rows, supermat at
585 + 31 = 616 → 1232; the 5000 floor needs 2500 positives per side and is
waived (explicitly for nfix in the config) rather than met with synthetic
padding.

The three `probe_augment_cache.json` files (content-hash-keyed `gpt-oss`
responses; pond 26 MB, nfix/supermat ~4 MB) are committed — `gpt-oss` at
`temperature=0.2` with no seed is not bit-reproducible, so the cache is the
determinism artifact and a regeneration only calls the model on a miss. The
per-split `*.diff.txt` unified-diff spot-check reports are derived from
cache + code and are gitignored, not committed.

The requested history rewrite was **not** done. The probe-augmentation commits
are interleaved with unrelated work (`representation_lm`, the validation-set
sampler, the harness/`devlog` infrastructure), the retired side-car scheme
(built `f08910f` / `9f35b3f` / `12a3dd1`, deliberately replaced by the inline
`context_override` field in `7aa6cf6` / `8492be4`) is already gone from the
tree, and both `devlog/` and the private note reference these hashes with
per-commit rationale — rewriting would orphan every one of those references,
which is the "mess up the git history elsewhere" case the request excluded.

### Commits

- `936c75f` probe augmentation: supermat axis-2 swaps the material formula in `name`

The `_v2` datasets and `gpt-oss` caches this session generated were **not**
committed: the augmentation procedure was reworked in the following session
(2026-09-08) and every `_v2` output regenerated under the corrected design, so
the interim blobs would have been superseded on the next commit.
