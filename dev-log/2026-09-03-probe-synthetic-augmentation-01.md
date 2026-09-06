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

> Implement the changes outlined in `probe_augment_rewrite_protocol_change.md`.
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
