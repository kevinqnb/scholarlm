<!-- Public devlog entry. Trimmed version of the private build note at
notes/scholarlm/builds/2026-09-08-probe-full-paper-context-01.md. -->

---
id: 2026-09-08-probe-full-paper-context-01
kind: build
config: configs/2026-09-03-probe-synthetic-augmentation-01.yaml
---

# Synthetic probe augmentation edits and stores the full paper, not one page

## Session 2026-09-08

### Prompts

> I'd like to make a quick change to creation of the probe dataset: instead of
> using only page text, use the full paper text. When gpt-oss is given the text
> to edit, it should be the full paper text. Likewise, if we're storing the text
> corresponding to real or synthetic measurements, it should be the full paper
> text.

> yes, go ahead with the all-rows stamping

### Implemented

In the `--augment` path of `src/scholarlm/utils/probe_augment.py`, `run_and_write`
now attaches the full OCR paper as each record's context instead of the
measurement's page(s) via `extract_page_text` (removed, along with the unused
`page_numbers_key` parameter). gpt-oss is prompted with `## PAPER TEXT` and told
to change every occurrence throughout the paper; the rewrite cache key gains a
`context_scope: "full"` field so page-scoped entries are permanent misses.
`inline_context_overrides` now stamps `context_override` onto **every** output
row — edited rows their edited paper, unedited rows the source verbatim — so the
judge reads one consistent context scope across valid/invalid and
edited/unedited; `assert_wellformed` enforces it. `configs/2026-09-03-probe-synthetic-augmentation-01.yaml`
records `context_scope: full` and expands the stale-`prompt_budget_multiple` note
(full-paper context lowers positive yield and tightens the negative pools, so
rung 3 must re-derive it upward and the caches regenerate from scratch). Rung 1:
full suite 260 passed / 1 skipped (net +7 tests). Rung 2: `--augment-stub`
end-to-end on pond and supermat — every row carries the full multi-page paper,
primary-test rows verbatim, edits land deep in the document.

### Commits

- `5daefe8` probe augmentation: gpt-oss edits and stores the full paper, not one page
