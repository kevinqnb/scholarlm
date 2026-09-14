<!-- Public devlog entry: scholarlm/devlog/2026-09-13-experiment-config-restructure-01.md.
Written by /devlog as a trimmed version of the private build note at
notes/scholarlm/builds/2026-09-13-experiment-config-restructure-01.md -->

---
id: 2026-09-13-experiment-config-restructure-01
kind: build
---

## Session 2026-09-13

### Prompts

Kickoff: "Our goal is to clean up the current experiment framework, which
has become too messy to be nicely reproducible," with a target layout
(`dataset-configs/`, `experiment-configs/{dataset}/{type}/{id}/`,
`model-configs/`) sketched directly in the request. Plan-mode
clarifications that shaped the approved plan: `out/` is SGE stdout/stderr
only, heavy pipeline output moves to `experiments/results/{dataset}/{type}/
{id}/` addressed by explicit experiment-id rather than "most recent date
wins"; extraction-result migration must use the exact dates
`analysis/calibration_updated.py`/`analysis/ablation.py` already key off,
not "most recent on disk"; `experiments/results/`/`out/` stay
gitignored with no per-file exceptions; every runner takes one config
argument instead of individual CLI flags ("wouldn't it just be cleaner to
have every runner take an input config, rather than doing a whole override
process?"); `run_ocr.py`/`run_table_cleaning.py` output moves under
`experiments/results/` too, so it's tied to an experiment id. Phase
transitions ("let's finish out Phase A" / "let's move into Phase B now" /
"go ahead to stage C") were explicit checkpoints, each confirmed before
proceeding; one migration-sizing bug was caught by the user directly ("How
are there 63 directories? ... I would expect 18") before being resolved
(63 is correct: extraction + 6 ablations x 3 models). Phase C was
explicitly scoped as an infrastructure acceptance gate, not a scientific
experiment ("No explog. We should not be running any experiments here.").
This session's own prompt: recover and confirm Phase C's finish (the
driving chat context had been compacted away) and close out the build
record, including more of the original plan's own language in this
section.

### Implemented

Retrofitted this repo's 14 `run_*.py` entry points onto the harness's
config-path-CLI / id-addressing contract. New layout: `model-configs/`
(one YAML per model, `serve:` split from `resources:`), `dataset-configs/`
(renamed from `experiments/configs/`), `experiment-configs/{dataset}/
{type}/{id}/{id}.yaml` (relocated from the 14 root `configs/<id>.yaml`),
and `experiments/results/{dataset}/{type}/{id}/` as the new id-addressed
output tree (`data/experiments/` stays frozen). `experiments/paths.py`
merged into `utils.py` with new `result_dir`/`find_result_dir`/
`load_model_config`/`classify_gpu_need` helpers; `experiments/submit.sh` +
`_resolve_job.py` + `_submit_job.sh` replace the old fixed-table dispatch
with per-model resourcing. All 14 runners converted to a single `config`
positional arg; the judge pipeline (`run_judge_interp.py`/
`run_judge_local.py`/`run_judge_combine.py`) redesigned around explicit
`extraction_id`/`judge_ids` rather than directory-scanning lookup, since
the old lookup didn't work against new-framework extraction output at all.
Phase C added 5 acceptance-gate configs (3 `judge_local` + 1
`judge_interp` + 1 `judge_combine`) against a real Phase-B-migrated
extraction, to exercise the new judge pipeline end-to-end. Full test suite
green throughout (343 -> 356 passed, 1 skipped, unrelated).

Phase C's ladder, recovered this session from the raw (un-compacted)
session transcript: a plumbing-only smoke check ran `run_judge_combine.py`
directly against real historical judge data staged under new-framework
ids and came back byte-identical (sha256-verified) to the pre-restructure
baseline, confirming the combine logic itself. The four real judge jobs
then ran for real (after fixing a genuine `_submit_job.sh` bug this
surfaced, see below) and all four succeeded per cluster job accounting;
partial diff results were recorded for two of them (gpt-oss-120b:
small, already-anticipated token-count discrepancy; qwen-2.5-72b: exact
match) before the session's finish. The real `judge_combine` step against
these four actual outputs — the step that would tie the acceptance gate
together — was never executed, and its would-be output directories don't
survive on disk. Phase D (deleting now-dead `configs/`,
`scripts/run_experiment.py`, `gen_serve_script.py`, `serve_*.sh`) is not
yet started.

### Commits

402a022 Relocate experiments/configs/ to experiments/dataset-configs/
cbc60f2 Split model_registry.py + config.yaml into experiments/model-configs/
4a26430 gitignore: cover experiments/results/ and the deeper out/ nesting
8362875 gitignore: no exceptions for experiments/results/, confirmed with user
3bf0c8e Move judge_common.py into src/scholarlm/utils/judge_prompts.py; delete validation.py
4382518 Update judge_prompts/validation.py call sites (missed by the previous commit)
787b004 Merge paths.py into utils.py; add id-based addressing + model-config loading
73736f8 Add experiments/submit.sh: general-purpose, dispatches by experiment-type
9718d6f Convert run_judge_combine.py + process_pdfs.py to config-path-only CLI
6a9b8cf Convert run_extraction.py to config-path CLI + model-configs/ lookup
7514077 Convert run_ablation.py + run_table_cleaning.py to config-path CLI
a989145 Convert the 3 baseline runners to config-path CLI
33bb677 Redesign the judge pipeline around explicit experiment-id references
fae6576 Convert the NNsight family (jacobian_lens, representation_lm, attribution)
fd34d89 Convert run_ocr.py to config-path CLI -- last of the 14 runners
8a38d75 Add run_probe_augment.py: last piece of Phase A
40fd878 Phase B: relocate the 14 root configs/ into experiment-configs/
097e517 Fix resolve_job() model-param mismatch across experiment types
680b45f Fix submit.sh/_submit_job.sh path + env bugs hit by first real GPU job
c678d73 Add Phase C acceptance-gate configs: pond ablation1/gpt-oss-120b judges

## Session 2026-09-14

### Prompts

Kickoff: "Phase D of
devlog/2026-09-13-experiment-config-restructure-01.md was never completed.
Thus, we still have a lot of leftover junk that we should clean up. Please
help me work through this." Two scoping decisions along the way: update
`CLAUDE.md` and the other stale docs as part of the cleanup ("Yes, update
CLAUDE.md + README.md + experiments/README.md + devlog/_TEMPLATE.md"), and
leave `demo.ipynb`'s now-broken `serve_olmocr.sh` cell alone for now. On an
unprompted finding (an in-flight config edit was landing in a dead file):
"Port fix to the live file now."

### Implemented

Deleted the confirmed-dead pre-restructure scaffolding -- `scripts/` in
full, `experiments/gen_serve_script.py`, all `experiments/serve_*.sh`,
`experiments/gen_augment.sh`, the empty `configs/` -- after confirming no
live imports anywhere. Rewrote `CLAUDE.md`'s Entry points, Repo layout, Key
concepts, Output directory schema, and Adding a new dataset/model sections,
plus `README.md`, `experiments/README.md`, and `devlog/_TEMPLATE.md`, to
match the contract Phases A-C actually built rather than the one that
predated them. Fixed a live bug found along the way: qwen-2.5-72b's judge
YaRN override was being edited in `experiments/config.yaml`, which nothing
reads anymore -- ported the validated fix into
`experiments/model-configs/vllm_judge/qwen-2.5-72b.yaml`, the file the real
serving path actually resolves. Full suite green except one pre-existing,
unrelated failure.

### Commits

898d43b Phase D: delete dead pre-restructure scaffolding, catch docs up to the new contract
276ee4f Fix qwen-2.5-72b judge YaRN override: factor 2.0/65536 -> factor 3.0/98304

### Prompts (continued, same session)

"What about the .vllm_endpoint files that are still hanging? Do we need
these now that models and experiments are run within the same job?" ->
"Yes, delete them." "Why do we still need @experiments/config.yaml?", then
"instead of comments saying what this file used to do, explain what that
seed is used for!!" on a first draft, then "Do it now." Then: "Keep going
on the Phase E list" -- covering both remaining items in one pass, with no
further direction given.

### Implemented (continued)

Deleted 21 leftover `.vllm_endpoint_*.txt` files (gitignored, unread by
anything since `serve_*.sh`/`gen_serve_script.py` were removed). Gutted
`experiments/config.yaml` to just `defaults.seed` after confirming every
`load_config()` call site only reads that one field -- the file's
`interp_judges` block turned out to be a stale, incomplete third copy of
`model_registry.py`'s live registry, not a duplicate of it. Finished
migrating the judge/interpretability runner family (`run_judge_local.py`,
`run_judge_interp.py`, `run_jacobian_lens.py`, `run_representation_lm.py`,
`run_attribution.py`) onto `experiments/model-configs/`, after verifying
field-for-field that the YAML files already matched their registry
counterparts and confirming (by reading the installed transformers source)
that the `nnsight_kwargs.torch_dtype` string-vs-object type change this
causes is handled safely. Deleted `experiments/model_registry.py` outright.
A `utils.resolve_job()` sweep across every committed experiment-config
surfaced the real scope of a pre-existing gap -- two model-configs missing
`resources:` blocks, blocking 26 committed configs combined -- which this
session flagged rather than fixed, since the values need inferring from
real prior runs, not inventing.

### Commits (continued)

9a91280 Gut experiments/config.yaml to just defaults.seed
1249633 CLAUDE.md: update config.yaml description now that the dead blocks are gone
cbb152e Finish the judge/interpretability family's migration onto model-configs/
