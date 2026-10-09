---
id: 2026-10-09-analysis-cleanup-01
kind: build
---

## Session 2026-10-09

### Prompts
- "Please help me undergo a large cleanup to the `analysis/` directory. Note that to kick things off, I created a backup copy to live in `older/analysis/`. This gives us some breathing room, but let's still be careful about doing this. Let's start by first examining all of the scripts directly in the parent `analysis/` directory. The aim is to find anything (a) not in use, which can be cleaned up, and (b) anything shared between scripts that can be centralized under a collection of analysis utils. Here are the main scripts I definitely want to keep, which may rely on other dependencies in the directory (that we may or may not want to keep): `calibration_latex.py`, `calibration_updated_v4.py` (which should get renamed to `calibration.py`), `calibration_validated.py`, `clustering.py` (which should get renamed to `pond_clustering.py`), `decision_threshold.py`, `deduplicate_cache.py`, `deduplication.py`, `match_cache.py`, `measeval_evaluation.py`, `meta_updated_v2.py` (which should get renamed to `pond_meta_analysis.py`), `platt_scaling_v2.py` (which should get renamed to `platt_scaling.py`), `postprocessing.py`, `recovery_validity_latex.py`, `recovery_validity.py`, `synthetic_probe_train.py`. Other stuff that is not in the list and is not a dependency for something in the list is an automatic candidate for removal."
- "That's another thing to be aware of: while I was building out these v2, v3 type scripts, I often re-used parts of them in their updated versions. That's something we want to clean up -- we don't want to keep an older script around JUST because its a dependency. If the relevant code in it can be taken out and centralized in a cleaner way, that's preferable."
- "(1) Defer, leave as is for now and make note that it needs to be fixed. (2) Same, defer. (3) Drop v3. (4) Delete." (calibration_latex and calibration_validated stay as they are for now; the meta analysis drops v3 calibration inputs; match_cache.sh is deleted.)
- "What about `calibration_ids.py`? That one seems like a strange mix of old and new code. Also I will mention: don't worry too much about making sure things are still runnable with old configs. I'm pretty much going to refresh the entire config directory later."
- Questions on how validity labels pair judgements with fuzzy-threshold matches, which extraction artifact each script reads, and whether reuse of cached calibration predictions is centralized.

### Implemented
`analysis/` now holds only config-driven entry points. Shared code moved to `analysis/common/` (`config`, `matching`, `recovery`, `provenance`, `dedup`, `pond_meta`, `calibration_ids`, …), and no entry point imports another. Superseded and unused scripts were removed (calibration v1–v3, the v1 Platt sweep, `meta_updated.py`, and others), along with dead loaders and config loaders. The keep-list scripts were renamed to `calibration.py`, `platt_scaling.py`, `pond_meta_analysis.py` and `pond_clustering.py`. Config section names and output directories are unchanged. This is a pure refactor with no eval or metric logic changed. Outputs recorded on real data before the moves were identical after every commit, and running six scripts on smoke configs with the pre-cleanup and the refactored code produced identical outputs. `analysis/README.md` documents the layout and two deferred known issues.

### Commits
- `b19e73d` analysis: remove superseded and unused scripts
- `14a3dd7` analysis: move shared modules into analysis/common/
- `aa7c05f` analysis: retire meta_updated.py; its shared half moves to common/pond_meta
- `91f07c7` analysis: move match-cache access out of match_cache.py into common/
- `809f1b9` analysis: move recovery/validity building blocks into common/recovery
- `1be05d5` analysis: move dedup paths and config loaders into common/dedup
- `08d9dba` analysis/common: drop dead loaders, dedupe copied helpers
- `a61d844` analysis: rename the current scripts; document the new layout
- `eb806ee` analysis/README: restore the judge_combine_ids fixture-verified caveat

## Session 2026-10-09 (config refresh)

### Prompts
- "Please help me continue the `analysis/` directory cleanup happening over the last few commits. We're going to focus on the configs in the `analysis-configs` directory. Essentially, I want to re-run all the experiments so I'm looking to create a totally fresh set of configs. These can be based off the old ones: there's already a lot of good material. But we need to be very concrete about parameter choices. In addition, this config directory needs to be more organized. It CANNOT live as a single list: configs should be separated by analysis *type* like they are in the `results/` directory. One thing I don't like about that current setup is that platt scaling analyses fall under the `calibration/` type. That should no longer be the case, they should get their own `platt-scaling/` type. Before writing anything we should go to the scripts that use these and make sure this is adjusted." The request then defined the dataset / extractor / baseline / ablation / judge sets, the 11 analyses to configure (setup, synthetic probes with Platt scaling on, main and ablation recovery-validity with latex tables, MeasEval, calibration, validated calibration, Platt scaling, decision threshold, pond meta analysis, pond clustering) with their parameters, asked whether NTP also uses Platt scaling when the flag is on, required the supermat v3s probe set wherever the probe split is used, and asked for ~2000 bootstrap/resample draws, flagging where that doesn't fit.
- Decisions: hyphenated type directories for configs and results; seed 0 everywhere; write the calibration-latex and calibration-validated configs now, best effort; "Run on recovery configs, but could we also configure deduplication to run on these as well? That way we really have one uniform setup config."
- "Good point -- it wasn't intended to and I should have left supermat out of the ablation datasets."
- On deduplication failing for baselines' scalar provenance fields: "Is this something that would be better placed into postprocessing?" then "I wouldn't fail for a mix of lists and non-lists. Parse every entry individually. Go ahead an implement it there."
- "Just to confirm -- the standard notation used by the full extraction pipeline has all provenance fields as lists and not just page numbers?"
- "Drop the resolve_job test, then commit with your proposed split. Then add a /devlog note."

### Implemented
Analysis configs now live in `analysis/analysis-configs/<type>/<id>.yaml`, with outputs in `analysis/results/<type>/`. Platt scaling and validated calibration have their own types, every loader rejects a config filed under the wrong type, and cross-config ids resolve through `common.config.analysis_config_path`. One recovery-validity config now drives postprocessing, match caching, both deduplication steps and recovery/validity. `analysis/postprocessing.py` gives every record list-valued provenance fields, so baselines can be deduplicated; no matching rule reads those fields. `common/calibration_ids.py` accepts the supermat `v3s` corpus version. The 116 old configs are replaced by 43 new ones dated 2026-10-09 across all nine types. All of them load and cross-check, except the latex configs held back for the pending `calibration_latex.py` update. Nothing has been run yet.

### Commits
- `7cc648a` calibration_ids: accept the supermat v3s corpus version
- `7b67b66` analysis: configs by analysis type; one recovery-validity config drives setup
- `1a7a841` postprocessing: give every record list-valued provenance fields
- `eb85920` analysis-configs: fresh 2026-10-09 config set for the full rerun

## Session 2026-10-09 (docstrings)

### Prompts
- "Let's work on updating the comments and docstrings everywhere across `analysis/`. These need to be much less dense and wordy -- you don't need to describe the history of the entire file. Just what it is, what it does, and why. All functions should have meaningful docstrings with a short description overall, as well as descriptions of each parameter and output item. Again: concise, specific, and clearly understandable for someone taking a glance. Don't update any code, just the comments and docstrings."

### Implemented
A comments-and-docstrings pass over every module in `analysis/` and `analysis/common/`. Module docstrings now say what each file is, what it does, why, and how to run it, without the change history. Every function, method and nested helper has a short summary plus Args, Returns and Raises. Docstrings that disagreed with the code were corrected, for example `loaders.cached_match`'s cache behaviour and `deduplicate_cache.compute_edges`'s return values. No code changed: each file's syntax tree with docstrings removed matches the previous commit. No configs were touched.

### Commits
- `8405b67` analysis: rewrite comments and docstrings for brevity; document every function
