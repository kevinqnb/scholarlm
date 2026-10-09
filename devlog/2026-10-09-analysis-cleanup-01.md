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
