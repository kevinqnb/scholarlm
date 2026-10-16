<!-- Public devlog entry: scholarlm/devlog/2026-10-03-match-cache-relocate-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-10-03-match-cache-relocate-01
kind: build
---

## Session 2026-10-03

### Prompts

- Before rebuilding the match caches, make sure they are stored under `analysis/results/match_cache/`: fix the saving mechanism first, then any loaders that pull matchings in.

### Implemented

`analysis/match_cache.py` gains `MATCH_CACHE_ROOT` (`analysis/results/match_cache`); `match_cache_path` and `match_cache_meta_path` now resolve to `MATCH_CACHE_ROOT/<id>/` and `build_match_cache` uses them instead of hand-building a path inside the run directory. The id-addressed loaders (`recovery_validity.py`, `calibration_ids.py`) already read through those helpers, so they follow with no change, and there is no fallback to the old per-run location. Tests patch the root and add checks that the default location is correct and that a cache left in the old location is never served. Rung 1 only; legacy date-based scripts under `data/experiments/` are untouched and no caches were rebuilt.

### Commits

- be3f295 match_cache: store caches under analysis/results/match_cache/<id>/, not the run directory
