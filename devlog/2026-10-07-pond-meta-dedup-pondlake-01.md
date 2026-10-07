<!-- Public devlog entry: scholarlm/devlog/2026-10-07-pond-meta-dedup-pondlake-01.md. Written by /devlog as a trimmed
version of the private build note at notes/scholarlm/builds/2026-10-07-pond-meta-dedup-pondlake-01.md — same shape,
minus anything sensitive and minus the session id. Committed to the code repo. -->

---
id: 2026-10-07-pond-meta-dedup-pondlake-01
kind: build
config: analysis/analysis-configs/2026-10-07-pond-meta-dedup-pondlake-01.yaml
---

## Session 2026-10-07

### Prompts
- "Are there any attributes for the pond ecosystem type that just have more document data points available, and are maybe better to test with?"
- "Ok let's first make this update to the analysis: for ponds and lakes lets isolate to surface area, max depth, ph, chla. Drop wetlands entirely"
- "run it directly here"
- "Commit these changes and start a /devlog note. Then work on 2."

### Implemented
The meta analysis is narrowed to the ecosystem/attribute cells with the broadest ground-truth document support. `analysis/meta_updated.py` now analyses only the cells named by three new required `params.meta` keys in `analysis/meta_inputs.py`: `ecosystems`, `attributes`, and an explicit `poster` flag. Bootstrap streams stay keyed on the canonical cell lists, so subsetting leaves every retained cell's statistics and CIs unchanged; only the single-draw `w2_shuffled` control moves, and that is now documented. `analysis/analysis-configs/2026-10-07-pond-meta-dedup-pondlake-01.yaml` (plus `-smoke-01`) covers pond and lake × surface_area, max_depth, ph, chla, with wetland dropped. The existing `2026-10-06-pond-meta-*` configs carry the full scope explicitly. Unit tests pass, and the smoke and full runs reproduce the corresponding full-scope runs exactly on every retained cell.

### Commits
4706ea0 meta_updated: config-selected ecosystem/attribute cells, explicit poster flag
