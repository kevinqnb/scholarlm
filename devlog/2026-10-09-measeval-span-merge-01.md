---
id: 2026-10-09-measeval-span-merge-01
kind: build
---

## Session 2026-10-09

### Prompts
- "Please help me audit and review the @analysis/measeval_evaluation.py script. Walk me through the outline of the code and point to specific lines to review."
- (Follow-up, summarized) Asked why duplicate spans affect recall, and whether changing span matching would stay faithful to the MeasEval benchmark; accepted the leftmost tie-break as a cost of evaluating a model that doesn't output spans.
- "Ok understood. I think we should prioritize working on (1). Let's try this out by dropping duplicate spans, and keeping the entity shared by most of the duplicated items. Run an initial diagnostic here for llama, gemma, gpt oss, and langextract"
- "Implement it in build_submission_tsv with unit tests. I think it would also be helpful to make "majority" a little more robust -- there might be slight variations in entity names that are really talking about the same thing. In that case, all those slight variations should count as a vote towards a common entity. You can do this by computing a fuzzy string similarity score across all pairs of entities. The entity with the largest sum of pairwise similarities should be the one we use."

### Implemented
`analysis/measeval_evaluation.py`'s `build_submission_tsv` now writes one Quantity row per distinct span. Before, every extraction record that resolved to the same offsets got its own row, and the official MeasEval scorer counts each repeated row as another true positive. Merged records take the entity whose text has the largest summed fuzzy similarity (`rapidfuzz` `fuzz.ratio`, as in `scholarlm.utils.deduplication`) to the others, and the most common unit; ties go to the earliest record, and merges, conflicts and ties are counted in `Coverage`. This is an eval-logic change that invalidates earlier measeval official-scorer numbers; no config changed. Rung 1 (unit tests in `tests/test_measeval_evaluation.py`) and an in-shell rung-3 check against a prior diagnostic passed; the full `measeval-official-eval` config has not been run.

### Commits
- `1f10ad8` measeval_evaluation: one Quantity row per distinct span
