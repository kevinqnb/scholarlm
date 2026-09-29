<!-- Public devlog entry: scholarlm/devlog/2026-09-29-nfix-ablation1-llama8b-reppen-cap100-noqualifiers-full-01.md.
Written by /devlog as a trimmed version of the private build note. -->

---
id: 2026-09-29-nfix-ablation1-llama8b-reppen-cap100-noqualifiers-full-01
kind: build
config: experiments/experiment-configs/nfix/ablation/2026-09-29-nfix-ablation1-llama8b-reppen-cap100-noqualifiers-full-01/2026-09-29-nfix-ablation1-llama8b-reppen-cap100-noqualifiers-full-01.yaml
---

## Session 2026-09-29

### Prompts

- Diagnose why the nfix ablation-1 run with llama-3.1-8b (repetition penalty, no qualifiers) stalls and makes no progress, and find fixes -- "even if it means we'll need to skip a paper or two".
- Try higher repetition penalties (1.25, 1.3, 1.4), including on non-looping papers to check for degradation; try frequency / presence penalties.
- Explain a `maxItems` cap and try it.
- "Go for option 1, but keep the repetition penalty at 1.1. Write new configs using the cap parameter for both pond ablation 1 with llama reppen, and nfix ablation 1 with llama reppen." Check the still-running job; if stalled, delete it and submit the new config.
- Delete the stalled job, submit the two new jobs, log the experiment, commit the code and write the devlog.

### Implemented

Added an opt-in `params.max_items` to Ablation 1 (`experiments/run_ablation.py`, `src/scholarlm/measurementlm_ablation1.py`): a JSON-Schema `maxItems` on the response's `items` list, so a document can no longer generate an unbounded record list. Absent, behaviour is unchanged. `run_metadata.json` now records `max_items`, `capped_papers` (documents that reached the cap, i.e. truncated or padded) and `failed_papers` (responses that failed validation and contributed no records). Unit tests are in `tests/test_ablation1_max_items.py`; the staged ladder ran unit tests, a smoke run and a tiny end-to-end on each dataset before submission. New configs: `experiments/experiment-configs/{nfix,pond}/ablation/2026-09-29-{nfix,pond}-ablation1-llama8b-reppen-cap100-noqualifiers-{tinye2e,full}-01/`.

### Commits

- e3cff20 Ablation 1: optional max_items cap (JSON-Schema maxItems) with capped/failed paper bookkeeping
