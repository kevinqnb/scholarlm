---
id: 2026-10-20-representation-lm-bidirectional-01
kind: build
---

# RepresentationLM: bidirectional models, OCR cleaning, paragraph windows, mean pooling

## Session 2026-10-20

### Prompts

> Please help me build upon the current `src/scholarlm/representationlm.py` module. I want to include robust support for a few new things: (1) bidirectional language models such as SciBERT or an LLM2Vec model (test with `allenai/scibert_scivocab_uncased`); (2) text cleaning: remove references, tables and the manually added `<page></page>` tags, by default; (3) optional context windows: for a term in paragraph x, pass only paragraphs x-1, x, x+1; (4) default to the mean representation over all tokens of the term instead of the last token.

> Use ModernBERT instead: `answerdotai/ModernBERT-large`. Investigate whether it works with the current Nnsight setup, or run it with HF transformers if efficiency doesn't suffer. For a bidirectional model without windowing: always raise on over-length input.

> Propose a reasonable configuration for ModernBERT under `experiments/model-configs`, smoke-test it on pond with the term "pond", and run it through `submit.sh`. Then resubmit with a new id, run a Llama causal smoke, and run rung 3 with ModernBERT.

> Tighten up the comments and docstrings: clear, concise, specific; no historical changes; every function documents all inputs and outputs.

### Implemented

`RepresentationLM` now takes a required `architecture` (`causal` via nnterp, `bidirectional` via HF `AutoModel`, restricted to ModernBERT), cleans OCR text (tables, references, page tags, with a kept-fraction guard on the reference cut), optionally cuts +/-r paragraph windows, and pools the term's tokens by `mean` (default) or `last`. Over-length input raises. `verify_read_point` gained a future-dependence check that separates bidirectional from causal attention. The runner requires every new param; added `experiments/model-configs/representation_lm/modernbert-large.yaml`, `architecture: causal` on the Llama model-config, and five experiment configs under `experiments/experiment-configs/pond/representation_lm/` (two ModernBERT smokes, one Llama causal smoke, two rung-3 runs). Rungs 1-3 passed for ModernBERT and rung 2 for the Llama causal path (prediction-matched row counts, shuffled-label control at chance); the old-vs-new Llama regression was skipped and no full run has been made. The two earlier representation-lm configs now fail loud until given the new keys.

### Commits

- `181335e` representation_lm: bidirectional backend, OCR cleaning, paragraph windows, mean pooling
