# ScholarlM :microscope: :books:

[![attested by humans](https://github.com/kevinqnb/scholarlm/actions/workflows/signoff.yml/badge.svg)](https://github.com/kevinqnb/scholarlm/actions/workflows/signoff.yml)

Extract structured measurements from scientific PDFs using large language models.
Supports local open-weight models via vLLM.

Core capabilities:
- **Document OCR** — convert PDF pages to markdown text with HTML table extraction
- **Measurement extraction** — extract (entity, attribute, value) triplets from text and tables
- **Judgement** - Judge and assign confidence scores to extracted data

<div align="center">
  <img src="figures/extraction_flowchart.svg" alt="Extraction Flowchart" width="500">
</div>

## Installation

**Prerequisites:** Python 3.12+; GPU required for local inference (vLLM / transformers / nnsight).

```bash
git clone https://github.com/yourusername/scholarlm.git
cd scholarlm

uv sync --no-extra gpu   # CPU-only / API-backed workflows
uv sync                  # Full install including local GPU inference
```

Alternative (pip):
```bash
python -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install -e .          # CPU-only
pip install -e ".[gpu]"   # Full
```

## Data

While we do not directly share the PDF documents used for our experiments, they are documented by title, author, and identifying information 
for both the [pond (PLW)](data/pond/directory.json) and the [nitrogen fixation (NF)](data/nfix/directory.json) datasets. Please note the 
sources which these datasets originated from:
1. Richardson, David C., et al. "A functional definition to distinguish ponds from lakes and wetlands." Scientific reports 12.1 (2022): 10472.
2. Fulweiler, Robinson W., et al. "A global dataset of nitrogen fixation rates across inland and coastal waters." Limnology and oceanography letters 10.3 (2025): 412-429.


In addition, we share pre-processed reviewed datasets for [PLW](data/pond/ground_truth_review.json) and [NF](data/pond/ground_truth_review.json), which 
are used for comparison against our extracted data. In addition, we share a sample [extracted dataset](data/experiments/pond/extraction/gemma-3-27b/2026_05_05/final.json) from `gemma-3-27b`. 

## Prompts, Schemas, and Configs
The core set of [prompts](src/scholarlm/instruction_prompts.py) for all experiments is shared, as well as complete schemas for both [PLW](experiments/dataset-configs/pond.py) and [NF](experiments/dataset-configs/nfix.py)

In addition all LLM model information (including parameters and source repository names) are shared, one YAML file per model, under [experiments/model-configs/](experiments/model-configs/). 

## Experiments
Please see the [experiments](experiments/README.md) directory for the full workflow guide. The following are some quick examples.

```bash
# Extract
python experiments/run_extraction.py experiments/experiment-configs/pond/extraction/<id>/<id>.yaml

# Judge with a local model
python experiments/run_judge_local.py experiments/experiment-configs/pond/judge_local/<id>/<id>.yaml

# Judge and collect model activations (attention head & layer output)
python experiments/run_judge_interp.py experiments/experiment-configs/pond/judge_interp/<id>/<id>.yaml

# Or submit any of the above to SGE
bash experiments/submit.sh <id>
```

## Development log

Much of the experiment and build work in this repo is done with AI coding
assistance. [`devlog/`](devlog/) records it: one file per experiment/build id
(`YYYY-MM-DD-slug-NN`, the same id as the experiment's config), pairing the
human-written prompts that drove the work with a short summary of what landed and
the commits that carry it. Individual commits also carry a `Claude-Session:`
trailer. See [`devlog/README.md`](devlog/README.md) for the format.

## Examples
Please see the [demo](demo.ipynb) notebook for a look at how the extraction system operates. 

## License

MIT — see [LICENSE](LICENSE).
