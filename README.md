# ScholarlM :microscope: :books:

Extract structured (entity, attribute, value) triplets from scientific PDFs using large language models.
Supports API-backed models (Anthropic, OpenAI, Gemini) and local open-source models via vLLM.

Core capabilities:
- **Document OCR** — convert PDF pages to markdown text with HTML table extraction
- **Measurement extraction** — extract (entity, attribute, value) triplets from text and tables
- **Hallucination detection** *(experimental)* — mechanistic intervention on model activations

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

## Usage

### Experiments
Please see the [experiments](experiments/README.md) directory for the full workflow guide. The following are some quick examples.

```bash
# Extract
python experiments/run_extraction.py --dataset pond --model gemma-3-27b

# Judge with a local model
python experiments/run_judge_local.py \
        --dataset pond  --extraction-model gemma-3-27b \
        --judge gpt-oss-120b --api-base http://localhost:{PORT}/v1

# Judge and collect model activations (attention head & layer output) 
python experiments/run_judge_interp.py \
    --dataset pond --extraction-model gemma-3-27b \
    --judge llama-3.1-8b --extraction-date 2026_04_01
```

### Analysis Notebooks
- `analysis/extraction_analysis.ipynb` — recovery rate and hallucination for one extraction run
- `analysis/probe_analysis.ipynb` — probe accuracy, calibration, greedy head selection


## License

MIT — see [LICENSE](LICENSE).
