# ScholarLM :microscope: :books:

**Extract structured data from scientific research papers using large language models.**

*Please note:* This project is a work in progress.

This library implements a pipeline for extracting data from scientific papers (PDFs) using large language models.
It supports both API-backed models (Anthropic, OpenAI, Google Gemini) and local open-source models (via vLLM / HuggingFace Transformers).

Core capabilities:
* **Document OCR**: convert PDF pages to markdown text, with HTML table extraction.
* **Measurement extraction**: systematically collect (entity, attribute, value) triplets from document text and tables.
* **Table cleaning**: normalize and restructure OCR-extracted HTML tables using a VLM.
* **Hallucination detection** *(experimental)*: mechanistic intervention on model activations to detect and reduce hallucinated responses.

## Installation

### Prerequisites
- **Python**: 3.12+
- **GPU** *(optional)*: required for local model inference with vLLM, `transformers`, or `nnsight`. API-backed workflows run on CPU.

### Install (recommended: `uv`)

```bash
# Clone the repository
git clone https://github.com/yourusername/scholarlm.git
cd scholarlm

# CPU-only / API-backed workflows
uv sync --no-extra gpu

# Full install including local GPU inference
uv sync
```

### Install (alternative: pip + venv)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# CPU-only / API-backed workflows
pip install -e .

# Full install including local GPU inference
pip install -e ".[gpu]"
```

## Usage

### Examples (notebooks)
Interactive walkthroughs are in `examples/` and `analysis/`:
- `examples/ocr.ipynb` — OCR a PDF to markdown
- `analysis/extraction_analysis.ipynb` — recovery rate and hallucination for one extraction run
- `analysis/probe_analysis.ipynb` — probe accuracy, calibration, greedy head selection

### Experiments (scripts)
Runnable experiment scripts are in `experiments/`. See [EXPERIMENTS.md](EXPERIMENTS.md) for the full guide.

```bash
# Extract
python experiments/run_extraction.py --dataset pond_ten --model gemma-3-27b

# Judge (frontier, async) + combine
python experiments/run_judge_frontier_v2.py \
    --dataset pond_ten --extraction-model gemma-3-27b \
    --judge openai --frontier-model gpt-4o-mini --extraction-date 2026_04_14
python experiments/run_judge_combine.py \
    --dataset pond_ten --extraction-model gemma-3-27b --extraction-date 2026_04_14
```

### Ground truth preprocessing
Each dataset ships a preprocessing script that builds the ground truth CSVs:

```bash
python data/pond/preprocessing.py   # → data/pond/ground_truth{_ten}.csv
python data/nfix/preprocessing.py   # → data/nfix/ground_truth{_ten}.csv
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
