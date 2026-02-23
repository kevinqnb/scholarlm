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
Interactive walkthroughs are in `examples/`:
- `examples/ocr.ipynb` — OCR a PDF to markdown
- `examples/pond_lake_extraction.ipynb` — end-to-end measurement extraction pipeline

### Experiments (scripts)
Runnable experiment scripts are in `experiments/`:
- `experiments/pond/ocr.py` — OCR a batch of PDFs
- `experiments/pond/pond.py` — run the measurement extraction pipeline
- `experiments/pond/judge_llama.py` — validate extracted measurements with a local LLM judge
- `experiments/pond/validate.py` — evaluate extraction results against a ground-truth dataset

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
