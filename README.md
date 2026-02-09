# ScholarLM :microscope: :books:

**Extract data from scientific research papers using local, large language models.**

*Please note:* This project is a work in progress.

This library implements a system for extracting data from scientific papers (PDFs) using large language models.
We apply local and open source LLMs towards organized tasks for:
* Document OCR: translating PDF images into markdown/text and splitting into paragraph-sized chunks.
* Document extraction: systematically collecting data points from chunks of text.
* (Experimental) Hallucination detection: mechanistic intervention on model activations to detect and reduce hallucinated responses.

Our focus is on using small, local models for OCR and text generation tasks, and this library is designed to be compatible with Hugging Face Transformers / vLLM-style models.

## Installation

### Prerequisites
- **Python**: 3.10+ recommended.
- **GPU (recommended)**: Most workflows are intended for CUDA GPUs (see `experiments/ocr.py`, `experiments/pond.py`, `experiments/judge_llama.py`).

### Install (recommended: `uv`)
To get started, install dependencies with the `uv` package manager. If you do not already have `uv` installed, follow the instructions here:
https://docs.astral.sh/uv/getting-started/installation/

```bash
# Clone the repository
git clone https://github.com/yourusername/scholarlm.git
cd scholarlm

# Create/resolve an environment and install dependencies
uv sync
```

### Install (alternative: pip + venv)
If you prefer a standard virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Basic Usage (high-level)
- **OCR PDFs → text**: see `experiments/ocr.py`
- **Extract measurements**: see `experiments/pond.py`
- **Judge/validate extracted points**: see `experiments/judge_llama.py` and `experiments/validate.py`

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
