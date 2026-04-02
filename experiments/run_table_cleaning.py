"""
Table-cleaning pipeline runner.

Cleans and normalizes tables in OCR-processed text files using either a local
vLLM model or the OpenAI API.  For each PDF/text pair the cleaner re-renders
pages that contain ``<table>`` tags using the PDF image, then rewrites the table
into a normalized, machine-readable format.

Results are written to:

    data/{dataset}/ocr_output_cleaned_{backend}_{model_tag}/

where ``model_tag`` is derived from the model name with ``/``, ``-``, and ``.``
replaced by ``_``.  After running, update ``ocr_dir`` in the relevant
``experiments/configs/<dataset>.py`` to point at this output directory before
running ``run_extraction.py``.

Usage
-----
    # Local vLLM model (GPU required):
    python experiments/run_table_cleaning.py \\
        --dataset pond --backend vllm --model gemma-3-27b

    # OpenAI API:
    python experiments/run_table_cleaning.py \\
        --dataset pond --backend openai --model gpt-4o-mini

    # Resume a partial run (skip files already in the output directory):
    python experiments/run_table_cleaning.py \\
        --dataset pond --backend openai --model gpt-4o-mini --resume

    # Override the input OCR directory:
    python experiments/run_table_cleaning.py \\
        --dataset pond --backend openai --model gpt-4o-mini \\
        --input-dir data/pond/ocr_output_raw

    # Process a subset of papers:
    python experiments/run_table_cleaning.py \\
        --dataset pond --backend openai --model gpt-4o-mini \\
        --paper-subset physical_and_chemical_limnological prairie_wetland

Available datasets : any file in experiments/configs/<name>.py that exports CONFIG.
vLLM model keys    : see MODEL_REGISTRY in this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm importable when run directly from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = Path(__file__).parent / "configs"
sys.path.insert(0, str(_REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv()

from scholarlm import TableCleaner
from scholarlm.config import DatasetConfig, ModelConfig
from scholarlm.utils import get_filenames_in_directory

# ---------------------------------------------------------------------------
# vLLM model registry (short name → ModelConfig)
# Add entries here to register models for the vllm backend.
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gemma-3-27b": ModelConfig(
        name="gemma-3-27b",
        model_id="gaunernst/gemma-3-27b-it-qat-autoawq",
        tensor_parallel_size=1,
        sampling_params={
            "temperature": 0.1,
            "max_tokens": 16384,
        },
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b",
        model_id="openai/gpt-oss-120b",
        tensor_parallel_size=2,
        sampling_params={
            "temperature": 0.1,
            "max_tokens": 16384,
        },
    ),
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_dataset_config(name: str) -> DatasetConfig:
    """Load a DatasetConfig by name from experiments/configs/<name>.py."""
    config_path = _CONFIGS_DIR / f"{name}.py"
    if not config_path.exists():
        available = sorted(p.stem for p in _CONFIGS_DIR.glob("*.py") if p.stem != "__init__")
        raise FileNotFoundError(
            f"No config found for dataset '{name}'. "
            f"Available datasets: {available}"
        )
    spec = importlib.util.spec_from_file_location(f"_dataset_config_{name}", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CONFIG"):
        raise AttributeError(
            f"Config file {config_path} must define a module-level 'CONFIG' variable "
            f"of type DatasetConfig."
        )
    return mod.CONFIG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_tag(model: str) -> str:
    """Return a filesystem-safe tag derived from the model name."""
    return model.replace("/", "_").replace("-", "_").replace(".", "_").lower()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_table_cleaning(
    dataset_config: DatasetConfig,
    backend: str,
    model: str,
    input_dir: str | None = None,
    output_dir: str | None = None,
    rate_limit: int = 100,
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
) -> None:
    """Clean tables in OCR text files for a dataset.

    Args:
        dataset_config: Dataset configuration loaded from experiments/configs/.
        backend: ``"vllm"`` for a local model or ``"openai"`` for the API.
        model: For vllm — a key from ``MODEL_REGISTRY`` (e.g. ``"gemma-3-27b"``).
            For openai — the API model name (e.g. ``"gpt-4o-mini"``).
        input_dir: Directory containing OCR ``.txt`` files.  Defaults to
            ``{data_dir}/ocr_output_raw/``.
        output_dir: Destination directory for cleaned ``.txt`` files.  Defaults
            to ``{data_dir}/ocr_output_cleaned_{backend}_{model_tag}/``.
        rate_limit: Max requests per minute (openai backend only).
        paper_subset_override: If provided, process only these paper codes
            (filename stems without .pdf).
        resume: If True, skip files whose output .txt already exists.
    """
    data_dir = Path(dataset_config.data_dir)
    pdf_dir = data_dir / "pdfs"

    ocr_input = Path(input_dir) if input_dir else data_dir / "ocr_output_raw"
    tag = f"{backend}_{_model_tag(model)}"
    ocr_output = Path(output_dir) if output_dir else data_dir / f"ocr_output_cleaned_{tag}"
    ocr_output.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset  : {dataset_config.name}")
    print(f"Backend  : {backend}")
    print(f"Model    : {model}")
    print(f"Input    : {ocr_input}")
    print(f"Output   : {ocr_output}\n")

    # Discover PDFs — drives the file list so we can pair PDFs with text files.
    pdf_files = get_filenames_in_directory(str(pdf_dir), ignore=[".DS_Store", ".gitkeep"])
    pdf_files = [f for f in pdf_files if f.endswith(".pdf")]
    pdf_files.sort()

    if paper_subset_override is not None:
        subset_set = set(paper_subset_override)
        pdf_files = [f for f in pdf_files if f.replace(".pdf", "") in subset_set]

    # Only keep PDFs that have an OCR text file in the input directory.
    available = [f for f in pdf_files if (ocr_input / f.replace(".pdf", ".txt")).exists()]
    missing = len(pdf_files) - len(available)
    if missing:
        print(f"Warning: {missing} PDF(s) have no OCR text in {ocr_input} — skipping.")
    pdf_files = available

    if resume:
        before = len(pdf_files)
        pdf_files = [
            f for f in pdf_files
            if not (ocr_output / f.replace(".pdf", ".txt")).exists()
        ]
        print(f"Resume: {before - len(pdf_files)} already done, {len(pdf_files)} remaining.")

    if not pdf_files:
        print("No files to process.")
        return

    print(f"Processing {len(pdf_files)} file(s)...")

    pdf_filepaths = [str(pdf_dir / f) for f in pdf_files]
    txt_in = [str(ocr_input / f.replace(".pdf", ".txt")) for f in pdf_files]
    txt_out = [str(ocr_output / f.replace(".pdf", ".txt")) for f in pdf_files]

    texts = []
    for p in txt_in:
        with open(p, "r", encoding="utf-8") as fh:
            texts.append(fh.read())

    if backend == "vllm":
        if model not in MODEL_REGISTRY:
            raise KeyError(
                f"Unknown vllm model '{model}'. "
                f"Available models: {sorted(MODEL_REGISTRY.keys())}"
            )
        model_config = MODEL_REGISTRY[model]
        cleaner = TableCleaner(
            backend="vllm",
            model_name=model_config.model_id,
            sampling_params=model_config.sampling_params,
            target_longest_dim=1536,
        )
    elif backend == "openai":
        cleaner = TableCleaner(
            backend="openai",
            openai_model=model,
            openai_rate_limit=rate_limit,
            sampling_params={"max_completion_tokens": 16384},
            target_longest_dim=1536,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Choose 'vllm' or 'openai'.")

    cleaned_texts = cleaner.clean(texts=texts, pdf_paths=pdf_filepaths)
    cleaner.save(cleaned_texts, txt_out)
    print(f"\nDone. Cleaned texts written to {ocr_output}")
    print(f"\nNext step: update ocr_dir in experiments/configs/{dataset_config.name}.py")
    print(f"  to point at: {ocr_output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean tables in OCR text files for a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match a file in experiments/configs/<name>.py).",
    )
    p.add_argument(
        "--backend",
        required=True,
        choices=["vllm", "openai"],
        help="Inference backend: 'vllm' (local GPU) or 'openai' (API).",
    )
    p.add_argument(
        "--model",
        required=True,
        help=(
            "For vllm: a key from MODEL_REGISTRY (e.g. 'gemma-3-27b'). "
            "For openai: the API model name (e.g. 'gpt-4o-mini')."
        ),
    )
    p.add_argument(
        "--input-dir",
        default=None,
        metavar="DIR",
        help="OCR input directory (default: data/{dataset}/ocr_output_raw/).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for cleaned texts "
            "(default: data/{dataset}/ocr_output_cleaned_{backend}_{model_tag}/)."
        ),
    )
    p.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        metavar="RPM",
        help="Max requests per minute (openai backend only; default: 100).",
    )
    p.add_argument(
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Process only these paper codes (filename stems without .pdf).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip files whose output .txt already exists in the output directory.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dataset_config = load_dataset_config(args.dataset)
    run_table_cleaning(
        dataset_config=dataset_config,
        backend=args.backend,
        model=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rate_limit=args.rate_limit,
        paper_subset_override=args.paper_subset,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
