"""
Legacy table-cleaning script using the OpenAI API.

Cleans and normalizes tables in OCR-processed text files using an OpenAI model.
For each PDF/text pair the cleaner re-renders pages that contain ``<table>`` tags
using the PDF image, then rewrites the table into a normalized, machine-readable
format.

For local (vLLM) table cleaning, use ``run_extraction.py`` without ``--ocr-dir``:
the extraction model cleans tables automatically before running extraction.

Results are written to:

    data/{dataset}/ocr_output_cleaned_openai_{model_tag}/

where ``model_tag`` is derived from the model name with ``/``, ``-``, and ``.``
replaced by ``_``.  The output directory can then be passed to
``run_extraction.py`` via ``--ocr-dir`` to skip the integrated cleaning step.

Usage
-----
    python experiments/run_table_cleaning.py \\
        --dataset pond --model gpt-4o-mini

    # Resume a partial run:
    python experiments/run_table_cleaning.py \\
        --dataset pond --model gpt-4o-mini --resume

    # Override the input OCR directory:
    python experiments/run_table_cleaning.py \\
        --dataset pond --model gpt-4o-mini \\
        --input-dir data/pond/ocr_output_raw

    # Process a subset of papers:
    python experiments/run_table_cleaning.py \\
        --dataset pond --model gpt-4o-mini \\
        --paper-subset physical_and_chemical_limnological prairie_wetland

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
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
from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory


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
    model: str,
    input_dir: str | None = None,
    output_dir: str | None = None,
    rate_limit: int = 100,
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
) -> None:
    """Clean tables in OCR text files using the OpenAI API.

    Args:
        dataset_config: Dataset configuration loaded from experiments/configs/.
        model: OpenAI model name (e.g. ``"gpt-4o-mini"``).
        input_dir: Directory containing OCR ``.txt`` files.  Defaults to
            ``{data_dir}/ocr_output_raw/``.
        output_dir: Destination directory for cleaned ``.txt`` files.  Defaults
            to ``{data_dir}/ocr_output_cleaned_openai_{model_tag}/``.
        rate_limit: Max requests per minute.
        paper_subset_override: If provided, process only these paper codes
            (filename stems without .pdf).
        resume: If True, skip files whose output .txt already exists.
    """
    data_dir = Path(dataset_config.data_dir)
    pdf_dir = data_dir / "pdfs"

    ocr_input = Path(input_dir) if input_dir else data_dir / "ocr_output_raw"
    tag = f"openai_{_model_tag(model)}"
    ocr_output = Path(output_dir) if output_dir else data_dir / f"ocr_output_cleaned_{tag}"
    ocr_output.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset  : {dataset_config.name}")
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

    cleaner = TableCleaner(
        backend="openai",
        openai_model=model,
        openai_rate_limit=rate_limit,
        sampling_params={"max_completion_tokens": 16384},
        target_longest_dim=1536,
    )

    cleaned_texts = cleaner.clean(texts=texts, pdf_paths=pdf_filepaths)
    cleaner.save(cleaned_texts, txt_out)
    print(f"\nDone. Cleaned texts written to {ocr_output}")
    print(f"\nTo use these cleaned texts for extraction, pass:")
    print(f"  --ocr-dir {ocr_output}")
    print(f"to run_extraction.py.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean tables in OCR text files using the OpenAI API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match a file in experiments/configs/<name>.py).",
    )
    p.add_argument(
        "--model",
        required=True,
        help="OpenAI model name (e.g. 'gpt-4o-mini', 'gpt-4o').",
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
            "(default: data/{dataset}/ocr_output_cleaned_openai_{model_tag}/)."
        ),
    )
    p.add_argument(
        "--rate-limit",
        type=int,
        default=100,
        metavar="RPM",
        help="Max requests per minute (default: 100).",
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
        model=args.model,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        rate_limit=args.rate_limit,
        paper_subset_override=args.paper_subset,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
