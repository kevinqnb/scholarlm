"""
OCR pipeline runner.

Runs olmOCR on all PDF files for a dataset, writing plain-text output to:

    data/{dataset}/ocr_output_raw/

Use ``--resume`` to skip PDFs that already have a corresponding ``.txt`` file in
the output directory.

Usage
-----
    # From the repo root:
    python experiments/run_ocr.py --dataset pond
    python experiments/run_ocr.py --dataset nfix --resume
    python experiments/run_ocr.py --dataset pond \\
        --paper-subset physical_and_chemical_limnological prairie_wetland

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
"""
from __future__ import annotations

import argparse
import importlib.util
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm importable when run directly from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = Path(__file__).parent / "configs"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from scholarlm import DocumentLM
from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

# Reproducibility
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)

# ---------------------------------------------------------------------------
# OCR model
# ---------------------------------------------------------------------------

OCR_MODEL = "allenai/olmOCR-2-7B-1025-FP8"

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
# Main pipeline
# ---------------------------------------------------------------------------


def run_ocr(
    dataset_config: DatasetConfig,
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
) -> None:
    """Run olmOCR on all PDFs for a dataset.

    Args:
        dataset_config: Dataset configuration loaded from experiments/configs/.
        paper_subset_override: If provided, process only these paper codes
            (filename stems without .pdf).
        resume: If True, skip PDFs whose output .txt file already exists.
    """
    data_dir = Path(dataset_config.data_dir)
    pdf_dir = data_dir / "pdfs"
    output_dir = data_dir / "ocr_output_raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = get_filenames_in_directory(str(pdf_dir), ignore=[".DS_Store", ".gitkeep"])
    pdf_files = [f for f in pdf_files if f.endswith(".pdf")]
    pdf_files.sort()

    if paper_subset_override is not None:
        subset_set = set(paper_subset_override)
        pdf_files = [f for f in pdf_files if f.replace(".pdf", "") in subset_set]

    if resume:
        before = len(pdf_files)
        pdf_files = [
            f for f in pdf_files
            if not (output_dir / f.replace(".pdf", ".txt")).exists()
        ]
        print(f"Resume: {before - len(pdf_files)} already done, {len(pdf_files)} remaining.")

    if not pdf_files:
        print("No PDFs to process.")
        return

    print(f"\nDataset : {dataset_config.name}")
    print(f"Model   : {OCR_MODEL}")
    print(f"Input   : {pdf_dir}")
    print(f"Output  : {output_dir}")
    print(f"Papers  : {len(pdf_files)}\n")

    filepaths = [str(pdf_dir / f) for f in pdf_files]
    out_filepaths = [str(output_dir / f.replace(".pdf", ".txt")) for f in pdf_files]

    doclm = DocumentLM(
        model=OCR_MODEL,
        ocr_prompt=olmocr_prompt(),
        sampling_params={"temperature": 0.1, "max_tokens": 8192, "seed": 342},
    )

    doclm.fit(filepaths)
    doclm.save(out_filepaths)
    print(f"\nDone. OCR output written to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run olmOCR on all PDFs for a dataset.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match a file in experiments/configs/<name>.py).",
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
        help="Skip PDFs whose output .txt already exists in the output directory.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dataset_config = load_dataset_config(args.dataset)
    run_ocr(
        dataset_config=dataset_config,
        paper_subset_override=args.paper_subset,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
