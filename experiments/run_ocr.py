"""
OCR pipeline runner.

Runs olmOCR on all PDF files for a dataset by calling a running vLLM server
(OpenAI-compatible API), writing plain-text output to:

    data/{dataset}/ocr_output_raw/

Start the OCR model server first:
    qsub experiments/serve_olmocr.sh

Use ``--resume`` to skip PDFs that already have a corresponding ``.txt`` file
in the output directory.

Usage
-----
    # From the repo root:
    python experiments/run_ocr.py --dataset pond
    python experiments/run_ocr.py --dataset nfix --resume
    python experiments/run_ocr.py --dataset pond \\
        --paper-subset physical_and_chemical_limnological prairie_wetland

    # Point at a non-default server:
    python experiments/run_ocr.py --dataset pond \\
        --api-base http://node042:8081/v1

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm and utils importable when run from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm import DocumentLM
from scholarlm.utils import get_filenames_in_directory
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

from run_extraction import load_dataset_config
from utils import load_config, set_seeds, write_run_metadata


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_ocr(
    dataset_config,
    model_id: str,
    sampling_params: dict,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
    processed_pdfs_dir: str | None = None,
) -> None:
    """Run olmOCR on all PDFs for a dataset via a vLLM server.

    Args:
        dataset_config: Dataset configuration loaded from experiments/configs/.
        model_id: HuggingFace model ID string (must match what the server is serving).
        sampling_params: Sampling parameters forwarded to DocumentLM.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the server (use "EMPTY" for local vLLM).
        paper_subset_override: If provided, process only these paper codes
            (filename stems without .pdf).
        resume: If True, skip PDFs whose output .txt file already exists.
        processed_pdfs_dir: If provided, load pre-rendered page images from this
            directory instead of rendering PDFs at runtime.  The expected layout
            is ``{processed_pdfs_dir}/{paper_code}/{page_index}.b64``, which
            matches the output of ``experiments/process_pdfs.py``.
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
    print(f"Model   : {model_id}")
    print(f"Server  : {api_base}")
    print(f"Input   : {processed_pdfs_dir if processed_pdfs_dir else pdf_dir}")
    print(f"Output  : {output_dir}")
    print(f"Papers  : {len(pdf_files)}\n")

    filepaths = [str(pdf_dir / f) for f in pdf_files]
    out_filepaths = [str(output_dir / f.replace(".pdf", ".txt")) for f in pdf_files]

    doclm = DocumentLM(
        model_name=model_id,
        ocr_prompt=olmocr_prompt(),
        sampling_params=sampling_params,
        api_base=api_base,
        api_key=api_key,
    )

    start_time = time.time()
    doclm.fit(filepaths, processed_pdfs_dir=processed_pdfs_dir)
    doclm.save(out_filepaths)
    print(f"\nDone. OCR output written to {output_dir}")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        model_id=model_id,
        api_base=api_base,
        papers_processed=len(pdf_files),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run olmOCR on all PDFs for a dataset via a vLLM server.",
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
    p.add_argument(
        "--use-processed-pdfs",
        action="store_true",
        help=(
            "Load pre-rendered page images from data/{dataset}/processed_pdfs/ "
            "(produced by process_pdfs.py) instead of rendering PDFs at runtime."
        ),
    )
    p.add_argument(
        "--processed-pdfs-dir",
        default=None,
        metavar="DIR",
        help=(
            "Explicit path to a pre-processed PDFs directory.  "
            "Overrides --use-processed-pdfs when set."
        ),
    )
    p.add_argument(
        "--api-base",
        default="http://localhost:8081/v1",
        metavar="URL",
        help="Base URL of the vLLM server (default: http://localhost:8081/v1).",
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        metavar="KEY",
        help="API key for the vLLM server (default: EMPTY).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    cfg = load_config()
    seed = cfg.get("defaults", {}).get("seed", 342)
    set_seeds(seed)

    ocr_cfg = cfg.get("models", {}).get("olmocr", {})
    model_id = ocr_cfg.get("model_id", "allenai/olmOCR-7B-0225-preview")
    sampling_params = ocr_cfg.get("sampling_params", {"temperature": 0.1, "max_tokens": 8192, "seed": 342})

    dataset_config = load_dataset_config(args.dataset)

    processed_pdfs_dir = None
    if args.processed_pdfs_dir:
        processed_pdfs_dir = args.processed_pdfs_dir
    elif args.use_processed_pdfs:
        processed_pdfs_dir = str(Path(dataset_config.data_dir) / "processed_pdfs")

    run_ocr(
        dataset_config=dataset_config,
        model_id=model_id,
        sampling_params=sampling_params,
        api_base=args.api_base,
        api_key=args.api_key,
        paper_subset_override=args.paper_subset,
        resume=args.resume,
        processed_pdfs_dir=processed_pdfs_dir,
    )


if __name__ == "__main__":
    main()
