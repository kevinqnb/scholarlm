"""
Table-cleaning script using a vLLM server (OpenAI-compatible API).

Cleans and normalizes tables in OCR-processed text files using a local
open-source model served by vLLM.  For each page that contains ``<table>``
tags the model is shown the pre-rendered page image and asked to correct
the table markup.  Pages without tables are returned unchanged.

The vLLM server must be started separately before running this script.
See the vLLM startup examples below.

Results are written to:

    data/{dataset}/ocr_output_cleaned_{model_name}/

where ``model_name`` is the short key from the model registry (e.g.
``gemma-3-27b``).  This directory can then be passed to ``run_extraction.py``
via ``--ocr-dir`` to skip the integrated cleaning step.

Prerequisites
-------------
1. Run ``process_pdfs.py`` first (preprocessing environment) to produce
   pre-rendered page images at ``data/{dataset}/processed_pdfs/``.

2. Start a vLLM server serving the chosen model, e.g.:

       vllm serve gaunernst/gemma-3-27b-it-qat-autoawq \\
           --tensor-parallel-size 1 --port 8000

   Wait for "Application startup complete" before running this script.

Usage
-----
    python experiments/run_vllm_table_cleaning.py \\
        --dataset pond --model gemma-3-27b

    # Resume a partial run (skip papers whose output file already exists):
    python experiments/run_vllm_table_cleaning.py \\
        --dataset pond --model gemma-3-27b --resume

    # Custom server URL:
    python experiments/run_vllm_table_cleaning.py \\
        --dataset pond --model gemma-3-27b \\
        --api-base http://gpu-node-01:8000/v1

    # Custom input/output directories:
    python experiments/run_vllm_table_cleaning.py \\
        --dataset pond --model gemma-3-27b \\
        --ocr-dir data/pond/ocr_output_raw \\
        --output-dir data/pond/ocr_output_cleaned_gemma-3-27b

    # Process a subset of papers:
    python experiments/run_vllm_table_cleaning.py \\
        --dataset pond --model gemma-3-27b \\
        --paper-subset paper_a paper_b

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
Available models:   keys of MODEL_REGISTRY in experiments/run_extraction.py.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm importable when run directly from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

# Import shared registry and helpers from run_extraction to keep model list
# and config loading in sync.
from run_extraction import MODEL_REGISTRY, load_dataset_config, get_model_config, load_papers
from scholarlm import MeasurementLM
from scholarlm.config import DatasetConfig, ModelConfig

random.seed(342)


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------


def run_vllm_table_cleaning(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    ocr_dir: str | None = None,
    output_dir: str | None = None,
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
) -> None:
    """Clean tables in OCR text files using a vLLM-served model.

    Loads raw OCR text, passes each page with tables through the model
    (alongside its pre-rendered page image), and writes cleaned texts to
    ``output_dir``.

    Args:
        dataset_config: Dataset configuration loaded from ``experiments/configs/``.
        model_config: Model configuration from ``MODEL_REGISTRY``.
        ocr_dir: Input directory of ``.txt`` OCR files.  Defaults to
            ``{data_dir}/ocr_output_raw/``.
        output_dir: Destination directory for cleaned ``.txt`` files.  Defaults
            to ``{data_dir}/ocr_output_cleaned_{model_name}/``.
        paper_subset_override: If provided, process only these paper codes.
        resume: If ``True``, skip papers whose output ``.txt`` already exists.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the vLLM server (any non-empty string works).
    """
    data_dir = Path(dataset_config.data_dir)
    effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")
    effective_output_dir = Path(output_dir) if output_dir else data_dir / f"ocr_output_cleaned_{model_config.name}"

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"OCR input : {effective_ocr_dir}")
    print(f"Output    : {effective_output_dir}")
    print(f"API base  : {api_base}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} papers.")

    if resume:
        pending_text = []
        pending_info = []
        for t, info in zip(text, text_info):
            out_file = effective_output_dir / f"{info['paper_code']}.txt"
            if out_file.exists():
                print(f"  Skipping {info['paper_code']} (already cleaned).")
            else:
                pending_text.append(t)
                pending_info.append(info)
        skipped = len(text) - len(pending_text)
        print(f"Resume: {skipped} already done, {len(pending_text)} remaining.\n")
        text = pending_text
        text_info = pending_info

    if not text:
        print("Nothing to clean.")
        return

    processed_pdf_root = data_dir / "processed_pdfs"
    if not processed_pdf_root.exists():
        raise FileNotFoundError(
            f"Processed PDF directory not found: {processed_pdf_root}\n"
            f"Run 'python experiments/process_pdfs.py --dataset {dataset_config.name}' first."
        )
    processed_pdf_dirs = [str(processed_pdf_root / info["paper_code"]) for info in text_info]

    mlm = MeasurementLM(
        model_name=model_config.model_id,
        entity_identification_prompt="",
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        sampling_params=model_config.sampling_params,
        api_base=api_base,
        api_key=api_key,
        clean_tables=True,
        cleaned_ocr_output_dir=str(effective_output_dir),
    )

    print(f"Cleaning tables for {len(text)} paper(s)...")
    mlm._clean_tables(text, processed_pdf_dirs)

    print(f"\nDone. Cleaned texts written to {effective_output_dir}")
    print(f"To use these cleaned texts for extraction, pass:")
    print(f"  --ocr-dir {effective_output_dir}")
    print(f"to run_extraction.py.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean tables in OCR text files using a vLLM-served model.",
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
        choices=sorted(MODEL_REGISTRY.keys()),
        help="Model key from MODEL_REGISTRY in run_extraction.py.",
    )
    p.add_argument(
        "--ocr-dir",
        default=None,
        metavar="DIR",
        help="Input OCR directory (default: data/{dataset}/ocr_output_raw/).",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        metavar="DIR",
        help=(
            "Output directory for cleaned texts "
            "(default: data/{dataset}/ocr_output_cleaned_{model_name}/)."
        ),
    )
    p.add_argument(
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Process only these paper codes (overrides the config's default subset).",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip papers whose output .txt already exists in the output directory.",
    )
    p.add_argument(
        "--api-base",
        default="http://localhost:8000/v1",
        metavar="URL",
        help="Base URL of the vLLM OpenAI-compatible server (default: http://localhost:8000/v1).",
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        metavar="KEY",
        help="API key for the vLLM server (any non-empty string; default: EMPTY).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dataset_config = load_dataset_config(args.dataset)
    model_config = get_model_config(args.model)
    run_vllm_table_cleaning(
        dataset_config=dataset_config,
        model_config=model_config,
        ocr_dir=args.ocr_dir,
        output_dir=args.output_dir,
        paper_subset_override=args.paper_subset,
        resume=args.resume,
        api_base=args.api_base,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
