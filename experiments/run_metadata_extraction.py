"""
Metadata extraction runner for MetadataLM.

Runs document-level metadata extraction for any registered dataset and model,
writing results to a structured output directory:

    data/experiments/{dataset}/metadata/{model}/{YYYY_mm_dd}/

Each document produces exactly one record in final.json.  The dataset config
must supply metadata_extraction_schema and metadata_extraction_prompt.

Usage
-----
    # From the repo root:
    python experiments/run_metadata_extraction.py --dataset govscape --model gpt-4o

    python experiments/run_metadata_extraction.py \\
        --dataset govscape --model gpt-4o \\
        --ocr-dir data/govscape/ocr_output_raw \\
        --max-concurrent 8

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG
                    with metadata_extraction_schema and metadata_extraction_prompt set.
Available models:   keys of MODEL_REGISTRY in run_extraction.py.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm.measurementlm import NumpyEncoder
from scholarlm.metadata_extraction import MetadataLM

from run_extraction import (
    MODEL_REGISTRY,
    load_dataset_config,
    get_model_config,
    load_papers,
)
import paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_metadata_extraction(
    dataset_config,
    model_config,
    output_dir: Path,
    ocr_dir: str | None = None,
    paper_subset_override: list[str] | None = None,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
    max_tokens: int | None = None,
    max_concurrent: int | None = None,
    max_input_tokens: int | None = None,
) -> None:
    """Run metadata extraction for a dataset / model pair.

    Writes a single ``final.json`` to ``output_dir`` with one record per document.

    Args:
        dataset_config: Dataset configuration with metadata_extraction_schema and
            metadata_extraction_prompt set.
        model_config: Model configuration from MODEL_REGISTRY.
        output_dir: Directory for the output file (created if needed).
        ocr_dir: Directory of pre-cleaned .txt files.  If None, raw OCR is loaded
            from {data_dir}/ocr_output_raw/ and table cleaning is performed
            automatically (local models only).
        paper_subset_override: If provided, overrides dataset_config.paper_subset.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key (any non-empty string for vLLM; real key for frontier).
    """
    if dataset_config.metadata_extraction_schema is None:
        raise ValueError(
            f"Dataset '{dataset_config.name}' has no metadata_extraction_schema. "
            "Set metadata_extraction_schema and metadata_extraction_prompt in the "
            "dataset config before running metadata extraction."
        )
    if dataset_config.metadata_extraction_prompt is None:
        raise ValueError(
            f"Dataset '{dataset_config.name}' has no metadata_extraction_prompt. "
            "Set metadata_extraction_prompt in the dataset config."
        )

    data_dir = Path(dataset_config.data_dir)
    is_frontier = model_config.api_base is not None

    if is_frontier:
        effective_api_base = model_config.api_base
        if api_key == "EMPTY":
            if "openai.com" in model_config.api_base:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            elif "anthropic.com" in model_config.api_base:
                api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            else:
                api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                f"API key required for frontier model '{model_config.name}'. "
                "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or GEMINI_API_KEY, or pass --api-key."
            )
    else:
        effective_api_base = api_base

    if ocr_dir is not None or is_frontier:
        effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")
        clean_tables = False
        cleaned_ocr_output_dir = None
    else:
        effective_ocr_dir = str(data_dir / "ocr_output_raw")
        clean_tables = True
        cleaned_ocr_output_dir = str(data_dir / f"ocr_output_cleaned_{model_config.name}")

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"Task      : metadata extraction")
    print(f"OCR dir   : {effective_ocr_dir}")
    if clean_tables:
        print(f"Cleaned   : {cleaned_ocr_output_dir}")
    print(f"Output    : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} documents.\n")

    mlm_kwargs = dict(
        model_name=model_config.model_id,
        entity_identification_prompt=None,
        entity_identification_schema=None,
        attribute_info_dict=None,
        sampling_params=model_config.sampling_params,
        api_base=effective_api_base,
        api_key=api_key,
        clean_tables=clean_tables,
        cleaned_ocr_output_dir=cleaned_ocr_output_dir,
        measurement_event_schema=None,
        measurement_event_prompt=None,
        use_extra_body=not is_frontier,
        metadata_extraction_schema=dataset_config.metadata_extraction_schema,
        metadata_extraction_prompt=dataset_config.metadata_extraction_prompt,
    )
    if max_concurrent is not None:
        mlm_kwargs["max_concurrent"] = max_concurrent
    if max_tokens is not None:
        mlm_kwargs["extract_max_tokens"] = max_tokens
    if max_input_tokens is not None:
        mlm_kwargs["max_input_tokens"] = max_input_tokens

    mlm = MetadataLM(**mlm_kwargs)

    if clean_tables:
        processed_pdf_root = data_dir / "processed_pdfs"
        if not processed_pdf_root.exists():
            raise FileNotFoundError(
                f"Processed PDF directory not found: {processed_pdf_root}\n"
                f"Run 'python experiments/process_pdfs.py --dataset {dataset_config.name}' first."
            )
        processed_pdf_dirs = [
            str(processed_pdf_root / info["document_id"]) for info in text_info
        ]
        text = mlm._clean_tables(text, processed_pdf_dirs)
        mlm.clean_tables = False

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running metadata extraction...")
    start_time = time.time()
    data = mlm.fit(text)

    dataset = [
        info | dp | {"document_id": info["document_id"]}
        for dp in data
        for info in [text_info[dp["document_id"]]]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "final.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        model=model_config.name,
        model_id=model_config.model_id,
        hf_revision=model_config.hf_revision,
        gpu_compatibility_warnings=gpu_warnings,
    )
    print(f"\nDone. Final dataset: {out_path}")
    print(f"     Documents saved: {len(dataset)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run MetadataLM document metadata extraction.",
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
        help="Extraction model key from MODEL_REGISTRY.",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Output date tag YYYY_mm_dd (default: today).",
    )
    p.add_argument(
        "--ocr-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory of pre-cleaned OCR .txt files to use as extraction input. "
            "If omitted, raw OCR is loaded from {data_dir}/ocr_output_raw/ and "
            "table cleaning is performed automatically (local models only)."
        ),
    )
    p.add_argument(
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Override dataset paper_subset with an explicit list of document codes.",
    )
    p.add_argument(
        "--api-base",
        default="http://localhost:8081/v1",
        metavar="URL",
        help=(
            "Base URL of the vLLM OpenAI-compatible server "
            "(default: http://localhost:8081/v1)."
        ),
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        metavar="KEY",
        help="API key for the server (any non-empty string for vLLM; real key for frontier).",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        metavar="N",
        help="Maximum output tokens per document call (default: 4096).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        metavar="N",
        help="Maximum concurrent document calls (default: 4).",
    )
    p.add_argument(
        "--max-input-tokens",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Maximum tokens of document text to send per call (default: 40,000). "
            "Documents exceeding this are truncated from the tail — metadata fields are almost always "
            "at the document head, so this loses nothing relevant."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from utils import load_config
    cfg = load_config()
    seed = cfg.get("defaults", {}).get("seed", 342)
    set_seeds(seed)

    dataset_config = load_dataset_config(args.dataset)
    model_config = get_model_config(args.model)
    output_dir = paths.metadata_extraction(args.dataset, args.model, args.date)

    run_metadata_extraction(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        ocr_dir=args.ocr_dir,
        paper_subset_override=args.paper_subset,
        api_base=args.api_base,
        api_key=args.api_key,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        max_input_tokens=args.max_input_tokens,
    )


if __name__ == "__main__":
    main()
