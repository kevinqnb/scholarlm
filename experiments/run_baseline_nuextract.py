"""
NuExtract-2.0-8B baseline runner.

Runs the NuExtract-2.0-8B extraction baseline for any registered dataset,
writing results to the *standard extraction path* so it can be loaded and
compared against MeasurementLM using the existing analysis code with no
modification:

    data/experiments/{dataset}/extraction/nuextract-2.0-8b/{YYYY_mm_dd}/final.json

Unlike run_extraction.py / run_ablation.py, this runner does not consume OCR
text — NuExtract-2.0-8B is a vision-language model that reads rendered page
images directly. It requires that `experiments/process_pdfs.py` has already
been run for the target dataset (producing `{data_dir}/processed_pdfs/`).

NuExtract has a fixed calling convention (see `measurementlm_nuextract.py`):
the JSON extraction schema is sent out-of-band via `extra_body`, and message
content must contain only image blocks — there is no field for freeform
instructions, so this baseline (unlike Ablation 1) does not use a dataset's
`direct_extraction_prompt`.

Usage
-----
    # From the repo root:
    python experiments/run_baseline_nuextract.py --dataset pond
    python experiments/run_baseline_nuextract.py --dataset nfix \\
        --paper-subset physical_and_chemical_limnological

    # Point at a vLLM server hosting NuExtract-2.0-8B (see experiments/config.yaml's
    # nuextract-2.0-8b entry and experiments/serve_nuextract_2_0_8b.sh, generated via
    # `python experiments/gen_serve_script.py nuextract-2.0-8b`):
    #   vllm serve numind/NuExtract-2.0-8B --trust-remote-code \\
    #       --chat-template-content-format openai --limit-mm-per-prompt '{"image": 50}'
    python experiments/run_baseline_nuextract.py --dataset pond --api-base http://localhost:8081/v1

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm and run_extraction importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm.measurementlm import NumpyEncoder
from scholarlm.measurementlm_nuextract import MeasurementLMNuExtract

from run_extraction import load_dataset_config, load_papers
from model_registry import BASELINE_MODEL_REGISTRY
import paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_baseline_nuextract(
    dataset_config,
    output_dir: Path,
    paper_subset_override: list[str] | None = None,
    api_base: str = "http://localhost:8082/v1",
    api_key: str = "EMPTY",
    max_concurrent: int = 2,
) -> None:
    """Run the NuExtract-2.0-8B baseline for a dataset.

    Requires `{data_dir}/processed_pdfs/{document_id}/` to already exist
    (produced by `experiments/process_pdfs.py`).

    Writes a single `final.json` to `output_dir`, in the standard extraction
    record schema (same fields as MeasurementLM/ablation final.json output),
    so it can be loaded via `analysis.loaders.load_extraction` unmodified.

    Args:
        dataset_config: Dataset configuration loaded from `experiments/configs/`.
        output_dir: Directory for the output file (created if needed).
        paper_subset_override: If provided, overrides `dataset_config.paper_subset`.
        api_base: Base URL of the vLLM OpenAI-compatible server hosting NuExtract.
        api_key: API key for the vLLM server (any non-empty string works).
        max_concurrent: Maximum concurrent in-flight requests.
    """
    model_config = BASELINE_MODEL_REGISTRY["nuextract-2.0-8b"]
    data_dir = Path(dataset_config.data_dir)

    if dataset_config.direct_extraction_schema is None:
        raise ValueError(
            f"Dataset '{dataset_config.name}' does not define direct_extraction_schema, "
            f"which is required for the NuExtract baseline (the same schema Ablation 1 uses)."
        )

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"Output    : {output_dir}\n")

    # OCR text is only used here to enumerate papers and get metadata; the
    # actual text content is discarded — NuExtract reads page images.
    ocr_dir = str(data_dir / "ocr_output_raw")
    _text, text_info = load_papers(dataset_config, ocr_dir, paper_subset_override)
    print(f"Loaded {len(text_info)} papers.\n")

    processed_pdf_root = data_dir / "processed_pdfs"
    if not processed_pdf_root.exists():
        raise FileNotFoundError(
            f"Processed PDF directory not found: {processed_pdf_root}\n"
            f"Run 'python experiments/process_pdfs.py --dataset {dataset_config.name}' first."
        )
    processed_pdf_dirs = [
        str(processed_pdf_root / info["document_id"]) for info in text_info
    ]

    mlm = MeasurementLMNuExtract(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        direct_extraction_schema=dataset_config.direct_extraction_schema,
        examples=dataset_config.nuextract_examples,
        sampling_params=model_config.sampling_params,
        api_base=api_base,
        api_key=api_key,
        max_concurrent=max_concurrent,
        clean_tables=False,
        measurement_event_schema=dataset_config.measurement_event_schema,
    )

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running NuExtract baseline...")
    start_time = time.time()
    data = mlm.fit(processed_pdf_dirs)

    dataset = [
        info | dp | {"document_id": info["document_id"], "measurement_id": i}
        for i, dp in enumerate(data)
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
        baseline="nuextract",
        gpu_compatibility_warnings=gpu_warnings,
    )
    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the NuExtract-2.0-8B extraction baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (must match a file in experiments/configs/<name>.py).",
    )
    p.add_argument(
        "--date",
        default=None,
        help="Output date tag YYYY_mm_dd (default: today).",
    )
    p.add_argument(
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Override dataset paper_subset with an explicit list of paper codes.",
    )
    p.add_argument(
        "--api-base",
        default="http://localhost:8081/v1",
        metavar="URL",
        help=(
            "Base URL of the vLLM OpenAI-compatible server hosting NuExtract-2.0-8B "
            "(default: http://localhost:8081/v1)."
        ),
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        metavar="KEY",
        help="API key for the vLLM server (any non-empty string; default: EMPTY).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent in-flight requests (default: 2).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from utils import load_config
    cfg = load_config()
    seed = cfg.get("defaults", {}).get("seed", 342)
    set_seeds(seed)

    dataset_config = load_dataset_config(args.dataset)
    output_dir = paths.extraction(args.dataset, "nuextract-2.0-8b", args.date)

    run_baseline_nuextract(
        dataset_config=dataset_config,
        output_dir=output_dir,
        paper_subset_override=args.paper_subset,
        api_base=args.api_base,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
