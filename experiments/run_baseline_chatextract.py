"""
ChatExtract baseline runner.

Runs the ChatExtract extraction baseline (Polak & Morgan, Nat. Commun. 2024;
arXiv:2303.05352) for any registered dataset, on any registered backbone model,
writing results to the *standard extraction path* so it can be loaded and
compared against MeasurementLM using the existing analysis code with no
modification:

    data/experiments/{dataset}/extraction/chatextract-{model}/{YYYY_mm_dd}/final.json

Unlike the NuExtract baseline, ChatExtract is a *text-based* method: it reads the
OCR'd `<page>/<table>` tagged text (same input as MeasurementLM / run_extraction),
so it does NOT require `experiments/process_pdfs.py`. It is a multi-turn
conversational method — classify each sentence, gate single vs. multiple values,
extract Material/Value/Unit, then verify each field with redundant strict yes/no
questions — run once per dataset attribute (see `measurementlm_chatextract.py`).
Because it wraps an ordinary chat model, `--model` selects any entry from the
standard `MODEL_REGISTRY`, so ChatExtract can be run on the same backbones as
MeasurementLM for a fair same-model comparison.

Usage
-----
    # From the repo root, against a vLLM server hosting the chosen model:
    python experiments/run_baseline_chatextract.py --dataset pond --model gemma-3-27b
    python experiments/run_baseline_chatextract.py --dataset pond --model gemma-3-27b \\
        --paper-subset agricultural_freshwater --api-base http://localhost:8000/v1

    # Toggle off the real-table workflow or single-branch verification if desired:
    python experiments/run_baseline_chatextract.py --dataset nfix --model gemma-3-27b \\
        --no-tables --no-single-verification

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
Available models: any entry in MODEL_REGISTRY (experiments/model_registry.py).
"""
from __future__ import annotations

import argparse
import json
import os
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
from scholarlm.measurementlm_chatextract import MeasurementLMChatExtract

from run_extraction import load_dataset_config, load_papers, get_model_config
import paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_baseline_chatextract(
    dataset_config,
    model_config,
    output_dir: Path,
    paper_subset_override: list[str] | None = None,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    max_concurrent: int = 32,
    extract_tables: bool = True,
    include_single_verification: bool = True,
) -> None:
    """Run the ChatExtract baseline for a dataset on a given backbone model.

    Writes a single `final.json` to `output_dir`, in the standard extraction
    record schema (same fields as MeasurementLM/ablation final.json output),
    so it can be loaded via `analysis.loaders.load_extraction` unmodified.
    """
    data_dir = Path(dataset_config.data_dir)

    is_frontier = model_config.api_base is not None
    if is_frontier:
        effective_api_base = model_config.api_base
        if api_key == "EMPTY":
            if "openai.com" in model_config.api_base:
                api_key = os.environ.get("OPENAI_API_KEY", "")
            else:
                api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError(
                f"API key required for frontier model '{model_config.name}'. "
                "Set OPENAI_API_KEY or GEMINI_API_KEY, or pass --api-key."
            )
    else:
        effective_api_base = api_base

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"Output    : {output_dir}\n")

    # ChatExtract reads OCR text directly.
    ocr_dir = str(data_dir / "ocr_output_raw")
    text, text_info = load_papers(dataset_config, ocr_dir, paper_subset_override)
    print(f"Loaded {len(text_info)} papers.\n")

    # Paper titles (from directory.json metadata) form the head of each passage.
    titles = [info.get("title", "") or "" for info in text_info]

    mlm = MeasurementLMChatExtract(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        attribute_property_names=dataset_config.chatextract_property_names,
        include_single_verification=include_single_verification,
        extract_tables=extract_tables,
        measurement_event_schema=dataset_config.measurement_event_schema,
        sampling_params=model_config.sampling_params,
        api_base=effective_api_base,
        api_key=api_key,
        max_concurrent=max_concurrent,
        use_extra_body=not is_frontier,
    )

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running ChatExtract baseline...")
    start_time = time.time()
    data = mlm.fit(text, titles)

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
        baseline="chatextract",
        extract_tables=extract_tables,
        include_single_verification=include_single_verification,
        gpu_compatibility_warnings=gpu_warnings,
    )
    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the ChatExtract extraction baseline.",
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
        default="gemma-3-27b",
        help="Backbone model name (must match an entry in MODEL_REGISTRY). Default: gemma-3-27b.",
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
        default="http://localhost:8000/v1",
        metavar="URL",
        help=(
            "Base URL of the vLLM OpenAI-compatible server hosting the backbone model "
            "(default: http://localhost:8000/v1). Ignored for frontier models, which "
            "use their registered api_base."
        ),
    )
    p.add_argument(
        "--api-key",
        default="EMPTY",
        metavar="KEY",
        help="API key (any non-empty string for vLLM; resolved from env for frontier).",
    )
    p.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        help=(
            "Maximum concurrent in-flight conversations (default: 32). Each conversation "
            "is a whole sentence/table dialogue, so most turns after classification are skipped."
        ),
    )
    p.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable the real-document-table extraction workflow (sentences only).",
    )
    p.add_argument(
        "--no-single-verification",
        action="store_true",
        help="Skip the single-valued-branch yes/no verification (matches the reference script, not the paper).",
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
    output_dir = paths.extraction(args.dataset, f"chatextract-{args.model}", args.date)

    run_baseline_chatextract(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        paper_subset_override=args.paper_subset,
        api_base=args.api_base,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        extract_tables=not args.no_tables,
        include_single_verification=not args.no_single_verification,
    )


if __name__ == "__main__":
    main()
