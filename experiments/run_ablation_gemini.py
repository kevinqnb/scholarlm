"""
Ablation 6 (direct triple extraction) runner for Google Gemini frontier models.

Uses Gemini's OpenAI-compatible endpoint, so the same MeasurementLMAblation6
class and JSON schema structured output are used unchanged.

Writes results to:

    data/experiments/{dataset}/ablations/ablation6/{model}/{YYYY_mm_dd}/final.json

Requires GEMINI_API_KEY environment variable or --api-key flag.
The dataset config must define direct_extraction_schema and direct_extraction_prompt.

Usage
-----
    python experiments/run_ablation_gemini.py --dataset pond --model gemini-2-flash-lite \\
        --paper-subset physical_and_chemical_limnological prairie_wetland
    python experiments/run_ablation_gemini.py --dataset nfix --model gemini-3-flash-lite \\
        --paper-subset R163 R164 R172
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm.measurementlm import NumpyEncoder
from scholarlm.measurementlm_ablation6 import MeasurementLMAblation6
from run_extraction import load_dataset_config, load_papers

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

GEMINI_MODEL_REGISTRY: dict[str, str] = {
    "gemini-2-flash-lite": "gemini-2.0-flash-lite",
    "gemini-2-flash": "gemini-2.0-flash",
    "gemini-3-flash-lite": "gemini-3-flash-lite",
    "gemini-1.5-flash": "gemini-1.5-flash",
}

# Gemini's OpenAI-compatible endpoint
_GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------


def get_output_dir(dataset_name: str, model_name: str, date: str | None = None) -> Path:
    if date is None:
        date = datetime.now().strftime("%Y_%m_%d")
    return (
        _REPO_ROOT / "data" / "experiments" / dataset_name
        / "ablations" / "ablation6" / model_name / date
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    dataset_config,
    model_id: str,
    model_name: str,
    output_dir: Path,
    ocr_dir: str | None = None,
    paper_subset_override: list[str] | None = None,
    api_key: str | None = None,
) -> None:
    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("Gemini API key required. Set GEMINI_API_KEY or pass --api-key.")

    if dataset_config.direct_extraction_schema is None or dataset_config.direct_extraction_prompt is None:
        raise ValueError(
            f"Dataset '{dataset_config.name}' does not define direct_extraction_schema or "
            "direct_extraction_prompt. Add them to the dataset config before running ablation 6."
        )

    data_dir = Path(dataset_config.data_dir)
    effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")

    print(f"\nDataset  : {dataset_config.name}")
    print(f"Model    : {model_name} ({model_id})")
    print(f"Ablation : 6 — Direct triple extraction")
    print(f"OCR dir  : {effective_ocr_dir}")
    print(f"Output   : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} papers.\n")

    mlm = MeasurementLMAblation6(
        model_name=model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        sampling_params={"temperature": 0.6, "top_p": 0.95, "max_tokens": 8192},
        api_base=_GEMINI_API_BASE,
        api_key=api_key,
        clean_tables=False,
        direct_extraction_schema=dataset_config.direct_extraction_schema,
        direct_extraction_prompt=dataset_config.direct_extraction_prompt,
    )
    # Remove vLLM-specific sampling params that Gemini's API does not accept.
    for key in ("top_k", "repetition_penalty", "enable_thinking"):
        mlm.sampling_params.pop(key, None)

    print("Running ablation 6 pipeline...")
    data = mlm.fit(text)

    dataset = [
        text_info[dp["document_id"]] | dp | {"measurement_id": i}
        for i, dp in enumerate(data)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "final.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ablation 6 (direct triple extraction) using Google Gemini frontier models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (experiments/configs/<name>.py).")
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(GEMINI_MODEL_REGISTRY.keys()),
        help="Gemini model key.",
    )
    p.add_argument("--date", default=None, help="Output date tag YYYY_mm_dd (default: today).")
    p.add_argument(
        "--ocr-dir",
        default=None,
        metavar="DIR",
        help="Directory of OCR .txt files. Defaults to {data_dir}/ocr_output_raw/.",
    )
    p.add_argument(
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Override the dataset paper_subset with an explicit list of paper codes.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="Gemini API key (default: GEMINI_API_KEY env var).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dataset_config = load_dataset_config(args.dataset)
    model_id = GEMINI_MODEL_REGISTRY[args.model]
    output_dir = get_output_dir(args.dataset, args.model, args.date)
    run(
        dataset_config=dataset_config,
        model_id=model_id,
        model_name=args.model,
        output_dir=output_dir,
        ocr_dir=args.ocr_dir,
        paper_subset_override=args.paper_subset,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
