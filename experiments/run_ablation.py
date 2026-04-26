"""
Ablation experiment runner for MeasurementLM.

Runs a named ablation variant of the MeasurementLM extraction pipeline for any
registered dataset and model, writing results to a structured output directory:

    data/experiments/{dataset}/ablations/ablation{N}/{model}/{YYYY_mm_dd}/

Usage
-----
    # From the repo root:
    python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 1
    python experiments/run_ablation.py --dataset nfix --model qwen-2.5-72b --ablation 3
    python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 2 \\
        --paper-subset physical_and_chemical_limnological prairie_wetland

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
Available models:   keys of MODEL_REGISTRY in run_extraction.py.
Available ablations: 1–6 (see ABLATION_REGISTRY below).

Notes
-----
Ablation 3 requires the dataset config to define ablation3_entity_schema and
ablation3_entity_identification_prompt. The schema must include two reserved
fields beyond the usual entity fields:
    - attribute (str)             : one of the keys in attribute_info_dict
    - attribute_terms (list[str]) : terminology used in the document
The prompt must instruct the model to emit one item per (entity, attribute) pair.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm and run_extraction importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm.measurementlm import NumpyEncoder
from scholarlm.measurementlm_ablation1 import MeasurementLMAblation1
from scholarlm.measurementlm_ablation2 import MeasurementLMAblation2
from scholarlm.measurementlm_ablation3 import MeasurementLMAblation3
from scholarlm.measurementlm_ablation4 import MeasurementLMAblation4
from scholarlm.measurementlm_ablation5 import MeasurementLMAblation5
from scholarlm.measurementlm_ablation6 import MeasurementLMAblation6

# Reuse shared utilities from run_extraction (model registry, config loading, etc.)
from run_extraction import (
    MODEL_REGISTRY,
    load_dataset_config,
    get_model_config,
    load_papers,
)
import paths

# ---------------------------------------------------------------------------
# Ablation registry
# ---------------------------------------------------------------------------

ABLATION_REGISTRY: dict[str, tuple[type, str]] = {
    "1": (
        MeasurementLMAblation1,
        "Direct triple extraction; the entire pipeline is replaced by a single LLM call "
        "per document that extracts all (entity, attribute, value) triples at once.",
    ),
    "2": (
        MeasurementLMAblation2,
        "Direct table value extraction; the model returns the value directly from the "
        "table instead of first identifying row/column indices for programmatic lookup.",
    ),
    "3": (
        MeasurementLMAblation3,
        "Combined entity-attribute extraction; entity detection and attribute detection "
        "merged into a single step, plus a combined per-page provenance step.",
    ),
    "4": (
        MeasurementLMAblation4,
        "Full-document context for value extraction and event resolution; the entire "
        "document (not just the relevant page/table) is sent to the value extractor "
        "and event resolver.",
    ),
    "5": (
        MeasurementLMAblation5,
        "No chain-of-thought explanations; all structured JSON responses drop the "
        "'explanation' field so the model does not produce reasoning traces.",
    ),
    "6": (
        MeasurementLMAblation6,
        "Full-document pair provenance; both provenance steps are replaced by a single "
        "full-document query per (entity, attribute) pair that returns a list of locations.",
    ),
}



# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_ablation(
    dataset_config,
    model_config,
    ablation: str,
    output_dir: Path,
    ocr_dir: str | None = None,
    paper_subset_override: list[str] | None = None,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
) -> None:
    """Run a single ablation experiment for a dataset / model pair.

    When ``ocr_dir`` is not given, raw OCR texts are loaded from
    ``{data_dir}/ocr_output_raw/`` and table cleaning is performed before
    running the ablation pipeline.  Cleaned texts are cached to
    ``{data_dir}/ocr_output_cleaned_{model_name}/``.

    When ``ocr_dir`` is given, texts are loaded directly from that directory
    and table cleaning is skipped.

    Writes a single ``final.json`` to ``output_dir``.

    Args:
        dataset_config: Dataset configuration loaded from ``experiments/configs/``.
        model_config: Model configuration from ``MODEL_REGISTRY``.
        ablation: Ablation key string (``"1"`` … ``"6"``).
        output_dir: Directory for the output file (created if needed).
        ocr_dir: Directory of pre-cleaned ``.txt`` files.  If ``None``, raw OCR
            is used and table cleaning is performed automatically.
        paper_subset_override: If provided, overrides ``dataset_config.paper_subset``.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the vLLM server (any non-empty string works).
    """
    ablation_class, ablation_desc = ABLATION_REGISTRY[ablation]
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
    print(f"Ablation  : {ablation} — {ablation_desc}")
    print(f"OCR dir   : {effective_ocr_dir}")
    if clean_tables:
        print(f"Cleaned   : {cleaned_ocr_output_dir}")
    print(f"Output    : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} papers.\n")

    # Ablation 3 runtime check: dataset config must provide ablation3_entity_schema
    # and ablation3_entity_identification_prompt with the required reserved fields.
    if ablation == "3":
        if dataset_config.ablation3_entity_schema is None:
            raise ValueError(
                f"Ablation 3 requires 'ablation3_entity_schema' to be set in the "
                f"dataset config for '{dataset_config.name}'. "
                f"Define the schema (entity fields + attribute + attribute_terms) and "
                f"set ablation3_entity_schema in the DatasetConfig."
            )
        schema_fields = set(dataset_config.ablation3_entity_schema.model_fields.keys())
        missing = {"attribute", "attribute_terms"} - schema_fields
        if missing:
            raise ValueError(
                f"Ablation 3 requires the ablation3_entity_schema to include the "
                f"fields {sorted(missing)}. The schema "
                f"({dataset_config.ablation3_entity_schema.__name__}) is missing "
                f"these fields. Please add them."
            )
        if dataset_config.ablation3_entity_identification_prompt is None:
            raise ValueError(
                f"Ablation 3 requires 'ablation3_entity_identification_prompt' to be "
                f"set in the dataset config for '{dataset_config.name}'."
            )
        # Use the ablation-specific schema and prompt for this run
        entity_schema = dataset_config.ablation3_entity_schema
        entity_identification_prompt = dataset_config.ablation3_entity_identification_prompt
    else:
        entity_schema = dataset_config.entity_schema
        entity_identification_prompt = dataset_config.entity_identification_prompt

    mlm_kwargs = dict(
        model_name=model_config.model_id,
        entity_identification_prompt=entity_identification_prompt,
        entity_identification_schema=entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        sampling_params=model_config.sampling_params,
        api_base=effective_api_base,
        api_key=api_key,
        clean_tables=clean_tables,
        cleaned_ocr_output_dir=cleaned_ocr_output_dir,
        measurement_event_schema=dataset_config.measurement_event_schema,
        measurement_event_prompt=dataset_config.measurement_event_prompt,
        use_extra_body=not is_frontier,
    )
    if ablation == "1":
        mlm_kwargs["direct_extraction_schema"] = dataset_config.direct_extraction_schema
        mlm_kwargs["direct_extraction_prompt"] = dataset_config.direct_extraction_prompt
    mlm = ablation_class(**mlm_kwargs)

    # Pre-clean tables if needed (same pattern as run_extraction.py)
    processed_pdf_dirs = None
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
        # Tables are now cleaned; disable the check inside fit() to avoid a second pass.
        mlm.clean_tables = False

    print("Running ablation pipeline...")
    data = mlm.fit(text)

    dataset = [
        info | dp | {"document_id": info["document_id"], "measurement_id": i}
        for i, dp in enumerate(data)
        for info in [text_info[dp["document_id"]]]
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
        description="Run a MeasurementLM ablation experiment.",
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
        "--ablation",
        required=True,
        choices=sorted(ABLATION_REGISTRY.keys()),
        metavar="N",
        help=f"Ablation to run. Choices: {{{', '.join(sorted(ABLATION_REGISTRY.keys()))}}}.",
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
            "table cleaning is performed automatically using the extraction model."
        ),
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
            "Base URL of the vLLM OpenAI-compatible server "
            "(default: http://localhost:8081/v1)."
        ),
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
    output_dir = paths.ablation(args.dataset, args.ablation, args.model, args.date)

    run_ablation(
        dataset_config=dataset_config,
        model_config=model_config,
        ablation=args.ablation,
        output_dir=output_dir,
        ocr_dir=args.ocr_dir,
        paper_subset_override=args.paper_subset,
        api_base=args.api_base,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
