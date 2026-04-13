"""
Ablation experiment runner for MeasurementLM.

Runs a named ablation variant of the MeasurementLM extraction pipeline for any
registered dataset and model, writing results to a structured output directory:

    data/experiments/{dataset}/ablations/ablation{N}/{model}/{YYYY_mm_dd}/

Usage
-----
    # From the repo root:
    python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 1
    python experiments/run_ablation.py --dataset nfix --model qwen-2.5-72b --ablation 6
    python experiments/run_ablation.py --dataset pond --model gemma-3-27b --ablation 3 \\
        --paper-subset physical_and_chemical_limnological prairie_wetland

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
Available models:   keys of MODEL_REGISTRY in run_extraction.py.
Available ablations: 1–6 (see ABLATION_REGISTRY below).

Notes
-----
Ablation 1 requires the dataset's entity_identification_schema to include two
additional fields beyond the usual entity-identifying fields:
    - attribute (str)             : one of the keys in attribute_info_dict
    - attribute_terms (list[str]) : terminology used in the document
The entity_identification_prompt must also instruct the model to emit one item
per (entity, attribute) pair rather than one item per entity.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
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

# ---------------------------------------------------------------------------
# Ablation registry
# ---------------------------------------------------------------------------

ABLATION_REGISTRY: dict[str, tuple[type, str]] = {
    "1": (
        MeasurementLMAblation1,
        "Combined entity-attribute extraction; entity detection and attribute detection "
        "merged into a single step, plus a combined per-page provenance step.",
    ),
    "2": (
        MeasurementLMAblation2,
        "Full-document context for value extraction; the entire document (not just the "
        "relevant page/table) is sent to the value extractor.",
    ),
    "3": (
        MeasurementLMAblation3,
        "Direct table value extraction; the model returns the value directly from the "
        "table instead of first identifying row/column indices for programmatic lookup.",
    ),
    "4": (
        MeasurementLMAblation4,
        "Full-document pair provenance; both provenance steps are replaced by a single "
        "full-document query per (entity, attribute) pair that returns a list of locations.",
    ),
    "5": (
        MeasurementLMAblation5,
        "No chain-of-thought explanations; all structured JSON responses drop the "
        "'explanation' field so the model does not produce reasoning traces.",
    ),
    "6": (
        MeasurementLMAblation6,
        "Direct triple extraction; the entire pipeline is replaced by a single LLM call "
        "per document that extracts all (entity, attribute, value) triples at once.",
    ),
}

# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------


def get_output_dir(
    dataset_name: str,
    ablation: str,
    model_name: str,
    date: str | None = None,
) -> Path:
    """Return the output directory for a given (dataset, ablation, model, date) tuple.

    Path convention:
        ``data/experiments/{dataset}/ablations/ablation{N}/{model}/{YYYY_mm_dd}/``

    Args:
        dataset_name: Dataset identifier (e.g. ``"pond"``).
        ablation: Ablation number as a string (e.g. ``"1"``).
        model_name: Model identifier (e.g. ``"qwen-2.5-72b"``).
        date: Optional date string ``"YYYY_mm_dd"``. Defaults to today.

    Returns:
        A ``Path`` object (not yet created on disk).
    """
    if date is None:
        date = datetime.now().strftime("%Y_%m_%d")
    return (
        _REPO_ROOT
        / "data"
        / "experiments"
        / dataset_name
        / "ablations"
        / f"ablation{ablation}"
        / model_name
        / date
    )


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

    if ocr_dir is not None:
        effective_ocr_dir = ocr_dir
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

    # Ablation 1 runtime check: entity schema must include 'attribute' and 'attribute_terms'
    if ablation == "1":
        schema_fields = set(dataset_config.entity_schema.model_fields.keys())
        missing = {"attribute", "attribute_terms"} - schema_fields
        if missing:
            raise ValueError(
                f"Ablation 1 requires the entity_identification_schema to include the "
                f"fields {sorted(missing)}. The dataset config's entity_schema "
                f"({dataset_config.entity_schema.__name__}) is missing these fields. "
                f"Please add them and update entity_identification_prompt accordingly."
            )

    mlm = ablation_class(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        sampling_params=model_config.sampling_params,
        api_base=api_base,
        api_key=api_key,
        clean_tables=clean_tables,
        cleaned_ocr_output_dir=cleaned_ocr_output_dir,
        direct_extraction_schema=dataset_config.direct_extraction_schema,
        direct_extraction_prompt=dataset_config.direct_extraction_prompt,
    )

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
            str(processed_pdf_root / info["paper_code"]) for info in text_info
        ]
        text = mlm._clean_tables(text, processed_pdf_dirs)
        # Tables are now cleaned; disable the check inside fit() to avoid a second pass.
        mlm.clean_tables = False

    print("Running ablation pipeline...")
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
    output_dir = get_output_dir(args.dataset, args.ablation, args.model, args.date)

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
