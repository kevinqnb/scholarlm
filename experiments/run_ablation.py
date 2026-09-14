"""
Ablation experiment runner for MeasurementLM.

Runs a named ablation variant of the MeasurementLM extraction pipeline for any
registered dataset and model, writing results to a structured output directory:

    experiments/results/{dataset}/ablation/{experiment_id}/

Usage
-----
    python experiments/run_ablation.py experiments/experiment-configs/pond/ablation/<id>/<id>.yaml

Required params: dataset, model, ablation (one of ABLATION_REGISTRY's keys, "1"-"6").
Optional params: ocr_dir, paper_subset (list), api_base, api_key.

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
Available models:   any file in experiments/model-configs/extraction/<name>.yaml.
Available ablations: 1–6 (see ABLATION_REGISTRY below).

Notes
-----
Ablation 2 requires the dataset config to define ablation2_entity_schema and
ablation2_entity_identification_prompt. The schema must include two reserved
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
from scholarlm.measurementlm_ablation1 import MeasurementLMAblation1
from scholarlm.measurementlm_ablation2 import MeasurementLMAblation2
from scholarlm.measurementlm_ablation3 import MeasurementLMAblation3
from scholarlm.measurementlm_ablation4 import MeasurementLMAblation4
from scholarlm.measurementlm_ablation5 import MeasurementLMAblation5
from scholarlm.measurementlm_ablation6 import MeasurementLMAblation6

# Reuse shared utilities from run_extraction (config loading, etc.)
from run_extraction import (
    load_dataset_config,
    get_model_config,
    load_papers,
)
import utils as paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata

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
        "Combined entity-attribute extraction; entity detection and attribute detection "
        "merged into a single step, plus a combined per-page provenance step.",
    ),
    "3": (
        MeasurementLMAblation3,
        "Full-document pair provenance; both provenance steps are replaced by a single "
        "full-document query per (entity, attribute) pair that returns a list of locations.",
    ),
    "4": (
        MeasurementLMAblation4,
        "Full-document context for value extraction and event resolution; the entire "
        "document (not just the relevant page/table) is sent to the value extractor "
        "and event resolver.",
    ),
    "5": (
        MeasurementLMAblation5,
        "Direct table value extraction; the model returns the value directly from the "
        "table instead of first identifying row/column indices for programmatic lookup.",
    ),
    "6": (
        MeasurementLMAblation6,
        "No chain-of-thought explanations; all structured JSON responses drop the "
        "'explanation' field so the model does not produce reasoning traces.",
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
        dataset_config: Dataset configuration loaded from ``experiments/dataset-configs/``.
        model_config: Model configuration from experiments/model-configs/extraction/.
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

    # Ablation 2 runtime check: dataset config must provide ablation2_entity_schema
    # and ablation2_entity_identification_prompt with the required reserved fields.
    if ablation == "2":
        if dataset_config.ablation2_entity_schema is None:
            raise ValueError(
                f"Ablation 2 requires 'ablation2_entity_schema' to be set in the "
                f"dataset config for '{dataset_config.name}'. "
                f"Define the schema (entity fields + attribute + attribute_terms) and "
                f"set ablation2_entity_schema in the DatasetConfig."
            )
        schema_fields = set(dataset_config.ablation2_entity_schema.model_fields.keys())
        missing = {"attribute", "attribute_terms"} - schema_fields
        if missing:
            raise ValueError(
                f"Ablation 2 requires the ablation2_entity_schema to include the "
                f"fields {sorted(missing)}. The schema "
                f"({dataset_config.ablation2_entity_schema.__name__}) is missing "
                f"these fields. Please add them."
            )
        if dataset_config.ablation2_entity_identification_prompt is None:
            raise ValueError(
                f"Ablation 2 requires 'ablation2_entity_identification_prompt' to be "
                f"set in the dataset config for '{dataset_config.name}'."
            )
        # Use the ablation-specific schema and prompt for this run
        entity_schema = dataset_config.ablation2_entity_schema
        entity_identification_prompt = dataset_config.ablation2_entity_identification_prompt
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
        collect_attribute_terms=dataset_config.collect_attribute_terms,
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

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running ablation pipeline...")
    start_time = time.time()
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

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        model=model_config.name,
        model_id=model_config.model_id,
        hf_revision=model_config.hf_revision,
        ablation=ablation,
        gpu_compatibility_warnings=gpu_warnings,
    )
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
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    p.add_argument(
        "--api-base", default=None, metavar="URL",
        help="Override params.api_base (submit.sh injects the compute node's vLLM endpoint here).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", "model", "ablation", config_path=config_path)

    ablation = str(params["ablation"])
    if ablation not in ABLATION_REGISTRY:
        raise ValueError(
            f"{config_path}: params.ablation {ablation!r} not in ABLATION_REGISTRY "
            f"(choices: {sorted(ABLATION_REGISTRY.keys())})"
        )

    exp_defaults = paths.load_config().get("defaults", {})
    repo_seed = exp_defaults.get("seed")
    if repo_seed is not None and cfg["seed"] != repo_seed:
        raise ValueError(
            f"{config_path}: seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed}) -- the repo's seed is a fixed, repo-wide value, "
            "not a per-run knob."
        )
    set_seeds(cfg["seed"])

    dataset_config = load_dataset_config(params["dataset"])
    model_config = get_model_config(params["model"])
    output_dir = paths.result_dir(params["dataset"], "ablation", cfg["id"])

    run_ablation(
        dataset_config=dataset_config,
        model_config=model_config,
        ablation=ablation,
        output_dir=output_dir,
        ocr_dir=params.get("ocr_dir"),
        paper_subset_override=params.get("paper_subset"),
        api_base=args.api_base or params.get("api_base") or "http://localhost:8081/v1",
        api_key=params.get("api_key", "EMPTY"),
    )


if __name__ == "__main__":
    main()
