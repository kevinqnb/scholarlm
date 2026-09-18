"""
MeasurementLMv2 (quantity-first) extraction pipeline runner.

Runs MeasurementLMv2 for any registered dataset and model, writing output to:

    experiments/results/{dataset}/extraction_v2/{experiment_id}/

Unlike run_extraction.py, this runner never performs table cleaning --
MeasurementLMv2 assumes the OCR text it is given is already cleaned by a
separate process (see run_table_cleaning.py), so params.ocr_dir must point at
an already-cleaned directory (e.g. data/{dataset}/ocr_output_cleaned_{model}/).

Usage
-----
    python experiments/run_extraction_v2.py experiments/experiment-configs/{dataset}/extraction_v2/<id>/<id>.yaml

Required params: dataset, model, ocr_dir.
Optional params: paper_subset (list), resume (bool), step (one of STEP_NAMES),
api_base, api_key.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from scholarlm import MeasurementLMv2
from scholarlm.config import DatasetConfig, ModelConfig
from scholarlm.measurementlm import NumpyEncoder
import utils as paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata
from run_extraction import load_dataset_config, get_model_config, load_papers

STEP_NAMES = ("quantities", "standardized", "final")


def _resolve_api(model_config: ModelConfig, api_base: str, api_key: str) -> tuple[str, str]:
    """Mirrors run_extraction.py's frontier/local api_base + api_key resolution."""
    if model_config.api_base is None:
        return api_base, api_key
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
    return effective_api_base, api_key


def _build_mlm(dataset_config: DatasetConfig, model_config: ModelConfig, api_base: str, api_key: str) -> MeasurementLMv2:
    return MeasurementLMv2(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        sampling_params=model_config.sampling_params,
        api_base=api_base,
        api_key=api_key,
        clean_tables=False,
        measurement_event_schema=dataset_config.measurement_event_schema,
        measurement_event_prompt=dataset_config.measurement_event_prompt,
        use_extra_body=model_config.api_base is None,
    )


def _finalize_records(records: list[dict], text_info: list[dict]) -> list[dict]:
    """Merge document metadata into each contextualized record and assign
    measurement_id, mirroring run_extraction.py's step_standardize_and_deduplicate.
    """
    dataset = []
    for i, dp in enumerate(records):
        info = text_info[dp["document_id"]]
        record = {k: v for k, v in dp.items() if k != "document_id"}
        record["document_id"] = info["document_id"]
        record["measurement_id"] = i
        dataset.append(record)
    return dataset


def run_pipeline(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    output_dir: Path,
    ocr_dir: str,
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
    step: str | None = None,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
) -> None:
    """Runs (or resumes) the three MeasurementLMv2 checkpoints:
    quantities.json (step 1) -> standardized.json (step 1.5 + code-only
    dedup) -> final.json (step 3). ``step`` runs exactly one named step,
    reading its inputs from files already in ``output_dir`` (mirrors
    run_extraction.py's run_single_step, at v2's coarser 3-checkpoint
    granularity).
    """
    effective_api_base, api_key = _resolve_api(model_config, api_base, api_key)

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"OCR dir   : {ocr_dir}")
    print(f"Output    : {output_dir}\n")

    text, text_info = load_papers(dataset_config, ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} papers.\n")

    mlm = _build_mlm(dataset_config, model_config, effective_api_base, api_key)
    mlm.data = [{"document_id": i, "context": p} for i, p in enumerate(text)]
    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    output_dir.mkdir(parents=True, exist_ok=True)
    f_quantities = output_dir / "quantities.json"
    f_standardized = output_dir / "standardized.json"
    f_final = output_dir / "final.json"

    start_time = time.time()

    if step in (None, "quantities"):
        if not (resume and f_quantities.exists()):
            print("Step 1 — Collecting quantities...")
            quantities = mlm._collect_quantities()
            with open(f_quantities, "w") as f:
                json.dump(quantities, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        else:
            print("Step 1 — Skipping (quantities.json exists).")
        if step == "quantities":
            write_run_metadata(
                output_dir, start_time=start_time, dataset=dataset_config.name, model=model_config.name,
                model_id=model_config.model_id, hf_revision=model_config.hf_revision, ocr_dir=ocr_dir,
                gpu_compatibility_warnings=gpu_warnings, max_prompt_tokens=mlm.max_prompt_tokens,
                hostname=urlparse(effective_api_base).hostname, step=step,
            )
            return

    if step in (None, "standardized"):
        with open(f_quantities) as f:
            quantities = json.load(f)
        if not (resume and f_standardized.exists()):
            print("Step 1.5+2 — Standardizing and deduplicating...")
            quantities = mlm._standardize_quantities(quantities)
            quantities = mlm._deduplicate_quantities(quantities)
            with open(f_standardized, "w") as f:
                json.dump(quantities, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        else:
            print("Step 1.5+2 — Skipping (standardized.json exists).")
        if step == "standardized":
            write_run_metadata(
                output_dir, start_time=start_time, dataset=dataset_config.name, model=model_config.name,
                model_id=model_config.model_id, hf_revision=model_config.hf_revision, ocr_dir=ocr_dir,
                gpu_compatibility_warnings=gpu_warnings, max_prompt_tokens=mlm.max_prompt_tokens,
                hostname=urlparse(effective_api_base).hostname, step=step,
            )
            return

    if step in (None, "final"):
        with open(f_standardized) as f:
            quantities = json.load(f)
        if not (resume and f_final.exists()):
            print("Step 3 — Contextualizing...")
            records = mlm._contextualize_quantities(quantities)
            dataset = _finalize_records(records, text_info)
            with open(f_final, "w") as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)
        else:
            print("Step 3 — Skipping (final.json exists).")

    write_run_metadata(
        output_dir, start_time=start_time, dataset=dataset_config.name, model=model_config.name,
        model_id=model_config.model_id, hf_revision=model_config.hf_revision, ocr_dir=ocr_dir,
        gpu_compatibility_warnings=gpu_warnings, max_prompt_tokens=mlm.max_prompt_tokens,
        hostname=urlparse(effective_api_base).hostname, step=step,
    )
    print(f"\nDone. Final dataset: {f_final}")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the MeasurementLMv2 extraction pipeline.",
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
    paths.require_params(params, "dataset", "model", "ocr_dir", config_path=config_path)

    step = params.get("step")
    if step is not None and step not in STEP_NAMES:
        raise ValueError(f"{config_path}: params.step must be one of {STEP_NAMES}, got {step!r}")

    exp_defaults = paths.load_config().get("defaults", {})
    repo_seed = exp_defaults.get("seed")
    if repo_seed is not None and cfg["seed"] != repo_seed:
        raise ValueError(
            f"{config_path}: seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed})."
        )
    set_seeds(cfg["seed"])

    dataset_config = load_dataset_config(params["dataset"])
    model_config = get_model_config(params["model"])
    output_dir = paths.result_dir(params["dataset"], "extraction_v2", cfg["id"])

    api_base = args.api_base or params.get("api_base") or "http://localhost:8081/v1"
    api_key = params.get("api_key", "EMPTY")

    run_pipeline(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        ocr_dir=params["ocr_dir"],
        paper_subset_override=params.get("paper_subset"),
        resume=params.get("resume", False),
        step=step,
        api_base=api_base,
        api_key=api_key,
    )


if __name__ == "__main__":
    main()
