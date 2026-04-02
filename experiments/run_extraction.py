"""
Unified extraction pipeline runner.

Runs the full MeasurementLM extraction pipeline for any registered dataset and
model, writing intermediate and final results to a structured output directory:

    data/experiments/{dataset}/extraction/{model}/{YYYY_mm_dd}/

Usage
-----
    # From the repo root:
    python experiments/run_extraction.py --dataset pond --model gemma-3-27b
    python experiments/run_extraction.py --dataset nfix --model qwen-2.5-72b
    python experiments/run_extraction.py --dataset pond --model llama-3.3-70b \\
        --paper-subset physical_and_chemical_limnological prairie_wetland

    # Resume from a specific step (skips steps whose output files already exist):
    python experiments/run_extraction.py --dataset pond --model gemma-3-27b --resume

Available datasets: any file in experiments/configs/<name>.py that exports CONFIG.
Available models:   keys of MODEL_REGISTRY in this file.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm importable when run directly from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = Path(__file__).parent / "configs"
sys.path.insert(0, str(_REPO_ROOT / "src"))

import torch
from scholarlm import MeasurementLM
from scholarlm.config import DatasetConfig, ModelConfig
from scholarlm.measurementlm import NumpyEncoder
from scholarlm.utils import get_filenames_in_directory

# Reproducibility
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, ModelConfig] = {
    "gemma-3-27b": ModelConfig(
        name="gemma-3-27b",
        model_id="gaunernst/gemma-3-27b-it-qat-autoawq",
        tensor_parallel_size=1,
        sampling_params={
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
            "seed": 342,
        },
    ),
    "qwen-2.5-72b": ModelConfig(
        name="qwen-2.5-72b",
        model_id="Qwen/Qwen2.5-72B-Instruct",
        tensor_parallel_size=2,
        sampling_params={
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
            "seed": 342,
        },
    ),
    "llama-3.3-70b": ModelConfig(
        name="llama-3.3-70b",
        model_id="meta-llama/Llama-3.3-70B-Instruct",
        tensor_parallel_size=2,
        sampling_params={
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 64,
            "max_tokens": 8192,
            "seed": 342,
        },
    ),
    "qwen-3.5-35b": ModelConfig(
        name="qwen-3.5-35b",
        model_id="Qwen/Qwen3.5-35B-A3B-FP8",
        tensor_parallel_size=1,
        sampling_params={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": 81920,
            "seed": 342,
        },
    ),
    "gpt-oss-120b": ModelConfig(
        name="gpt-oss-120b",
        model_id="openai/gpt-oss-120b",
        tensor_parallel_size=1,
        sampling_params={
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "max_tokens": 8192,
            "seed": 342,
        },
    ),
}

# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_dataset_config(name: str) -> DatasetConfig:
    """Load a DatasetConfig by name from experiments/configs/<name>.py.

    The config file must define a module-level ``CONFIG`` variable of type
    ``DatasetConfig``.

    Args:
        name: Dataset identifier matching a file in ``experiments/configs/``.

    Returns:
        The ``DatasetConfig`` instance exported by the config file.

    Raises:
        FileNotFoundError: If no config file exists for ``name``.
        AttributeError: If the config file does not define ``CONFIG``.
    """
    config_path = _CONFIGS_DIR / f"{name}.py"
    if not config_path.exists():
        available = sorted(p.stem for p in _CONFIGS_DIR.glob("*.py") if p.stem != "__init__")
        raise FileNotFoundError(
            f"No config found for dataset '{name}'. "
            f"Available datasets: {available}"
        )
    spec = importlib.util.spec_from_file_location(f"_dataset_config_{name}", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CONFIG"):
        raise AttributeError(
            f"Config file {config_path} must define a module-level 'CONFIG' variable "
            f"of type DatasetConfig."
        )
    return mod.CONFIG


def get_model_config(name: str) -> ModelConfig:
    """Retrieve a ModelConfig from MODEL_REGISTRY by short name.

    Args:
        name: Model key in ``MODEL_REGISTRY``.

    Returns:
        The corresponding ``ModelConfig``.

    Raises:
        KeyError: If ``name`` is not in the registry.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model '{name}'. "
            f"Available models: {sorted(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------


def get_output_dir(dataset_name: str, model_name: str, date: str | None = None) -> Path:
    """Return the output directory for a given (dataset, model, date) triple.

    Path convention: ``data/experiments/{dataset}/extraction/{model}/{YYYY_mm_dd}/``

    Args:
        dataset_name: Dataset identifier (e.g. ``"pond"``).
        model_name: Model identifier (e.g. ``"qwen-2.5-72b"``).
        date: Optional date string ``"YYYY_mm_dd"``. Defaults to today.

    Returns:
        A ``Path`` object (not yet created on disk).
    """
    if date is None:
        date = datetime.now().strftime("%Y_%m_%d")
    return _REPO_ROOT / "data" / "experiments" / dataset_name / "extraction" / model_name / date


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_papers(
    dataset_config: DatasetConfig,
    paper_subset_override: list[str] | None = None,
) -> tuple[list[str], list[dict]]:
    """Load OCR text and metadata for all papers selected by the dataset config.

    Applies ``dataset_config.paper_filter`` first (if set), then intersects with
    ``paper_subset_override`` if provided, otherwise uses ``dataset_config.paper_subset``.

    Args:
        dataset_config: The dataset configuration to load papers for.
        paper_subset_override: If given, overrides ``dataset_config.paper_subset``.

    Returns:
        A tuple ``(text, text_info)`` where:
        - ``text[i]`` is the full OCR text of the i-th paper.
        - ``text_info[i]`` is the metadata dict for the i-th paper
          (includes ``paper_code`` plus whatever keys are in ``directory.json``).
    """
    with open(dataset_config.metadata_file) as f:
        paper_info: dict[str, dict] = json.load(f)

    text_files = get_filenames_in_directory(
        dataset_config.ocr_dir, ignore=[".DS_Store", ".gitkeep"]
    )
    text_files.sort()

    # Apply metadata filter
    if dataset_config.paper_filter is not None:
        registered_ids = {
            k for k, v in paper_info.items() if dataset_config.paper_filter(v)
        }
        text_files = [f for f in text_files if f.replace(".txt", "") in registered_ids]

    # Determine effective paper subset
    effective_subset = paper_subset_override if paper_subset_override is not None else dataset_config.paper_subset
    if effective_subset is not None:
        subset_set = set(effective_subset)
        text_files = [f for f in text_files if f.replace(".txt", "") in subset_set]

    text: list[str] = []
    text_info: list[dict] = []
    for fname in text_files:
        paper_code = fname.replace(".txt", "")
        filepath = os.path.join(dataset_config.ocr_dir, fname)
        with open(filepath, "r", encoding="utf-8") as fh:
            text.append(fh.read())
        metadata = dict(paper_info.get(paper_code, {}))
        metadata["paper_code"] = paper_code
        text_info.append(metadata)

    return text, text_info


# ---------------------------------------------------------------------------
# Provenance serialization helpers (shared across both datasets)
# ---------------------------------------------------------------------------


def _serialize_prov(prov_dict: dict) -> dict:
    """Serialize tuple-keyed provenance dict for JSON storage.

    Converts ``{(doc_id, item_id): value}`` to ``{"doc_id|item_id": value}``.
    """
    return {f"{k[0]}|{k[1]}": v for k, v in prov_dict.items()}


def _deserialize_prov(json_dict: dict) -> dict:
    """Deserialize JSON provenance dict back to tuple keys."""
    out: dict = {}
    for k, v in json_dict.items():
        parts = k.split("|", 1)
        try:
            doc_id: int | str = int(parts[0])
        except ValueError:
            doc_id = parts[0]
        out[(doc_id, parts[1])] = v
    return out


# ---------------------------------------------------------------------------
# Pipeline step functions
# ---------------------------------------------------------------------------


def step_extract_entities(
    mlm: MeasurementLM,
    text: list[str],
    outfile: Path,
) -> None:
    """Step 1: Extract entities from each document and save to JSON.

    Args:
        mlm: Configured ``MeasurementLM`` instance.
        text: List of OCR text strings, one per document.
        outfile: Destination JSON path.
    """
    print("Step 1 — Extracting entities...")
    mlm.data = [{"document_id": i, "context": paper} for i, paper in enumerate(text)]
    data = mlm._extract_entities()
    # Strip context before saving; it is re-injected from text[] in later steps.
    save_data = [{k: v for k, v in r.items() if k != "context"} for r in data]
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)


def step_detect_attributes(
    mlm: MeasurementLM,
    text: list[str],
    outfile: Path,
) -> None:
    """Step 2: Document-level attribute detection and save to JSON.

    Args:
        mlm: Configured ``MeasurementLM`` instance.
        text: List of OCR text strings, one per document.
        outfile: Destination JSON path.
    """
    print("Step 2 — Detecting attributes...")
    mlm.data = [{"document_id": i, "context": paper} for i, paper in enumerate(text)]
    doc_attributes = mlm._detect_attributes()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(doc_attributes, f, indent=4, ensure_ascii=False)


def step_entity_provenance(
    mlm: MeasurementLM,
    text: list[str],
    entities_file: Path,
    outfile: Path,
) -> None:
    """Step 3a: Per-page entity provenance and save to JSON.

    Args:
        mlm: Configured ``MeasurementLM`` instance.
        text: List of OCR text strings, one per document.
        entities_file: Path to the entities JSON produced by step 1.
        outfile: Destination JSON path.
    """
    print("Step 3a — Running entity provenance...")
    with open(entities_file) as f:
        entity_data: list[dict] = json.load(f)
    for record in entity_data:
        record["context"] = text[record["document_id"]]
    prov = mlm._entity_provenance(entity_data)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(_serialize_prov(prov), f, indent=4, ensure_ascii=False)


def step_attribute_provenance(
    mlm: MeasurementLM,
    text: list[str],
    attributes_file: Path,
    outfile: Path,
) -> None:
    """Step 3b: Per-page attribute provenance and save to JSON.

    Args:
        mlm: Configured ``MeasurementLM`` instance.
        text: List of OCR text strings, one per document.
        attributes_file: Path to the attributes JSON produced by step 2.
        outfile: Destination JSON path.
    """
    print("Step 3b — Running attribute provenance...")
    with open(attributes_file) as f:
        doc_attributes: dict = json.load(f)
    mlm.data = [{"document_id": i, "context": paper} for i, paper in enumerate(text)]
    prov = mlm._attribute_provenance(doc_attributes)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(_serialize_prov(prov), f, indent=4, ensure_ascii=False)


def step_extract_values(
    mlm: MeasurementLM,
    text: list[str],
    entities_file: Path,
    attributes_file: Path,
    entity_prov_file: Path,
    attr_prov_file: Path,
    outfile: Path,
) -> None:
    """Steps 4+5: Extract values from text and tables and save to JSON.

    Args:
        mlm: Configured ``MeasurementLM`` instance.
        text: List of OCR text strings, one per document.
        entities_file: Path to entities JSON (step 1 output).
        attributes_file: Path to attributes JSON (step 2 output).
        entity_prov_file: Path to entity provenance JSON (step 3a output).
        attr_prov_file: Path to attribute provenance JSON (step 3b output).
        outfile: Destination JSON path.
    """
    print("Steps 4+5 — Extracting values from text and tables...")
    with open(entities_file) as f:
        entity_data: list[dict] = json.load(f)
    with open(attributes_file) as f:
        doc_attributes: dict = json.load(f)
    with open(entity_prov_file) as f:
        entity_prov = _deserialize_prov(json.load(f))
    with open(attr_prov_file) as f:
        attr_prov = _deserialize_prov(json.load(f))

    doc_attributes = {int(k): v for k, v in doc_attributes.items()}
    for record in entity_data:
        record["context"] = text[record["document_id"]]

    text_values = mlm._extract_values_from_text(entity_data, doc_attributes, entity_prov, attr_prov)
    table_values = mlm._extract_values_from_tables(entity_data, doc_attributes, entity_prov, attr_prov)

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(text_values + table_values, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


def step_standardize_and_deduplicate(
    mlm: MeasurementLM,
    text_info: list[dict],
    infile: Path,
    outfile: Path,
) -> None:
    """Steps 6+7: Standardize units and deduplicate, then save final dataset.

    Merges each deduplicated measurement with its document metadata and assigns
    a sequential ``measurement_id``.

    Args:
        mlm: Configured ``MeasurementLM`` instance.
        text_info: List of paper metadata dicts (same order as ``text``).
        infile: Path to values JSON (step 4+5 output).
        outfile: Destination JSON path for the final dataset.
    """
    print("Steps 6+7 — Standardizing and deduplicating...")
    with open(infile) as f:
        mlm.data = json.load(f)
    standardized = mlm._standardize()
    deduplicated = mlm._deduplicate(standardized)

    dataset = [
        text_info[dp["document_id"]] | dp | {"measurement_id": i}
        for i, dp in enumerate(deduplicated)
    ]

    outfile.parent.mkdir(parents=True, exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)


# ---------------------------------------------------------------------------
# Main pipeline orchestrator
# ---------------------------------------------------------------------------


def _run_all_steps(
    mlm: MeasurementLM,
    text: list[str],
    text_info: list[dict],
    work_dir: Path,
    resume: bool = False,
) -> None:
    """Run all pipeline steps, writing outputs into work_dir."""
    f_entities = work_dir / "entities.json"
    f_attributes = work_dir / "attributes.json"
    f_entity_prov = work_dir / "entity_prov.json"
    f_attr_prov = work_dir / "attribute_prov.json"
    f_values = work_dir / "values.json"
    f_final = work_dir / "final.json"

    if not (resume and f_entities.exists()):
        step_extract_entities(mlm, text, f_entities)
    else:
        print("Step 1 — Skipping (entities.json exists).")

    if not (resume and f_attributes.exists()):
        step_detect_attributes(mlm, text, f_attributes)
    else:
        print("Step 2 — Skipping (attributes.json exists).")

    if not (resume and f_entity_prov.exists()):
        step_entity_provenance(mlm, text, f_entities, f_entity_prov)
    else:
        print("Step 3a — Skipping (entity_prov.json exists).")

    if not (resume and f_attr_prov.exists()):
        step_attribute_provenance(mlm, text, f_attributes, f_attr_prov)
    else:
        print("Step 3b — Skipping (attribute_prov.json exists).")

    if not (resume and f_values.exists()):
        step_extract_values(mlm, text, f_entities, f_attributes, f_entity_prov, f_attr_prov, f_values)
    else:
        print("Steps 4+5 — Skipping (values.json exists).")

    if not (resume and f_final.exists()):
        step_standardize_and_deduplicate(mlm, text_info, f_values, f_final)
    else:
        print("Steps 6+7 — Skipping (final.json exists).")


def run_pipeline(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    output_dir: Path,
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
    final_only: bool = False,
) -> None:
    """Run the full extraction pipeline for a dataset / model pair.

    When ``final_only=False`` (default), writes six files to ``output_dir``:
    - ``entities.json``       — Step 1: identified entities
    - ``attributes.json``     — Step 2: detected attributes per document
    - ``entity_prov.json``    — Step 3a: entity provenance per page
    - ``attribute_prov.json`` — Step 3b: attribute provenance per page
    - ``values.json``         — Steps 4+5: raw extracted values
    - ``final.json``          — Steps 6+7: standardized, deduplicated dataset

    When ``final_only=True``, intermediate files are written to a temporary
    directory that is cleaned up automatically; only ``final.json`` is copied
    to ``output_dir``.

    Args:
        dataset_config: Dataset configuration loaded from ``experiments/configs/``.
        model_config: Model configuration from ``MODEL_REGISTRY``.
        output_dir: Directory for output files (created if needed).
        paper_subset_override: If provided, overrides ``dataset_config.paper_subset``.
        resume: If ``True``, skip steps whose output files already exist.
        final_only: If ``True``, keep only ``final.json``; discard intermediates.
    """
    print(f"\nDataset : {dataset_config.name}")
    print(f"Model   : {model_config.name} ({model_config.model_id})")
    print(f"Output  : {output_dir}\n")

    text, text_info = load_papers(dataset_config, paper_subset_override)
    print(f"Loaded {len(text)} papers.\n")

    mlm = MeasurementLM(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        sampling_params=model_config.sampling_params,
        tensor_parallel_size=model_config.tensor_parallel_size,
    )

    if final_only:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _run_all_steps(mlm, text, text_info, tmp_path, resume=False)
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tmp_path / "final.json", output_dir / "final.json")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        _run_all_steps(mlm, text, text_info, output_dir, resume=resume)

    print(f"\nDone. Final dataset: {output_dir / 'final.json'}")


STEP_NAMES = ("entities", "attributes", "entity_prov", "attribute_prov", "values", "final")


def run_single_step(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    output_dir: Path,
    step: str,
    paper_subset_override: list[str] | None = None,
) -> None:
    """Run a single named pipeline step, reading inputs from and writing output to output_dir.

    Previous steps' output files must already exist in ``output_dir``.

    Args:
        dataset_config: Dataset configuration.
        model_config: Model configuration from ``MODEL_REGISTRY``.
        output_dir: Directory containing prior step outputs and receiving this step's output.
        step: One of ``entities``, ``attributes``, ``entity_prov``, ``attribute_prov``,
              ``values``, ``final``.
        paper_subset_override: If provided, overrides ``dataset_config.paper_subset``.
    """
    if step not in STEP_NAMES:
        raise ValueError(f"Unknown step '{step}'. Choose from: {STEP_NAMES}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset : {dataset_config.name}")
    print(f"Model   : {model_config.name} ({model_config.model_id})")
    print(f"Step    : {step}")
    print(f"Output  : {output_dir}\n")

    text, text_info = load_papers(dataset_config, paper_subset_override)
    print(f"Loaded {len(text)} papers.\n")

    mlm = MeasurementLM(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        sampling_params=model_config.sampling_params,
        tensor_parallel_size=model_config.tensor_parallel_size,
    )

    f_entities = output_dir / "entities.json"
    f_attributes = output_dir / "attributes.json"
    f_entity_prov = output_dir / "entity_prov.json"
    f_attr_prov = output_dir / "attribute_prov.json"
    f_values = output_dir / "values.json"
    f_final = output_dir / "final.json"

    if step == "entities":
        step_extract_entities(mlm, text, f_entities)
    elif step == "attributes":
        step_detect_attributes(mlm, text, f_attributes)
    elif step == "entity_prov":
        step_entity_provenance(mlm, text, f_entities, f_entity_prov)
    elif step == "attribute_prov":
        step_attribute_provenance(mlm, text, f_attributes, f_attr_prov)
    elif step == "values":
        step_extract_values(mlm, text, f_entities, f_attributes, f_entity_prov, f_attr_prov, f_values)
    elif step == "final":
        step_standardize_and_deduplicate(mlm, text_info, f_values, f_final)

    print(f"\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the MeasurementLM extraction pipeline.",
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
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Override dataset paper_subset with an explicit list of paper codes.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Skip pipeline steps whose output files already exist (full pipeline only).",
    )
    p.add_argument(
        "--final-only",
        action="store_true",
        help=(
            "Run the full pipeline but save only final.json to the output directory; "
            "intermediate files are written to a temporary directory and discarded. "
            "Mutually exclusive with --step."
        ),
    )
    p.add_argument(
        "--step",
        choices=list(STEP_NAMES),
        default=None,
        metavar="STEP",
        help=(
            "Run a single named step and exit. Previous steps' outputs must already "
            f"exist in the output directory. Choices: {{{', '.join(STEP_NAMES)}}}. "
            "Mutually exclusive with --final-only."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.final_only and args.step:
        raise SystemExit("error: --final-only and --step are mutually exclusive.")
    if args.resume and args.step:
        raise SystemExit("error: --resume has no effect when --step is given.")

    dataset_config = load_dataset_config(args.dataset)
    model_config = get_model_config(args.model)
    output_dir = get_output_dir(args.dataset, args.model, args.date)

    if args.step:
        run_single_step(
            dataset_config=dataset_config,
            model_config=model_config,
            output_dir=output_dir,
            step=args.step,
            paper_subset_override=args.paper_subset,
        )
    else:
        run_pipeline(
            dataset_config=dataset_config,
            model_config=model_config,
            output_dir=output_dir,
            paper_subset_override=args.paper_subset,
            resume=args.resume,
            final_only=args.final_only,
        )


if __name__ == "__main__":
    main()
