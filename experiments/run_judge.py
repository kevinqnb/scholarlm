"""
Unified judge pipeline runner.

Runs validation for a given (dataset, extraction_model, judge_model) triple,
writing results to a structured output directory:

    data/experiments/{dataset}/judge/{extraction_model}/{judge_model}/{YYYY_mm_dd}/

For local judge models the runner saves:
  - ``responses.json``          — per-measurement judgement + probability scores
  - ``attention_outputs.npz``   — per-layer, per-head attention output activations

For frontier (API) judge models the runner saves:
  - ``responses.json``          — per-measurement judgement + raw text + token prob

Usage
-----
    # Local judge (NNsight / JudgementLM):
    python experiments/run_judge.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge llama-3.1-8b \\
        --extraction-date 2026_04_01

    # Frontier judge (batch API):
    python experiments/run_judge.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge openai \\
        --frontier-model gpt-4o-mini \\
        --extraction-date 2026_04_01

    # Step-by-step frontier (submit → poll → process):
    python experiments/run_judge.py \\
        --dataset pond --extraction-model gemma-3-27b \\
        --judge anthropic --frontier-model claude-haiku-4-5 \\
        --extraction-date 2026_04_01 \\
        submit
    python experiments/run_judge.py ... poll --state .batch_state_anthropic.json
    python experiments/run_judge.py ... process --state .batch_state_anthropic.json

Local judge models  : llama-3.1-8b, qwen-3-8b, gemma-3-12b
Frontier providers  : openai, anthropic, gemini
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = Path(__file__).parent / "configs"
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
# Make 'batch' importable as a package (experiments/batch/)
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch

from scholarlm import JudgementLM
from scholarlm.config import DatasetConfig
from scholarlm.instruction_prompts import JUDGE_INSTRUCTIONS_TABLE, JUDGE_INSTRUCTIONS_TEXT
from scholarlm.utils import get_filenames_in_directory

random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)

from model_registry import (
    INTERP_JUDGE_REGISTRY as LOCAL_JUDGE_REGISTRY,
    FRONTIER_JUDGE_PROVIDERS as FRONTIER_PROVIDERS,
)

# ---------------------------------------------------------------------------
# Config / path helpers (shared with run_extraction.py)
# ---------------------------------------------------------------------------


def _load_dataset_config(name: str) -> DatasetConfig:
    """Load a DatasetConfig by name from experiments/configs/<name>.py."""
    config_path = _CONFIGS_DIR / f"{name}.py"
    if not config_path.exists():
        available = sorted(p.stem for p in _CONFIGS_DIR.glob("*.py") if p.stem != "__init__")
        raise FileNotFoundError(
            f"No config found for dataset '{name}'. Available: {available}"
        )
    spec = importlib.util.spec_from_file_location(f"_dataset_config_{name}", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CONFIG


def get_judge_output_dir(
    dataset_name: str,
    extraction_model: str,
    judge_model: str,
    date: str | None = None,
) -> Path:
    """Return the output directory for a judge run.

    Path: ``data/experiments/{dataset}/judge/{extraction_model}/{judge_model}/{YYYY_mm_dd}/``
    """
    if date is None:
        date = datetime.now().strftime("%Y_%m_%d")
    return (
        _REPO_ROOT
        / "data" / "experiments"
        / dataset_name / "judge"
        / extraction_model / judge_model / date
    )


def _find_extraction_final(
    dataset_name: str,
    extraction_model: str,
    extraction_date: str | None,
) -> Path:
    """Locate the final.json produced by run_extraction for a given model.

    If ``extraction_date`` is provided, look in that specific date directory.
    Otherwise find the most recent date directory.
    """
    base = _REPO_ROOT / "data" / "experiments" / dataset_name / "extraction" / extraction_model
    if extraction_date:
        candidate = base / extraction_date / "final.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Extraction results not found: {candidate}")
        return candidate
    # Find most recent date directory
    date_dirs = sorted(base.iterdir(), reverse=True) if base.exists() else []
    for d in date_dirs:
        candidate = d / "final.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No extraction results found for dataset='{dataset_name}' model='{extraction_model}' "
        f"under {base}. Run run_extraction.py first."
    )


# ---------------------------------------------------------------------------
# Prompt builder (dataset-agnostic)
# ---------------------------------------------------------------------------


def _build_judge_messages(
    data: list[dict],
    documents: list[str],
    dataset_config: DatasetConfig,
) -> tuple[list[tuple[str, str, str]], list[int]]:
    """Build (instructions, document, query) triples for JudgementLM.

    Args:
        data: Extraction records from ``final.json``.
        documents: OCR texts indexed by ``document_id``.
        dataset_config: Provides entity fields, attribute catalogue, and entity
            type description.

    Returns:
        A tuple ``(messages, measurement_ids)`` where each message is an
        ``(instructions, context, query)`` triple.
    """
    entity_fields = set(dataset_config.entity_schema.model_fields.keys())
    attr_dict = dataset_config.attribute_info_dict
    entity_type_desc = dataset_config.entity_type_description

    messages: list[tuple[str, str, str]] = []
    measurement_ids: list[int] = []

    for entry in data:
        document = documents[entry["document_id"]]
        attribute = entry.get("attribute")
        attribute_description = attr_dict[attribute]["description"]
        attribute_terms = entry.get("attribute_terms", [])
        entity_description = {k: v for k, v in entry.items() if k in entity_fields}
        page_number = entry.get("page_number")
        table_number = entry.get("table_number")
        source = entry.get("source", "text")
        units = entry.get("units")
        units_str = units if units is not None else "not reported"

        entity_section = (
            f"Target entity type: {entity_type_desc}\n"
            f"Extracted entity: {entity_description}"
        )
        attribute_section = (
            f"Target attribute: {attribute_description}\n"
            f"Attribute terminology: {attribute_terms}"
        )

        location_parts = []
        if page_number is not None:
            location_parts.append(f"Page number: {page_number}")
        if source == "table" and table_number is not None:
            location_parts.append(f"Table number: {table_number}")
        location_section = "\n".join(location_parts)

        if source == "table":
            instructions = JUDGE_INSTRUCTIONS_TABLE
            row_index = entry.get("row_index")
            column_index = entry.get("column_index")
            value_section = (
                f"Extracted row index: {row_index}\n"
                f"Extracted column index: {column_index}\n"
                f"Extracted units: {units_str}"
            )
            closing = (
                "Is the extracted (entity, attribute, row index, column index) tuple fully valid — "
                "meaning the entity is correctly identified and together the row index and column index "
                "correctly locate the value for that (entity, target attribute) pair in the specified table?"
            )
        else:
            instructions = JUDGE_INSTRUCTIONS_TEXT
            measurement_val = entry["value"]
            value_section = (
                f"Extracted value: {measurement_val}\n"
                f"Extracted units: {units_str}"
            )
            closing = (
                "Is the extracted (entity, attribute, value) triplet fully valid — "
                "meaning the entity is correctly identified and the extracted value "
                "correctly corresponds to the target attribute for that entity, as evidenced by the document?"
            )

        sections = [entity_section, attribute_section]
        if location_section:
            sections.append(location_section)
        sections.append(value_section)
        sections.append(closing)

        messages.append((instructions, document, "\n\n".join(sections)))
        measurement_ids.append(entry["measurement_id"])

    return messages, measurement_ids


# ---------------------------------------------------------------------------
# Local judge runner
# ---------------------------------------------------------------------------


def run_local_judge(
    dataset_config: DatasetConfig,
    extraction_model: str,
    judge_key: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
) -> None:
    """Run a local (NNsight) judge and save responses + activations.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model whose results to judge.
        judge_key: Key in ``LOCAL_JUDGE_REGISTRY``.
        output_dir: Directory to write ``responses.json`` and ``attention_outputs.npz``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files used as document context.
            Defaults to ``{data_dir}/ocr_output_raw/``.
    """
    if judge_key not in LOCAL_JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown local judge '{judge_key}'. "
            f"Available: {sorted(LOCAL_JUDGE_REGISTRY.keys())}"
        )
    judge_cfg = LOCAL_JUDGE_REGISTRY[judge_key]

    input_file = _find_extraction_final(dataset_config.name, extraction_model, extraction_date)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")

    # Load documents in the same sorted order used during extraction
    text_files = get_filenames_in_directory(
        effective_ocr_dir, ignore=[".DS_Store", ".gitkeep"]
    )
    text_files.sort()
    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(effective_ocr_dir, fname), "r", encoding="utf-8") as fh:
            documents.append(fh.read())

    messages, measurement_ids = _build_judge_messages(data, documents, dataset_config)

    llm = JudgementLM(
        model_name=judge_cfg["model_id"],
        sampling_params=judge_cfg["sampling_params"],
        nnsight_kwargs=judge_cfg["nnsight_kwargs"],
    )

    responses = llm.predict(messages)

    output_dir.mkdir(parents=True, exist_ok=True)
    judged_data: list[dict] = []
    attn_output_dict: dict[str, Any] = {}

    for i, response in enumerate(responses):
        mid = str(measurement_ids[i])
        judged_data.append(
            data[i] | {
                "judgement": "true" in response["response"].strip().lower(),
                "judgement_prob": math.exp(float(response["logprob"])),
                "judgement_p_true": float(response["p_true"]),
                "judgement_p_false": float(response["p_false"]),
                "judgement_logit_p_true": float(response["logit_p_true"]),
                "judgement_logit_p_false": float(response["logit_p_false"]),
                "judgement_model": judge_cfg["model_id"],
            }
        )
        if response.get("attn_output") is not None:
            attn_output_dict[mid] = response["attn_output"]

    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(judged_data, f, indent=4, ensure_ascii=False)
    print(f"Responses saved to {responses_file}")

    if attn_output_dict:
        attn_file = output_dir / "attention_outputs.npz"
        np.savez_compressed(attn_file, **attn_output_dict)
        print(f"Activations saved to {attn_file}")


# ---------------------------------------------------------------------------
# Frontier judge runner (delegates to pond batch infrastructure)
# ---------------------------------------------------------------------------


def _build_frontier_chat_entries(
    data: list[dict],
    documents: list[str],
    dataset_config: DatasetConfig,
) -> list[dict]:
    """Build provider-agnostic batch chat entries using the dataset config.

    Delegates to the updated ``batch.common.prepare_chat_entries`` passing the
    dataset config so entity fields, attribute descriptions, and the entity
    type description are sourced from the config rather than pond defaults.
    """
    from batch import common as batch_common
    return batch_common.prepare_chat_entries(data, documents, dataset_config)


def run_frontier_judge(
    dataset_config: DatasetConfig,
    extraction_model: str,
    provider: str,
    frontier_model: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    dest_gcs: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    interval: int = 60,
    state_file: str | None = None,
) -> None:
    """Run a frontier (API) batch judge and save responses.

    Uses the provider batch modules from ``experiments/pond/judge/batch/``.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model.
        provider: One of ``"openai"``, ``"anthropic"``, ``"gemini"``.
        frontier_model: Provider-specific model name (e.g. ``"gpt-4o-mini"``).
        output_dir: Directory to write ``responses.json``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files used as document context.
            Defaults to ``{data_dir}/ocr_output_raw/``.
        dest_gcs: GCS URI (Gemini only).
        gcp_project: GCP project (Gemini only).
        gcp_location: GCP region (Gemini only).
        interval: Poll interval in seconds.
        state_file: Optional path for batch state JSON.
    """
    if provider not in FRONTIER_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {FRONTIER_PROVIDERS}")

    input_file = _find_extraction_final(dataset_config.name, extraction_model, extraction_date)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")

    text_files = get_filenames_in_directory(
        effective_ocr_dir, ignore=[".DS_Store", ".gitkeep"]
    )
    text_files.sort()
    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(effective_ocr_dir, fname), "r", encoding="utf-8") as fh:
            documents.append(fh.read())

    chat_entries = _build_frontier_chat_entries(data, documents, dataset_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if state_file is None:
        state_file = str(output_dir / f".batch_state_{provider}.json")

    # Import batch modules (relative imports inside batch/ are resolved since
    # _BATCH_PARENT = experiments/pond/judge/ is on sys.path)
    from batch import common as batch_common
    from batch import openai_batch, anthropic_batch, gemini_batch

    # Submit
    state: dict[str, Any] = {"provider": provider, "model": frontier_model}
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        requests = openai_batch.build_requests(chat_entries, frontier_model)
        batch_ids = openai_batch.submit_batch(requests, client=client)
        state["batch_ids"] = batch_ids

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        requests = anthropic_batch.build_requests(chat_entries, frontier_model)
        batch_ids = anthropic_batch.submit_batch(requests, client=client)
        state["batch_ids"] = batch_ids

    elif provider == "gemini":
        requests = gemini_batch.build_requests(chat_entries, frontier_model)
        batch_names = gemini_batch.submit_batch(
            requests, frontier_model,
            dest_gcs=dest_gcs, project=gcp_project, location=gcp_location,
        )
        state["batch_names"] = batch_names
        state["dest_gcs"] = dest_gcs
        if gcp_project:
            state["gcp_project"] = gcp_project
        if gcp_location:
            state["gcp_location"] = gcp_location

    Path(state_file).write_text(json.dumps(state, indent=2))
    print(f"Batch submitted. State saved to {state_file}")

    # Poll then fetch results
    if provider == "openai":
        from openai import OpenAI
        openai_batch.poll_batch(state["batch_ids"], client=OpenAI(), interval=interval)
        results = openai_batch.fetch_results(state["batch_ids"], client=OpenAI(), model=frontier_model)

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        anthropic_batch.poll_batch(state["batch_ids"], client=client, interval=interval)
        results = anthropic_batch.fetch_results(state["batch_ids"], client=client, model=frontier_model)

    elif provider == "gemini":
        gemini_batch.poll_batch(
            state["batch_names"],
            project=state.get("gcp_project"),
            location=state.get("gcp_location"),
            interval=interval,
        )
        results = gemini_batch.fetch_results(
            state["batch_names"], model=frontier_model,
            dest_gcs=state["dest_gcs"],
            project=state.get("gcp_project"),
        )

    data_out = batch_common.merge_results(data, results)
    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(data_out, f, indent=4, ensure_ascii=False)
    print(f"Responses saved to {responses_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run judge pipeline for a (dataset, extraction_model, judge) triple.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'pond', 'nfix').")
    p.add_argument(
        "--extraction-model", required=True,
        help="Short name of the extraction model whose results to judge.",
    )
    p.add_argument(
        "--judge", required=True,
        help=(
            "Judge identifier: a key from LOCAL_JUDGE_REGISTRY "
            f"({sorted(LOCAL_JUDGE_REGISTRY.keys())}) or a frontier provider "
            f"({sorted(FRONTIER_PROVIDERS)})."
        ),
    )
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd of extraction run.")
    p.add_argument("--judge-date", default=None, help="Date tag for output directory (default: today).")
    p.add_argument(
        "--ocr-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory of OCR .txt files to use as document context. "
            "Defaults to {data_dir}/ocr_output_raw/. Pass the cleaned OCR dir "
            "to match the texts used during extraction."
        ),
    )
    p.add_argument("--frontier-model", default=None, help="Provider model name (frontier judges only).")
    p.add_argument("--dest-gcs", default=None, help="GCS URI (Gemini only).")
    p.add_argument("--gcp-project", default=None, help="GCP project (Gemini only).")
    p.add_argument("--gcp-location", default=None, help="GCP region (Gemini only).")
    p.add_argument("--interval", type=int, default=60, help="Poll interval in seconds (frontier only).")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    dataset_config = _load_dataset_config(args.dataset)
    judge_label = args.judge

    output_dir = get_judge_output_dir(
        args.dataset, args.extraction_model, judge_label, args.judge_date
    )
    print(f"\nDataset          : {args.dataset}")
    print(f"Extraction model : {args.extraction_model}")
    print(f"Judge            : {judge_label}")
    print(f"Output           : {output_dir}\n")

    if judge_label in LOCAL_JUDGE_REGISTRY:
        run_local_judge(
            dataset_config=dataset_config,
            extraction_model=args.extraction_model,
            judge_key=judge_label,
            output_dir=output_dir,
            extraction_date=args.extraction_date,
            ocr_dir=args.ocr_dir,
        )
    elif judge_label in FRONTIER_PROVIDERS:
        if args.frontier_model is None:
            raise ValueError("--frontier-model is required for frontier judges.")
        run_frontier_judge(
            dataset_config=dataset_config,
            extraction_model=args.extraction_model,
            provider=judge_label,
            frontier_model=args.frontier_model,
            output_dir=output_dir,
            extraction_date=args.extraction_date,
            ocr_dir=args.ocr_dir,
            dest_gcs=args.dest_gcs,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
            interval=args.interval,
        )
    else:
        raise ValueError(
            f"Unknown judge '{judge_label}'. "
            f"Use a local model key ({sorted(LOCAL_JUDGE_REGISTRY.keys())}) "
            f"or a frontier provider ({sorted(FRONTIER_PROVIDERS)})."
        )


if __name__ == "__main__":
    main()
