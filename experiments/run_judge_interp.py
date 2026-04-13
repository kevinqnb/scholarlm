"""
Interpretability judge pipeline (NNsight / JudgementLM).

Runs judge validation for a given (dataset, extraction_model, judge_model) triple
using a local model loaded through NNsight, collecting per-layer, per-head
attention output activations alongside binary judgement probabilities.

Output path:
    data/experiments/{dataset}/judge/{extraction_model}/{judge_model}/{YYYY_mm_dd}/

Saves:
  - ``responses.json``        — per-measurement judgement + probability scores
  - ``attention_outputs.npz`` — per-layer, per-head attention output activations

Usage
-----
    python experiments/run_judge_interp.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge llama-3.1-8b \\
        --extraction-date 2026_04_01

Available judge models: llama-3.1-8b, gemma-2-9b, mistral-7b (see JUDGE_REGISTRY in code for details).
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
sys.path.insert(0, str(_REPO_ROOT / "src"))

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

from model_registry import INTERP_JUDGE_REGISTRY as JUDGE_REGISTRY

# ---------------------------------------------------------------------------
# Config / path helpers
# ---------------------------------------------------------------------------


def _load_dataset_config(name: str) -> DatasetConfig:
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
    extraction_date: str,
    judge_model: str,
    judge_date: str | None = None,
) -> Path:
    """Return ``data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_model}/{judge_date}/``."""
    if judge_date is None:
        judge_date = datetime.now().strftime("%Y_%m_%d")
    return (
        _REPO_ROOT
        / "data" / "experiments"
        / dataset_name / "judge"
        / extraction_model / extraction_date / judge_model / judge_date
    )


def _find_extraction_final(
    dataset_name: str,
    extraction_model: str,
    extraction_date: str | None,
) -> Path:
    base = _REPO_ROOT / "data" / "experiments" / dataset_name / "extraction" / extraction_model
    if extraction_date:
        candidate = base / extraction_date / "final.json"
        if not candidate.exists():
            raise FileNotFoundError(f"Extraction results not found: {candidate}")
        return candidate
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
# Prompt builder
# ---------------------------------------------------------------------------


def _build_judge_messages(
    data: list[dict],
    documents: list[str],
    dataset_config: DatasetConfig,
) -> tuple[list[tuple[str, str, str]], list[int]]:
    """Build (instructions, document, query) triples for JudgementLM.

    Returns:
        ``(messages, measurement_ids)`` where each message is an
        ``(instructions, context, query)`` triple formatted for JudgementLM.
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
# Runner
# ---------------------------------------------------------------------------


def run_interp_judge(
    dataset_config: DatasetConfig,
    extraction_model: str,
    judge_key: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
) -> None:
    """Run a local NNsight judge and save responses + attention activations.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model whose results to judge.
        judge_key: Key in ``JUDGE_REGISTRY``.
        output_dir: Directory to write ``responses.json`` and ``attention_outputs.npz``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
    """
    if judge_key not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge '{judge_key}'. Available: {sorted(JUDGE_REGISTRY.keys())}"
        )
    judge_cfg = JUDGE_REGISTRY[judge_key]

    input_file = _find_extraction_final(dataset_config.name, extraction_model, extraction_date)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    text_files = get_filenames_in_directory(effective_ocr_dir, ignore=[".DS_Store", ".gitkeep"])
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
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run interpretability judge (NNsight/JudgementLM).",
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
        choices=sorted(JUDGE_REGISTRY.keys()),
        help=f"Judge model key. Available: {sorted(JUDGE_REGISTRY.keys())}",
    )
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd of extraction run.")
    p.add_argument("--judge-date", default=None, help="Date tag for output directory (default: today).")
    p.add_argument(
        "--ocr-dir", default=None, metavar="DIR",
        help=(
            "Directory of OCR .txt files to use as document context. "
            "Defaults to {data_dir}/ocr_output_raw/."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    dataset_config = _load_dataset_config(args.dataset)
    input_file = _find_extraction_final(args.dataset, args.extraction_model, args.extraction_date)
    extraction_date_resolved = input_file.parent.name
    output_dir = get_judge_output_dir(
        args.dataset, args.extraction_model, extraction_date_resolved, args.judge, args.judge_date
    )
    print(f"\nDataset          : {args.dataset}")
    print(f"Extraction model : {args.extraction_model}")
    print(f"Extraction date  : {extraction_date_resolved}")
    print(f"Judge            : {args.judge}")
    print(f"Output           : {output_dir}\n")

    run_interp_judge(
        dataset_config=dataset_config,
        extraction_model=args.extraction_model,
        judge_key=args.judge,
        output_dir=output_dir,
        extraction_date=extraction_date_resolved,
        ocr_dir=args.ocr_dir,
    )


if __name__ == "__main__":
    main()
