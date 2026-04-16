"""
Interpretability judge pipeline (NNsight / JudgementLM).

Runs judge validation for a given (dataset, extraction_model, judge_model) triple
using a local model loaded through NNsight, collecting per-layer, per-head
attention output activations alongside binary judgement probabilities.

Output path:
    data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_model}/{judge_date}/

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
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))  # makes 'batch' importable

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch

from scholarlm import JudgementLM
from scholarlm.config import DatasetConfig
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
    ablation: str | None = None,
) -> Path:
    """Return the judge output directory.

    Without ablation:
        ``data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_model}/{judge_date}/``
    With ablation:
        ``data/experiments/{dataset}/ablations/ablation{N}/{extraction_model}/{extraction_date}/judge/{judge_model}/{judge_date}/``
    """
    if judge_date is None:
        judge_date = datetime.now().strftime("%Y_%m_%d")
    if ablation is not None:
        return (
            _REPO_ROOT
            / "data" / "experiments"
            / dataset_name / "ablations" / f"ablation{ablation}"
            / extraction_model / extraction_date / "judge" / judge_model / judge_date
        )
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
    ablation: str | None = None,
) -> Path:
    if ablation is not None:
        base = _REPO_ROOT / "data" / "experiments" / dataset_name / "ablations" / f"ablation{ablation}" / extraction_model
    else:
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
# Runner
# ---------------------------------------------------------------------------


def run_interp_judge(
    dataset_config: DatasetConfig,
    extraction_model: str,
    judge_key: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    ablation: str | None = None,
) -> None:
    """Run a local NNsight judge and save responses + attention activations.

    Prompts are built via ``batch.common.prepare_chat_entries`` — the same
    function used by the local vLLM and frontier judge runners — so the query
    content (entity description, attribute description, value/indices, closing
    question) is identical across all judge backends.  JudgementLM receives
    the three parts separately as (instructions, context, query), which it
    wraps into a single user message internally.

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

    input_file = _find_extraction_final(dataset_config.name, extraction_model, extraction_date, ablation)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    from batch import common as batch_common
    documents = batch_common.load_documents_for_dataset(dataset_config, effective_ocr_dir)

    # Build prompts using the shared batch prompt builder.
    # prepare_chat_entries sorts by document_id for cache locality; custom_id
    # preserves the original index so results can be merged back in order.
    chat_entries = batch_common.prepare_chat_entries(data, documents, dataset_config)

    # JudgementLM takes (instructions, context, query) triples separately.
    # instructions = system prompt, context = extracted page(s), query = ## QUERY content.
    messages: list[tuple[str, str, str]] = [
        (entry["system"], entry["page_text"], entry["user_query"])
        for entry in chat_entries
    ]

    llm = JudgementLM(
        model_name=judge_cfg["model_id"],
        sampling_params=judge_cfg["sampling_params"],
        nnsight_kwargs=judge_cfg["nnsight_kwargs"],
    )

    responses = llm.predict(messages)

    # Map results back to original data order via custom_id.
    result_by_orig_idx: dict[int, dict] = {}
    attn_output_dict: dict[str, Any] = {}

    for entry, response in zip(chat_entries, responses):
        orig_idx = int(entry["custom_id"])
        result_by_orig_idx[orig_idx] = response
        if response.get("attn_output") is not None:
            mid = str(data[orig_idx]["measurement_id"])
            attn_output_dict[mid] = response["attn_output"]

    output_dir.mkdir(parents=True, exist_ok=True)
    judged_data: list[dict] = []

    for i, record in enumerate(data):
        response = result_by_orig_idx.get(i, {})
        judged_data.append(
            record | {
                "judgement": "true" in response.get("response", "").strip().lower(),
                "judgement_prob": math.exp(float(response["logprob"])) if "logprob" in response else None,
                "judgement_p_true": float(response["p_true"]) if "p_true" in response else None,
                "judgement_p_false": float(response["p_false"]) if "p_false" in response else None,
                "judgement_logit_p_true": float(response["logit_p_true"]) if "logit_p_true" in response else None,
                "judgement_logit_p_false": float(response["logit_p_false"]) if "logit_p_false" in response else None,
                "judgement_model": judge_cfg["model_id"],
            }
        )

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
        "--ablation", default=None, metavar="N",
        help="Ablation number (e.g. 2). If set, reads from ablations/ablation{N}/ and writes judge output there.",
    )
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
    input_file = _find_extraction_final(args.dataset, args.extraction_model, args.extraction_date, args.ablation)
    extraction_date_resolved = input_file.parent.name
    output_dir = get_judge_output_dir(
        args.dataset, args.extraction_model, extraction_date_resolved, args.judge, args.judge_date,
        ablation=args.ablation,
    )
    print(f"\nDataset          : {args.dataset}")
    print(f"Extraction model : {args.extraction_model}")
    print(f"Extraction date  : {extraction_date_resolved}")
    if args.ablation:
        print(f"Ablation         : {args.ablation}")
    print(f"Judge            : {args.judge}")
    print(f"Output           : {output_dir}\n")

    run_interp_judge(
        dataset_config=dataset_config,
        extraction_model=args.extraction_model,
        judge_key=args.judge,
        output_dir=output_dir,
        extraction_date=extraction_date_resolved,
        ocr_dir=args.ocr_dir,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()
