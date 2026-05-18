"""
Interpretability judge pipeline (NNsight / JudgementLM).

Runs judge validation for a given (dataset, extraction_model, judge_model) triple
using a local model loaded through NNsight, collecting per-layer, per-head
attention output activations alongside binary judgement probabilities.

Standard output path:
    data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_model}/{judge_date}/

Synthetic probe output path (when --synthetic is used):
    data/experiments/{dataset}/synthetic_probe/{judge_model}/{judge_date}/

Saves:
  - ``responses.json``        — per-measurement judgement + probability scores
  - ``attention_outputs.npz`` — per-layer, per-head attention output activations
  - ``layer_outputs.npz``     — per-layer residual stream outputs (last generated token)

Usage
-----
    # Standard extraction run
    python experiments/run_judge_interp.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge llama-3.1-8b \\
        --extraction-date 2026_04_01

    # Synthetic probe dataset
    python experiments/run_judge_interp.py \\
        --dataset pond \\
        --synthetic \\
        --judge llama-3.1-8b

Available judge models: llama-3.1-8b, gemma-2-9b, mistral-7b (see JUDGE_REGISTRY in code for details).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = Path(__file__).parent / "configs"
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch

from scholarlm import JudgementLM
from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory

from model_registry import INTERP_JUDGE_REGISTRY as JUDGE_REGISTRY
from run_extraction import load_dataset_config
import paths
from utils import set_seeds, write_run_metadata


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_interp_judge(
    dataset_config: DatasetConfig,
    extraction_model: str | None,
    judge_key: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    ablation: str | None = None,
    input_file: Path | None = None,
) -> None:
    """Run a local NNsight judge and save responses + attention activations.

    Prompts are built via ``judge_common.prepare_chat_entries`` — the same
    function used by all judge runners — so the query content (entity
    description, attribute description, value/units, closing question) is
    identical across all judge backends.  JudgementLM receives
    the three parts separately as (instructions, context, query), which it
    wraps into a single user message internally.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model whose results to judge.
            Not used when ``input_file`` is provided explicitly (synthetic mode).
        judge_key: Key in ``JUDGE_REGISTRY``.
        output_dir: Directory to write ``responses.json``, ``attention_outputs.npz``, and ``layer_outputs.npz``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
        input_file: If provided, load data from this path instead of looking up
            the extraction run (synthetic probe mode).
    """
    if judge_key not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge '{judge_key}'. Available: {sorted(JUDGE_REGISTRY.keys())}"
        )
    judge_cfg = JUDGE_REGISTRY[judge_key]

    if input_file is None:
        input_file = paths.find_extraction_final(dataset_config.name, extraction_model, extraction_date, ablation)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    import judge_common
    documents = judge_common.load_documents_for_dataset(dataset_config, effective_ocr_dir)

    # prepare_chat_entries sorts by document_id for cache locality; custom_id
    # preserves the original index so results can be merged back in order.
    chat_entries = judge_common.prepare_chat_entries(
        data, documents, dataset_config,
    )

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

    start_time = time.time()
    responses = llm.predict(messages)

    # Map results back to original data order via custom_id.
    result_by_orig_idx: dict[int, dict] = {}
    attn_output_dict: dict[str, Any] = {}
    layer_output_dict: dict[str, Any] = {}

    for entry, response in zip(chat_entries, responses):
        orig_idx = int(entry["custom_id"])
        result_by_orig_idx[orig_idx] = response
        mid = str(data[orig_idx]["measurement_id"])
        if response.get("attn_output") is not None:
            attn_output_dict[mid] = response["attn_output"]
        if response.get("layer_output") is not None:
            layer_output_dict[mid] = response["layer_output"]

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
        print(f"Attention activations saved to {attn_file}")

    if layer_output_dict:
        layer_file = output_dir / "layer_outputs.npz"
        np.savez_compressed(layer_file, **layer_output_dict)
        print(f"Layer outputs saved to {layer_file}")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        extraction_model=extraction_model,
        judge_model=judge_key,
        judge_model_id=judge_cfg["model_id"],
        max_prompt_tokens=llm.max_prompt_tokens,
    )


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
        "--extraction-model", default=None,
        help="Short name of the extraction model whose results to judge. Required unless --synthetic is used.",
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
    p.add_argument(
        "--synthetic", action="store_true", default=False,
        help=(
            "Run on the synthetic probe dataset instead of an extraction run. "
            "--extraction-model and --ablation are ignored."
        ),
    )
    p.add_argument(
        "--synthetic-split", choices=["train", "test"], default=None,
        help=(
            "Which synthetic split to run (only relevant with --synthetic). "
            "'train' → probe_dataset.json → synthetic_probe/; "
            "'test'  → probe_dataset_test.json → synthetic_probe_test/. "
            "If omitted, both splits are run in sequence."
        ),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from utils import load_config
    cfg = load_config()
    seed = cfg.get("defaults", {}).get("seed", 342)
    set_seeds(seed)

    dataset_config = load_dataset_config(args.dataset)

    if args.synthetic:
        splits = [args.synthetic_split] if args.synthetic_split else ["train", "test"]
        for split in splits:
            probe_filename = "probe_dataset_test.json" if split == "test" else "probe_dataset.json"
            probe_file = _REPO_ROOT / "data" / args.dataset / probe_filename
            if not probe_file.exists():
                raise FileNotFoundError(
                    f"Probe dataset not found: {probe_file}. "
                    f"Run data/{args.dataset}/create_probe_dataset.py first."
                )
            output_dir = (
                paths.synthetic_probe_test(args.dataset, args.judge, args.judge_date)
                if split == "test"
                else paths.synthetic_probe(args.dataset, args.judge, args.judge_date)
            )
            print(f"\nDataset          : {args.dataset}")
            print(f"Mode             : synthetic probe ({split})")
            print(f"Input            : {probe_file}")
            print(f"Judge            : {args.judge}")
            print(f"Output           : {output_dir}\n")
            run_interp_judge(
                dataset_config=dataset_config,
                extraction_model=None,
                judge_key=args.judge,
                output_dir=output_dir,
                ocr_dir=args.ocr_dir,
                input_file=probe_file,
            )
    else:
        if args.extraction_model is None:
            _build_parser().error("--extraction-model is required unless --synthetic is used.")
        input_file = paths.find_extraction_final(
            args.dataset, args.extraction_model, args.extraction_date, args.ablation
        )
        extraction_date_resolved = input_file.parent.name
        output_dir = paths.judge(
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
