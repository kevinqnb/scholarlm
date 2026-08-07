"""
Jacobian-lens j-score computation (NNsight / JacobianLensLM).

Computes the j-score matrix S for each judge example — how strongly the
'true' direction (transported through a pretrained per-layer Jacobian) shows
up in the residual stream at each context-token position and each fitted
layer — and persists it. This is Thrust 1 of
``notes/scholarlm/threads/Jacobian Lens.md``: pure compute-and-persist, no
analysis of what S shows.

Deliberately separate from run_judge_interp.py / JudgementLM: see the design
note for this build for why (prompt ordering and trace shape both diverge
from JudgementLM.generate()).

Standard output path:
    data/experiments/{dataset}/jacobian_lens/{extraction_model}/{extraction_date}/{lens_model}/{lens_date}/

Saves:
  - ``jacobian_scores.npz`` — per-example S matrices (keyed by measurement_id)
    plus a ``layer_indices`` array giving the source-layer each row of S
    corresponds to (constant across all examples in a run).
  - ``run_metadata.json``

Usage
-----
    python experiments/run_jacobian_lens.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --model llama-3.1-8b-base \\
        --jacobian-lens-path "neuronpedia/jacobian-lens:llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt" \\
        --extraction-date 2026_04_01 \\
        --limit 5

Available models: llama-3.1-8b-base (see JACOBIAN_LENS_REGISTRY in code for details).
"""
from __future__ import annotations

import argparse
import json
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

from scholarlm import JacobianLensLM
from scholarlm.config import DatasetConfig

from model_registry import JACOBIAN_LENS_REGISTRY
from run_extraction import load_dataset_config
import paths
from utils import set_seeds, write_run_metadata


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _load_chat_entries(
    dataset_config: DatasetConfig,
    input_file: Path,
    ocr_dir: str | None,
    limit: int | None,
) -> tuple[list[dict], list[dict]]:
    """Load extraction records and build chat entries for a dataset.

    Shared by run_judge_interp.py-style loading logic; kept local to this
    script rather than added to run_judge_interp.py since this build doesn't
    touch that file.

    Args:
        dataset_config: Dataset configuration.
        input_file: Path to a ``final.json``-shaped extraction results file.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to
            ``{data_dir}/ocr_output_raw/``.
        limit: If given, truncate ``data`` to the first ``limit`` records
            before document loading / chat-entry prep.

    Returns:
        (data, chat_entries) — the (possibly truncated) raw records and the
        chat entries built from them via ``judge_common.prepare_chat_entries``.
    """
    with open(input_file) as f:
        data: list[dict] = json.load(f)

    if limit is not None:
        data = data[:limit]

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    import judge_common
    documents = judge_common.load_documents_for_dataset(dataset_config, effective_ocr_dir)

    chat_entries = judge_common.prepare_chat_entries(data, documents, dataset_config)
    return data, chat_entries


def run_jacobian_lens(
    dataset_config: DatasetConfig,
    extraction_model: str,
    model_key: str,
    jacobian_lens_path: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    ablation: str | None = None,
    limit: int | None = None,
) -> None:
    """Run JacobianLensLM over a dataset and save j-score matrices.

    Prompts are built via ``judge_common.prepare_chat_entries`` — the same
    function used by all judge runners — so the query content (entity
    description, attribute description, value/units, closing question) is
    identical across all judge/lens backends. ``JacobianLensLM`` receives the
    three parts separately as (instructions, context, query); its own
    ``tokenize()`` reorders them to instructions/query/context internally.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model whose results to score.
        model_key: Key in ``JACOBIAN_LENS_REGISTRY``.
        jacobian_lens_path: Local path or ``repo_id:filename`` HuggingFace Hub
            spec for the pretrained Jacobian-lens checkpoint.
        output_dir: Directory to write ``jacobian_scores.npz`` and ``run_metadata.json``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
        ablation: Optional ablation number.
        limit: If given, only score the first ``limit`` records.
    """
    if model_key not in JACOBIAN_LENS_REGISTRY:
        raise KeyError(
            f"Unknown model '{model_key}'. Available: {sorted(JACOBIAN_LENS_REGISTRY.keys())}"
        )
    model_cfg = JACOBIAN_LENS_REGISTRY[model_key]

    input_file = paths.find_extraction_final(dataset_config.name, extraction_model, extraction_date, ablation)
    print(f"Input   : {input_file}")

    data, chat_entries = _load_chat_entries(dataset_config, input_file, ocr_dir, limit)

    messages: list[tuple[str, str, str]] = [
        (entry["system"], entry["page_text"], entry["user_query"])
        for entry in chat_entries
    ]

    llm = JacobianLensLM(
        model_name=model_cfg["model_id"],
        jacobian_lens_path=jacobian_lens_path,
        nnsight_kwargs=model_cfg["nnsight_kwargs"],
        use_chat_template=model_cfg.get("use_chat_template", False),
        hf_cache_dir=os.environ.get("HF_CACHE"),
    )

    start_time = time.time()
    results = llm.predict(messages)

    scores_dict: dict[str, Any] = {}
    layer_indices = None
    for entry, result in zip(chat_entries, results):
        orig_idx = int(entry["custom_id"])
        mid = str(data[orig_idx]["measurement_id"])
        scores_dict[mid] = result["S"]
        layer_indices = result["layer_indices"]

    output_dir.mkdir(parents=True, exist_ok=True)

    if scores_dict:
        scores_file = output_dir / "jacobian_scores.npz"
        np.savez_compressed(scores_file, layer_indices=layer_indices, **scores_dict)
        print(f"Jacobian scores saved to {scores_file}")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        extraction_model=extraction_model,
        lens_model=model_key,
        lens_model_id=model_cfg["model_id"],
        jacobian_lens_path=jacobian_lens_path,
        n_examples=len(scores_dict),
        max_prompt_tokens=llm.max_prompt_tokens,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute Jacobian-lens j-scores (NNsight/JacobianLensLM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'pond', 'nfix').")
    p.add_argument(
        "--extraction-model", required=True,
        help="Short name of the extraction model whose results to score.",
    )
    p.add_argument(
        "--model", required=True,
        choices=sorted(JACOBIAN_LENS_REGISTRY.keys()),
        help=f"Jacobian-lens model key. Available: {sorted(JACOBIAN_LENS_REGISTRY.keys())}",
    )
    p.add_argument(
        "--jacobian-lens-path", required=True,
        help=(
            "Local filesystem path to a Jacobian-lens .pt checkpoint, or a "
            "'repo_id:filename' HuggingFace Hub spec, e.g. "
            "'neuronpedia/jacobian-lens:llama3.1-8b/jlens/Salesforce-wikitext/Llama-3.1-8B_jacobian_lens.pt'."
        ),
    )
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd of extraction run.")
    p.add_argument("--lens-date", default=None, help="Date tag for output directory (default: today).")
    p.add_argument(
        "--ablation", default=None, metavar="N",
        help="Ablation number (e.g. 2). If set, reads from ablations/ablation{N}/.",
    )
    p.add_argument(
        "--ocr-dir", default=None, metavar="DIR",
        help=(
            "Directory of OCR .txt files to use as document context. "
            "Defaults to {data_dir}/ocr_output_raw/."
        ),
    )
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only score the first N records (for small-subset spot checks).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    from utils import load_config
    cfg = load_config()
    seed = cfg.get("defaults", {}).get("seed", 342)
    set_seeds(seed)

    dataset_config = load_dataset_config(args.dataset)

    input_file = paths.find_extraction_final(
        args.dataset, args.extraction_model, args.extraction_date, args.ablation
    )
    extraction_date_resolved = input_file.parent.name
    output_dir = paths.jacobian_lens(
        args.dataset, args.extraction_model, extraction_date_resolved, args.model, args.lens_date,
    )
    print(f"\nDataset          : {args.dataset}")
    print(f"Extraction model : {args.extraction_model}")
    print(f"Extraction date  : {extraction_date_resolved}")
    if args.ablation:
        print(f"Ablation         : {args.ablation}")
    print(f"Model            : {args.model}")
    print(f"Jacobian lens    : {args.jacobian_lens_path}")
    if args.limit:
        print(f"Limit            : {args.limit}")
    print(f"Output           : {output_dir}\n")
    run_jacobian_lens(
        dataset_config=dataset_config,
        extraction_model=args.extraction_model,
        model_key=args.model,
        jacobian_lens_path=args.jacobian_lens_path,
        output_dir=output_dir,
        extraction_date=extraction_date_resolved,
        ocr_dir=args.ocr_dir,
        ablation=args.ablation,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
