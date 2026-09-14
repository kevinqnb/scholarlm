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

Output path (id-addressed, like every other Tier-1 type):
    experiments/results/{dataset}/jacobian_lens/{experiment_id}/

Saves:
  - ``jacobian_scores.npz`` — per-example S matrices (keyed by measurement_id)
    plus a ``layer_indices`` array giving the source-layer each row of S
    corresponds to (constant across all examples in a run).
  - ``run_metadata.json``

Usage
-----
    python experiments/run_jacobian_lens.py experiments/experiment-configs/pond/jacobian_lens/<id>/<id>.yaml

Required params: dataset, extraction_id (an extraction or ablation experiment
id, resolved via utils.find_result_dir), model (a key in
experiments/model-configs/jacobian_lens/), jacobian_lens_path.
Optional params: ocr_dir, limit.

Available models: the YAML files in experiments/model-configs/jacobian_lens/.
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

from run_extraction import load_dataset_config
import utils as paths
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
        chat entries built from them via ``judge_prompts.prepare_chat_entries``.
    """
    with open(input_file) as f:
        data: list[dict] = json.load(f)

    if limit is not None:
        data = data[:limit]

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    from scholarlm.utils import judge_prompts
    documents = judge_prompts.load_documents_for_dataset(dataset_config, effective_ocr_dir)

    chat_entries = judge_prompts.prepare_chat_entries(data, documents, dataset_config)
    return data, chat_entries


def run_jacobian_lens(
    dataset_config: DatasetConfig,
    model_key: str,
    jacobian_lens_path: str,
    output_dir: Path,
    input_file: Path,
    ocr_dir: str | None = None,
    limit: int | None = None,
    extraction_id: str | None = None,
) -> None:
    """Run JacobianLensLM over a dataset and save j-score matrices.

    Prompts are built via ``judge_prompts.prepare_chat_entries`` — the same
    function used by all judge runners — so the query content (entity
    description, attribute description, value/units, closing question) is
    identical across all judge/lens backends. ``JacobianLensLM`` receives the
    three parts separately as (instructions, context, query); its own
    ``tokenize()`` reorders them to instructions/query/context internally.

    Args:
        dataset_config: Dataset configuration.
        model_key: Key in ``experiments/model-configs/jacobian_lens/``.
        jacobian_lens_path: Local path or ``repo_id:filename`` HuggingFace Hub
            spec for the pretrained Jacobian-lens checkpoint.
        output_dir: Directory to write ``jacobian_scores.npz`` and ``run_metadata.json``.
        input_file: Path to the ``final.json``-shaped extraction/ablation output to score.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
        limit: If given, only score the first ``limit`` records.
        extraction_id: The upstream extraction/ablation experiment id being
            scored, recorded in run_metadata.json.
    """
    model_cfg = paths.load_model_config("jacobian_lens", model_key)

    print(f"Input   : {input_file}")

    data, chat_entries = _load_chat_entries(dataset_config, input_file, ocr_dir, limit)

    messages: list[tuple[str, str, str]] = [
        (entry["system"], entry["context_text"], entry["user_query"])
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
        extraction_id=extraction_id,
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
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(
        params, "dataset", "extraction_id", "model", "jacobian_lens_path", config_path=config_path
    )

    set_seeds(cfg["seed"])

    dataset = params["dataset"]
    model_key = params["model"]
    paths.load_model_config("jacobian_lens", model_key)  # fail loud on an unknown model before any work starts
    dataset_config = load_dataset_config(dataset)

    extraction_id = params["extraction_id"]
    extraction_dir = paths.find_result_dir(extraction_id)
    resolved_dataset = extraction_dir.parts[-3]
    if resolved_dataset != dataset:
        raise ValueError(
            f"{config_path}: params.dataset {dataset!r} does not match the dataset "
            f"of params.extraction_id {extraction_id!r} ({resolved_dataset!r})"
        )
    input_file = extraction_dir / "final.json"
    output_dir = paths.result_dir(dataset, "jacobian_lens", cfg["id"])

    print(f"\nDataset          : {dataset}")
    print(f"Extraction id    : {extraction_id}")
    print(f"Model            : {model_key}")
    print(f"Jacobian lens    : {params['jacobian_lens_path']}")
    if params.get("limit"):
        print(f"Limit            : {params['limit']}")
    print(f"Output           : {output_dir}\n")
    run_jacobian_lens(
        dataset_config=dataset_config,
        model_key=model_key,
        jacobian_lens_path=params["jacobian_lens_path"],
        output_dir=output_dir,
        input_file=input_file,
        ocr_dir=params.get("ocr_dir"),
        limit=params.get("limit"),
        extraction_id=extraction_id,
    )


if __name__ == "__main__":
    main()
