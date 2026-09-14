"""
Interpretability judge pipeline (NNsight / JudgementLM).

Runs judge validation for a given (dataset, extraction_model, judge_model) triple
using a local model loaded through NNsight, collecting per-layer, per-head
attention output activations alongside binary judgement probabilities.

Standard mode output path (id-addressed, like every other Tier-1 type):
    experiments/results/{dataset}/judge_interp/{experiment_id}/

Synthetic probe mode output path (params.synthetic) is deliberately UNCHANGED
-- still the old date-addressed tree, since analysis/*.py's probe-calibration
pipeline (explicitly out of scope for this restructure) reads it directly:
    data/experiments/{dataset}/synthetic_probe/{judge_model}/{judge_date}/

Saves:
  - ``responses.json``        — per-measurement judgement + probability scores
  - ``attention_outputs.npz`` — per-layer, per-head attention output activations
  - ``layer_outputs.npz``     — per-layer residual stream outputs (last generated token)

Usage
-----
    python experiments/run_judge_interp.py experiments/experiment-configs/pond/judge_interp/<id>/<id>.yaml

Required params: dataset, judge.
Standard mode requires params.extraction_id (an extraction or ablation
experiment id, resolved via utils.find_result_dir -- its final.json is judged).
Synthetic mode (params.synthetic: true, or params.synthetic_file) ignores
extraction_id; see below for its params (unchanged from before this restructure).
Optional params: extraction_id, judge_date, ocr_dir, synthetic (bool),
synthetic_split ('train'|'test'), synthetic_file, synthetic_name.

Available judge models: the YAML files in experiments/model-configs/interp_judge/.
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

from run_extraction import load_dataset_config
import utils as paths
from utils import set_seeds, write_run_metadata


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_interp_judge(
    dataset_config: DatasetConfig,
    judge_key: str,
    output_dir: Path,
    input_file: Path,
    ocr_dir: str | None = None,
    extraction_id: str | None = None,
) -> None:
    """Run a local NNsight judge and save responses + attention activations.

    Prompts are built via ``judge_prompts.prepare_chat_entries`` — the same
    function used by all judge runners — so the query content (entity
    description, attribute description, value/units, closing question) is
    identical across all judge backends.  JudgementLM receives
    the three parts separately as (instructions, context, query), which it
    wraps into a single user message internally.  The context is the full OCR
    paper text; a row carrying a ``context_override`` field (set by the
    synthetic-probe augmentation pipeline) uses that text verbatim instead.

    Args:
        dataset_config: Dataset configuration.
        judge_key: Key in ``experiments/model-configs/interp_judge/``.
        output_dir: Directory to write ``responses.json``, ``attention_outputs.npz``, and ``layer_outputs.npz``.
        input_file: Path to the ``final.json``-shaped file to judge (an
            extraction/ablation run's output, or a synthetic probe file).
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
        extraction_id: The upstream extraction/ablation experiment id being
            judged, recorded in run_metadata.json. ``None`` for synthetic mode
            (there is no upstream extraction run).
    """
    judge_cfg = paths.load_model_config("interp_judge", judge_key)

    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    from scholarlm.utils import judge_prompts
    documents = judge_prompts.load_documents_for_dataset(dataset_config, effective_ocr_dir)

    # prepare_chat_entries sorts by document_id for cache locality; custom_id
    # preserves the original index so results can be merged back in order.
    chat_entries = judge_prompts.prepare_chat_entries(data, documents, dataset_config)

    # JudgementLM takes (instructions, context, query) triples separately.
    # instructions = system prompt, context = full paper text, query = ## QUERY content.
    messages: list[tuple[str, str, str]] = [
        (entry["system"], entry["context_text"], entry["user_query"])
        for entry in chat_entries
    ]

    llm = JudgementLM(
        model_name=judge_cfg["model_id"],
        sampling_params=judge_cfg["sampling_params"],
        nnsight_kwargs=judge_cfg["nnsight_kwargs"],
        use_chat_template=judge_cfg.get("use_chat_template", True),
        answer_cue=judge_cfg.get("answer_cue", None),
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
        extraction_id=extraction_id,
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
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", "judge", config_path=config_path)

    set_seeds(cfg["seed"])

    dataset = params["dataset"]
    judge = params["judge"]
    paths.load_model_config("interp_judge", judge)  # fail loud on an unknown judge before any work starts
    dataset_config = load_dataset_config(dataset)

    synthetic_file = params.get("synthetic_file")
    synthetic = params.get("synthetic", False)

    if synthetic_file:
        synthetic_name = params.get("synthetic_name")
        if not synthetic_name:
            raise ValueError(f"{config_path}: params.synthetic_name is required with params.synthetic_file.")
        probe_file = Path(synthetic_file)
        if not probe_file.exists():
            raise FileNotFoundError(f"params.synthetic_file not found: {probe_file}")
        output_dir = paths.synthetic_probe_named(dataset, synthetic_name, judge, params.get("judge_date"))
        print(f"\nDataset          : {dataset}")
        print(f"Mode             : synthetic probe (named: {synthetic_name})")
        print(f"Input            : {probe_file}")
        print(f"Judge            : {judge}")
        print(f"Output           : {output_dir}\n")
        run_interp_judge(
            dataset_config=dataset_config,
            judge_key=judge,
            output_dir=output_dir,
            input_file=probe_file,
            ocr_dir=params.get("ocr_dir"),
        )
        return

    if synthetic:
        splits = [params["synthetic_split"]] if params.get("synthetic_split") else ["train", "test"]
        for split in splits:
            probe_filename = "probe_dataset_test_v2.json" if split == "test" else "probe_dataset_v2.json"
            probe_file = _REPO_ROOT / "data" / dataset / probe_filename
            if not probe_file.exists():
                raise FileNotFoundError(
                    f"Probe dataset not found: {probe_file}. "
                    f"Run data/{dataset}/create_probe_dataset.py first."
                )
            output_dir = (
                paths.synthetic_probe_test(dataset, judge, params.get("judge_date"))
                if split == "test"
                else paths.synthetic_probe(dataset, judge, params.get("judge_date"))
            )
            print(f"\nDataset          : {dataset}")
            print(f"Mode             : synthetic probe ({split})")
            print(f"Input            : {probe_file}")
            print(f"Judge            : {judge}")
            print(f"Output           : {output_dir}\n")
            run_interp_judge(
                dataset_config=dataset_config,
                judge_key=judge,
                output_dir=output_dir,
                input_file=probe_file,
                ocr_dir=params.get("ocr_dir"),
            )
        return

    paths.require_params(params, "extraction_id", config_path=config_path)
    extraction_id = params["extraction_id"]
    extraction_dir = paths.find_result_dir(extraction_id)
    resolved_dataset = extraction_dir.parts[-3]
    if resolved_dataset != dataset:
        raise ValueError(
            f"{config_path}: params.dataset {dataset!r} does not match the dataset "
            f"of params.extraction_id {extraction_id!r} ({resolved_dataset!r})"
        )
    input_file = extraction_dir / "final.json"

    output_dir = paths.result_dir(dataset, "judge_interp", cfg["id"])
    print(f"\nDataset          : {dataset}")
    print(f"Extraction id    : {extraction_id}")
    print(f"Judge            : {judge}")
    print(f"Output           : {output_dir}\n")
    run_interp_judge(
        dataset_config=dataset_config,
        judge_key=judge,
        output_dir=output_dir,
        input_file=input_file,
        ocr_dir=params.get("ocr_dir"),
        extraction_id=extraction_id,
    )


if __name__ == "__main__":
    main()
