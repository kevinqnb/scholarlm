"""
Local vLLM judge pipeline.

Runs judge validation using a local model served via vLLM's OpenAI-compatible
API. Unlike the interpretability judge (run_judge_interp.py), this runner does
not load the model locally — it sends async HTTP requests to a running vLLM
server, making it practical for large models that would otherwise require
NNsight's full GPU memory.

Output path:
    data/experiments/{dataset}/judge/{extraction_model}/{judge_model}/{YYYY_mm_dd}/

Saves:
  - ``responses.json`` — per-measurement judgement (true/false from text response)

Usage
-----
    # Start a vLLM server first, e.g.:
    #   vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8081

    python experiments/run_judge_local.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge llama-3.1-8b \\
        --extraction-date 2026_04_01

    # Use a different server or larger model:
    python experiments/run_judge_local.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge qwen-2.5-72b \\
        --api-base http://localhost:8082/v1

Available judge models: see JUDGE_REGISTRY below.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
from pathlib import Path

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

from openai import AsyncOpenAI

from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory

random.seed(342)

from model_registry import VLLM_JUDGE_REGISTRY as JUDGE_REGISTRY
from run_extraction import load_dataset_config
import paths


# ---------------------------------------------------------------------------
# Async judge call
# ---------------------------------------------------------------------------


async def _judge_one(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    sem: asyncio.Semaphore,
) -> dict:
    """Send a single judge request and return judgement based on text response."""
    async with sem:
        try:
            response = await client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": entry["system"]},
                    {"role": "user", "content": entry["user"]},
                ],
                max_tokens=2048,
                temperature=0.0,
            )
        except Exception as e:
            print(f"  [idx={idx}] API error: {e}")
            return {
                "judgement": None,
                "judgement_model": model_id,
            }

    choice = response.choices[0]
    response_text = choice.message.content or ""

    print(f"  [idx={idx}] Response: {response_text}")

    # Derive judgement from the response text
    judgement: bool | None = None
    if response_text:
        t = response_text.strip().lower()
        if "true" in t:
            judgement = True
        elif "false" in t:
            judgement = False

    return {
        "judgement": judgement,
        "judgement_model": response.model,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_local_vllm_judge(
    dataset_config: DatasetConfig,
    extraction_model: str,
    judge_key: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
    max_concurrent: int = 64,
    ablation: str | None = None,
) -> None:
    """Run a local vLLM judge and save responses.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model whose results to judge.
        judge_key: Key in ``JUDGE_REGISTRY``.
        output_dir: Directory to write ``responses.json``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the vLLM server.
        max_concurrent: Maximum concurrent requests to the server.
    """
    if judge_key not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge '{judge_key}'. Available: {sorted(JUDGE_REGISTRY.keys())}"
        )
    judge_cfg = JUDGE_REGISTRY[judge_key]
    model_id = judge_cfg["model_id"]

    input_file = paths.find_extraction_final(dataset_config.name, extraction_model, extraction_date, ablation)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    from batch import common as batch_common
    documents = batch_common.load_documents_for_dataset(dataset_config, effective_ocr_dir)
    print(f"Documents: {len(documents)} loaded from {effective_ocr_dir}")

    chat_entries = batch_common.prepare_chat_entries(data, documents, dataset_config)

    # chat_entries are sorted by document_id for cache locality; we need to
    # track the original indices to merge results back in order.
    # prepare_chat_entries sets custom_id = str(original index in data).

    print(f"Sending {len(chat_entries)} requests to {api_base} (model: {model_id}) ...")
    print(f"max_concurrent={max_concurrent}\n")

    client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=300.0)
    sem = asyncio.Semaphore(max_concurrent)

    async def _run_all() -> list[dict]:
        tasks = [
            _judge_one(client, model_id, entry, int(entry["custom_id"]), sem)
            for entry in chat_entries
        ]
        return await asyncio.gather(*tasks)

    raw_results = asyncio.run(_run_all())

    # Map results back to the original data order using custom_id
    result_by_orig_idx: dict[int, dict] = {}
    for entry, result in zip(chat_entries, raw_results):
        orig_idx = int(entry["custom_id"])
        result_by_orig_idx[orig_idx] = result

    judged_data: list[dict] = []
    for i, record in enumerate(data):
        result = result_by_orig_idx.get(i, {})
        judged_data.append(record | result)

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(judged_data, f, indent=4, ensure_ascii=False)
    print(f"Responses saved to {responses_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run local vLLM judge (text-based judgement).",
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
        help="Directory of OCR .txt files. Defaults to {data_dir}/ocr_output_raw/.",
    )
    p.add_argument(
        "--api-base", default="http://localhost:8081/v1", metavar="URL",
        help="Base URL of the vLLM OpenAI-compatible server (default: http://localhost:8081/v1).",
    )
    p.add_argument(
        "--api-key", default="EMPTY", metavar="KEY",
        help="API key for the vLLM server (default: EMPTY).",
    )
    p.add_argument(
        "--max-concurrent", type=int, default=64,
        help="Maximum concurrent requests to the vLLM server (default: 64).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    dataset_config = load_dataset_config(args.dataset)
    input_file = paths.find_extraction_final(args.dataset, args.extraction_model, args.extraction_date, args.ablation)
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
    print(f"API base         : {args.api_base}")
    print(f"Output           : {output_dir}\n")

    run_local_vllm_judge(
        dataset_config=dataset_config,
        extraction_model=args.extraction_model,
        judge_key=args.judge,
        output_dir=output_dir,
        extraction_date=extraction_date_resolved,
        ocr_dir=args.ocr_dir,
        api_base=args.api_base,
        api_key=args.api_key,
        max_concurrent=args.max_concurrent,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()
