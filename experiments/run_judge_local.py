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
  - ``responses.json`` — per-measurement judgement + next-token P(true)/P(false)

Logprob extraction
------------------
Each request is sent with ``max_tokens=1``, ``logprobs=True``, and
``top_logprobs=N`` (default 20, matching vLLM's default ``--max-logprobs``
server-side limit). The runner inspects the returned top-N tokens for
"true", "True", "false", "False" and computes a normalized P(true) via
logsumexp + softmax over those two classes. If neither token appears in the
top-N list, ``judgement_p_true`` and ``judgement_p_false`` are ``null`` in
the output; ``judgement`` is still derived from the generated token text.

To raise the top_logprobs ceiling, start vLLM with ``--max-logprobs <N>``
and pass ``--top-logprobs <N>`` to this script.

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
import importlib.util
import json
import math
import os
import random
import sys
from datetime import datetime
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

# ---------------------------------------------------------------------------
# Judge model registry
# ---------------------------------------------------------------------------

JUDGE_REGISTRY: dict[str, dict] = {
    "gemma-3-27b": {
        "model_id": "gaunernst/gemma-3-27b-it-int4-awq",
    },
    "qwen-3.5-27b": {
        "model_id": "Qwen/Qwen3.5-27B-FP8"
    },
    "llama-3.3-70b": {
        "model_id": "ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4",
    },
    "qwen-2.5-72b": {
        "model_id": "Qwen/Qwen2.5-72B-Instruct-AWQ",
    },
    "gpt-oss-120b": {
        "model_id": "openai/gpt-oss-120b",
    },
}

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
# Logprob helpers
# ---------------------------------------------------------------------------

_TRUE_TOKENS = {"true", "True"}
_FALSE_TOKENS = {"false", "False"}


def _logsumexp2(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def _extract_binary_probs(
    top_logprobs: list,  # list of TopLogprob objects (token: str, logprob: float)
    generated_token: str,
    generated_logprob: float | None,
) -> dict:
    """Compute normalized P(true) and P(false) from a top-logprobs list.

    Collects log-probabilities for "true"/"True" and "false"/"False" tokens,
    then marginalizes via logsumexp and normalizes via softmax over the two
    classes.

    Returns a dict with keys:
        judgement_p_true, judgement_p_false,
        judgement_logit_p_true, judgement_logit_p_false
    All values are float or None if the tokens were not found.
    """
    # Build a token → logprob map from the top-N list
    lp_map: dict[str, float] = {}
    for tlp in top_logprobs:
        lp_map[tlp.token] = tlp.logprob

    # Ensure the generated token is always represented
    if generated_token and generated_logprob is not None:
        lp_map.setdefault(generated_token, generated_logprob)

    true_lps = [lp_map[t] for t in _TRUE_TOKENS if t in lp_map]
    false_lps = [lp_map[t] for t in _FALSE_TOKENS if t in lp_map]

    if not true_lps or not false_lps:
        return {
            "judgement_p_true": None,
            "judgement_p_false": None,
            "judgement_logit_p_true": None,
            "judgement_logit_p_false": None,
        }

    # Marginalize over casing
    log_p_true = true_lps[0] if len(true_lps) == 1 else _logsumexp2(*true_lps[:2])
    log_p_false = false_lps[0] if len(false_lps) == 1 else _logsumexp2(*false_lps[:2])

    # Softmax normalization over the two classes
    max_lp = max(log_p_true, log_p_false)
    e_true = math.exp(log_p_true - max_lp)
    e_false = math.exp(log_p_false - max_lp)
    denom = e_true + e_false

    return {
        "judgement_p_true": e_true / denom,
        "judgement_p_false": e_false / denom,
        "judgement_logit_p_true": log_p_true,
        "judgement_logit_p_false": log_p_false,
    }


# ---------------------------------------------------------------------------
# Async judge call
# ---------------------------------------------------------------------------


async def _judge_one(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    top_logprobs: int,
    sem: asyncio.Semaphore,
) -> dict:
    """Send a single judge request and return binary probability fields."""
    async with sem:
        try:
            if model_id == "openai/gpt-oss-120b":
                # GPT-OSS-120B thinks before generating a token, 
                # so we set max_tokens=32 to get logprobs for the first generated token.
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": entry["system"]},
                        {"role": "user", "content": entry["user"]},
                    ],
                    max_tokens=8192,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                )
                print("Response text: ", response.choices[0].message.content)
            
            else:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": entry["system"]},
                        {"role": "user", "content": entry["user"]},
                    ],
                    max_tokens=1,
                    temperature=0.0,
                    logprobs=True,
                    top_logprobs=top_logprobs,
                )       
        except Exception as e:
            print(f"  [idx={idx}] API error: {e}")
            return {
                "judgement": None,
                "judgement_prob": None,
                "judgement_p_true": None,
                "judgement_p_false": None,
                "judgement_logit_p_true": None,
                "judgement_logit_p_false": None,
                "judgement_model": model_id,
            }

    choice = response.choices[0]
    content_lps = (choice.logprobs.content or []) if choice.logprobs else []

    if content_lps:
        generated_token = content_lps[0].token
        generated_logprob = content_lps[0].logprob
        top_lps_list = content_lps[0].top_logprobs or []
    else:
        generated_token = ""
        generated_logprob = None
        top_lps_list = []

    print(f"  [idx={idx}] Generated token: '{generated_token}' with logprob {generated_logprob}")
    print()

    # Derive judgement from the generated token text
    judgement: bool | None = None
    if generated_token:
        t = generated_token.strip().lower()
        if "true" in t:
            judgement = True
        elif "false" in t:
            judgement = False

    prob_fields = _extract_binary_probs(top_lps_list, generated_token, generated_logprob)

    return {
        "judgement": judgement,
        "judgement_prob": math.exp(generated_logprob) if generated_logprob is not None else None,
        "judgement_model": response.model,
        **prob_fields,
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
    top_logprobs: int = 20,
) -> None:
    """Run a local vLLM judge and save responses with token-level probabilities.

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
        top_logprobs: Number of top tokens to request from vLLM (must be ≤ the
            server's ``--max-logprobs`` setting, which defaults to 20).
    """
    if judge_key not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge '{judge_key}'. Available: {sorted(JUDGE_REGISTRY.keys())}"
        )
    judge_cfg = JUDGE_REGISTRY[judge_key]
    model_id = judge_cfg["model_id"]

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

    # Build prompts using the shared batch prompt builder
    from batch import common as batch_common
    chat_entries = batch_common.prepare_chat_entries(data, documents, dataset_config)

    # chat_entries are sorted by document_id for cache locality; we need to
    # track the original indices to merge results back in order.
    # prepare_chat_entries sets custom_id = str(original index in data).

    print(f"Sending {len(chat_entries)} requests to {api_base} (model: {model_id}) ...")
    print(f"top_logprobs={top_logprobs}, max_concurrent={max_concurrent}\n")

    client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=300.0)
    sem = asyncio.Semaphore(max_concurrent)

    async def _run_all() -> list[dict]:
        tasks = [
            _judge_one(client, model_id, entry, int(entry["custom_id"]), top_logprobs, sem)
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
    n_missing_probs = 0
    for i, record in enumerate(data):
        result = result_by_orig_idx.get(i, {})
        judged_data.append(record | result)
        if result.get("judgement_p_true") is None:
            n_missing_probs += 1

    if n_missing_probs:
        print(
            f"Note: {n_missing_probs}/{len(data)} records have null P(true)/P(false) — "
            f"'true'/'false' tokens were not found in top_{top_logprobs} logprobs. "
            f"Consider raising --top-logprobs (and starting vLLM with --max-logprobs <N>)."
        )

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
        description="Run local vLLM judge (binary probabilities via logprobs API).",
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
    p.add_argument(
        "--top-logprobs", type=int, default=20,
        help=(
            "Number of top tokens to request for logprob extraction (default: 20). "
            "Must be ≤ the server's --max-logprobs setting."
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
        top_logprobs=args.top_logprobs,
    )


if __name__ == "__main__":
    main()
