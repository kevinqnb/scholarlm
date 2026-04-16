"""
Frontier judge pipeline v2 — direct async API calls (OpenAI / Anthropic / Gemini).

Unlike v1, this makes normal chat-completion calls with async concurrency rather
than using provider batch APIs.  No state files, no polling; results are written
as soon as all calls finish.

Output path (same convention as v1):
    data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_model}/{judge_date}/

Saves:
  - ``responses.json`` — per-measurement judgement + raw text response

Usage
-----
    python experiments/run_judge_frontier_v2.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge openai \\
        --frontier-model gpt-5-mini \\
        --extraction-date 2026_04_01

    python experiments/run_judge_frontier_v2.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge anthropic \\
        --frontier-model claude-haiku-4-5-20251001 \\
        --extraction-date 2026_04_01

    python experiments/run_judge_frontier_v2.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge gemini \\
        --frontier-model gemini-2.5-flash-lite \\
        --extraction-date 2026_04_01

Available frontier providers: openai, anthropic, gemini

Environment variables
---------------------
    OPENAI_API_KEY      — required for --judge openai
    ANTHROPIC_API_KEY   — required for --judge anthropic
    GEMINI_API_KEY      — required for --judge gemini
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
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
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

from scholarlm.config import DatasetConfig

FRONTIER_PROVIDERS = {"openai", "anthropic", "gemini"}

# ---------------------------------------------------------------------------
# Config / path helpers (mirrors run_judge_frontier.py)
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
# Async judge runners — one per provider
# ---------------------------------------------------------------------------


_MAX_RETRIES = 8


async def _judge_openai(
    entries: list[dict],
    model: str,
    max_tokens: int,
    temperature: float | None,
    sem: asyncio.Semaphore,
    counter: list[int],
    total: int,
) -> list[tuple[str, dict]]:
    import openai
    client = openai.AsyncOpenAI()

    async def _call(entry: dict) -> tuple[str, dict]:
        raw = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with sem:
                    call_kwargs: dict[str, Any] = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": entry["system"]},
                            {"role": "user", "content": entry["user"]},
                        ],
                        "max_completion_tokens": max_tokens,
                    }
                    if temperature is not None:
                        call_kwargs["temperature"] = temperature
                    resp = await client.chat.completions.create(**call_kwargs)
                raw = (resp.choices[0].message.content or "").strip()
                break
            except openai.RateLimitError:
                if attempt < _MAX_RETRIES:
                    delay = min(60.0, (2.0 ** attempt) * (0.5 + random.random()))
                    await asyncio.sleep(delay)
                else:
                    print(f"  [openai] request {entry['custom_id']} exhausted retries (rate limit).")
            except Exception as e:
                print(f"  [openai] request {entry['custom_id']} failed: {e}")
                break
        counter[0] += 1
        if counter[0] % 50 == 0 or counter[0] == total:
            print(f"  {counter[0]}/{total} complete", flush=True)
        from batch.common import normalize_bool_text
        return entry["custom_id"], {
            "judgement": normalize_bool_text(raw),
            "prob": None,
            "model": model,
            "raw_text": raw,
        }

    return list(await asyncio.gather(*[_call(e) for e in entries]))


async def _judge_anthropic(
    entries: list[dict],
    model: str,
    max_tokens: int,
    temperature: float | None,
    sem: asyncio.Semaphore,
    counter: list[int],
    total: int,
) -> list[tuple[str, dict]]:
    import anthropic
    client = anthropic.AsyncAnthropic()

    async def _call(entry: dict) -> tuple[str, dict]:
        raw = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with sem:
                    call_kwargs: dict[str, Any] = dict(
                        model=model,
                        max_tokens=max_tokens,
                        system=[
                            {
                                "type": "text",
                                "text": entry["system"],
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": entry["user_document"],
                                        "cache_control": {"type": "ephemeral"},
                                    },
                                    {
                                        "type": "text",
                                        "text": f"## QUERY:\n{entry['user_query']}",
                                    },
                                ],
                            }
                        ],
                    )
                    if temperature is not None:
                        call_kwargs["temperature"] = temperature
                    resp = await client.messages.create(**call_kwargs)
                raw = "".join(
                    block.text for block in (resp.content or []) if hasattr(block, "text")
                ).strip()
                break
            except anthropic.RateLimitError:
                if attempt < _MAX_RETRIES:
                    delay = min(60.0, (2.0 ** attempt) * (0.5 + random.random()))
                    await asyncio.sleep(delay)
                else:
                    print(f"  [anthropic] request {entry['custom_id']} exhausted retries (rate limit).")
            except Exception as e:
                print(f"  [anthropic] request {entry['custom_id']} failed: {e}")
                break
        counter[0] += 1
        if counter[0] % 50 == 0 or counter[0] == total:
            print(f"  {counter[0]}/{total} complete", flush=True)
        from batch.common import normalize_bool_text
        return entry["custom_id"], {
            "judgement": normalize_bool_text(raw),
            "prob": None,
            "model": model,
            "raw_text": raw,
        }

    return list(await asyncio.gather(*[_call(e) for e in entries]))


async def _judge_gemini(
    entries: list[dict],
    model: str,
    max_tokens: int,
    temperature: float | None,
    sem: asyncio.Semaphore,
    counter: list[int],
    total: int,
) -> list[tuple[str, dict]]:
    import openai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is required for --judge gemini.")
    client = openai.AsyncOpenAI(
        api_key=api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    async def _call(entry: dict) -> tuple[str, dict]:
        raw = ""
        for attempt in range(_MAX_RETRIES + 1):
            try:
                async with sem:
                    call_kwargs = {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": entry["system"]},
                            {"role": "user", "content": entry["user"]},
                        ],
                        "max_tokens": max_tokens,
                    }
                    if temperature is not None:
                        call_kwargs["temperature"] = temperature
                    resp = await client.chat.completions.create(**call_kwargs)
                raw = (resp.choices[0].message.content or "").strip()
                break
            except openai.RateLimitError:
                if attempt < _MAX_RETRIES:
                    delay = min(60.0, (2.0 ** attempt) * (0.5 + random.random()))
                    await asyncio.sleep(delay)
                else:
                    print(f"  [gemini] request {entry['custom_id']} exhausted retries (rate limit).")
            except Exception as e:
                print(f"  [gemini] request {entry['custom_id']} failed: {e}")
                break
        counter[0] += 1
        if counter[0] % 50 == 0 or counter[0] == total:
            print(f"  {counter[0]}/{total} complete", flush=True)
        from batch.common import normalize_bool_text
        return entry["custom_id"], {
            "judgement": normalize_bool_text(raw),
            "prob": None,
            "model": model,
            "raw_text": raw,
        }

    return list(await asyncio.gather(*[_call(e) for e in entries]))


async def _run_judge_async(
    entries: list[dict],
    provider: str,
    model: str,
    max_tokens: int,
    temperature: float,
    max_concurrent: int,
) -> dict[str, dict]:
    sem = asyncio.Semaphore(max_concurrent)
    counter: list[int] = [0]
    total = len(entries)

    if provider == "openai":
        pairs = await _judge_openai(entries, model, max_tokens, temperature, sem, counter, total)
    elif provider == "anthropic":
        pairs = await _judge_anthropic(entries, model, max_tokens, temperature, sem, counter, total)
    elif provider == "gemini":
        pairs = await _judge_gemini(entries, model, max_tokens, temperature, sem, counter, total)
    else:
        raise ValueError(f"Unknown provider '{provider}'.")

    return dict(pairs)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_frontier_judge_v2(
    dataset_config: DatasetConfig,
    extraction_model: str,
    provider: str,
    frontier_model: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    max_concurrent: int = 32,
    max_tokens: int = 5,
    temperature: float | None = None,
    ablation: str | None = None,
) -> None:
    """Run the frontier judge using direct async API calls and save responses.json."""
    if provider not in FRONTIER_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {FRONTIER_PROVIDERS}")

    input_file = _find_extraction_final(dataset_config.name, extraction_model, extraction_date, ablation)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    from batch import common as batch_common
    documents = batch_common.load_documents_for_dataset(dataset_config, effective_ocr_dir)
    chat_entries = batch_common.prepare_chat_entries(data, documents, dataset_config)

    print(f"Calling {provider} ({frontier_model}) for {len(chat_entries)} entries "
          f"[max_concurrent={max_concurrent}] ...")

    results = asyncio.run(
        _run_judge_async(chat_entries, provider, frontier_model, max_tokens, temperature, max_concurrent)
    )

    data_out = batch_common.merge_results(data, results)
    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(data_out, f, indent=4, ensure_ascii=False)
    print(f"Responses saved to {responses_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run frontier judge via direct async API calls (v2).",
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
        choices=sorted(FRONTIER_PROVIDERS),
        help=f"Frontier provider. Available: {sorted(FRONTIER_PROVIDERS)}",
    )
    p.add_argument("--frontier-model", required=True, help="Provider model name (e.g. 'gpt-5-mini').")
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd of extraction run.")
    p.add_argument("--judge-date", default=None, help="Date tag for output directory (default: today).")
    p.add_argument(
        "--ablation", default=None, metavar="N",
        help="Ablation number. If set, reads from ablations/ablation{N}/ and writes judge output there.",
    )
    p.add_argument(
        "--ocr-dir", default=None, metavar="DIR",
        help="Directory of OCR .txt files. Defaults to {data_dir}/ocr_output_raw/.",
    )
    p.add_argument(
        "--max-concurrent", type=int, default=32, metavar="N",
        help="Maximum number of concurrent API calls (default: 32).",
    )
    p.add_argument(
        "--max-tokens", type=int, default=5, metavar="N",
        help="Max tokens per judge response (default: 5; judge outputs 'true'/'false').",
    )
    p.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature. If omitted, the provider default is used (recommended for models that restrict this parameter, e.g. gpt-5-mini).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    dataset_config = _load_dataset_config(args.dataset)
    input_file = _find_extraction_final(args.dataset, args.extraction_model, args.extraction_date, args.ablation)
    extraction_date_resolved = input_file.parent.name
    output_dir = get_judge_output_dir(
        args.dataset, args.extraction_model, extraction_date_resolved,
        args.judge, args.judge_date, ablation=args.ablation,
    )

    print(f"\nDataset          : {args.dataset}")
    print(f"Extraction model : {args.extraction_model}")
    print(f"Extraction date  : {extraction_date_resolved}")
    if args.ablation:
        print(f"Ablation         : {args.ablation}")
    print(f"Judge            : {args.judge} / {args.frontier_model}")
    print(f"Output           : {output_dir}\n")

    run_frontier_judge_v2(
        dataset_config=dataset_config,
        extraction_model=args.extraction_model,
        provider=args.judge,
        frontier_model=args.frontier_model,
        output_dir=output_dir,
        extraction_date=extraction_date_resolved,
        ocr_dir=args.ocr_dir,
        max_concurrent=args.max_concurrent,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()
