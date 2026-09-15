"""
Local vLLM judge pipeline.

Runs judge validation using a local model served via vLLM's OpenAI-compatible
API. Unlike the interpretability judge (run_judge_interp.py), this runner does
not load the model locally — it sends async HTTP requests to a running vLLM
server, making it practical for large models that would otherwise require
NNsight's full GPU memory.

Standard mode output path (id-addressed, like every other Tier-1 type):
    experiments/results/{dataset}/judge_local/{experiment_id}/

Synthetic probe mode output path (params.synthetic) is deliberately UNCHANGED
-- still the old date-addressed tree, since analysis/*.py's probe-calibration
pipeline (explicitly out of scope for this restructure) reads it directly.

Saves:
  - ``responses.json`` — per-measurement judgement (true/false from text response)

Usage
-----
    python experiments/run_judge_local.py experiments/experiment-configs/pond/judge_local/<id>/<id>.yaml

Required params: dataset, judge.
Standard mode requires params.extraction_id (an extraction or ablation
experiment id, resolved via utils.find_result_dir -- its final.json is judged).
Synthetic mode (params.synthetic: true, or params.synthetic_file) ignores
extraction_id; see below for its params (unchanged from before this restructure).
Optional params: extraction_id, judge_date, ocr_dir, api_base, api_key,
synthetic (bool), synthetic_split ('train'|'test'),
synthetic_file, synthetic_name.

max_concurrent and request_timeout are NOT experiment params (2026-09-14) --
they come from the judge's own model-config
(experiments/model-configs/vllm_judge/<judge>.yaml), since they're serving
characteristics of a given judge model under this stack, not something that
varies per experiment. Setting either in params.* is a hard error.

Available judge models: the YAML files in experiments/model-configs/vllm_judge/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

from openai import AsyncOpenAI

from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory

from run_extraction import load_dataset_config
import utils as paths
from utils import set_seeds, write_run_metadata


# ---------------------------------------------------------------------------
# Async judge call
# ---------------------------------------------------------------------------


async def _judge_one(
    client: AsyncOpenAI,
    model_id: str,
    entry: dict,
    idx: int,
    sem: asyncio.Semaphore,
    sampling_params: dict | None = None,
) -> dict:
    """Send a single judge request and return its true/false judgement.

    Fails loud: a request that errors, comes back with no content, or whose
    text yields neither ``true`` nor ``false`` raises. It must not resolve to a
    null judgement — ``run_judge_combine`` counts anything that is not ``True``
    as a non-affirmative vote, so a silently-dropped request would bias the
    majority-vote ground truth without leaving a trace.
    """
    # Per-judge sampling overrides (temperature/seed/reasoning_effort) come
    # from the judge's own model-config now -- see
    # experiments/model-configs/vllm_judge/gpt-oss-120b.yaml's sampling_params
    # comment for the debugging history behind its specific values. A judge
    # with no sampling_params block (every one except gpt-oss-120b today)
    # falls back to the defaults below, unchanged from before this was
    # config-driven.
    sp = sampling_params or {}
    temperature = sp.get("temperature", 0.0)
    seed = sp.get("seed")
    extra_body = (
        {"chat_template_kwargs": {"reasoning_effort": sp["reasoning_effort"]}}
        if "reasoning_effort" in sp else None
    )

    async with sem:
        try:
            kwargs: dict = dict(
                model=model_id,
                messages=[
                    {"role": "system", "content": entry["system"]},
                    {"role": "user", "content": entry["user"]},
                ],
                # 8192: reasoning models (gpt-oss-120b) emit a long analysis
                # channel before the verdict; 2048 truncated it on ~0.5% of
                # pond rows (fixed by the 8192 bump, commit 7fcf19d). See
                # experiments/model-configs/vllm_judge/gpt-oss-120b.yaml's
                # sampling_params comment for why gpt-oss-120b's remaining
                # supermat truncations were a temperature issue, not a
                # max_tokens issue -- this cap is unchanged from 7fcf19d.
                # Non-reasoning judges are unaffected (they stop far short of
                # this cap).
                max_tokens=8192,
                temperature=temperature,
                extra_body=extra_body,
            )
            if seed is not None:
                kwargs["seed"] = seed
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            raise RuntimeError(
                f"[idx={idx}] judge API call failed: {type(e).__name__}: {e}"
            ) from e

    choice = response.choices[0]
    response_text = (choice.message.content or "").strip()

    print(f"  [idx={idx}] Response: {response_text}")

    # Derive judgement from the response text. Parse semantics are unchanged
    # from the original ("true" wins if both appear); only the no-verdict case,
    # which used to fall through to None, now raises.
    t = response_text.lower()
    if "true" in t:
        judgement = True
    elif "false" in t:
        judgement = False
    else:
        raise ValueError(
            f"[idx={idx}] judge response has no true/false verdict "
            f"(finish_reason={choice.finish_reason!r}): {response_text!r}"
        )

    return {
        "judgement": judgement,
        "judgement_model": response.model,
        "_prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_local_vllm_judge(
    dataset_config: DatasetConfig,
    judge_key: str,
    output_dir: Path,
    input_file: Path,
    max_concurrent: int,
    request_timeout: float,
    ocr_dir: str | None = None,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
    extraction_id: str | None = None,
) -> None:
    """Run a local vLLM judge and save responses.

    The judge sees the full OCR paper as ``## CONTEXT``. A row carrying a
    ``context_override`` field (set by the synthetic-probe augmentation
    pipeline) uses that text verbatim instead — see
    ``judge_prompts.prepare_chat_entries``.

    Args:
        dataset_config: Dataset configuration.
        judge_key: Key in ``experiments/model-configs/vllm_judge/``.
        output_dir: Directory to write ``responses.json``.
        input_file: Path to the ``final.json``-shaped file to judge (an
            extraction/ablation run's output, or a synthetic probe file).
        max_concurrent: Maximum concurrent requests to the server. Comes from
            the judge's model-config (experiments/model-configs/vllm_judge/),
            not from experiment params -- see module docstring.
        request_timeout: Per-request client timeout (seconds) for the
            OpenAI-compatible HTTP client. Same source as max_concurrent.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the vLLM server.
        extraction_id: The upstream extraction/ablation experiment id being
            judged, recorded in run_metadata.json. ``None`` for synthetic mode.
    """
    judge_cfg = paths.load_model_config("vllm_judge", judge_key)
    model_id = judge_cfg["model_id"]

    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    from scholarlm.utils import judge_prompts
    documents = judge_prompts.load_documents_for_dataset(dataset_config, effective_ocr_dir)
    print(f"Documents: {len(documents)} loaded from {effective_ocr_dir}")

    chat_entries = judge_prompts.prepare_chat_entries(data, documents, dataset_config)

    # chat_entries are sorted by document_id for cache locality; we need to
    # track the original indices to merge results back in order.
    # prepare_chat_entries sets custom_id = str(original index in data).

    print(f"Sending {len(chat_entries)} requests to {api_base} (model: {model_id}) ...")
    print(f"max_concurrent={max_concurrent}\n")

    start_time = time.time()
    client = AsyncOpenAI(api_key=api_key, base_url=api_base, timeout=request_timeout)
    sem = asyncio.Semaphore(max_concurrent)

    judge_sampling_params = judge_cfg.get("sampling_params")

    async def _run_all() -> list[dict | BaseException]:
        tasks = [
            _judge_one(client, model_id, entry, int(entry["custom_id"]), sem, judge_sampling_params)
            for entry in chat_entries
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)

    raw_results = asyncio.run(_run_all())

    # Any failed request aborts the run before responses.json is written — a
    # partial file would feed a biased vote into run_judge_combine unnoticed.
    failures = [
        (int(entry["custom_id"]), res)
        for entry, res in zip(chat_entries, raw_results)
        if isinstance(res, BaseException)
    ]
    if failures:
        preview = "\n".join(f"    idx={i}: {res}" for i, res in failures[:10])
        more = "" if len(failures) <= 10 else f"\n    ... and {len(failures) - 10} more"
        raise RuntimeError(
            f"{len(failures)}/{len(chat_entries)} judge requests failed; "
            f"responses.json NOT written.\n{preview}{more}"
        )

    max_pt = max((r.get("_prompt_tokens", 0) or 0) for r in raw_results)

    # Map results back to the original data order using custom_id
    result_by_orig_idx: dict[int, dict] = {}
    for entry, result in zip(chat_entries, raw_results):
        orig_idx = int(entry["custom_id"])
        result_by_orig_idx[orig_idx] = result

    judged_data: list[dict] = []
    for i, record in enumerate(data):
        result = result_by_orig_idx.get(i, {})
        judged_data.append(record | {k: v for k, v in result.items() if k != "_prompt_tokens"})

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(judged_data, f, indent=4, ensure_ascii=False)
    print(f"Responses saved to {responses_file}")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        extraction_id=extraction_id,
        judge_model=judge_key,
        judge_model_id=model_id,
        max_prompt_tokens=max_pt,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run local vLLM judge (text-based judgement).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    p.add_argument(
        "--api-base", default=None, metavar="URL",
        help="Override params.api_base (submit.sh injects the compute node's vLLM endpoint here).",
    )
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
    judge_cfg = paths.load_model_config("vllm_judge", judge)  # fail loud on an unknown judge before any work starts
    if "max_concurrent" in params or "request_timeout" in params:
        raise ValueError(
            f"{config_path}: params.max_concurrent/params.request_timeout are no "
            f"longer read from experiment params (2026-09-14) -- they come from "
            f"experiments/model-configs/vllm_judge/{judge}.yaml instead. Remove "
            f"them from this config's params."
        )
    dataset_config = load_dataset_config(dataset)
    api_base = args.api_base or params.get("api_base") or "http://localhost:8081/v1"
    api_key = params.get("api_key", "EMPTY")
    max_concurrent = judge_cfg["max_concurrent"]
    request_timeout = judge_cfg["request_timeout"]

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
        print(f"API base         : {api_base}")
        print(f"Output           : {output_dir}\n")
        run_local_vllm_judge(
            dataset_config=dataset_config,
            judge_key=judge,
            output_dir=output_dir,
            input_file=probe_file,
            ocr_dir=params.get("ocr_dir"),
            api_base=api_base,
            api_key=api_key,
            max_concurrent=max_concurrent,
            request_timeout=request_timeout,
        )
        return

    if synthetic:
        splits = [params["synthetic_split"]] if params.get("synthetic_split") else ["train", "test"]
        for split in splits:
            probe_filename = "probe_dataset_test.json" if split == "test" else "probe_dataset.json"
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
            print(f"API base         : {api_base}")
            print(f"Output           : {output_dir}\n")
            run_local_vllm_judge(
                dataset_config=dataset_config,
                judge_key=judge,
                output_dir=output_dir,
                input_file=probe_file,
                ocr_dir=params.get("ocr_dir"),
                api_base=api_base,
                api_key=api_key,
                max_concurrent=max_concurrent,
                request_timeout=request_timeout,
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
    output_dir = paths.result_dir(dataset, "judge_local", cfg["id"])

    print(f"\nDataset          : {dataset}")
    print(f"Extraction id    : {extraction_id}")
    print(f"Judge            : {judge}")
    print(f"API base         : {api_base}")
    print(f"Output           : {output_dir}\n")
    run_local_vllm_judge(
        dataset_config=dataset_config,
        judge_key=judge,
        output_dir=output_dir,
        input_file=input_file,
        ocr_dir=params.get("ocr_dir"),
        api_base=api_base,
        api_key=api_key,
        max_concurrent=max_concurrent,
        extraction_id=extraction_id,
        request_timeout=request_timeout,
    )


if __name__ == "__main__":
    main()
