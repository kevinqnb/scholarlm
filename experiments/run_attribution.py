"""
Input-token attribution for the interpretability judge (NNsight / attribution.py).

Runs one attribution method (``src/scholarlm/attribution.py``'s
``ATTRIBUTION_REGISTRY``) over every measurement of a dataset's judge inputs and
persists one Input×Gradient score per context token per measurement, joined back
to the pre-existing interp-judge ``responses.json`` record so the judgement rides
alongside the attribution scores.

This is the persisted-output-schema + batch-runner scope that
``notes/scholarlm/builds/2026-08-18-token-attribution-01.md`` explicitly deferred.
It calls ``AttributionMethod.attribute()`` (its own single ``llm.trace()``,
never ``generate()``); it does not touch ``attribution.py``'s internals.

The paired interp-judge run is a HARD PREREQUISITE, not produced here — the
runner fails loud if it is missing (see ``--judge-date`` / ``paths.find_judge_responses``).

Standard output path (real-extraction mode):
    data/experiments/{dataset}/attribution/{extraction_model}/{extraction_date}/{judge_model}/{method}/{date}/

Synthetic probe output path (--synthetic):
    data/experiments/{dataset}/attribution_synthetic[_test]/{judge_model}/{method}/{date}/

Saves:
  - ``attribution_scores.npz`` — per-measurement float32 ``(n_context_tokens,)``
    score arrays keyed by ``measurement_id``, each with a parallel
    ``{measurement_id}__context_token_indices`` int32 array (token positions in
    the full tokenized prompt), plus a ``measurement_ids`` string array.
  - ``attribution.json`` — per-measurement sidecar: method scalar
    (``target`` / ``probe_output``), joined judgement fields, ``n_context_tokens``.
  - ``run_metadata.json`` — adds resolved judge date, context-token-count stats,
    and peak GPU memory (the 2026-08-18 build's "measure, don't assume" item).

Usage
-----
    # Real extraction run
    python experiments/run_attribution.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --extraction-date 2026_05_05 \\
        --judge qwen-2.5-7b \\
        --method probe

    # Synthetic probe dataset (train split)
    python experiments/run_attribution.py \\
        --dataset pond --synthetic \\
        --judge llama-3.1-8b --method contrastive_gradient

NOTE: ``scripts/submit.sh <id>`` cannot run this — the contract adapter only
covers entry_point extraction|ablation, and the interp-judge models are not in
the extraction MODEL_REGISTRY. Full runs are hand-authored single-GPU qsub jobs
(see the build note).
"""
from __future__ import annotations

import argparse
import gc
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
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import torch

from scholarlm.attribution import ATTRIBUTION_REGISTRY
from scholarlm.config import DatasetConfig

import paths
from model_registry import INTERP_JUDGE_REGISTRY as JUDGE_REGISTRY
from run_extraction import load_dataset_config
from utils import load_config, set_seeds, write_run_metadata


# Which extra key each method's attribute() returns its scalar under (the
# runner handles both). See attribution.py's ContrastiveGradientAttribution /
# ProbeAttribution return dicts.
SCALAR_KEYS: dict[str, str] = {
    "contrastive_gradient": "target",
    "probe": "probe_output",
}


# ---------------------------------------------------------------------------
# Input loading (mirrors run_jacobian_lens._load_chat_entries)
# ---------------------------------------------------------------------------


def _load_inputs(
    dataset_config: DatasetConfig,
    input_file: Path,
    ocr_dir: str | None,
    limit: int | None,
) -> tuple[list[dict], list[dict]]:
    """Load raw records + chat entries, exactly as run_judge_interp.py does.

    Args:
        dataset_config: Dataset configuration.
        input_file: A ``final.json``-shaped extraction file, or a synthetic
            ``probe_dataset*.json``.
        ocr_dir: Directory of OCR ``.txt`` files. ``None`` -> ``{data_dir}/ocr_output_raw``
            (the same default run_judge_interp.py uses — every paired judge run
            for this build's matrix used that default, so the attributed context
            must be built from it too).
        limit: If given, truncate ``data`` to the first ``limit`` records before
            document loading / chat-entry prep.

    Returns:
        (data, chat_entries) — the (possibly truncated) raw records and the chat
        entries built via ``judge_common.prepare_chat_entries``. ``chat_entries``
        may be shorter than ``data`` (prepare_chat_entries skips records with an
        unknown document_id or attribute).
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


def _load_responses_by_mid(responses_path: Path) -> dict[str, dict]:
    """Load an interp-judge responses.json and index it by ``str(measurement_id)``.

    Raises:
        AssertionError: If a measurement_id is duplicated.
    """
    with open(responses_path) as f:
        responses: list[dict] = json.load(f)
    by_mid: dict[str, dict] = {}
    for r in responses:
        mid = str(r["measurement_id"])
        assert mid not in by_mid, (
            f"duplicate measurement_id {mid!r} in {responses_path}"
        )
        by_mid[mid] = r
    return by_mid


# ---------------------------------------------------------------------------
# Core attribution loop (model-free-testable: `method` is any object with
# `.attribute(instructions, context, query) -> dict`)
# ---------------------------------------------------------------------------


def attribute_dataset(
    method: Any,
    method_name: str,
    data: list[dict],
    chat_entries: list[dict],
    responses_by_mid: dict[str, dict],
    output_dir: Path,
    *,
    empty_cache_every: int = 1,
) -> dict[str, Any]:
    """Run ``method.attribute()`` over every chat entry and persist the results.

    Writes ``attribution_scores.npz`` and ``attribution.json`` to ``output_dir``.
    The caller is responsible for ``run_metadata.json``.

    Args:
        method: Object exposing ``attribute(instructions, context, query) -> dict``
            with keys ``scores``, ``context_token_indices``, and the method's
            scalar under ``SCALAR_KEYS[method_name]``.
        method_name: Key in ``ATTRIBUTION_REGISTRY`` (selects the scalar key).
        data: Raw records; ``data[int(entry["custom_id"])]["measurement_id"]`` is
            the join key.
        chat_entries: Entries from ``judge_common.prepare_chat_entries``.
        responses_by_mid: ``str(measurement_id) -> interp-judge responses.json record``.
        output_dir: Written to (created if absent).
        empty_cache_every: Call ``torch.cuda.empty_cache()`` / ``gc.collect()``
            every N measurements (memory hygiene — the attribution trace keeps
            the autograd graph live, ~2x peak vs a forward-only pass).

    Returns:
        Summary dict (n_measurements, per-measurement context-token counts).
    """
    assert method_name in SCALAR_KEYS, (
        f"unknown method {method_name!r}; expected one of {sorted(SCALAR_KEYS)}"
    )
    scalar_key = SCALAR_KEYS[method_name]

    output_dir.mkdir(parents=True, exist_ok=True)

    npz_dict: dict[str, np.ndarray] = {}
    sidecar: list[dict] = []
    mids: list[str] = []
    ctx_lens: list[int] = []
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()

    for n_done, entry in enumerate(chat_entries):
        orig_idx = int(entry["custom_id"])
        mid = str(data[orig_idx]["measurement_id"])
        assert mid not in mids, (
            f"measurement_id {mid!r} appears twice in the input — npz keys would collide"
        )

        instructions = entry["system"]
        context = entry["page_text"]
        query = entry["user_query"]

        res = method.attribute(instructions, context, query)

        scores = np.asarray(res["scores"], dtype=np.float32)
        idx = np.asarray(res["context_token_indices"], dtype=np.int32)
        assert scores.ndim == 1 and idx.ndim == 1, (
            f"measurement_id {mid}: expected 1-D scores/indices, got "
            f"{scores.shape} / {idx.shape}"
        )
        assert len(scores) == len(idx), (
            f"measurement_id {mid}: len(scores)={len(scores)} != "
            f"len(context_token_indices)={len(idx)}"
        )
        assert np.isfinite(scores).all(), f"measurement_id {mid}: non-finite scores"
        assert scalar_key in res, (
            f"measurement_id {mid}: method {method_name!r} returned no {scalar_key!r} key "
            f"(got {sorted(res)})"
        )
        scalar_val = float(res[scalar_key])

        # Join to the pre-existing interp-judge record. Missing id or a
        # None p_true (a prepare_chat_entries-skipped judge row, which lands in
        # responses.json as judgement=False / p_true=None) is a hard error.
        assert mid in responses_by_mid, (
            f"measurement_id {mid} has no record in the paired interp-judge "
            f"responses.json — the judge run is stale or incomplete"
        )
        jr = responses_by_mid[mid]
        assert jr.get("judgement_p_true") is not None, (
            f"measurement_id {mid}: paired judge record has judgement_p_true=None "
            f"(skipped judge row?) — join would silently pair to a non-judgement"
        )
        assert jr["document_id"] == entry["document_id"], (
            f"measurement_id {mid}: paired judge record document_id="
            f"{jr['document_id']!r} != attributed entry document_id="
            f"{entry['document_id']!r} — responses.json is not row-aligned with "
            f"this extraction / chat-entry set (stale or mismatched judge run)"
        )

        idx_key = f"{mid}__context_token_indices"
        assert idx_key not in npz_dict, f"npz key collision on {idx_key!r}"
        npz_dict[mid] = scores
        npz_dict[idx_key] = idx
        mids.append(mid)
        ctx_lens.append(int(len(idx)))

        sidecar.append({
            "measurement_id": data[orig_idx]["measurement_id"],
            "document_id": entry["document_id"],
            "method": method_name,
            "scalar_name": scalar_key,
            scalar_key: scalar_val,
            "judgement": jr.get("judgement"),
            "judgement_p_true": jr.get("judgement_p_true"),
            "judgement_p_false": jr.get("judgement_p_false"),
            "n_context_tokens": int(len(idx)),
        })

        del res, scores, idx
        if cuda and (n_done + 1) % empty_cache_every == 0:
            torch.cuda.empty_cache()
            gc.collect()

    assert len(mids) == len(chat_entries), (
        f"wrote {len(mids)} measurements but had {len(chat_entries)} chat entries"
    )
    assert len(set(mids)) == len(mids), "duplicate measurement_id in output"

    npz_path = output_dir / "attribution_scores.npz"
    np.savez_compressed(
        npz_path,
        measurement_ids=np.asarray(mids, dtype=object).astype("U"),
        **npz_dict,
    )
    print(f"Attribution scores : {npz_path}  ({npz_path.stat().st_size / 1e6:.1f} MB)")

    sidecar_path = output_dir / "attribution.json"
    with open(sidecar_path, "w") as f:
        json.dump(sidecar, f, indent=2, ensure_ascii=False)
    print(f"Sidecar            : {sidecar_path}")

    summary: dict[str, Any] = {
        "n_measurements": len(mids),
        "context_token_count_min": min(ctx_lens) if ctx_lens else 0,
        "context_token_count_max": max(ctx_lens) if ctx_lens else 0,
        "context_token_count_mean": (sum(ctx_lens) / len(ctx_lens)) if ctx_lens else 0.0,
    }
    if cuda:
        summary["peak_gpu_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        print(
            f"Peak GPU mem       : {summary['peak_gpu_mem_bytes'] / 1e9:.2f} GB   "
            f"context tokens: min={summary['context_token_count_min']} "
            f"max={summary['context_token_count_max']} "
            f"mean={summary['context_token_count_mean']:.0f}"
        )
    return summary


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_attribution(
    dataset_config: DatasetConfig,
    method_name: str,
    judge_key: str,
    input_file: Path,
    responses_path: Path,
    resolved_judge_date: str,
    output_dir: Path,
    seed: int,
    *,
    extraction_model: str | None,
    extraction_date: str | None,
    ocr_dir: str | None,
    limit: int | None,
) -> None:
    if judge_key not in JUDGE_REGISTRY:
        raise KeyError(
            f"Unknown judge {judge_key!r}. Available: {sorted(JUDGE_REGISTRY)}"
        )
    if method_name not in ATTRIBUTION_REGISTRY:
        raise KeyError(
            f"Unknown method {method_name!r}. Available: {sorted(ATTRIBUTION_REGISTRY)}"
        )
    judge_cfg = JUDGE_REGISTRY[judge_key]

    print(f"Input              : {input_file}")
    print(f"Paired judge run   : {responses_path}")

    data, chat_entries = _load_inputs(dataset_config, input_file, ocr_dir, limit)
    print(f"Records            : {len(data)}   chat entries: {len(chat_entries)}")
    responses_by_mid = _load_responses_by_mid(responses_path)

    from scholarlm import JudgementLM
    llm = JudgementLM(
        model_name=judge_cfg["model_id"],
        sampling_params=judge_cfg["sampling_params"],
        nnsight_kwargs=judge_cfg["nnsight_kwargs"],
        use_chat_template=judge_cfg.get("use_chat_template", True),
        answer_cue=judge_cfg.get("answer_cue", None),
    )

    if method_name == "probe":
        from analysis.loaders import load_trained_probe
        # Fails loud (FileNotFoundError) if head_probe_noplatt.pkl is absent for
        # this (dataset, judge) pair.
        probe_data = load_trained_probe(
            dataset_config.name, judge_key, ptype="head", variant="noplatt"
        )
        method = ATTRIBUTION_REGISTRY["probe"](llm, probe_data)
    else:
        method = ATTRIBUTION_REGISTRY[method_name](llm)

    start_time = time.time()
    summary = attribute_dataset(
        method=method,
        method_name=method_name,
        data=data,
        chat_entries=chat_entries,
        responses_by_mid=responses_by_mid,
        output_dir=output_dir,
    )

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        extraction_model=extraction_model,
        extraction_date=extraction_date,
        judge_model=judge_key,
        judge_model_id=judge_cfg["model_id"],
        method=method_name,
        scalar_name=SCALAR_KEYS[method_name],
        resolved_judge_date=resolved_judge_date,
        seed=seed,
        max_prompt_tokens=llm.max_prompt_tokens,
        **summary,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run input-token attribution over a dataset's interp-judge inputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'pond', 'nfix', 'supermat').")
    p.add_argument(
        "--method", required=True, choices=sorted(ATTRIBUTION_REGISTRY),
        help=f"Attribution method. Available: {sorted(ATTRIBUTION_REGISTRY)}",
    )
    p.add_argument(
        "--judge", required=True, choices=sorted(JUDGE_REGISTRY),
        help=f"Interp-judge model key. Available: {sorted(JUDGE_REGISTRY)}",
    )
    p.add_argument(
        "--extraction-model", default=None,
        help="Extraction model whose results were judged. Required unless --synthetic.",
    )
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd of the extraction run.")
    p.add_argument(
        "--judge-date", default=None,
        help="Date tag of the paired interp-judge run (default: latest with a responses.json).",
    )
    p.add_argument(
        "--synthetic", action="store_true", default=False,
        help="Attribute over the synthetic probe dataset instead of an extraction run.",
    )
    p.add_argument(
        "--synthetic-split", choices=["train", "test"], default="train",
        help="Which synthetic split (only with --synthetic). Default: train.",
    )
    p.add_argument(
        "--ocr-dir", default=None, metavar="DIR",
        help="OCR .txt directory for document context. Default: {data_dir}/ocr_output_raw.",
    )
    p.add_argument("--date", default=None, help="Output date tag YYYY_mm_dd (default: today).")
    p.add_argument(
        "--limit", type=int, default=None, metavar="N",
        help="Only attribute the first N records (smoke / tiny-e2e).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # DatasetConfig paths (metadata_file, data_dir, ...) are repo-root-relative.
    os.chdir(_REPO_ROOT)

    cfg = load_config()
    seed = cfg["defaults"]["seed"]  # no fallback — CLAUDE.md's no-magic-number rule
    set_seeds(seed)

    dataset_config = load_dataset_config(args.dataset)

    if args.synthetic:
        split = args.synthetic_split
        probe_filename = "probe_dataset_test.json" if split == "test" else "probe_dataset.json"
        input_file = _REPO_ROOT / "data" / args.dataset / probe_filename
        if not input_file.exists():
            raise FileNotFoundError(
                f"Synthetic probe dataset not found: {input_file}. "
                f"Run data/{args.dataset}/create_probe_dataset.py first."
            )
        responses_path = paths.find_synthetic_responses(
            args.dataset, args.judge, args.judge_date, split
        )
        resolved_judge_date = responses_path.parent.name
        output_dir = paths.attribution_synthetic(
            args.dataset, args.judge, args.method, args.date, split
        )
        extraction_model = None
        extraction_date = None
        print(f"\nDataset            : {args.dataset}")
        print(f"Mode               : synthetic probe ({split})")
    else:
        if args.extraction_model is None:
            parser.error("--extraction-model is required unless --synthetic is used.")
        input_file = paths.find_extraction_final(
            args.dataset, args.extraction_model, args.extraction_date
        )
        extraction_model = args.extraction_model
        extraction_date = input_file.parent.name
        responses_path, resolved_judge_date = paths.find_judge_responses(
            args.dataset, extraction_model, extraction_date, args.judge, args.judge_date
        )
        output_dir = paths.attribution(
            args.dataset, extraction_model, extraction_date, args.judge, args.method, args.date
        )
        print(f"\nDataset            : {args.dataset}")
        print(f"Extraction model   : {extraction_model}")
        print(f"Extraction date    : {extraction_date}")

    print(f"Judge              : {args.judge}")
    print(f"Method             : {args.method}")
    print(f"Judge date         : {resolved_judge_date}")
    print(f"Seed               : {seed}")
    if args.limit:
        print(f"Limit              : {args.limit}")
    print(f"Output             : {output_dir}\n")

    run_attribution(
        dataset_config=dataset_config,
        method_name=args.method,
        judge_key=args.judge,
        input_file=input_file,
        responses_path=responses_path,
        resolved_judge_date=resolved_judge_date,
        output_dir=output_dir,
        seed=seed,
        extraction_model=extraction_model,
        extraction_date=extraction_date,
        ocr_dir=args.ocr_dir,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
