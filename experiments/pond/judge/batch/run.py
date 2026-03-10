"""CLI for batch judging: generate → submit → poll → process.

Usage — all-in-one (runs the full pipeline from start to finish):

    uv run python -m experiments.pond.judge.batch.run \\
        --provider openai run \\
        --input  data/experiments/2026_03_04/pond_final.json \\
        --docs   data/pond/ocr_output_cleaned_gpt_5_mini \\
        --model  gpt-5-mini \\
        --output data/experiments/2026_03_04/pond_judged_gpt_batch.json

    uv run python -m experiments.pond.judge.batch.run \\
        --provider anthropic run \\
        --input  data/experiments/2026_03_04/pond_final.json \\
        --docs   data/pond/ocr_output_cleaned_gpt_5_mini \\
        --model  claude-haiku-4-5 \\
        --output data/experiments/2026_03_04/pond_judged_claude_batch.json

    uv run python -m experiments.pond.judge.batch.run \\
        --provider gemini run \\
        --input      data/experiments/2026_03_04/pond_final.json \\
        --docs       data/pond/ocr_output_cleaned_gpt_5_mini \\
        --model      gemini-3.1-flash-lite-preview \\
        --output     data/experiments/2026_03_04/pond_judged_gemini_batch.json \\
        --dest-gcs   gs://my-bucket/judge-output/ \\
        --gcp-project my-gcp-project \\
        --gcp-location us-central1

Usage — step by step (useful when you want to inspect the batch before
processing, or resume after a crash):

    # 1. Submit and save the batch id to a state file
    uv run python -m experiments.pond.judge.batch.run \\
        --provider openai submit \\
        --input data/experiments/2026_03_04/pond_final.json \\
        --docs  data/pond/ocr_output_cleaned_gpt_5_mini \\
        --model gpt-5-mini \\
        --state batch_state.json

    # 2. Poll until done (reads batch id from state file)
    uv run python -m experiments.pond.judge.batch.run \\
        --provider openai poll --state batch_state.json

    # 3. Download and merge results
    uv run python -m experiments.pond.judge.batch.run \\
        --provider openai process \\
        --state  batch_state.json \\
        --input  data/experiments/2026_03_04/pond_final.json \\
        --output data/experiments/2026_03_04/pond_judged_gpt_batch.json

State file format (JSON):
    {"provider": "openai",    "batch_ids":   ["batch_xxx", "batch_yyy"], "model": "gpt-5-mini"}
    {"provider": "anthropic", "batch_ids":   ["msgbatch_xxx"],           "model": "claude-haiku-4-5"}
    {"provider": "gemini",    "batch_names": ["projects/.../jobs/xxx"],  "model": "...", "dest_gcs": "gs://..."}

Multiple ids/names appear when the input exceeds the provider's per-file size
limit and is automatically split into several concurrent batches.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .common import load_data, load_documents, merge_results, prepare_chat_entries


# ─── Provider dispatch ────────────────────────────────────────────────────────


def _get_openai_client():
    from openai import OpenAI

    return OpenAI()


def _get_anthropic_client():
    import anthropic

    return anthropic.Anthropic()


# ─── Core pipeline steps ──────────────────────────────────────────────────────


def step_submit(
    provider: str,
    input_file: str,
    docs_dir: str,
    model: str,
    state_file: str,
    dest_gcs: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
) -> list[str]:
    """Build and submit batches (splitting if needed). Saves state. Returns ids/names."""
    data = load_data(input_file)
    documents = load_documents(docs_dir)
    chat_entries = prepare_chat_entries(data, documents)

    state: dict[str, Any] = {"provider": provider, "model": model}

    if provider == "openai":
        from . import openai_batch

        client = _get_openai_client()
        requests = openai_batch.build_requests(chat_entries, model)
        batch_ids = openai_batch.submit_batch(requests, client=client)
        state["batch_ids"] = batch_ids

    elif provider == "anthropic":
        from . import anthropic_batch

        client = _get_anthropic_client()
        requests = anthropic_batch.build_requests(chat_entries, model)
        batch_ids = anthropic_batch.submit_batch(requests, client=client)
        state["batch_ids"] = batch_ids

    elif provider == "gemini":
        from . import gemini_batch

        if not dest_gcs:
            raise ValueError("--dest-gcs is required for the gemini provider")
        requests = gemini_batch.build_requests(chat_entries, model)
        batch_names = gemini_batch.submit_batch(
            requests, model,
            dest_gcs=dest_gcs,
            project=gcp_project,
            location=gcp_location,
        )
        state["batch_names"] = batch_names
        state["dest_gcs"] = dest_gcs
        if gcp_project:
            state["gcp_project"] = gcp_project
        if gcp_location:
            state["gcp_location"] = gcp_location

    else:
        raise ValueError(f"Unknown provider: {provider}")

    Path(state_file).write_text(json.dumps(state, indent=2))
    print(f"State saved to {state_file}")
    return state.get("batch_ids") or state.get("batch_names", [])


def step_poll(provider: str, state_file: str, interval: int = 60) -> None:
    """Poll all batches recorded in state_file until they all complete."""
    state = json.loads(Path(state_file).read_text())
    provider = state.get("provider", provider)

    if provider == "openai":
        from . import openai_batch

        client = _get_openai_client()
        openai_batch.poll_batch(state["batch_ids"], client=client, interval=interval)

    elif provider == "anthropic":
        from . import anthropic_batch

        client = _get_anthropic_client()
        anthropic_batch.poll_batch(state["batch_ids"], client=client, interval=interval)

    elif provider == "gemini":
        from . import gemini_batch

        gemini_batch.poll_batch(
            state["batch_names"],
            project=state.get("gcp_project"),
            location=state.get("gcp_location"),
            interval=interval,
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def step_process(
    provider: str,
    state_file: str,
    input_file: str,
    output_file: str,
) -> None:
    """Fetch results from all batch chunks, merge, and write output_file."""
    state = json.loads(Path(state_file).read_text())
    provider = state.get("provider", provider)
    model = state.get("model", "")

    data = load_data(input_file)

    if provider == "openai":
        from . import openai_batch

        client = _get_openai_client()
        results = openai_batch.fetch_results(state["batch_ids"], client=client, model=model)

    elif provider == "anthropic":
        from . import anthropic_batch

        client = _get_anthropic_client()
        results = anthropic_batch.fetch_results(state["batch_ids"], client=client, model=model)

    elif provider == "gemini":
        from . import gemini_batch

        results = gemini_batch.fetch_results(
            state["batch_names"], model=model,
            dest_gcs=state["dest_gcs"],
            project=state.get("gcp_project"),
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")

    data_out = merge_results(data, results)
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(data_out, f, indent=4, ensure_ascii=False)
    print(f"Wrote {len(data_out)} records to {output_file}")


def run_all(
    provider: str,
    input_file: str,
    docs_dir: str,
    model: str,
    output_file: str,
    *,
    dest_gcs: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    interval: int = 60,
    state_file: str = ".batch_state.json",
) -> None:
    """Run the full generate → submit → poll → process pipeline."""
    step_submit(
        provider, input_file, docs_dir, model, state_file,
        dest_gcs=dest_gcs, gcp_project=gcp_project, gcp_location=gcp_location,
    )
    step_poll(provider, state_file, interval=interval)
    step_process(provider, state_file, input_file, output_file)


# ─── CLI ──────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="batch/run.py",
        description="Batch judging pipeline for OpenAI, Anthropic, and Gemini.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--provider",
        required=True,
        choices=["openai", "anthropic", "gemini"],
        help="Which provider's batch API to use.",
    )

    sub = p.add_subparsers(dest="command", required=True)

    # ── run (all-in-one) ──────────────────────────────────────────────────────
    r = sub.add_parser("run", help="Full pipeline: submit → poll → process.")
    r.add_argument("--input", required=True, help="Path to extracted data JSON.")
    r.add_argument("--docs", required=True, help="Directory of OCR text files.")
    r.add_argument("--model", required=True, help="Model name for this provider.")
    r.add_argument("--output", required=True, help="Output JSON path.")
    r.add_argument("--dest-gcs", help="GCS URI for Gemini output (required for gemini).")
    r.add_argument("--gcp-project", help="GCP project id (gemini only; or set GOOGLE_CLOUD_PROJECT).")
    r.add_argument("--gcp-location", help="GCP region (gemini only; default us-central1).")
    r.add_argument("--interval", type=int, default=60, help="Poll interval in seconds.")
    r.add_argument("--state", default=".batch_state.json", help="Temp state file.")

    # ── submit ────────────────────────────────────────────────────────────────
    s = sub.add_parser("submit", help="Build and submit the batch; save state.")
    s.add_argument("--input", required=True, help="Path to extracted data JSON.")
    s.add_argument("--docs", required=True, help="Directory of OCR text files.")
    s.add_argument("--model", required=True, help="Model name for this provider.")
    s.add_argument("--dest-gcs", help="GCS URI for Gemini output (required for gemini).")
    s.add_argument("--gcp-project", help="GCP project id (gemini only; or set GOOGLE_CLOUD_PROJECT).")
    s.add_argument("--gcp-location", help="GCP region (gemini only; default us-central1).")
    s.add_argument("--state", default=".batch_state.json", help="State file to write.")

    # ── poll ──────────────────────────────────────────────────────────────────
    po = sub.add_parser("poll", help="Poll until the batch completes.")
    po.add_argument("--state", default=".batch_state.json", help="State file to read.")
    po.add_argument("--interval", type=int, default=60, help="Poll interval in seconds.")

    # ── process ───────────────────────────────────────────────────────────────
    pr = sub.add_parser("process", help="Download results and write output JSON.")
    pr.add_argument("--state", default=".batch_state.json", help="State file to read.")
    pr.add_argument("--input", required=True, help="Path to original extracted data JSON.")
    pr.add_argument("--output", required=True, help="Output JSON path.")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        run_all(
            provider=args.provider,
            input_file=args.input,
            docs_dir=args.docs,
            model=args.model,
            output_file=args.output,
            dest_gcs=getattr(args, "dest_gcs", None),
            gcp_project=getattr(args, "gcp_project", None),
            gcp_location=getattr(args, "gcp_location", None),
            interval=args.interval,
            state_file=args.state,
        )

    elif args.command == "submit":
        step_submit(
            provider=args.provider,
            input_file=args.input,
            docs_dir=args.docs,
            model=args.model,
            state_file=args.state,
            dest_gcs=getattr(args, "dest_gcs", None),
            gcp_project=getattr(args, "gcp_project", None),
            gcp_location=getattr(args, "gcp_location", None),
        )

    elif args.command == "poll":
        step_poll(
            provider=args.provider,
            state_file=args.state,
            interval=args.interval,
        )

    elif args.command == "process":
        step_process(
            provider=args.provider,
            state_file=args.state,
            input_file=args.input,
            output_file=args.output,
        )


if __name__ == "__main__":
    main()
