"""
Frontier judge pipeline (OpenAI / Anthropic / Gemini Batch API).

Runs judge validation for a given (dataset, extraction_model, judge_model) triple
using a frontier model via its provider's batch API.

Output path:
    data/experiments/{dataset}/judge/{extraction_model}/{judge_model}/{YYYY_mm_dd}/

Saves:
  - ``responses.json`` — per-measurement judgement + raw text response

Usage
-----
    # Submit, poll, and process in one shot:
    python experiments/run_judge_frontier.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judge openai \\
        --frontier-model gpt-4o-mini \\
        --extraction-date 2026_04_01

    # Step-by-step (submit → poll → process):
    python experiments/run_judge_frontier.py \\
        --dataset pond --extraction-model gemma-3-27b \\
        --judge anthropic --frontier-model claude-haiku-4-5 \\
        --extraction-date 2026_04_01 \\
        submit
    python experiments/run_judge_frontier.py ... poll --state .batch_state_anthropic.json
    python experiments/run_judge_frontier.py ... process --state .batch_state_anthropic.json

Available frontier providers: openai, anthropic, gemini
"""
from __future__ import annotations

import argparse
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
# Make 'batch' importable as a package (experiments/batch/)
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from dotenv import load_dotenv
load_dotenv()

from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory

random.seed(342)

FRONTIER_PROVIDERS = {"openai", "anthropic", "gemini"}

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
# Runner
# ---------------------------------------------------------------------------


def run_frontier_judge(
    dataset_config: DatasetConfig,
    extraction_model: str,
    provider: str,
    frontier_model: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    dest_gcs: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
    interval: int = 60,
    state_file: str | None = None,
) -> None:
    """Run a frontier batch judge and save responses.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model.
        provider: One of ``"openai"``, ``"anthropic"``, ``"gemini"``.
        frontier_model: Provider-specific model name (e.g. ``"gpt-4o-mini"``).
        output_dir: Directory to write ``responses.json``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to ``{data_dir}/ocr_output_raw/``.
        dest_gcs: GCS URI (Gemini only).
        gcp_project: GCP project (Gemini only).
        gcp_location: GCP region (Gemini only).
        interval: Poll interval in seconds.
        state_file: Optional path for batch state JSON.
    """
    if provider not in FRONTIER_PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {FRONTIER_PROVIDERS}")

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

    from batch import common as batch_common
    chat_entries = batch_common.prepare_chat_entries(data, documents, dataset_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    if state_file is None:
        state_file = str(output_dir / f".batch_state_{provider}.json")

    from batch import openai_batch, anthropic_batch, gemini_batch

    # Submit
    state: dict[str, Any] = {"provider": provider, "model": frontier_model}
    if provider == "openai":
        from openai import OpenAI
        client = OpenAI()
        requests = openai_batch.build_requests(chat_entries, frontier_model)
        batch_ids = openai_batch.submit_batch(requests, client=client)
        state["batch_ids"] = batch_ids

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        requests = anthropic_batch.build_requests(chat_entries, frontier_model)
        batch_ids = anthropic_batch.submit_batch(requests, client=client)
        state["batch_ids"] = batch_ids

    elif provider == "gemini":
        requests = gemini_batch.build_requests(chat_entries, frontier_model)
        batch_names = gemini_batch.submit_batch(
            requests, frontier_model,
            dest_gcs=dest_gcs, project=gcp_project, location=gcp_location,
        )
        state["batch_names"] = batch_names
        state["dest_gcs"] = dest_gcs
        if gcp_project:
            state["gcp_project"] = gcp_project
        if gcp_location:
            state["gcp_location"] = gcp_location

    Path(state_file).write_text(json.dumps(state, indent=2))
    print(f"Batch submitted. State saved to {state_file}")

    # Poll then fetch results
    if provider == "openai":
        from openai import OpenAI
        openai_batch.poll_batch(state["batch_ids"], client=OpenAI(), interval=interval)
        results = openai_batch.fetch_results(state["batch_ids"], client=OpenAI(), model=frontier_model)

    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        anthropic_batch.poll_batch(state["batch_ids"], client=client, interval=interval)
        results = anthropic_batch.fetch_results(state["batch_ids"], client=client, model=frontier_model)

    elif provider == "gemini":
        gemini_batch.poll_batch(
            state["batch_names"],
            project=state.get("gcp_project"),
            location=state.get("gcp_location"),
            interval=interval,
        )
        results = gemini_batch.fetch_results(
            state["batch_names"], model=frontier_model,
            dest_gcs=state["dest_gcs"],
            project=state.get("gcp_project"),
        )

    data_out = batch_common.merge_results(data, results)
    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(data_out, f, indent=4, ensure_ascii=False)
    print(f"Responses saved to {responses_file}")


# ---------------------------------------------------------------------------
# Step-by-step helpers (submit / poll / process separately)
# ---------------------------------------------------------------------------


def _submit(
    dataset_config: DatasetConfig,
    extraction_model: str,
    provider: str,
    frontier_model: str,
    output_dir: Path,
    state_file: str,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
    dest_gcs: str | None = None,
    gcp_project: str | None = None,
    gcp_location: str | None = None,
) -> None:
    """Submit batch requests and save state file."""
    input_file = _find_extraction_final(dataset_config.name, extraction_model, extraction_date)
    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    text_files = get_filenames_in_directory(effective_ocr_dir, ignore=[".DS_Store", ".gitkeep"])
    text_files.sort()
    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(effective_ocr_dir, fname), "r", encoding="utf-8") as fh:
            documents.append(fh.read())

    from batch import common as batch_common
    chat_entries = batch_common.prepare_chat_entries(data, documents, dataset_config)
    output_dir.mkdir(parents=True, exist_ok=True)

    from batch import openai_batch, anthropic_batch, gemini_batch

    state: dict[str, Any] = {"provider": provider, "model": frontier_model}
    if provider == "openai":
        from openai import OpenAI
        requests = openai_batch.build_requests(chat_entries, frontier_model)
        state["batch_ids"] = openai_batch.submit_batch(requests, client=OpenAI())
    elif provider == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        requests = anthropic_batch.build_requests(chat_entries, frontier_model)
        state["batch_ids"] = anthropic_batch.submit_batch(requests, client=client)
    elif provider == "gemini":
        requests = gemini_batch.build_requests(chat_entries, frontier_model)
        batch_names = gemini_batch.submit_batch(
            requests, frontier_model,
            dest_gcs=dest_gcs, project=gcp_project, location=gcp_location,
        )
        state["batch_names"] = batch_names
        state["dest_gcs"] = dest_gcs
        if gcp_project:
            state["gcp_project"] = gcp_project
        if gcp_location:
            state["gcp_location"] = gcp_location

    Path(state_file).write_text(json.dumps(state, indent=2))
    print(f"Batch submitted. State saved to {state_file}")


def _poll(state_file: str, interval: int) -> None:
    """Poll batch status until completion."""
    state = json.loads(Path(state_file).read_text())
    provider = state["provider"]
    from batch import openai_batch, anthropic_batch, gemini_batch

    if provider == "openai":
        from openai import OpenAI
        openai_batch.poll_batch(state["batch_ids"], client=OpenAI(), interval=interval)
    elif provider == "anthropic":
        import anthropic
        anthropic_batch.poll_batch(state["batch_ids"], client=anthropic.Anthropic(), interval=interval)
    elif provider == "gemini":
        gemini_batch.poll_batch(
            state["batch_names"],
            project=state.get("gcp_project"),
            location=state.get("gcp_location"),
            interval=interval,
        )


def _process(
    dataset_config: DatasetConfig,
    extraction_model: str,
    output_dir: Path,
    state_file: str,
    extraction_date: str | None = None,
) -> None:
    """Fetch results from a completed batch and write responses.json."""
    state = json.loads(Path(state_file).read_text())
    provider = state["provider"]
    frontier_model = state["model"]

    input_file = _find_extraction_final(dataset_config.name, extraction_model, extraction_date)
    with open(input_file) as f:
        data: list[dict] = json.load(f)

    from batch import common as batch_common
    from batch import openai_batch, anthropic_batch, gemini_batch

    if provider == "openai":
        from openai import OpenAI
        results = openai_batch.fetch_results(state["batch_ids"], client=OpenAI(), model=frontier_model)
    elif provider == "anthropic":
        import anthropic
        results = anthropic_batch.fetch_results(
            state["batch_ids"], client=anthropic.Anthropic(), model=frontier_model
        )
    elif provider == "gemini":
        results = gemini_batch.fetch_results(
            state["batch_names"], model=frontier_model,
            dest_gcs=state["dest_gcs"],
            project=state.get("gcp_project"),
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
        description="Run frontier batch judge (OpenAI / Anthropic / Gemini).",
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
    p.add_argument("--frontier-model", required=True, help="Provider model name (e.g. 'gpt-4o-mini').")
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd of extraction run.")
    p.add_argument("--judge-date", default=None, help="Date tag for output directory (default: today).")
    p.add_argument(
        "--ocr-dir", default=None, metavar="DIR",
        help="Directory of OCR .txt files. Defaults to {data_dir}/ocr_output_raw/.",
    )
    p.add_argument("--dest-gcs", default=None, help="GCS URI (Gemini only).")
    p.add_argument("--gcp-project", default=None, help="GCP project (Gemini only).")
    p.add_argument("--gcp-location", default=None, help="GCP region (Gemini only).")
    p.add_argument("--interval", type=int, default=60, help="Poll interval in seconds.")
    p.add_argument(
        "--state", default=None, metavar="FILE",
        help="Batch state JSON file (for poll/process sub-commands).",
    )
    sub = p.add_subparsers(dest="subcommand")
    sub.add_parser("submit", help="Submit batch and save state file, then exit.")
    sub.add_parser("poll", help="Poll an already-submitted batch until complete.")
    sub.add_parser("process", help="Fetch and write results from a completed batch.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    dataset_config = _load_dataset_config(args.dataset)
    input_file = _find_extraction_final(args.dataset, args.extraction_model, args.extraction_date)
    extraction_date_resolved = input_file.parent.name
    output_dir = get_judge_output_dir(
        args.dataset, args.extraction_model, extraction_date_resolved, args.judge, args.judge_date
    )
    state_file = args.state or str(output_dir / f".batch_state_{args.judge}.json")

    print(f"\nDataset          : {args.dataset}")
    print(f"Extraction model : {args.extraction_model}")
    print(f"Extraction date  : {extraction_date_resolved}")
    print(f"Judge            : {args.judge} / {args.frontier_model}")
    print(f"Output           : {output_dir}\n")

    if args.subcommand == "submit":
        _submit(
            dataset_config=dataset_config,
            extraction_model=args.extraction_model,
            provider=args.judge,
            frontier_model=args.frontier_model,
            output_dir=output_dir,
            state_file=state_file,
            extraction_date=extraction_date_resolved,
            ocr_dir=args.ocr_dir,
            dest_gcs=args.dest_gcs,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
        )
    elif args.subcommand == "poll":
        _poll(state_file=state_file, interval=args.interval)
    elif args.subcommand == "process":
        _process(
            dataset_config=dataset_config,
            extraction_model=args.extraction_model,
            output_dir=output_dir,
            state_file=state_file,
            extraction_date=extraction_date_resolved,
        )
    else:
        # Default: full submit → poll → process in one shot
        run_frontier_judge(
            dataset_config=dataset_config,
            extraction_model=args.extraction_model,
            provider=args.judge,
            frontier_model=args.frontier_model,
            output_dir=output_dir,
            extraction_date=extraction_date_resolved,
            ocr_dir=args.ocr_dir,
            dest_gcs=args.dest_gcs,
            gcp_project=args.gcp_project,
            gcp_location=args.gcp_location,
            interval=args.interval,
            state_file=state_file,
        )


if __name__ == "__main__":
    main()
