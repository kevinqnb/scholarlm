"""
Ablation 6 (direct triple extraction) runner for Anthropic Claude frontier models.

Anthropic's API does not support the OpenAI-style response_format JSON schema
parameter. Structured output is achieved via tool use: a single tool whose
input_schema matches the DirectExtractionList pydantic model is defined, and
tool_choice forces the model to call it. The tool input is already parsed JSON,
so no separate validation step is needed.

Writes results to:

    data/experiments/{dataset}/ablations/ablation6/{model}/{YYYY_mm_dd}/final.json

Requires ANTHROPIC_API_KEY environment variable or --api-key flag.
The dataset config must define direct_extraction_schema and direct_extraction_prompt.

Usage
-----
    python experiments/run_ablation_claude.py --dataset pond --model claude-haiku-4-5 \\
        --paper-subset physical_and_chemical_limnological prairie_wetland
    python experiments/run_ablation_claude.py --dataset nfix --model claude-haiku-4-5 \\
        --paper-subset R163 R164 R172
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

import anthropic
from pydantic import create_model

from scholarlm.measurementlm import NumpyEncoder
from scholarlm.instruction_prompts import DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS
from run_extraction import load_dataset_config, load_papers

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

CLAUDE_MODEL_REGISTRY: dict[str, str] = {
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20251001",
}

# ---------------------------------------------------------------------------
# Output path helper
# ---------------------------------------------------------------------------


def get_output_dir(dataset_name: str, model_name: str, date: str | None = None) -> Path:
    if date is None:
        date = datetime.now().strftime("%Y_%m_%d")
    return (
        _REPO_ROOT / "data" / "experiments" / dataset_name
        / "ablations" / "ablation6" / model_name / date
    )


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _extract_triples(
    client: anthropic.Anthropic,
    model_id: str,
    dataset_config,
    text: list[str],
) -> list[dict]:
    """
    Extract all measurement records from each document using Anthropic tool use.

    Structured output is enforced by defining a tool whose input_schema matches
    DirectExtractionList and setting tool_choice to force that tool. The model's
    response contains a tool_use block whose .input field is already a parsed dict.
    """
    DirectExtractionList = create_model(
        "DirectExtractionList",
        items=(list[dataset_config.direct_extraction_schema], ...),
    )
    schema = DirectExtractionList.model_json_schema()

    tool = {
        "name": "extract_records",
        "description": (
            "Extract all measurement records from the document. "
            "Return every (entity, measurement event, attribute, value, units) combination found."
        ),
        "input_schema": schema,
    }

    query = "Extract all measurement records from this document as described in the instructions."
    triple_data = []

    for i, context in enumerate(text):
        prompt = (
            f"## Instructions:\n{DIRECT_TRIPLE_EXTRACTION_INSTRUCTIONS}\n\n"
            f"## Dataset-specific extraction context:\n{dataset_config.direct_extraction_prompt}\n\n"
            f"## Context:\n{context}\n\n## Query:\n{query}"
        )

        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=8192,
                tools=[tool],
                tool_choice={"type": "tool", "name": "extract_records"},
                messages=[{"role": "user", "content": prompt}],
            )
            tool_block = next(
                (b for b in response.content if b.type == "tool_use"),
                None,
            )
            if tool_block is None:
                print(f"Warning: no tool_use block in response for document {i}.")
                validated = {"items": []}
            else:
                try:
                    validated = DirectExtractionList.model_validate(tool_block.input).model_dump()
                except Exception as e:
                    print(f"Validation error for document {i}: {e}")
                    validated = {"items": []}
        except Exception as e:
            print(f"API call failed for document {i}: {e}")
            validated = {"items": []}

        for j, item in enumerate(validated["items"]):
            if item.get("value") is None:
                continue
            triple_data.append(
                {"document_id": i, "context": context}
                | item
                | {"entity_id": f"doc_{i}_entity_{j}", "attribute_terms": []}
            )

    return triple_data


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run(
    dataset_config,
    model_id: str,
    model_name: str,
    output_dir: Path,
    ocr_dir: str | None = None,
    paper_subset_override: list[str] | None = None,
    api_key: str | None = None,
) -> None:
    if api_key is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError("Anthropic API key required. Set ANTHROPIC_API_KEY or pass --api-key.")

    if dataset_config.direct_extraction_schema is None or dataset_config.direct_extraction_prompt is None:
        raise ValueError(
            f"Dataset '{dataset_config.name}' does not define direct_extraction_schema or "
            "direct_extraction_prompt. Add them to the dataset config before running ablation 6."
        )

    data_dir = Path(dataset_config.data_dir)
    effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")

    print(f"\nDataset  : {dataset_config.name}")
    print(f"Model    : {model_name} ({model_id})")
    print(f"Ablation : 6 — Direct triple extraction")
    print(f"OCR dir  : {effective_ocr_dir}")
    print(f"Output   : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} papers.\n")

    client = anthropic.Anthropic(api_key=api_key)

    print("Running ablation 6 pipeline...")
    data = _extract_triples(client, model_id, dataset_config, text)

    dataset = [
        text_info[dp["document_id"]] | dp | {"measurement_id": i}
        for i, dp in enumerate(data)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "final.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run ablation 6 (direct triple extraction) using Anthropic Claude frontier models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (experiments/configs/<name>.py).")
    p.add_argument(
        "--model",
        required=True,
        choices=sorted(CLAUDE_MODEL_REGISTRY.keys()),
        help="Claude model key.",
    )
    p.add_argument("--date", default=None, help="Output date tag YYYY_mm_dd (default: today).")
    p.add_argument(
        "--ocr-dir",
        default=None,
        metavar="DIR",
        help="Directory of OCR .txt files. Defaults to {data_dir}/ocr_output_raw/.",
    )
    p.add_argument(
        "--paper-subset",
        nargs="+",
        default=None,
        metavar="PAPER_CODE",
        help="Override the dataset paper_subset with an explicit list of paper codes.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        metavar="KEY",
        help="Anthropic API key (default: ANTHROPIC_API_KEY env var).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dataset_config = load_dataset_config(args.dataset)
    model_id = CLAUDE_MODEL_REGISTRY[args.model]
    output_dir = get_output_dir(args.dataset, args.model, args.date)
    run(
        dataset_config=dataset_config,
        model_id=model_id,
        model_name=args.model,
        output_dir=output_dir,
        ocr_dir=args.ocr_dir,
        paper_subset_override=args.paper_subset,
        api_key=args.api_key,
    )


if __name__ == "__main__":
    main()
