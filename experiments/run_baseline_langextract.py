"""
LangExtract baseline runner.

Runs the langextract extraction baseline (https://github.com/google/langextract)
for any registered dataset, on any registered backbone model, writing to:

    experiments/results/{dataset}/baseline_langextract/{experiment_id}/final.json

(id-addressed, like every other Tier-1 type.)

Like ChatExtract, this is a *text-based* method: it reads the OCR'd
`<page>/<table>` tagged text (same input as MeasurementLM / run_extraction), so
it does NOT require `experiments/process_pdfs.py`. It wraps an ordinary chat
model served through a local vLLM OpenAI-compatible endpoint, so `model`
selects any entry under experiments/model-configs/extraction/, the same
backbones MeasurementLM runs on, for a fair same-model comparison (see
`measurementlm_langextract.py`).

langextract's chunking/recall/output-format knobs change what this baseline
actually measures, so they carry no runner-side default: every one of
`max_char_buffer`, `extraction_passes`, `max_workers`, `batch_length`,
`use_schema_constraints`, `fence_output` must be set explicitly in the
experiment config's `params`.

Usage
-----
    python experiments/run_baseline_langextract.py experiments/experiment-configs/pond/baseline_langextract/<id>/<id>.yaml

Required params: dataset, model, max_char_buffer, extraction_passes,
max_workers, batch_length, use_schema_constraints, fence_output.
Optional params: paper_subset (list), ocr_dir, api_base, api_key,
max_concurrent (default 32).

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
Available models: any file in experiments/model-configs/extraction/<name>.yaml.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm and run_extraction importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm.measurementlm import NumpyEncoder
from scholarlm.measurementlm_langextract import MeasurementLMLangExtract

from run_extraction import load_dataset_config, load_papers, get_model_config
import utils as paths
from utils import set_seeds, check_gpu_model_compatibility, write_run_metadata


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def run_baseline_langextract(
    dataset_config,
    model_config,
    output_dir: Path,
    *,
    max_char_buffer: int,
    extraction_passes: int,
    max_workers: int,
    batch_length: int,
    use_schema_constraints: bool,
    fence_output: bool,
    paper_subset_override: list[str] | None = None,
    ocr_dir: str | None = None,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
    max_concurrent: int = 32,
) -> None:
    """Run the langextract baseline for a dataset on a given backbone model.

    Writes a single `final.json` to `output_dir`, one record per langextract
    extraction (no `_standardize`/`_deduplicate` -- see
    `MeasurementLMLangExtract.fit`'s docstring for why: this baseline is meant
    to measure langextract's own method, not this repo's post-processing).
    Field names match the standard extraction record schema (same fields as
    MeasurementLM/ablation `final.json`), but unlike every deduplicated
    baseline, `page_number` here is a plain scalar, not a list.
    """
    data_dir = Path(dataset_config.data_dir)

    # langextract reads OCR text directly, same convention as ChatExtract: an
    # explicit --ocr-dir (e.g. a pre-cleaned ocr_output_cleaned_{model}
    # directory) takes precedence, otherwise raw OCR.
    effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"OCR dir   : {effective_ocr_dir}")
    print(f"Output    : {output_dir}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text_info)} papers.\n")

    mlm = MeasurementLMLangExtract(
        model_name=model_config.model_id,
        entity_identification_prompt=dataset_config.entity_identification_prompt,
        entity_identification_schema=dataset_config.entity_schema,
        attribute_info_dict=dataset_config.attribute_info_dict,
        direct_extraction_schema=dataset_config.direct_extraction_schema,
        direct_extraction_prompt=dataset_config.direct_extraction_prompt,
        nuextract_examples=dataset_config.nuextract_examples,
        measurement_event_schema=dataset_config.measurement_event_schema,
        sampling_params=model_config.sampling_params,
        api_base=api_base,
        api_key=api_key,
        max_concurrent=max_concurrent,
        max_char_buffer=max_char_buffer,
        extraction_passes=extraction_passes,
        max_workers=max_workers,
        batch_length=batch_length,
        use_schema_constraints=use_schema_constraints,
        fence_output=fence_output,
    )

    gpu_warnings = check_gpu_model_compatibility(model_config.model_id)

    print("Running LangExtract baseline...")
    start_time = time.time()
    data = mlm.fit(text)

    dataset = [
        info | dp | {"document_id": info["document_id"], "measurement_id": i}
        for i, dp in enumerate(data)
        for info in [text_info[dp["document_id"]]]
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "final.json"
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False, cls=NumpyEncoder)

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        model=model_config.name,
        model_id=model_config.model_id,
        hf_revision=model_config.hf_revision,
        baseline="langextract",
        max_char_buffer=max_char_buffer,
        extraction_passes=extraction_passes,
        max_workers=max_workers,
        batch_length=batch_length,
        use_schema_constraints=use_schema_constraints,
        fence_output=fence_output,
        gpu_compatibility_warnings=gpu_warnings,
        # lx.extract() manages its own HTTP calls, bypassing _acall/_call_batch
        # entirely (see module docstring) -- token_usage would be all-zero,
        # indistinguishable from "forgot to record", so it's marked n/a instead.
        token_accounting="n/a: langextract manages its own HTTP calls",
    )
    print(f"\nDone. Final dataset: {out_path}")
    print(f"       Records saved: {len(dataset)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the langextract extraction baseline.",
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
    paths.require_params(
        params,
        "dataset", "model", "max_char_buffer", "extraction_passes",
        "max_workers", "batch_length", "use_schema_constraints", "fence_output",
        config_path=config_path,
    )

    set_seeds(cfg["seed"])

    dataset_config = load_dataset_config(params["dataset"])
    model_config = get_model_config(params["model"])
    output_dir = paths.result_dir(params["dataset"], "baseline_langextract", cfg["id"])

    run_baseline_langextract(
        dataset_config=dataset_config,
        model_config=model_config,
        output_dir=output_dir,
        max_char_buffer=params["max_char_buffer"],
        extraction_passes=params["extraction_passes"],
        max_workers=params["max_workers"],
        batch_length=params["batch_length"],
        use_schema_constraints=params["use_schema_constraints"],
        fence_output=params["fence_output"],
        paper_subset_override=params.get("paper_subset"),
        ocr_dir=params.get("ocr_dir"),
        api_base=args.api_base or params.get("api_base") or "http://localhost:8081/v1",
        api_key=params.get("api_key", "EMPTY"),
        max_concurrent=params.get("max_concurrent", 32),
    )


if __name__ == "__main__":
    main()
