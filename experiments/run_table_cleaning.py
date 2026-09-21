"""
Table-cleaning script using a vLLM server (OpenAI-compatible API).

Cleans and normalizes tables in OCR-processed text files using a local
open-source model served by vLLM.  For each page that contains ``<table>``
tags the model is shown the pre-rendered page image and asked to correct
the table markup.  Pages without tables are returned unchanged.

The vLLM server must be started separately before running this script
(default endpoint: ``http://localhost:8081/v1``).

Results are written to:

    experiments/results/{dataset}/table_cleaning/{experiment_id}/

(tied to an experiment id like every other Tier-1 type -- unlike the old
data/{dataset}/ocr_output_cleaned_{model_name}/ convention, which stays
exactly as-is for datasets that already have it, but is no longer where new
runs default to). Pass this directory to run_extraction.py's params.ocr_dir
to use it for extraction.

Prerequisites
-------------
1. Run ``process_pdfs.py`` first (preprocessing environment) to produce
   pre-rendered page images at ``data/{dataset}/processed_pdfs/``.

2. Start a vLLM server serving the chosen model and wait for startup to complete
   (experiments/submit.sh does this for you automatically).

Usage
-----
    python experiments/run_table_cleaning.py experiments/experiment-configs/pond/table_cleaning/<id>/<id>.yaml

Required params: dataset, model.
Optional params: ocr_dir (default: data/{dataset}/ocr_output_raw/), output_dir
(default: experiments/results/{dataset}/table_cleaning/{id}/ -- override only
to write somewhere else, e.g. the legacy data/{dataset}/ocr_output_cleaned_{model}/
convention), paper_subset (list), resume (bool), api_base, api_key.

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
Available models:   any file in experiments/model-configs/extraction/<name>.yaml.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm importable when run directly from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

# Import shared helpers from run_extraction to keep config loading in sync.
from run_extraction import load_dataset_config, get_model_config, load_papers
from scholarlm import TableCleaner
from scholarlm.config import DatasetConfig, ModelConfig
import utils


# ---------------------------------------------------------------------------
# Main cleaning function
# ---------------------------------------------------------------------------


def run_vllm_table_cleaning(
    dataset_config: DatasetConfig,
    model_config: ModelConfig,
    ocr_dir: str | None = None,
    output_dir: str | None = None,
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
) -> None:
    """Clean tables in OCR text files using a vLLM-served model.

    Loads raw OCR text, passes each page with tables through the model
    (alongside its pre-rendered page image), and writes cleaned texts to
    ``output_dir``.

    Args:
        dataset_config: Dataset configuration loaded from ``experiments/dataset-configs/``.
        model_config: Model configuration from experiments/model-configs/extraction/.
        ocr_dir: Input directory of ``.txt`` OCR files.  Defaults to
            ``{data_dir}/ocr_output_raw/``.
        output_dir: Destination directory for cleaned ``.txt`` files.  Defaults
            to ``{data_dir}/ocr_output_cleaned_{model_name}/``.
        paper_subset_override: If provided, process only these paper codes.
        resume: If ``True``, skip papers whose output ``.txt`` already exists.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the vLLM server (any non-empty string works).
    """
    data_dir = Path(dataset_config.data_dir)
    effective_ocr_dir = ocr_dir or str(data_dir / "ocr_output_raw")
    effective_output_dir = Path(output_dir) if output_dir else data_dir / f"ocr_output_cleaned_{model_config.name}"

    print(f"\nDataset   : {dataset_config.name}")
    print(f"Model     : {model_config.name} ({model_config.model_id})")
    print(f"OCR input : {effective_ocr_dir}")
    print(f"Output    : {effective_output_dir}")
    print(f"API base  : {api_base}\n")

    text, text_info = load_papers(dataset_config, effective_ocr_dir, paper_subset_override)
    print(f"Loaded {len(text)} papers.")

    if resume:
        pending_text = []
        pending_info = []
        for t, info in zip(text, text_info):
            out_file = effective_output_dir / f"{info['document_id']}.txt"
            if out_file.exists():
                print(f"  Skipping {info['document_id']} (already cleaned).")
            else:
                pending_text.append(t)
                pending_info.append(info)
        skipped = len(text) - len(pending_text)
        print(f"Resume: {skipped} already done, {len(pending_text)} remaining.\n")
        text = pending_text
        text_info = pending_info

    if not text:
        print("Nothing to clean.")
        return

    processed_pdf_root = data_dir / "processed_pdfs"
    if not processed_pdf_root.exists():
        raise FileNotFoundError(
            f"Processed PDF directory not found: {processed_pdf_root}\n"
            f"Run 'python experiments/process_pdfs.py --dataset {dataset_config.name}' first."
        )
    processed_pdf_dirs = [str(processed_pdf_root / info["document_id"]) for info in text_info]

    cleaner = TableCleaner(
        model_name=model_config.model_id,
        sampling_params=model_config.sampling_params,
        api_base=api_base,
        api_key=api_key,
        output_dir=str(effective_output_dir),
    )

    print(f"Cleaning tables for {len(text)} paper(s)...")
    cleaner.clean(text, processed_pdf_dirs)

    print(f"\nDone. Cleaned texts written to {effective_output_dir}")
    print(f"To use these cleaned texts for extraction, pass:")
    print(f"  --ocr-dir {effective_output_dir}")
    print(f"to run_extraction.py.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Clean tables in OCR text files using a vLLM-served model.",
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
    cfg = utils.load_experiment_config(config_path)
    params = cfg["params"]
    utils.require_params(params, "dataset", "model", config_path=config_path)

    utils.set_seeds(cfg["seed"])

    dataset_config = load_dataset_config(params["dataset"])
    model_config = get_model_config(params["model"])
    output_dir = params.get("output_dir") or str(
        utils.result_dir(params["dataset"], "table_cleaning", cfg["id"])
    )
    run_vllm_table_cleaning(
        dataset_config=dataset_config,
        model_config=model_config,
        ocr_dir=params.get("ocr_dir"),
        output_dir=output_dir,
        paper_subset_override=params.get("paper_subset"),
        resume=params.get("resume", False),
        api_base=args.api_base or params.get("api_base") or "http://localhost:8081/v1",
        api_key=params.get("api_key", "EMPTY"),
    )


if __name__ == "__main__":
    main()
