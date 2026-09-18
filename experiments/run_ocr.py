"""
OCR pipeline runner.

Runs an OCR model (default: olmOCR) on all PDF files for a dataset by calling a
running vLLM server (OpenAI-compatible API), writing plain-text output to:

    experiments/results/{dataset}/ocr/{experiment_id}/

(id-addressed, like every other Tier-1 type -- not the old
data/{dataset}/ocr_output_raw/-style convention this runner used to default
to; see the run_table_cleaning.py commit for the same tradeoff, confirmed
with the user. params.output_dir remains available to explicitly write to
the legacy conventional location if that's what you want.)

Start the OCR model server first (default endpoint: ``http://localhost:8081/v1``,
or let experiments/submit.sh bring it up automatically).

Usage
-----
    python experiments/run_ocr.py experiments/experiment-configs/pond/ocr/<id>/<id>.yaml

Required params: dataset.
Optional params: model (default: olmocr; also: chandra-ocr-2 -- any file in
experiments/model-configs/ocr/), paper_subset (list), resume (bool),
use_processed_pdfs (bool), processed_pdfs_dir, fast (bool), output_dir
(default: experiments/results/{dataset}/ocr/{id}/), api_base, api_key.

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
Available models:   any file in experiments/model-configs/ocr/<name>.yaml.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — make scholarlm and utils importable when run from the repo root
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

from scholarlm import DocumentLM
from scholarlm.utils import get_filenames_in_directory
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

from run_extraction import load_dataset_config
import utils as paths
from utils import set_seeds, write_run_metadata


# ---------------------------------------------------------------------------
# Prompt sources
#
# experiments/model-configs/ocr/{model}.yaml's `prompt_source:` field is a
# marker string, not a baked prompt: olmocr's real prompt comes from a
# library function call (a fixed "no anchoring" template, called with no
# per-document args), not a static constant -- baking its output into YAML
# would drift from the olmocr package and duplicate rather than reference
# the real source. `prompt_source: null` (chandra-ocr-2) means "use
# DocumentLM's own default markdown-with-html-tables prompt" -- chandra's own
# repo recommends an HTML layout-block prompt instead; swap this in if the
# default prompt's output proves degenerate.
# ---------------------------------------------------------------------------

_PROMPT_SOURCES = {
    "olmocr_no_anchoring_v4": olmocr_prompt,
}


def _resolve_prompt(prompt_source: str | None) -> str | None:
    if prompt_source is None:
        return None
    if prompt_source not in _PROMPT_SOURCES:
        raise ValueError(
            f"Unknown prompt_source {prompt_source!r}. Known: {sorted(_PROMPT_SOURCES)}"
        )
    return _PROMPT_SOURCES[prompt_source]()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_ocr(
    dataset_config,
    model_id: str,
    sampling_params: dict,
    output_dir: Path,
    ocr_prompt: str | None = None,
    api_base: str = "http://localhost:8081/v1",
    api_key: str = "EMPTY",
    paper_subset_override: list[str] | None = None,
    resume: bool = False,
    processed_pdfs_dir: str | None = None,
    fast: bool = False,
) -> None:
    """Run an OCR model on all PDFs for a dataset via a vLLM server.

    Args:
        dataset_config: Dataset configuration loaded from experiments/dataset-configs/.
        model_id: HuggingFace model ID string (must match what the server is serving).
        sampling_params: Sampling parameters forwarded to DocumentLM.
        output_dir: Directory to write per-paper ``.txt`` OCR output to (created if needed).
        ocr_prompt: System prompt for the OCR task. ``None`` uses DocumentLM's
            default markdown prompt.
        api_base: Base URL of the vLLM OpenAI-compatible server.
        api_key: API key for the server (use "EMPTY" for local vLLM).
        paper_subset_override: If provided, process only these paper codes
            (filename stems without .pdf).
        resume: If True, skip PDFs whose output .txt file already exists.
        processed_pdfs_dir: If provided, load pre-rendered page images from this
            directory instead of rendering PDFs at runtime.  The expected layout
            is ``{processed_pdfs_dir}/{paper_code}/{page_index}.b64``, which
            matches the output of ``experiments/process_pdfs.py``.
        fast: If True, run DocumentLM in fast mode (lower-resolution page
            images, no retry loop) trading OCR quality for speed.
    """
    data_dir = Path(dataset_config.data_dir)
    pdf_dir = data_dir / "pdfs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = get_filenames_in_directory(str(pdf_dir), ignore=[".DS_Store", ".gitkeep"])
    pdf_files = [f for f in pdf_files if f.endswith(".pdf")]
    pdf_files.sort()

    if paper_subset_override is not None:
        subset_set = set(paper_subset_override)
        pdf_files = [f for f in pdf_files if f.replace(".pdf", "") in subset_set]

    if resume:
        before = len(pdf_files)
        pdf_files = [
            f for f in pdf_files
            if not (output_dir / f.replace(".pdf", ".txt")).exists()
        ]
        print(f"Resume: {before - len(pdf_files)} already done, {len(pdf_files)} remaining.")

    if not pdf_files:
        print("No PDFs to process.")
        return

    print(f"\nDataset : {dataset_config.name}")
    print(f"Model   : {model_id}")
    print(f"Server  : {api_base}")
    print(f"Input   : {processed_pdfs_dir if processed_pdfs_dir else pdf_dir}")
    print(f"Output  : {output_dir}")
    print(f"Papers  : {len(pdf_files)}\n")

    filepaths = [str(pdf_dir / f) for f in pdf_files]
    out_filepaths = [str(output_dir / f.replace(".pdf", ".txt")) for f in pdf_files]

    doclm = DocumentLM(
        model_name=model_id,
        ocr_prompt=ocr_prompt,
        sampling_params=sampling_params,
        api_base=api_base,
        api_key=api_key,
        fast=fast,
    )

    start_time = time.time()
    doclm.fit(filepaths, processed_pdfs_dir=processed_pdfs_dir)
    doclm.save(out_filepaths)
    print(f"\nDone. OCR output written to {output_dir}")

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset_config.name,
        model_id=model_id,
        api_base=api_base,
        papers_processed=len(pdf_files),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run an OCR model on all PDFs for a dataset via a vLLM server.",
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
    paths.require_params(params, "dataset", config_path=config_path)

    repo_seed = paths.load_config()["defaults"]["seed"]
    if cfg["seed"] != repo_seed:
        raise ValueError(
            f"{config_path}: seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed}) -- the repo's seed is a fixed, repo-wide value, "
            "not a per-run knob."
        )
    set_seeds(cfg["seed"])

    model_name = params.get("model", "olmocr")
    model_cfg = paths.load_model_config("ocr", model_name)
    model_id = model_cfg["model_id"]
    sampling_params = model_cfg.get("sampling_params", {})
    ocr_prompt = _resolve_prompt(model_cfg.get("prompt_source"))

    dataset = params["dataset"]
    dataset_config = load_dataset_config(dataset)
    output_dir = Path(params.get("output_dir") or paths.result_dir(dataset, "ocr", cfg["id"]))

    processed_pdfs_dir = params.get("processed_pdfs_dir")
    if not processed_pdfs_dir and params.get("use_processed_pdfs"):
        processed_pdfs_dir = str(Path(dataset_config.data_dir) / "processed_pdfs")

    run_ocr(
        dataset_config=dataset_config,
        model_id=model_id,
        sampling_params=sampling_params,
        output_dir=output_dir,
        ocr_prompt=ocr_prompt,
        api_base=args.api_base or params.get("api_base") or "http://localhost:8081/v1",
        api_key=params.get("api_key", "EMPTY"),
        paper_subset_override=params.get("paper_subset"),
        resume=params.get("resume", False),
        processed_pdfs_dir=processed_pdfs_dir,
        fast=params.get("fast", False),
    )


if __name__ == "__main__":
    main()
