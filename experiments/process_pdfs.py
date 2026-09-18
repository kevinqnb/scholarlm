"""
PDF image pre-processing for table cleaning.

Renders each page of every dataset PDF as a base64-encoded PNG and saves the
result to disk.  These pre-processed images are consumed by the integrated
table-cleaning step inside ``run_extraction.py``.

The rendering libraries required for this step (Pillow, pypdf, pdfinfo) are
not available in the same environment as vLLM, so this script must be run in
the preprocessing environment before launching extraction.

Output
------
    data/{dataset}/processed_pdfs/{paper_code}/{page_index}.b64

Each ``.b64`` file contains a single base64-encoded PNG string for that page.

Usage
-----
    python experiments/process_pdfs.py experiments/experiment-configs/pond/process_pdfs/<id>/<id>.yaml

Required params: dataset.
Optional params: paper_subset (list), target_longest_dim (default 1536), resume (bool).

Available datasets: any file in experiments/dataset-configs/<name>.py that exports CONFIG.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = Path(__file__).parent / "dataset-configs"
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory, process_pdf

# utils.py's own dependencies (yaml, stdlib only) are as lightweight as this
# script's -- safe to import here despite this script's separate-environment
# note, which is about Pillow/pypdf vs. vLLM/torch, not this.
import utils


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_dataset_config(name: str) -> DatasetConfig:
    """Load a DatasetConfig by name from experiments/dataset-configs/<name>.py."""
    config_path = _CONFIGS_DIR / f"{name}.py"
    if not config_path.exists():
        available = sorted(p.stem for p in _CONFIGS_DIR.glob("*.py") if p.stem != "__init__")
        raise FileNotFoundError(
            f"No config found for dataset '{name}'. "
            f"Available datasets: {available}"
        )
    spec = importlib.util.spec_from_file_location(f"_dataset_config_{name}", config_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "CONFIG"):
        raise AttributeError(
            f"Config file {config_path} must define a module-level 'CONFIG' variable "
            f"of type DatasetConfig."
        )
    return mod.CONFIG


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_dataset_pdfs(
    dataset_config: DatasetConfig,
    paper_subset_override: list[str] | None = None,
    target_longest_dim: int = 1536,
    resume: bool = False,
) -> None:
    """Render all dataset PDFs to per-page base64 images and save to disk.

    Args:
        dataset_config: Dataset configuration.
        paper_subset_override: If provided, process only these paper codes.
            All PDFs are processed if ``None``.
        target_longest_dim: Maximum pixels on the longest edge when rendering.
        resume: If ``True``, skip papers whose output directory already exists.
    """
    data_dir = Path(dataset_config.data_dir)
    pdf_dir = data_dir / "pdfs"
    output_root = data_dir / "processed_pdfs"

    pdf_files = get_filenames_in_directory(str(pdf_dir), ignore=[".DS_Store", ".gitkeep"])
    pdf_files = sorted(f for f in pdf_files if f.endswith(".pdf"))

    if paper_subset_override is not None:
        subset = set(paper_subset_override)
        pdf_files = [f for f in pdf_files if f.replace(".pdf", "") in subset]

    if resume:
        before = len(pdf_files)
        pdf_files = [
            f for f in pdf_files
            if not (output_root / f.replace(".pdf", "")).exists()
        ]
        print(f"Resume: {before - len(pdf_files)} already done, {len(pdf_files)} remaining.")

    if not pdf_files:
        print("No files to process.")
        return

    print(f"\nDataset  : {dataset_config.name}")
    print(f"PDFs     : {pdf_dir}")
    print(f"Output   : {output_root}")
    print(f"Files    : {len(pdf_files)}\n")

    for i, fname in enumerate(pdf_files):
        paper_code = fname.replace(".pdf", "")
        pdf_path = str(pdf_dir / fname)
        out_dir = output_root / paper_code

        print(f"[{i + 1}/{len(pdf_files)}] {paper_code}...")
        try:
            pages = process_pdf(pdf_path, target_longest_dim=target_longest_dim)
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        for page_idx, page_b64 in enumerate(pages):
            (out_dir / f"{page_idx}.b64").write_text(page_b64)

        print(f"  {len(pages)} pages → {out_dir}")

    print(f"\nDone. Processed PDFs saved to {output_root}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pre-process PDF pages to base64 images for table cleaning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = utils.load_experiment_config(config_path)
    params = cfg["params"]
    utils.require_params(params, "dataset", config_path=config_path)

    dataset_config = load_dataset_config(params["dataset"])
    process_dataset_pdfs(
        dataset_config=dataset_config,
        paper_subset_override=params.get("paper_subset"),
        target_longest_dim=params.get("target_longest_dim", 1536),
        resume=params.get("resume", False),
    )


if __name__ == "__main__":
    main()
