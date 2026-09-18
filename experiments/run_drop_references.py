"""
Retroactive reference/appendix-boundary re-processing for OCR or
table-cleaned OCR output that already exists under experiments/results/.

Applies scholarlm.utils.references.drop_references_section (and its
diagnostics twin, describe_drop_references) to every .txt file already
sitting under an existing ocr/ or table_cleaning/ result directory, writing
the reference-dropped text -- plus a per-paper diagnostics report -- to a
NEW experiment id under experiments/results/{dataset}/drop_references/{id}/.
The source directory is never modified.

Pure text post-processing: no model call, no vLLM server, no PDF re-render,
so the source pipeline is never recomputed. Composite/manual experiment-type
(see notes/hub/conventions.md) -- no submit.sh/SGE automation, since this
finishes in well under a minute over a few hundred files; run it directly.

Usage
-----
    python experiments/run_drop_references.py experiments/experiment-configs/pond/drop_references/<id>/<id>.yaml

Required params: dataset, source_type ("ocr" or "table_cleaning"), source_id
(the existing experiment id to re-process).
Optional params: paper_subset (list).
"""
from __future__ import annotations

import argparse
import json
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

from scholarlm.utils.references import _drop_references

import utils as paths
from utils import set_seeds, write_run_metadata

_VALID_SOURCE_TYPES = ("ocr", "table_cleaning")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_drop_references(
    dataset: str,
    source_type: str,
    source_id: str,
    output_dir: Path,
    paper_subset_override: list[str] | None = None,
) -> dict:
    """Apply drop_references_section to every .txt file in an existing
    ocr/ or table_cleaning/ result directory.

    Args:
        dataset: Dataset name -- cross-checked against source_id's own
            resolved path, not trusted blindly.
        source_type: "ocr" or "table_cleaning" -- the result-tree name
            source_id lives under; also cross-checked against the resolved path.
        source_id: The existing experiment id to re-process.
        output_dir: Destination directory for the reference-dropped .txt
            files and the diagnostics report (created if needed).
        paper_subset_override: If provided, process only these paper codes
            (filename stems without .txt).

    Returns:
        Summary dict (papers_processed, reference_heading_found,
        resume_heading_found, total_dropped_chars, total_rescued_chars).

    Raises:
        ValueError: If source_id's resolved (dataset, experiment_type)
            doesn't match the dataset/source_type params, or no .txt files
            are found to process.
    """
    source_dir = paths.find_result_dir(source_id)
    resolved_dataset, resolved_type = source_dir.parent.parent.name, source_dir.parent.name
    if resolved_dataset != dataset:
        raise ValueError(
            f"source_id={source_id!r} resolves to {source_dir}, whose dataset "
            f"({resolved_dataset!r}) does not match params.dataset={dataset!r}"
        )
    if resolved_type != source_type:
        raise ValueError(
            f"source_id={source_id!r} resolves to {source_dir}, whose experiment-type "
            f"({resolved_type!r}) does not match params.source_type={source_type!r}"
        )

    txt_files = sorted(source_dir.glob("*.txt"))
    if paper_subset_override is not None:
        subset = set(paper_subset_override)
        txt_files = [f for f in txt_files if f.stem in subset]
    if not txt_files:
        raise ValueError(f"No .txt files found under {source_dir} (after paper_subset filtering)")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nDataset     : {dataset}")
    print(f"Source      : {source_dir} ({source_type})")
    print(f"Output      : {output_dir}")
    print(f"Papers      : {len(txt_files)}\n")

    report = []
    n_ref_found = 0
    n_resumed = 0
    total_dropped_chars = 0
    total_rescued_chars = 0

    for f in txt_files:
        text = f.read_text()
        # One detection pass shared by the written text and the reported
        # diagnostics (via the private _drop_references) -- two separate
        # public calls would run detection twice and risk exactly the
        # text-vs-diagnostics drift that split was written to prevent.
        dropped_text, info = _drop_references(text)

        (output_dir / f.name).write_text(dropped_text)
        report.append({"paper": f.stem, **info})

        n_ref_found += int(info["reference_heading_found"])
        n_resumed += int(info["resume_heading_found"])
        total_dropped_chars += info["dropped_chars"]
        total_rescued_chars += info["rescued_chars"]

    report_path = output_dir / "drop_references_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)

    summary = {
        "papers_processed": len(txt_files),
        "reference_heading_found": n_ref_found,
        "resume_heading_found": n_resumed,
        "total_dropped_chars": total_dropped_chars,
        "total_rescued_chars": total_rescued_chars,
    }
    print(f"Done. Reference-dropped output written to {output_dir}")
    print(f"Per-paper report: {report_path}")
    print(summary)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Retroactively drop references (with appendix/figures/tables/"
        "supplementary resume-splicing) from an existing ocr/ or table_cleaning/ result.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", "source_type", "source_id", config_path=config_path)

    repo_seed = paths.load_config()["defaults"]["seed"]
    if cfg["seed"] != repo_seed:
        raise ValueError(
            f"{config_path}: seed ({cfg['seed']}) does not match experiments/config.yaml "
            f"defaults.seed ({repo_seed}) -- the repo's seed is a fixed, repo-wide value, "
            "not a per-run knob."
        )
    set_seeds(cfg["seed"])

    dataset = params["dataset"]
    source_type = params["source_type"]
    source_id = params["source_id"]
    if source_type not in _VALID_SOURCE_TYPES:
        raise ValueError(
            f"{config_path}: params.source_type must be one of {_VALID_SOURCE_TYPES}, "
            f"got {source_type!r}"
        )

    output_dir = paths.result_dir(dataset, "drop_references", cfg["id"])

    start_time = time.time()
    summary = run_drop_references(
        dataset=dataset,
        source_type=source_type,
        source_id=source_id,
        output_dir=output_dir,
        paper_subset_override=params.get("paper_subset"),
    )

    write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset,
        source_type=source_type,
        source_id=source_id,
        **summary,
    )


if __name__ == "__main__":
    main()
