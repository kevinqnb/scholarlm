"""
Ground truth preprocessing for the nfix (aquatic nitrogen fixation) dataset.

Pipeline
--------
    meta/aquatic_N2fix_rates.csv  (raw database export)
        ↓  filter to text/table-extractable papers
        ↓  reshape to long format (one row per nfix_rate measurement)
    ground_truth.csv              (all registered text/table papers)
    ground_truth_ten.csv          (top-10 paper development subset)

Paper inclusion filter
----------------------
Only papers whose ``extraction_location`` in ``directory.json`` does NOT
contain any of: "figure", "supplement", "archive", "author".  These are
papers whose data can be extracted from running text or tables.

No unit conversion is applied — values are kept as ``nfix_rate_original``
(the units as reported in each paper).  The matching step in
``analysis/metrics.py`` therefore compares extracted values directly to the
original reported values.

Usage
-----
Run from the repo root:

    python data/nfix/preprocessing.py

Or from data/nfix/:

    python preprocessing.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent  # data/nfix/

_TOP_PAPERS = [
    "R163", "R164", "R172", "R248", "R124",
    "R51", "R59", "R114", "R43", "R103",
]

_ID_COLS = [
    "nfix_rate_id", "reference_id", "site_name", "latitude", "longitude",
    "habitat", "year", "month", "day", "hour_minute", "season",
    "substrate", "substrate_details",
]


def _is_text_or_table(location: str) -> bool:
    """Return True if the paper's data can be extracted from text or tables."""
    return not any(x in location for x in ("figure", "supplement", "archive", "author"))


def build_ground_truth(raw_path: Path, directory_path: Path, out_dir: Path) -> None:
    """Build ground_truth.csv and ground_truth_ten.csv from the raw nfix database.

    Filters to papers with text/table extractable data, reshapes to long format
    (attribute=nfix_rate, value=nfix_rate_original), drops rows with missing
    values, and writes two output files.

    Args:
        raw_path: Path to ``meta/aquatic_N2fix_rates.csv``.
        directory_path: Path to ``directory.json``.
        out_dir: Directory where the output files are written.
    """
    with open(directory_path) as f:
        paper_info = json.load(f)

    registered_ids = [
        ref_id
        for ref_id, info in paper_info.items()
        if _is_text_or_table(info.get("extraction_location", ""))
    ]

    nfix_df = pd.read_csv(raw_path)
    df = nfix_df[nfix_df.reference_id.isin(registered_ids)].copy()

    gt = df[_ID_COLS].assign(
        attribute="nfix_rate",
        value=df["nfix_rate_original"],
        error=df["nfix_error_original"],
        error_type=df["nfix_error_type"],
        units=df["nfix_unit_original"],
    ).dropna(subset=["value"]).reset_index(drop=True)

    gt.to_csv(out_dir / "ground_truth.csv", index=False)
    print(f"  Saved {len(gt):,} rows → ground_truth.csv")

    gt_ten = gt.loc[gt.reference_id.isin(_TOP_PAPERS)].reset_index(drop=True)
    gt_ten.to_csv(out_dir / "ground_truth_ten.csv", index=False)
    print(f"  Saved {len(gt_ten):,} rows → ground_truth_ten.csv")


def main() -> None:
    print("Building nfix ground truth CSVs ...")
    build_ground_truth(
        raw_path=BASE / "meta" / "aquatic_N2fix_rates.csv",
        directory_path=BASE / "directory.json",
        out_dir=BASE,
    )


if __name__ == "__main__":
    main()
