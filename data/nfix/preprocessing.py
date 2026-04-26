"""
Ground truth preprocessing for the nfix (aquatic nitrogen fixation) dataset.

Pipeline
--------
    raw_data/aquatic_N2fix_rates.csv  (raw database export)
        ↓  filter to text/table-extractable papers
        ↓  reshape: one row per nfix_rate measurement
    ground_truth.csv                  (all registered text/table papers)
    ground_truth_ten.csv              (top-10 paper development subset)

Paper inclusion filter
----------------------
Only papers whose ``extraction_location`` in ``directory.json`` does NOT
contain any of: "figure", "supplement", "archive", "author".  These are
papers whose data can be extracted from running text or tables.

No unit conversion is applied — values are kept as ``nfix_rate_original``
(the units as reported in each paper).  The matching step in
``analysis/metrics.py`` therefore compares extracted values directly to the
original reported values.

Output columns
--------------
document_id, name, identifiers, location, site_type, date, nfix_method,
substrate_type, sample_depth, additional_details, attribute, value, units.

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


def _is_text_or_table(location: str) -> bool:
    """Return True if the paper's data can be extracted from text or tables."""
    return not any(x in location for x in ("figure", "supplement", "archive", "author"))


def _format_date(year, month, day) -> str | None:
    """Format year/month/day into a partial ISO date string.

    Returns YYYY-mm-dd if all three components are present, YYYY-mm if day is
    missing, YYYY if only year is known, and None if year is NaN.
    """
    if pd.isna(year):
        return None
    y = int(year)
    if pd.isna(month):
        return str(y)
    m = int(month)
    if pd.isna(day):
        return f"{y}-{m:02d}"
    return f"{y}-{m:02d}-{int(day):02d}"


def build_ground_truth(raw_path: Path, directory_path: Path, out_dir: Path) -> None:
    """Build ground_truth.csv and ground_truth_ten.csv from the raw nfix database.

    Filters to papers with text/table-extractable data, constructs the output
    columns, assigns attribute='nfix_rate' with value=nfix_rate_original and
    units=nfix_unit_original, and writes two output files.

    Output schema: document_id, name, identifiers, location, site_type, date,
    nfix_method, substrate_type, sample_depth, additional_details, attribute,
    value, units.

    Args:
        raw_path: Path to ``raw_data/aquatic_N2fix_rates.csv``.
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

    df = pd.read_csv(raw_path)
    df = df[df.reference_id.isin(registered_ids)].copy()

    df["_location"] = df.apply(
        lambda r: f"({r.latitude}, {r.longitude})"
        if not (pd.isna(r.latitude) or pd.isna(r.longitude))
        else None,
        axis=1,
    )
    df["_date"] = df.apply(lambda r: _format_date(r.year, r.month, r.day), axis=1)

    gt = pd.DataFrame({
        "document_id":        df["reference_id"],
        "name":               df["site_name"],
        "identifiers":        None,
        "location":           df["_location"],
        "site_type":          df["habitat"],
        "date":               df["_date"],
        "nfix_method":        df["nfix_method"],
        "substrate_type":     df["substrate"],
        "sample_depth":       df["sample_depth"],
        "additional_details": None,
        "attribute":          "nfix_rate",
        "value":              df["nfix_rate_original"],
        "units":              df["nfix_unit_original"],
    }).dropna(subset=["value"]).reset_index(drop=True)

    gt.to_csv(out_dir / "ground_truth.csv", index=False)
    print(f"  Saved {len(gt):,} rows → ground_truth.csv")

    gt_ten = gt[gt["document_id"].isin(_TOP_PAPERS)].reset_index(drop=True)
    gt_ten.to_csv(out_dir / "ground_truth_ten.csv", index=False)
    print(f"  Saved {len(gt_ten):,} rows → ground_truth_ten.csv")


def main() -> None:
    print("Building nfix ground truth CSVs ...")
    build_ground_truth(
        raw_path=BASE / "raw_data" / "aquatic_N2fix_rates.csv",
        directory_path=BASE / "directory.json",
        out_dir=BASE,
    )


if __name__ == "__main__":
    main()
