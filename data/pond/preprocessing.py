"""
Ground truth preprocessing for the pond (aquatic ecosystem) dataset.

Pipeline
--------
    pond_data.csv          (raw database export)
        ↓  step 1: reshape to long format, filter to registered papers
    pond_data_cleaned.csv  (reproducible; committed alongside raw data)
        ↓  manual corrections applied offline (not reproduced here)
    pond_data_corrected.csv  (source of truth for the ground truth)
        ↓  step 2: filter to registered/subset papers, strip index column
    ground_truth.csv         (all registered papers)
    ground_truth_ten.csv     (top-10 paper development subset)

Note on manual corrections
---------------------------
``pond_data_corrected.csv`` was produced by manually inspecting
``pond_data_cleaned.csv`` and fixing measurement errors.  The correction
step cannot be reproduced automatically; if new corrections are needed,
edit ``pond_data_corrected.csv`` directly and re-run step 2.

To reproduce ``pond_data_cleaned.csv`` (step 1) without corrections, pass
``use_corrected=False`` to ``build_ground_truth``, or run:

    python data/pond/preprocessing.py --uncorrected

Usage
-----
Run from the repo root:

    python data/pond/preprocessing.py

Or from data/pond/:

    python preprocessing.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent  # data/pond/

_TOP_PAPERS = [
    "classification_trees",
    "physical-chemical_influences",
    "habitat_characteristics",
    "physical_and_chemical_limnological",
    "prairie_wetland",
    "macroinvertebrate_size",
    "relationships_between_fish",
    "net_heterotrophy",
    "impact_of_macrophytes",
    "environmental_conditions",
]


def build_cleaned(raw_path: Path, out_path: Path) -> pd.DataFrame:
    """Reshape raw pond_data.csv into long format and save as pond_data_cleaned.csv.

    Selects the relevant columns, renames them to the canonical schema
    (author, title, name, location, ecosystem, attribute, value), melts
    to one row per (entity, attribute) measurement, and filters to papers
    present in directory.json.

    Args:
        raw_path: Path to the raw ``pond_data.csv``.
        out_path: Destination path for the cleaned CSV.

    Returns:
        The cleaned DataFrame.
    """
    with open(BASE / "directory.json") as f:
        paper_info = json.load(f)
    registered_titles = [entry["title"] for entry in paper_info.values()]

    df = pd.read_csv(raw_path, encoding_errors="ignore")
    df = df.loc[df.title.isin(registered_titles)].reset_index(drop=True)

    df = df.loc[
        :,
        [
            "author", "title", "pondname", "location", "author_term",
            "max_depth_m", "mean_surfacearea_m2", "macrophytes_percentcover",
            "ph", "tn_ugpl", "tp_ugpl", "chla_ugpl",
        ],
    ]
    df.columns = [
        "author", "title", "name", "location", "ecosystem",
        "max_depth", "surface_area", "vegetation_cover", "ph", "tn", "tp", "chla",
    ]

    df = df.melt(
        id_vars=["author", "title", "name", "location", "ecosystem"],
        value_vars=["max_depth", "surface_area", "vegetation_cover", "ph", "tn", "tp", "chla"],
        var_name="attribute",
        value_name="value",
    )
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    df["date"] = None
    df["state"] = None
    df = df[["author", "title", "name", "location", "ecosystem", "date", "state", "attribute", "value"]]

    df.to_csv(out_path, index=False)
    print(f"  Saved {len(df):,} rows → {out_path.name}")
    return df


def build_ground_truth(corrected_path: Path, out_dir: Path) -> None:
    """Filter pond_data_corrected.csv to registered papers and save ground truth CSVs.

    Produces two files:
    - ``ground_truth.csv``     — all papers present in directory.json
    - ``ground_truth_ten.csv`` — the top-10 paper development subset only

    Values are already in standard units in the corrected file (max_depth in m,
    surface_area in m², vegetation_cover in %, tn/tp/chla in µg/L, ph dimensionless).

    Args:
        corrected_path: Path to ``pond_data_corrected.csv``.
        out_dir: Directory where the output files are written.
    """
    with open(BASE / "directory.json") as f:
        paper_info = json.load(f)

    all_registered_titles = [entry["title"] for entry in paper_info.values()]
    ten_titles = [
        paper_info[code]["title"] for code in _TOP_PAPERS if code in paper_info
    ]

    # index_col=0 because pond_data_corrected.csv has an unnamed integer index column
    df = pd.read_csv(corrected_path, encoding_errors="ignore", index_col=0)

    gt_full = df.loc[df.title.isin(all_registered_titles)].reset_index(drop=True)
    gt_full.to_csv(out_dir / "ground_truth.csv", index=False)
    print(f"  Saved {len(gt_full):,} rows → ground_truth.csv")

    gt_ten = df.loc[df.title.isin(ten_titles)].reset_index(drop=True)
    gt_ten.to_csv(out_dir / "ground_truth_ten.csv", index=False)
    print(f"  Saved {len(gt_ten):,} rows → ground_truth_ten.csv")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--uncorrected",
        action="store_true",
        help="Build ground truth from pond_data_cleaned.csv instead of pond_data_corrected.csv.",
    )
    parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Skip step 1 (rebuilding pond_data_cleaned.csv) and go straight to step 2.",
    )
    args = parser.parse_args(argv)

    raw_path = BASE / "pond_data.csv"
    cleaned_path = BASE / "pond_data_cleaned.csv"
    corrected_path = BASE / "pond_data_corrected.csv"

    if not args.skip_clean:
        print("Step 1: Rebuilding pond_data_cleaned.csv from pond_data.csv ...")
        build_cleaned(raw_path, cleaned_path)
    else:
        print("Step 1: Skipped (--skip-clean).")

    source = cleaned_path if args.uncorrected else corrected_path
    if args.uncorrected:
        print(f"\nStep 2: Building ground truth from uncorrected data ({source.name}) ...")
    else:
        print(f"\nStep 2: Building ground truth from manually corrected data ({source.name}) ...")
    build_ground_truth(source, BASE)


if __name__ == "__main__":
    main()
