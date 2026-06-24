"""
Ground truth preprocessing for the pond (aquatic ecosystem) dataset.

Pipeline
--------
    raw_data/pond_data_corrected.csv
        ↓  filter to registered papers, add document_id + units
        ↓  page attribution via OCR scoring
    ground_truth.json                  (all registered papers)
    ground_truth_ten.json              (top-10 paper development subset)

Note on entity assignment
--------------------------
``pond_data_corrected.csv`` stores measurements in attribute-first order (all
values for one attribute, then all values for the next).  Within each
``(title, name)`` group every attribute appears the same number of times — one
occurrence per entity.

Usage
-----
Run from the repo root:

    python data/pond/preprocessing.py

Or from data/pond/:

    python preprocessing.py
"""
from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path


import pandas as pd

from scholarlm.utils.page_attribution import POND_WEIGHTS, attribute_page, parse_ocr

logger = logging.getLogger(__name__)

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

_UNITS: dict[str, str | None] = {
    "max_depth":        "m",
    "surface_area":     "m^2",
    "vegetation_cover": "percent",
    "tn":               "µg/L",
    "tp":               "µg/L",
    "chla":             "µg/L",
    "ph":               None,
}

# Conversion factors from each attribute's standard unit (above) to a paper's
# original unit.  paper_value = standard_value * factor.
# Add entries here as new paper units are encountered.
_UNIT_CONVERSION: dict[str, float] = {
    # surface_area (standard: m^2)
    "m^2":           1.0,
    "ha":            1e-4,
    "km^2":          1e-6,
    "acres":         1.0 / 4046.856,
    "x10^-2 km^2":   1e-4,   # 10^-2 km^2 = 1 ha
    "x10^-6 m^2":    1e-6,   # column in paper labelled ×10^6 m^2 (= km^2)
    # max_depth (standard: m)
    "m":             1.0,
    "cm":            100.0,
    "ft":            3.28084,
    # tn / tp / chla (standard: µg/L)
    "µg/L":          1.0,
    "mg/L":          1e-3,
    "mg/m^3":        1.0,    # 1 µg/L == 1 mg/m^3
    "µg/cm^2":       1.0,    # area-based unit; no volume conversion, kept as-is
    # vegetation_cover (standard: percent)
    "percent":       1.0,
    "fraction":      0.01,
}

# Papers excluded from ground truth (data quality issues).
_EXCLUDED_FROM_GT: frozenset[str] = frozenset({
    "analysis_of_biological", # data not found in the paper...
    "bacterioplankton",   # values digitised from figures, not tables
    "summer_assessment",  # data only in supplemental text
})


def _add_page_attribution(gt: pd.DataFrame, ocr_dir: Path) -> pd.DataFrame:
    """Append page_number, page_score, and page_confidence columns to *gt*.

    ``page_number`` is a list of all candidate page numbers within the confidence
    margin (e.g. ``[3]`` or ``[3, 4]``).  Scores all pages for
    each document (no table pre-filtering for pond).  Rows whose OCR file is
    missing receive NaN attribution columns.

    Prints a summary: rows attributed, missing-OCR count, confidence distribution.
    """
    gt = gt.copy()
    gt["page_number"] = pd.NA
    gt["page_score"] = pd.NA
    gt["page_confidence"] = pd.NA

    n_attributed = 0
    n_missing_ocr = 0
    confidence_counts: Counter[str] = Counter()

    for doc_id, group in gt.groupby("document_id"):
        ocr_path = ocr_dir / f"{doc_id}.txt"
        if not ocr_path.exists():
            n_missing_ocr += 1
            logger.warning("OCR file not found: %s", ocr_path)
            continue

        parsed = parse_ocr(ocr_path)

        for idx, row in group.iterrows():
            result = attribute_page(row.to_dict(), parsed, POND_WEIGHTS)
            gt.at[idx, "page_number"] = result["candidates"]
            gt.at[idx, "page_score"] = result["score"]
            gt.at[idx, "page_confidence"] = result["confidence"]
            confidence_counts[result["confidence"]] += 1
            n_attributed += 1

    total = len(gt)
    print(f"  Page attribution: {n_attributed:,}/{total:,} rows attributed "
          f"({n_missing_ocr} docs with missing OCR)")
    print(f"  Confidence distribution: {dict(confidence_counts)}")
    return gt


def build_ground_truth(corrected_path: Path, out_dir: Path) -> None:
    """Build ground_truth.json and ground_truth_ten.json from pond_data_corrected.csv.

    Filters to registered papers, adds document_id, and applies per-document
    unit conversions using the ``"units"`` field in ``directory.json``.  The
    ``units`` column in the output reflects each paper's original units (as
    recorded in ``directory.json``), not the standard internal units.

    Output schema: document_id, name, identifiers, location, ecosystem, date,
    additional_details, attribute, value, units, page_number, page_score,
    page_confidence.

    Args:
        corrected_path: Path to ``pond_data_corrected.csv``.
        out_dir: Directory where the output files are written.
    """
    with open(BASE / "directory.json") as f:
        paper_info = json.load(f)

    title_to_id = {info["title"]: doc_id for doc_id, info in paper_info.items()}

    # (document_id, attribute) → paper's original unit
    paper_unit_lookup: dict[tuple[str, str], str | None] = {
        (doc_id, attr): unit
        for doc_id, info in paper_info.items()
        for attr, unit in info.get("units", {}).items()
    }

    df = pd.read_csv(corrected_path, index_col=0, encoding_errors="ignore")
    df = df[df["title"].isin(title_to_id)].copy()
    df = df[df["attribute"].isin(_UNITS)].copy()
    df = df.dropna(subset=["value"]).reset_index(drop=True)

    df["document_id"] = df["title"].map(title_to_id)
    df = df[~df["document_id"].isin(_EXCLUDED_FROM_GT)].reset_index(drop=True)

    df["identifiers"] = None
    df["date"] = None
    df["additional_details"] = None

    # Resolve each row's paper unit from directory.json, falling back to the
    # standard unit when the paper's units field is absent.
    df["units"] = [
        paper_unit_lookup.get((doc_id, attr), _UNITS.get(attr))
        for doc_id, attr in zip(df["document_id"], df["attribute"])
    ]

    # Apply conversion: paper_value = standard_value * factor
    factors = [
        _UNIT_CONVERSION.get(u, 1.0) if u is not None and u != _UNITS.get(a) else 1.0
        for u, a in zip(df["units"], df["attribute"])
    ]
    df["value"] = df["value"] * factors

    final_cols = [
        "document_id", "name", "identifiers", "location", "ecosystem",
        "date", "additional_details", "attribute", "value", "units",
    ]
    df_final = df[final_cols].reset_index(drop=True)

    df_final = _add_page_attribution(df_final, BASE / "ocr_output_raw")

    df_final.to_json(out_dir / "ground_truth.json", orient="records", indent=2)
    print(f"  Saved {len(df_final):,} rows → ground_truth.json")

    gt_ten = df_final[df_final["document_id"].isin(_TOP_PAPERS)].reset_index(drop=True)
    gt_ten.to_json(out_dir / "ground_truth_ten.json", orient="records", indent=2)
    print(f"  Saved {len(gt_ten):,} rows → ground_truth_ten.json")


def main(argv: list[str] | None = None) -> None:
    corrected_path = BASE / "raw_data" / "pond_data_corrected.csv"
    print("Building ground truth JSONs ...")
    build_ground_truth(corrected_path, BASE)


if __name__ == "__main__":
    main()
