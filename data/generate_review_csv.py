"""
Generate a prioritized page-review CSV for manual verification.

Reads ground_truth.json for the specified dataset and writes page_review.csv
sorted by review priority:
  1. ambiguous confidence (multi-page first, then single)
  2. medium confidence
  3. high confidence, multi-page
  4. high confidence, single-page
  5. table-anchored (highest confidence, least likely to need review)

Correction columns (fill in as needed, then run apply_review.py):
  corrected_page    – integer or comma-separated list (e.g. "3" or "3, 4")
  corrected_name    – corrected entity name string
  corrected_value   – corrected numeric value
  corrected_units   – corrected units string
  excluded          – True to drop this record from ground_truth_review.json
  excluded_reason   – brief note explaining the exclusion

If page_review.csv already exists, re-running this script preserves all
filled-in correction columns (matched by gt_row_index).  New rows get blank
corrections; rows whose gt_row_index no longer appears in ground_truth.json
are dropped.

After filling in corrections, run:

    python data/apply_review.py --dataset <name>

Usage
-----
    python data/generate_review_csv.py --dataset pond
    python data/generate_review_csv.py --dataset nfix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent

_CONFIDENCE_RANK: dict[str, int] = {
    "ambiguous":      0,
    "medium":         1,
    "high":           2,
    "table-anchored": 3,
}

_CORRECTION_COLS = [
    "corrected_page",
    "corrected_name",
    "corrected_value",
    "corrected_units",
    "excluded",
    "excluded_reason",
]


def generate_review_csv(dataset: str) -> None:
    gt_path = DATA_DIR / dataset / "ground_truth.json"
    out_path = DATA_DIR / dataset / "page_review.csv"

    with open(gt_path) as f:
        records = json.load(f)

    # Load existing corrections indexed by gt_row_index (preserves user edits).
    existing_corrections: dict[int, dict[str, str]] = {}
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        for _, row in existing.iterrows():
            try:
                idx = int(float(row["gt_row_index"]))
            except (ValueError, KeyError):
                continue
            existing_corrections[idx] = {col: row.get(col, "") for col in _CORRECTION_COLS}

    rows = []
    for i, rec in enumerate(records):
        candidates = rec.get("page_number")
        confidence = rec.get("page_confidence") or "ambiguous"
        n_candidates = len(candidates) if isinstance(candidates, list) else 0
        page_str = ", ".join(str(p) for p in candidates) if isinstance(candidates, list) else ""
        conf_rank = _CONFIDENCE_RANK.get(confidence, 0)
        # Sort key: confidence first, then single-page after multi-page within same confidence
        priority = conf_rank * 10 + (1 if n_candidates <= 1 else 0)
        corrections = existing_corrections.get(i, {})
        rows.append({
            "_priority":       priority,
            "document_id":     rec.get("document_id"),
            "gt_row_index":    i,
            "name":            rec.get("name"),
            "attribute":       rec.get("attribute"),
            "value":           rec.get("value"),
            "units":           rec.get("units"),
            "page_number":     page_str,
            "page_confidence": confidence,
            **{col: corrections.get(col, "") for col in _CORRECTION_COLS},
        })

    df = pd.DataFrame(rows)
    # Rank each paper by its most urgent row, so all rows from the same paper
    # are grouped together and appear as early as the paper's best case warrants.
    paper_min = df.groupby("document_id")["_priority"].transform("min")
    df = (
        df.assign(_paper_priority=paper_min)
        .sort_values(["_paper_priority", "document_id", "_priority"])
        .drop(columns=["_priority", "_paper_priority"])
        .reset_index(drop=True)
    )
    df.to_csv(out_path, index=False)

    conf_counts = df["page_confidence"].value_counts().to_dict()
    multi_page = (df["page_number"].str.contains(",", na=False)).sum()
    n_carried = sum(
        1 for v in existing_corrections.values()
        if any(v.get(c, "") for c in _CORRECTION_COLS)
    )
    print(f"Wrote {len(df):,} rows → {out_path.relative_to(Path.cwd())}")
    print(f"Confidence distribution: {conf_counts}")
    print(f"Multi-candidate rows: {multi_page:,}")
    if existing_corrections:
        print(f"Carried over corrections from {n_carried} previously annotated rows.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. pond, nfix)")
    args = parser.parse_args()
    generate_review_csv(args.dataset)


if __name__ == "__main__":
    main()
