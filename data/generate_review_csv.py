"""
Generate a prioritized page-review CSV for manual page-number verification.

Reads ground_truth.json for the specified dataset and writes page_review.csv
sorted by review priority:
  1. ambiguous confidence (multi-page first, then single)
  2. medium confidence
  3. high confidence, multi-page
  4. high confidence, single-page
  5. table-anchored (highest confidence, least likely to need review)

Fill in the ``corrected_page`` column (a single integer) for any rows that
need correction, then run:

    python data/apply_page_corrections.py --dataset <name>

to patch ground_truth.json.  Rows left blank are untouched.

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


def generate_review_csv(dataset: str) -> None:
    gt_path = DATA_DIR / dataset / "ground_truth.json"
    out_path = DATA_DIR / dataset / "page_review.csv"

    with open(gt_path) as f:
        records = json.load(f)

    rows = []
    for i, rec in enumerate(records):
        candidates = rec.get("page_number")
        confidence = rec.get("page_confidence") or "ambiguous"
        n_candidates = len(candidates) if isinstance(candidates, list) else 0
        page_str = ", ".join(str(p) for p in candidates) if isinstance(candidates, list) else ""
        conf_rank = _CONFIDENCE_RANK.get(confidence, 0)
        # Sort key: confidence first, then single-page after multi-page within same confidence
        priority = conf_rank * 10 + (1 if n_candidates <= 1 else 0)
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
            "corrected_page":  "",
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
    print(f"Wrote {len(df):,} rows → {out_path.relative_to(Path.cwd())}")
    print(f"Confidence distribution: {conf_counts}")
    print(f"Multi-candidate rows: {multi_page:,}")


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
