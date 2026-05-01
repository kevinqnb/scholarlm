"""
Apply manual page-number corrections from page_review.csv to ground_truth.json.

Reads page_review.csv (filled by the user) and for each row where
``corrected_page`` is non-empty, updates the corresponding record in
ground_truth.json:
  - page_number      → parsed list of integers from corrected_page
  - page_confidence  → "manual"

``corrected_page`` accepts a single integer ("3") or a comma-separated list
("3, 4") for data points that genuinely span multiple pages.

Rows with an empty ``corrected_page`` are left unchanged.

After running this script, re-run create_probe_dataset.py to rebuild the
probe dataset with the corrected page numbers.

Note: re-running preprocessing.py will overwrite ground_truth.json and
discard manual corrections.  Keep page_review.csv as the source of truth
and re-apply after any preprocessing rerun.

Usage
-----
    python data/apply_page_corrections.py --dataset pond
    python data/apply_page_corrections.py --dataset nfix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent


def apply_corrections(dataset: str) -> None:
    gt_path = DATA_DIR / dataset / "ground_truth.json"
    review_path = DATA_DIR / dataset / "page_review.csv"

    with open(gt_path) as f:
        records = json.load(f)

    df = pd.read_csv(review_path, dtype={"corrected_page": str})
    filled = df[df["corrected_page"].notna() & (df["corrected_page"].str.strip() != "")]

    n_applied = 0
    n_skipped = 0
    for _, row in filled.iterrows():
        idx = int(row["gt_row_index"])
        raw = str(row["corrected_page"]).strip().strip("[]")
        try:
            pages = [int(float(t.strip())) for t in raw.split(",") if t.strip()]
            if not pages:
                raise ValueError("empty")
        except (ValueError, TypeError):
            print(f"  Warning: invalid corrected_page {raw!r} at gt_row_index={idx} — skipped")
            n_skipped += 1
            continue
        records[idx]["page_number"] = pages
        records[idx]["page_confidence"] = "manual"
        n_applied += 1

    with open(gt_path, "w") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"Applied {n_applied:,} corrections → {gt_path.relative_to(Path.cwd())}")
    if n_skipped:
        print(f"Skipped {n_skipped} rows with unparseable values.")
    print("Next: python data/pond/create_probe_dataset.py  (or nfix)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. pond, nfix)")
    args = parser.parse_args()
    apply_corrections(args.dataset)


if __name__ == "__main__":
    main()
