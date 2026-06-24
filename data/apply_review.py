"""
Apply manual corrections from page_review.csv to produce ground_truth_review.json.

Reads page_review.csv (filled by the user) and applies corrections to a copy
of ground_truth.json, writing ground_truth_review.json (and
ground_truth_ten_review.json if ground_truth_ten.json exists).

Correction columns handled:
  corrected_page    – integer or comma-separated list → updates page_number,
                      sets page_confidence to "manual"
  corrected_name    – non-empty string → updates name
  corrected_value   – numeric string → updates value
  corrected_units   – non-empty string → updates units
  excluded          – "True" / "true" / "1" / "yes" → record omitted from output
  excluded_reason   – informational only, not written to output

Rows with all correction columns empty are left unchanged.

ground_truth.json and ground_truth_ten.json are never modified.
ground_truth_review.json is the corrected output; use it as the source of
truth for probe dataset creation and analysis.

After running this script, re-run create_probe_dataset.py to rebuild the
probe dataset from the corrected ground truth.

Usage
-----
    python data/apply_review.py --dataset pond
    python data/apply_review.py --dataset nfix
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent

_EXCLUDED_VALUES = {"true", "1", "yes"}


def _parse_page_list(raw: str) -> list[int] | None:
    """Parse a corrected_page string into a list of ints, or None on failure."""
    cleaned = raw.strip().strip("[]")
    if not cleaned:
        return None
    try:
        pages = [int(float(t.strip())) for t in cleaned.split(",") if t.strip()]
        return pages if pages else None
    except (ValueError, TypeError):
        return None


def apply_review(dataset: str) -> None:
    gt_path = DATA_DIR / dataset / "ground_truth.json"
    gt_ten_path = DATA_DIR / dataset / "ground_truth_ten.json"
    review_path = DATA_DIR / dataset / "page_review.csv"
    out_path = DATA_DIR / dataset / "ground_truth_review.json"
    out_ten_path = DATA_DIR / dataset / "ground_truth_ten_review.json"

    with open(gt_path) as f:
        records: list[dict] = json.load(f)

    df = pd.read_csv(review_path, dtype=str, keep_default_na=False)

    n_page = n_name = n_value = n_units = n_excluded = n_skipped = 0
    excluded_indices: set[int] = set()

    for _, row in df.iterrows():
        try:
            idx = int(float(row["gt_row_index"]))
        except (ValueError, KeyError):
            print(f"  Warning: unparseable gt_row_index {row.get('gt_row_index')!r} — skipped")
            n_skipped += 1
            continue

        if idx < 0 or idx >= len(records):
            print(f"  Warning: gt_row_index {idx} out of range (0–{len(records)-1}) — skipped")
            n_skipped += 1
            continue

        # Exclusion takes priority over all other corrections.
        excluded_str = row.get("excluded", "").strip().lower()
        if excluded_str in _EXCLUDED_VALUES:
            excluded_indices.add(idx)
            n_excluded += 1
            continue

        # Page correction
        raw_page = row.get("corrected_page", "").strip()
        if raw_page:
            pages = _parse_page_list(raw_page)
            if pages is not None:
                records[idx]["page_number"] = pages
                records[idx]["page_confidence"] = "manual"
                n_page += 1
            else:
                print(f"  Warning: invalid corrected_page {raw_page!r} at gt_row_index={idx} — skipped")
                n_skipped += 1

        # Name correction
        corrected_name = row.get("corrected_name", "").strip()
        if corrected_name:
            records[idx]["name"] = corrected_name
            n_name += 1

        # Value correction
        corrected_value = row.get("corrected_value", "").strip()
        if corrected_value:
            try:
                records[idx]["value"] = float(corrected_value)
                n_value += 1
            except (ValueError, TypeError):
                print(f"  Warning: invalid corrected_value {corrected_value!r} at gt_row_index={idx} — skipped")
                n_skipped += 1

        # Units correction
        corrected_units = row.get("corrected_units", "").strip()
        if corrected_units:
            records[idx]["units"] = corrected_units
            n_units += 1

    reviewed = [rec for i, rec in enumerate(records) if i not in excluded_indices]

    with open(out_path, "w") as f:
        json.dump(reviewed, f, indent=2, ensure_ascii=False)

    print(f"Corrections applied: {n_page} page, {n_name} name, "
          f"{n_value} value, {n_units} units, {n_excluded} excluded"
          + (f", {n_skipped} skipped" if n_skipped else ""))
    print(f"Saved {len(reviewed):,} rows → {out_path.relative_to(Path.cwd())}")

    # Ten-paper subset: filter reviewed records to document_ids in ground_truth_ten.json.
    if gt_ten_path.exists():
        with open(gt_ten_path) as f:
            ten_records = json.load(f)
        ten_ids = {rec["document_id"] for rec in ten_records}
        reviewed_ten = [rec for rec in reviewed if rec.get("document_id") in ten_ids]
        with open(out_ten_path, "w") as f:
            json.dump(reviewed_ten, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(reviewed_ten):,} rows → {out_ten_path.relative_to(Path.cwd())}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g. pond, nfix)")
    args = parser.parse_args()
    apply_review(args.dataset)


if __name__ == "__main__":
    main()
