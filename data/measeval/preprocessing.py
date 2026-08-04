"""
Ground truth preprocessing for the measeval dataset.

Unlike pond/nfix/supermat, MeasEval (https://github.com/harperco/MeasEval,
SemEval-2021 Task 8) ships plain text and gold TSV annotations directly, with
exact character offsets -- there is no PDF/OCR step and no need for the fuzzy
page-attribution machinery those other datasets rely on.

Pipeline
--------
    data/measeval/raw/data/{train,trial,eval}/{text|txt}/*.txt, {tsv}/*.tsv
        v  every text file -> ocr_output_raw/{document_id}.txt (page-wrapped)
           and a directory.json entry, whether or not it has annotations
        v  every tsv row grouped by (docId, annotSet): at most one Quantity,
           MeasuredEntity, MeasuredProperty, and 0-3 Qualifiers per set
           (verified across the full corpus -- no relation-graph traversal
           needed, annotSet grouping is sufficient)
        v  value parsed from the Quantity span's first numeric token; rows
           whose mods mark them as a range/approximation/list are dropped
           (see _should_drop below), matching the "single unambiguous value"
           policy documented in data/supermat/README.md
    ground_truth.json                  (all three splits combined)
    ground_truth_ten.json              (top-10 document development subset)

data/iaa/ (a separate inter-annotator-agreement re-annotation of a subset of
train) is intentionally excluded -- including it would double-count rows for
documents already present in train.

Usage
-----
Run from the repo root (after data/measeval/download_measeval.py):

    python data/measeval/preprocessing.py

Or from data/measeval/:

    python preprocessing.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

from scholarlm.utils.page_attribution import parse_numeric

BASE = Path(__file__).parent  # data/measeval/
RAW_DATA_DIR = BASE / "raw" / "data"
OCR_DIR = BASE / "ocr_output_raw"

_TOP_PAPERS_N = 10

# (split name, text subdirectory name) -- trial's text lives under txt/, not text/.
_SPLITS: list[tuple[str, str]] = [
    ("train", "text"),
    ("trial", "txt"),
    ("eval", "text"),
]

# Quantity mods that make a value ambiguous (range/approximation/multi-value
# list) rather than a single reported number. Matched by substring, not exact
# set membership, since the corpus contains garbled concatenations like
# "IsRangeHasTolerance" or "IsMeanIsRange" alongside the clean single-mod case.
_DROP_MOD_SUBSTRINGS = ("IsRange", "IsApproximate", "IsList")

# First run of digits (with optional thousands-separator commas and a decimal
# point) in a Quantity span, e.g. "5318" in "5318 participants", "60,268" in
# "60,268 km". Mirrors supermat's _parse_tcvalue: take the first number found
# rather than trying to fully parse free-text quantity expressions.
_NUMBER_TOKEN_RE = re.compile(r"[-−]?\d[\d,]*(?:\.\d+)?")

# "<mantissa> × 10<exp>" / "<mantissa> x 10^<exp>" scientific notation, e.g.
# "3.7 × 106" (= 3.7e6) or "4.3 × 10−8" (= 4.3e-8). Checked before the plain
# token regex above, which would otherwise only pick up the mantissa and
# silently drop the exponent. Exponent digits follow "10" with no separator,
# same convention documented in scholarlm.utils.page_attribution.
_SCI_NOTATION_RE = re.compile(
    r"(?P<mantissa>[-−]?\d+(?:\.\d+)?)\s*[×x]\s*10\s*\^?\{?(?P<exp>[-−+]?\d+)\}?"
)
_UNICODE_MINUS = str.maketrans({"−": "-"})


def _should_drop(mods: list[str]) -> bool:
    return any(sub in mod for mod in mods for sub in _DROP_MOD_SUBSTRINGS)


def _extract_value(quantity_text: str) -> float | None:
    """Pull the reported numeric value out of a Quantity span's text.

    Tries scientific notation first, then falls back to the first plain
    numeric token found. Returns None if no digit is present at all (e.g.
    spelled-out numbers like "four", "twice") -- converting those to a number
    would require inference beyond what's directly written, which this
    pipeline avoids (see data/supermat/README.md's "Ground truth value policy").
    """
    sci_match = _SCI_NOTATION_RE.search(quantity_text)
    if sci_match:
        mantissa = sci_match.group("mantissa").translate(_UNICODE_MINUS)
        exp = sci_match.group("exp").translate(_UNICODE_MINUS)
        return parse_numeric(f"{mantissa}e{exp}")

    m = _NUMBER_TOKEN_RE.search(quantity_text)
    if m is None:
        return None
    return parse_numeric(m.group(0).replace(",", ""))


def _read_tsv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        rows = list(reader)
    for row in rows:
        row["other"] = json.loads(row["other"]) if row["other"] else {}
        row["startOffset"] = int(row["startOffset"])
        row["endOffset"] = int(row["endOffset"])
    return rows


def _build_rows_for_document(document_id: str, split: str, tsv_rows: list[dict]) -> list[dict]:
    by_set: dict[str, list[dict]] = {}
    for row in tsv_rows:
        by_set.setdefault(row["annotSet"], []).append(row)

    out_rows: list[dict] = []
    for annot_set, rows in by_set.items():
        by_type: dict[str, list[dict]] = {}
        for row in rows:
            by_type.setdefault(row["annotType"], []).append(row)

        quantity_rows = by_type.get("Quantity", [])
        if not quantity_rows:
            continue
        quantity = quantity_rows[0]

        entity = by_type.get("MeasuredEntity", [None])[0]
        prop = by_type.get("MeasuredProperty", [None])[0]
        qualifiers = by_type.get("Qualifier", [])

        mods = quantity["other"].get("mods", [])
        if _should_drop(mods):
            continue

        value = _extract_value(quantity["text"])
        if value is None:
            continue

        out_rows.append({
            "document_id": document_id,
            "split": split,
            # The raw Quantity span, units included as written ("54.8 years",
            # "5318"). This is the gold counterpart of the extraction side's
            # entity field, `EntitySchema.quantity`: the quantity-first design
            # in experiments/configs/measeval.py enumerates one entity per
            # reported quantity and copies the span verbatim, so both sides
            # hold the same kind of string. `value`/`units` below remain the
            # parsed number and its unit, and are what matching keys on --
            # this is carried for traceability and for the option of fuzzy-
            # matching on the span itself later.
            "quantity": quantity["text"],
            "name": entity["text"] if entity else None,
            # Constant across every row: matches the single abstract attribute
            # bucket in experiments/configs/measeval.py's attribute_info_dict,
            # so ground truth and extraction output strict-match trivially on
            # this field. The actual open-vocabulary property name lives in
            # `property` below (matched fuzzily instead, alongside `name`).
            "attribute": "measurement",
            "property": prop["text"] if prop else None,
            "value": value,
            "units": quantity["other"].get("unit"),
            "additional_details": "; ".join(q["text"] for q in qualifiers) or None,
            "mods": mods,
            "annot_set": annot_set,
            "entity_start": entity["startOffset"] if entity else None,
            "entity_end": entity["endOffset"] if entity else None,
            "property_start": prop["startOffset"] if prop else None,
            "property_end": prop["endOffset"] if prop else None,
            "quantity_start": quantity["startOffset"],
            "quantity_end": quantity["endOffset"],
        })
    return out_rows


def build_ground_truth() -> None:
    OCR_DIR.mkdir(exist_ok=True)

    directory: dict[str, dict] = {}
    all_rows: list[dict] = []
    n_no_value = 0
    n_dropped_mods = 0

    for split, text_subdir in _SPLITS:
        text_dir = RAW_DATA_DIR / split / text_subdir
        tsv_dir = RAW_DATA_DIR / split / "tsv"

        text_files = sorted(text_dir.glob("*.txt"))
        n_docs = 0
        n_rows_split = 0
        for text_path in text_files:
            document_id = text_path.stem
            if document_id in directory:
                raise ValueError(
                    f"document_id collision: {document_id!r} appears in both "
                    f"{directory[document_id]['source_split']} and {split}"
                )

            text = text_path.read_text(encoding="utf-8")
            (OCR_DIR / f"{document_id}.txt").write_text(
                f'<page number="0">\n{text}\n</page>\n', encoding="utf-8"
            )

            article_id = document_id.rsplit("-", 1)[0]
            directory[document_id] = {
                "title": None,
                "author": None,
                "year": None,
                "source_split": split,
                "article_id": article_id,
            }
            n_docs += 1

            tsv_path = tsv_dir / f"{document_id}.tsv"
            if not tsv_path.exists():
                continue  # real document with zero annotated measurements

            tsv_rows = _read_tsv(tsv_path)
            doc_rows = _build_rows_for_document(document_id, split, tsv_rows)
            n_rows_split += len(doc_rows)

            # Tally why quantities were dropped for the summary printed below.
            for row in tsv_rows:
                if row["annotType"] != "Quantity":
                    continue
                mods = row["other"].get("mods", [])
                if _should_drop(mods):
                    n_dropped_mods += 1
                elif _extract_value(row["text"]) is None:
                    n_no_value += 1
            all_rows.extend(doc_rows)

        print(f"  {split}: {n_docs} documents, {n_rows_split} ground-truth rows")

    with open(BASE / "directory.json", "w", encoding="utf-8") as fh:
        json.dump(directory, fh, indent=2)
    print(f"  Saved {len(directory):,} documents -> directory.json")

    print(f"  Dropped {n_dropped_mods:,} Quantity rows tagged IsRange/IsApproximate/IsList")
    print(f"  Dropped {n_no_value:,} Quantity rows with no parseable numeric token "
          f"(e.g. spelled-out numbers)")

    with open(BASE / "ground_truth.json", "w", encoding="utf-8") as fh:
        json.dump(all_rows, fh, indent=2)
    print(f"  Saved {len(all_rows):,} rows -> ground_truth.json")

    # Dev subset is drawn from train/trial only, never eval -- eval is the held-out
    # test split (see README's "Train/trial/eval and comparability" section), and a
    # debugging subset that quietly included test documents would let prompt
    # iteration peek at test-set answers.
    dev_rows = [row for row in all_rows if row["split"] != "eval"]
    doc_counts = Counter(row["document_id"] for row in dev_rows)
    top_docs = {doc_id for doc_id, _ in doc_counts.most_common(_TOP_PAPERS_N)}
    rows_ten = [row for row in dev_rows if row["document_id"] in top_docs]
    with open(BASE / "ground_truth_ten.json", "w", encoding="utf-8") as fh:
        json.dump(rows_ten, fh, indent=2)
    print(f"  Saved {len(rows_ten):,} rows -> ground_truth_ten.json "
          f"({len(top_docs)} documents, train/trial only: {sorted(top_docs)})")


def main(argv: list[str] | None = None) -> None:
    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_DIR} not found -- run data/measeval/download_measeval.py first"
        )
    print("Building ground truth JSONs ...")
    build_ground_truth()


if __name__ == "__main__":
    main()
