"""
Regex-based judge: verify that the extracted value appears in the document text.

This is a lightweight sanity check that complements the LLM judges. For each
extraction record, it searches the full document OCR text for any numerically
equivalent surface representation of the extracted value.  Formatting variants
handled include:

  - Comma as decimal separator (European): 3,14 ↔ 3.14
  - Comma or period as thousands separator: 1,000 ↔ 1.000 ↔ 1000
  - Trailing zero differences: 3.0 ↔ 3
  - Scientific notation: 0.001 ↔ 1e-3 ↔ 1×10^-3

A True judgement means the value (or a formatting variant) was found anywhere
in the document.  This does NOT verify entity or attribute correctness — it
only checks that the number was not entirely hallucinated.

The "regex" judge key is recognised by ``run_judge_combine.py`` as a
non-voting judge: it adds a ``judgement_regex`` column to ``combined.json``
without affecting the majority-vote ground-truth label.

Output path:
    data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/regex/{judge_date}/

Saves:
  - ``responses.json``

Usage
-----
    python experiments/run_judge_regex.py \\
        --dataset pond --extraction-model gemma-3-27b

    # Pin to a specific extraction run:
    python experiments/run_judge_regex.py \\
        --dataset pond --extraction-model gemma-3-27b \\
        --extraction-date 2026_04_01

    # Use cleaned OCR (same dir used during extraction):
    python experiments/run_judge_regex.py \\
        --dataset pond --extraction-model gemma-3-27b \\
        --ocr-dir data/pond/ocr_output_cleaned_gemma-3-27b
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
_CONFIGS_DIR = Path(__file__).parent / "configs"
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))  # makes 'batch' importable

from dotenv import load_dotenv
load_dotenv()

from scholarlm.config import DatasetConfig
from scholarlm.utils import get_filenames_in_directory

JUDGE_KEY = "regex"

from run_extraction import load_dataset_config
import paths


# ---------------------------------------------------------------------------
# Value matching
# ---------------------------------------------------------------------------


def _try_parse_float(s: str) -> float | None:
    """Parse a numeric string to float, handling common locale variants."""
    s = s.strip()
    attempts = [
        s,
        s.replace(",", ""),            # Remove thousands commas: 1,000 → 1000
        s.replace(",", "."),           # Comma decimal: 3,14 → 3.14
        s.replace(".", "").replace(",", "."),  # European: 1.234,56 → 1234.56
    ]
    for attempt in attempts:
        try:
            v = float(attempt)
            if math.isfinite(v):
                return v
        except ValueError:
            pass
    return None


def _value_patterns(value_str: str) -> list[str]:
    """Return regex-escaped candidate strings for a value, covering formatting variants.

    Always includes the literal string. When the value parses as a finite float,
    also adds:
      - g-format (trailing zeros removed)
      - Comma-decimal variant (European)
      - Integer form + thousands-separator variants (for whole numbers ≥ 1000)
      - Scientific notation variants (for |v| < 0.001 or |v| ≥ 1,000,000)
    """
    s = value_str.strip()
    if not s:
        return []

    raw: list[str] = [s]
    fv = _try_parse_float(s)

    if fv is not None:
        # Compact decimal form
        g = f"{fv:g}"
        raw.append(g)

        # European comma-decimal variant
        if "." in g:
            raw.append(g.replace(".", ","))

        # Integer variants for whole numbers
        if fv == int(fv) and abs(fv) < 1e15:
            iv = int(fv)
            raw.append(str(iv))
            if abs(iv) >= 1000:
                # English thousands: 1,000,000
                raw.append(f"{iv:,}")
                # European thousands: 1.000.000
                raw.append(f"{iv:,}".replace(",", "."))

        # Scientific notation for very small / very large values
        if fv != 0 and (abs(fv) < 0.001 or abs(fv) >= 1_000_000):
            exp = int(math.floor(math.log10(abs(fv))))
            mantissa = fv / (10.0 ** exp)
            m = f"{mantissa:g}"
            raw.extend([
                f"{m}e{exp}",
                f"{m}e{exp:+03d}",   # e.g. 1e-03
                f"{m}E{exp}",
                f"{m}×10^{exp}",
                f"{m}*10^{exp}",
            ])

    # Deduplicate while preserving order, then escape for regex
    seen: set[str] = set()
    patterns: list[str] = []
    for c in raw:
        if c and c not in seen:
            seen.add(c)
            patterns.append(re.escape(c))
    return patterns


def _found_in_document(value_str: str | None, document: str) -> bool | None:
    """Return True/False if value (or a formatting variant) appears in document.

    Returns None if value is absent or unparseable as any pattern.
    Uses negative digit lookbehind/lookahead to avoid matching the value as a
    substring of a larger number (e.g. 0.26 should not match inside 10.26).
    """
    if value_str is None:
        return None
    s = str(value_str).strip()
    if not s:
        return None

    for pattern in _value_patterns(s):
        if re.search(rf"(?<!\d){pattern}(?!\d)", document):
            return True
    return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_regex_judge(
    dataset_config: DatasetConfig,
    extraction_model: str,
    output_dir: Path,
    extraction_date: str | None = None,
    ocr_dir: str | None = None,
) -> None:
    """Search each record's document for the extracted value and save results.

    Args:
        dataset_config: Dataset configuration.
        extraction_model: Short name of the extraction model whose results to judge.
        output_dir: Directory to write ``responses.json``.
        extraction_date: Optional date tag for locating extraction results.
        ocr_dir: Directory of OCR ``.txt`` files. Defaults to
            ``{data_dir}/ocr_output_raw/``.
    """
    input_file = paths.find_extraction_final(dataset_config.name, extraction_model, extraction_date)
    print(f"Input   : {input_file}")

    with open(input_file) as f:
        data: list[dict] = json.load(f)

    effective_ocr_dir = ocr_dir or str(Path(dataset_config.data_dir) / "ocr_output_raw")
    from batch import common as batch_common
    documents = batch_common.load_documents_for_dataset(dataset_config, effective_ocr_dir)

    print(f"Checking {len(data)} records against {len(documents)} documents ...")

    n_true = n_false = n_null = 0
    judged_data: list[dict] = []

    for record in data:
        value = record.get("value")
        doc_id = record.get("document_id")
        document = documents[doc_id] if doc_id is not None and doc_id < len(documents) else ""

        judgement = _found_in_document(value, document)
        if judgement is True:
            n_true += 1
        elif judgement is False:
            n_false += 1
        else:
            n_null += 1

        judged_data.append(record | {
            "judgement": judgement,
            "judgement_model": JUDGE_KEY,
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    responses_file = output_dir / "responses.json"
    with open(responses_file, "w") as f:
        json.dump(judged_data, f, indent=4, ensure_ascii=False)

    print(f"Results : {n_true} found, {n_false} not found, {n_null} skipped (null value)")
    print(f"Saved   : {responses_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Regex judge: verify extracted values appear in document text.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'pond', 'nfix').")
    p.add_argument(
        "--extraction-model", required=True,
        help="Short name of the extraction model whose results to judge.",
    )
    p.add_argument("--extraction-date", default=None, help="Date tag YYYY_mm_dd of extraction run.")
    p.add_argument("--judge-date", default=None, help="Date tag for output directory (default: today).")
    p.add_argument(
        "--ocr-dir", default=None, metavar="DIR",
        help="Directory of OCR .txt files. Defaults to {data_dir}/ocr_output_raw/.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    dataset_config = load_dataset_config(args.dataset)
    input_file = paths.find_extraction_final(args.dataset, args.extraction_model, args.extraction_date)
    extraction_date_resolved = input_file.parent.name
    output_dir = paths.judge(
        args.dataset, args.extraction_model, extraction_date_resolved, JUDGE_KEY, args.judge_date
    )

    print(f"\nDataset          : {args.dataset}")
    print(f"Extraction model : {args.extraction_model}")
    print(f"Extraction date  : {extraction_date_resolved}")
    print(f"Judge            : {JUDGE_KEY}")
    print(f"Output           : {output_dir}\n")

    run_regex_judge(
        dataset_config=dataset_config,
        extraction_model=args.extraction_model,
        output_dir=output_dir,
        extraction_date=extraction_date_resolved,
        ocr_dir=args.ocr_dir,
    )


if __name__ == "__main__":
    main()
