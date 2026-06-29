"""
Extract OCR text from raw JSONL files and write each document to its own .txt file.

Input:  data/govscape/ocr_output_raw/output_*.jsonl
Output: data/govscape/ocr_output_cleaned/{id}.txt
"""

import json
from pathlib import Path

RAW_DIR = Path(__file__).parent / "ocr_output_raw"
CLEAN_DIR = Path(__file__).parent / "ocr_output_cleaned"


def main():
    CLEAN_DIR.mkdir(exist_ok=True)

    jsonl_files = sorted(RAW_DIR.glob("output_*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {RAW_DIR}")

    written = 0
    for jsonl_path in jsonl_files:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                out_path = CLEAN_DIR / f"{doc['id']}.txt"
                out_path.write_text(doc["text"])
                written += 1

    print(f"Wrote {written} documents to {CLEAN_DIR}")


if __name__ == "__main__":
    main()
