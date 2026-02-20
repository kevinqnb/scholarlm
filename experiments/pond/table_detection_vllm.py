"""
Step 1 of 3: Detect tables missing from OCR text.

Scans every page of every document and flags pages where a data table is visible
in the PDF image but absent (or only partially present) in the OCR text.

Output
------
Writes ``data/pond/reocr_candidates.json`` — a list of records with the PDF
filepath, OCR text filepath, and page number for each flagged page.  Pass this
file to ``table_reocr.py`` to re-run OCR on only those pages.
"""

import os
import json
from scholarlm import TableCleaner
from scholarlm.utils import get_filenames_in_directory

####################################################################################################
# Configuration

main_directory = "data/pond"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output_raw")
candidates_path = os.path.join(main_directory, "reocr_candidates.json")

MODEL_NAME = "gaunernst/gemma-3-27b-it-qat-autoawq"
TARGET_DIM = 1536

####################################################################################################
# Load files

pdf_files = get_filenames_in_directory(pdf_directory, ignore=[".DS_Store", ".gitkeep"])
pdf_files.sort()

pdf_filepaths = [os.path.join(pdf_directory, f) for f in pdf_files]
text_files = [f.replace(".pdf", ".txt") for f in pdf_files]
text_filepaths = [os.path.join(ocr_directory, f) for f in text_files]

# Filter to files that exist in the OCR output directory
existing = [
    (pdf, txt, pdf_f, txt_f)
    for pdf, txt, pdf_f, txt_f in zip(pdf_filepaths, text_filepaths, pdf_files, text_files)
    if os.path.exists(txt)
]
pdf_filepaths, text_filepaths, pdf_files, text_files = (
    list(x) for x in zip(*existing)
) if existing else ([], [], [], [])

print(f"Found {len(text_filepaths)} OCR text files to scan.")

texts = []
for filepath in text_filepaths:
    with open(filepath, "r", encoding="utf-8") as f:
        texts.append(f.read())

####################################################################################################
# Run detection

cleaner = TableCleaner(
    backend="vllm",
    model_name=MODEL_NAME,
    target_longest_dim=TARGET_DIM,
)

candidates = cleaner.detect(texts=texts, pdf_paths=pdf_filepaths)

####################################################################################################
# Save candidates

candidates_out = [
    {
        "doc_idx": doc_idx,
        "pdf_filepath": pdf_filepaths[doc_idx],
        "text_filepath": text_filepaths[doc_idx],
        "page_number": page_number,
    }
    for doc_idx, page_number in candidates
]

with open(candidates_path, "w") as f:
    json.dump(candidates_out, f, indent=2)

print(f"\n{len(candidates)} pages flagged for re-OCR → {candidates_path}")
print("Next step: run table_reocr.py")
