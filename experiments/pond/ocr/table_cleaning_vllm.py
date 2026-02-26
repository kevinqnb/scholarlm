"""
Step 3 of 3: Clean and normalize tables in OCR text.

Processes every page that contains <table> tags, using the page image to
verify and restructure each table into a normalized, machine-readable format
(long-form, index column, flattened multi-level headers, split composite cells).

Operates only on tables already present in the OCR text — missing-table
detection and re-OCR are handled by the earlier pipeline steps.

Corrected texts are written back to the OCR output directory in-place.
Originals are backed up with a ``.pre_clean.bak`` extension before writing.
"""

import os
import shutil
from scholarlm import TableCleaner
from scholarlm.utils import get_filenames_in_directory

####################################################################################################
# Configuration

main_directory = "data/pond"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output_raw")
ocr_out_directory = os.path.join(main_directory, "ocr_output_cleaned_vllm")

MODEL_NAME = "gaunernst/gemma-3-27b-it-qat-autoawq"
TARGET_DIM = 1536

####################################################################################################
# Load files

pdf_files = get_filenames_in_directory(pdf_directory, ignore=[".DS_Store", ".gitkeep"])
pdf_files.sort()

pdf_filepaths = [os.path.join(pdf_directory, f) for f in pdf_files]
text_files = [f.replace(".pdf", ".txt") for f in pdf_files]
text_filepaths = [os.path.join(ocr_directory, f) for f in text_files]
text_out_filepaths = [os.path.join(ocr_out_directory, f) for f in text_files]

existing = [
    (pdf, txt)
    for pdf, txt in zip(pdf_filepaths, text_filepaths)
    if os.path.exists(txt)
]
pdf_filepaths, text_filepaths = (
    list(x) for x in zip(*existing)
) if existing else ([], [])

print(f"Found {len(text_filepaths)} OCR text files to clean.")

texts = []
for filepath in text_filepaths:
    with open(filepath, "r", encoding="utf-8") as f:
        texts.append(f.read())

####################################################################################################
# Run cleaning

cleaner = TableCleaner(
    backend="vllm",
    model_name=MODEL_NAME,
    sampling_params={"temperature": 0.1, "max_tokens": 16384},
    target_longest_dim=TARGET_DIM,
)

cleaned_texts = cleaner.clean(texts=texts, pdf_paths=pdf_filepaths)

####################################################################################################
# Save results

cleaner.save(cleaned_texts, text_out_filepaths)
print(f"\nSaved {len(text_out_filepaths)} cleaned texts to {ocr_out_directory}.")
