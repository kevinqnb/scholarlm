"""
Table cleaning for the pond dataset using the OpenAI API.

Equivalent to table_correction_vllm.py but uses the openai backend.
Only runs the cleaning step — for detection of tables missing from OCR,
use table_detection_vllm.py followed by table_reocr.py first.

Corrected texts are written back to the OCR output directory in-place.
Originals are backed up with a .pre_clean.bak extension before writing.
"""

import os
import shutil
from dotenv import load_dotenv
load_dotenv()

from scholarlm import TableCleaner
from scholarlm.utils import get_filenames_in_directory

####################################################################################################
# Configuration

main_directory = "data/nfix"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output_raw")
ocr_out_directory_cleaned = os.path.join(main_directory, "ocr_output_cleaned_openai")

OPENAI_MODEL = "gpt-5-mini"
RATE_LIMIT = 100  # requests per minute
TARGET_DIM = 1536

# Skip files that have already been processed.
# Comment out or clear this list to process everything.
'''
precomputed_pdf_files = [
    'physical_and_chemical_limnological.pdf',
    'physical-chemical_influences.pdf',
    'prairie_wetland.pdf',
    'net_heterotrophy.pdf',
    'habitat_characteristics.pdf',
    'biodiversity_of_constructed.pdf',
    'fish_production_in_lakes.pdf',
    'long-term_stability.pdf',
    'diversity_of_macroinvertebrates.pdf',
    'impact_of_macrophytes.pdf',
    #"bacterioplankton.pdf",
    "conservation_of_pond.pdf",
    "distinct_optical.pdf",
    "fish_assemblages.pdf",
    "lake_morphometry.pdf",
    "natural_variability.pdf",
    "productivity_and_depth.pdf",
    "relationships_of_fish.pdf",
    "sediment_characteristics.pdf",
    "vegetation-environmental.pdf",
]
'''

####################################################################################################
# Load files

pdf_files = get_filenames_in_directory(pdf_directory, ignore=[".DS_Store", ".gitkeep"])
pdf_files.sort()
#pdf_files = [f for f in pdf_files if f not in precomputed_pdf_files]
#pdf_files = pdf_files[:10]  # Limit to 10 files for testing; remove this line to process all files

pdf_filepaths = [os.path.join(pdf_directory, f) for f in pdf_files]
text_files = [f.replace(".pdf", ".txt") for f in pdf_files]
text_filepaths = [os.path.join(ocr_directory, f) for f in text_files]
text_out_filepaths = [os.path.join(ocr_out_directory_cleaned, f) for f in text_files]

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
    backend="openai",
    openai_model=OPENAI_MODEL,
    openai_rate_limit=RATE_LIMIT,
    sampling_params={"max_completion_tokens": 16384},
    target_longest_dim=TARGET_DIM,
)

cleaned_texts = cleaner.clean(texts=texts, pdf_paths=pdf_filepaths)

####################################################################################################
# Save results

cleaner.save(cleaned_texts, text_out_filepaths)
print(f"\nSaved {len(text_filepaths)} cleaned texts to {ocr_out_directory_cleaned}.")
