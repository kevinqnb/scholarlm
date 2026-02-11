import os
import json
from dotenv import load_dotenv
load_dotenv()
from scholarlm import DocumentLM
from scholarlm.utils import get_filenames_in_directory

# OlmOCR specfic prompt:
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)

####################################################################################################
# Load PDFs and set up filepaths

main_directory = "data/pond"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output")
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

# Get all PDF files in the pdf directory
pdf_files = get_filenames_in_directory(pdf_directory, ignore = [".DS_Store", ".gitkeep"])
pdf_files.sort()

# Or specify specific files to process:
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
    "bacterioplankton.pdf",
    "conservation_of_pond.pdf",
    "distinct_optical.pdf",
    "fish_assemblages.pdf",
    "lake_morphometry.pdf",
    "natural_variability.pdf",
    "productivity_and_depth.pdf",
    "relationships_of_fish.pdf",
    "sediment_characteristics.pdf",
    "vegetation-environmental.pdf"
]

pdf_files = [pf for pf in pdf_files if pf not in precomputed_pdf_files]


filepaths = []
for f in pdf_files:
    filepath = os.path.join(pdf_directory, f)
    filepaths.append(filepath)

out_filepaths = []
for f in pdf_files:
    filename = f.replace('.pdf', '.txt')
    out_filepath = os.path.join(ocr_directory, filename)
    out_filepaths.append(out_filepath)


####################################################################################################
# Run the OCR model with DocumentLM

doclm = DocumentLM(
    model = "allenai/olmOCR-2-7B-1025-FP8",
    ocr_prompt = olmocr_prompt(),
    sampling_params = {
        "temperature": 0.1,
        "max_tokens": 8192,
        "seed": 342
    }
)

# Fit and save outputs
text = doclm.fit(filepaths)
doclm.save(out_filepaths)
