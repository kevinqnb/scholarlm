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

main_directory = "data/nfix"
pdf_directory = os.path.join(main_directory, "pdfs")
ocr_directory = os.path.join(main_directory, "ocr_output_raw")
with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

# Get all PDF files in the pdf directory
pdf_files = get_filenames_in_directory(pdf_directory, ignore = [".DS_Store", ".gitkeep"])
pdf_files.sort()

pdf_files = ["R93.pdf"]

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
