import os
import json
from dotenv import load_dotenv
load_dotenv()
from scholarlm import DocumentLM2
from scholarlm.utils import get_filenames_in_directory


# OlmOCR specfic prompt:
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)

main_directory = os.getenv("POND_PATH")
pdf_directory = os.getenv("POND_PDF_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH")
image_directory = os.getenv("POND_IMAGE_PATH")

with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

pdf_files = get_filenames_in_directory(pdf_directory, ignore = [".DS_Store"])
pdf_files.sort()


pdf_files = [
    'physical_and_chemical_limnological.pdf',
    'physical-chemical_influences.pdf',
    'prairie_wetland.pdf',
    'net_heterotrophy.pdf',
    'habitat_characteristics.pdf',
    'biodiversity_of_constructed.pdf',
    'fish_production_in_lakes.pdf',
    'long-term_stability.pdf',
    'diversity_of_macroinvertebrates.pdf',
    'impact_of_macrophytes.pdf'
]


filepaths = []
for f in pdf_files:
    filepath = os.path.join(pdf_directory, f)
    filepaths.append(filepath)

out_filepaths = []
for f in pdf_files:
    filename = f.replace('.pdf', '.txt')
    out_filepath = os.path.join(text_directory, filename)
    out_filepaths.append(out_filepath)

####################################################################################################

doclm = DocumentLM2(
    model = "allenai/olmOCR-2-7B-1025-FP8",
    ocr_prompt = olmocr_prompt(),
    sampling_params = {
        "temperature": 0.1,
        "max_tokens": 8192,
        "seed": 342
    }
)

text = doclm.fit(filepaths)
doclm.save(out_filepaths)
