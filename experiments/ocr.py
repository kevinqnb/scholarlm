import os
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from scholarlm import DocumentLM
from scholarlm.utils import get_filenames_in_directory

# OlmOCR specfic prompt:
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

#task_id = int(os.getenv('SGE_TASK_ID'))

main_directory = os.getenv("POND_PATH")
pdf_directory = os.getenv("POND_PDF_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH")
image_directory = os.getenv("POND_IMAGE_PATH")


with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

pdf_files = get_filenames_in_directory(pdf_directory, ignore = [".DS_Store"])
pdf_files.sort()

#pdf_files = pdf_files[(task_id - 1)*22 : min(task_id*22, len(pdf_files))]
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

#pdf_files = ['biodiversity_of_constructed.pdf']

filepaths = []
for f in pdf_files:
    filepath = os.path.join(pdf_directory, f)
    filepaths.append(filepath)

#filepaths = filepaths[:10]
#print(filepaths)

doclm = DocumentLM(
    model = "allenai/olmOCR-2-7B-1025-FP8",
    ocr = True,
    ocr_prompt = olmocr_prompt(),
    sampling_params = {"temperature": 0.1, "max_tokens": 8192},
)

#doclm.filepaths = filepaths
#doclm.chunk()
#doclm.save_images(image_directory)
#chunks = doclm.ocr_read()
#doclm.save_chunks(text_directory)

doclm.fit(filepaths)
doclm.save_images(image_directory)
doclm.save_chunks(text_directory)


