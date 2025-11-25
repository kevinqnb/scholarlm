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

filepaths = []
for f in pdf_files:
    filepath = os.path.join(pdf_directory, f)
    filepaths.append(filepath)

doclm = DocumentLM(
    model = "allenai/olmOCR-2-7B-1025-FP8",
    ocr = True,
    ocr_prompt = olmocr_prompt(),
    sampling_params = {"temperature": 0.1, "max_tokens": 8192, "seed": 342},
)

doclm.fit(filepaths)
#doclm.save_images(image_directory)
doclm.save_chunks(text_directory)


