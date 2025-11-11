import os
import re
import json
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from scholarlm import DocumentLM
from scholarlm.utils import get_filenames_in_directory, encode_pil_image

from pdf2image import convert_from_path
from vllm import LLM, SamplingParams

# OlmOCR specfic prompt:
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

main_directory = os.getenv("POND_PATH")
pdf_directory = os.getenv("POND_PDF_PATH")
md_directory = os.getenv("POND_MARKDOWN_PATH")
text_directory = os.getenv("POND_TEXT_PATH")
image_directory = os.getenv("POND_IMAGE_PATH")


with open(os.path.join(main_directory, "directory.json"), "r") as f:
    paper_info = json.load(f)

pdf_files = get_filenames_in_directory(pdf_directory, ignore = [".DS_Store"])
pdf_files.sort()
filepaths = []
for f in pdf_files:
    filepath = os.path.join(pdf_directory, f)
    filepaths.append(filepath)

filepaths = filepaths[:10]

####################################################################################################

vlm = LLM("allenai/olmOCR-2-7B-1025-FP8")
ocr_prompt = olmocr_prompt()
sampling_params = SamplingParams(temperature=0.1, max_tokens=8192)


messages = []
message_paper_ids = []
for i, filepath in enumerate(filepaths):
    images = convert_from_path(filepath, dpi=300)
    for j, img in enumerate(images):
        if img.mode == "RGBA":
            img = img.convert("RGB")

        base64_image = encode_pil_image(img)
        image_data_uri = f'data:image/png;base64,{base64_image}'
        message = [
            {"role": "system", "content": ocr_prompt},
            {
                "role": "user",
                "content": [{
                    "type": "image_url",
                    "image_url": {
                    "url": image_data_uri
                    }
                }],
            },
        ]
        messages.append(message)
        message_paper_ids.append(i)


responses = vlm.chat(messages = messages, sampling_params = sampling_params)
response_markdown = [r.outputs[0].text for r in responses]
markdown_documents = ["" for _ in range(len(filepaths))]
for i, markdown in enumerate(response_markdown):
    paper_id = message_paper_ids[i]
    # Remove any front-matter from the markdown
    cleaned_markdown = re.sub(r"^---[\s\S]*?---\s*", "", markdown)
    markdown_documents[paper_id] += cleaned_markdown + "\n\n"

for i, filepath in enumerate(filepaths):
    filename = os.path.basename(filepath).replace('.pdf', '.md')
    save_path = os.path.join(md_directory, filename)
    with open(save_path, 'w', encoding='utf-8') as file:
        file.write(markdown_documents[i])
