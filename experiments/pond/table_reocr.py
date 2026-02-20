"""
Step 2 of 3: Re-run olmOCR on pages flagged by table_detection_vllm.py.

Loads ``data/pond/reocr_candidates.json``, renders only the flagged pages from
each PDF, runs olmOCR on those pages, and splices the corrected text back into
the full OCR document.  Table numbers are re-assigned globally across each
document after splicing so they remain contiguous.

The corrected OCR texts are written back to the OCR output directory in-place.
Original files are backed up with a ``.pre_reocr.bak`` extension before writing.

Note on GPU / VLLM
------------------
olmOCR and the TableCleaner cleaning model are separate VLLM instances.  Run
this script **between** ``table_detection_vllm.py`` and
``table_correction_vllm.py`` so each model has the GPU to itself.
"""

import os
import re
import json
import shutil
from itertools import count as itercount

from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt
from vllm import LLM, SamplingParams

from scholarlm.utils import load_pdf_page, correct_image_orientation, encode_pil_image

####################################################################################################
# Configuration

main_directory = "data/pond"
ocr_directory = os.path.join(main_directory, "ocr_output_raw")
candidates_path = os.path.join(main_directory, "reocr_candidates.json")

OCR_MODEL = "allenai/olmOCR-2-7B-1025-FP8"
TARGET_DIM = 2048
SAMPLING_PARAMS = SamplingParams(temperature=0.1, max_tokens=8192, seed=342)

####################################################################################################
# Load candidates

with open(candidates_path, "r") as f:
    candidates = json.load(f)

if not candidates:
    print("No re-OCR candidates found. Nothing to do.")
    raise SystemExit(0)

# Group candidates by text file so we load each document only once
by_text_file: dict[str, list[dict]] = {}
for entry in candidates:
    by_text_file.setdefault(entry["text_filepath"], []).append(entry)

print(f"Re-OCR-ing {len(candidates)} pages across {len(by_text_file)} documents.")

####################################################################################################
# Load OCR model

llm = LLM(OCR_MODEL)
ocr_instruction = olmocr_prompt()

####################################################################################################
# Helper: get page tag bounds in text

def _page_bounds(text: str, page_number: int) -> tuple[int, int] | None:
    """Return (content_start, content_end) for <page number="N">…</page>."""
    open_tag = f'<page number="{page_number}">'
    close_tag = "</page>"
    start = text.find(open_tag)
    if start == -1:
        return None
    content_start = start + len(open_tag)
    content_end = text.find(close_tag, content_start)
    if content_end == -1:
        return None
    return content_start, content_end


def _renumber_tables(text: str) -> str:
    """Re-assign sequential <table number="N"> IDs across the full document."""
    counter = itercount(1)
    return re.sub(
        r'<table(?:\s+number="\d+")?>',
        lambda _: f'<table number="{next(counter)}">',
        text,
    )

####################################################################################################
# Process each document

for text_filepath, page_entries in by_text_file.items():
    # Load existing OCR text
    with open(text_filepath, "r", encoding="utf-8") as f:
        doc_text = f.read()

    # Sort entries by page number so splicing offsets stay consistent when
    # iterating from last page to first (avoids offset drift after each splice).
    page_entries_sorted = sorted(page_entries, key=lambda e: e["page_number"], reverse=True)

    # Build one OCR message per flagged page, then batch them all at once.
    pdf_filepath = page_entries[0]["pdf_filepath"]  # same PDF for all entries in group
    messages = []
    page_numbers = []

    for entry in sorted(page_entries, key=lambda e: e["page_number"]):
        page_number = entry["page_number"]
        # OCR page_number is 0-indexed; load_pdf_page uses 1-indexed PDF pages.
        pdf_page_num = page_number + 1
        try:
            pil_image = load_pdf_page(pdf_filepath, pdf_page_num, TARGET_DIM)
            pil_image = correct_image_orientation(pil_image)
            b64_image = encode_pil_image(pil_image)
        except Exception as exc:
            print(f"  Could not render page {page_number} of {pdf_filepath}: {exc}")
            continue

        image_data_uri = f"data:image/png;base64,{b64_image}"
        message = [
            {"role": "system", "content": ocr_instruction},
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": image_data_uri}}],
            },
        ]
        messages.append(message)
        page_numbers.append(page_number)

    if not messages:
        continue

    print(f"  {os.path.basename(text_filepath)}: re-OCR-ing pages {page_numbers}")
    responses = llm.chat(messages=messages, sampling_params=SAMPLING_PARAMS)

    # Splice each new page text back in, working from highest page number down
    # to keep string offsets valid across iterations.
    new_page_texts: dict[int, str] = {}
    for page_number, response in zip(page_numbers, responses):
        raw = response.outputs[0].text
        # Strip olmOCR front-matter (YAML block between --- markers)
        cleaned = re.sub(r"^---[\s\S]*?---\s*", "", raw)
        new_page_texts[page_number] = cleaned.strip()

    for page_number in sorted(new_page_texts, reverse=True):
        new_content = new_page_texts[page_number]
        bounds = _page_bounds(doc_text, page_number)
        if bounds is None:
            print(f"  Warning: could not locate page {page_number} in {text_filepath}")
            continue
        content_start, content_end = bounds
        doc_text = doc_text[:content_start] + "\n" + new_content + "\n" + doc_text[content_end:]

    # Re-number tables globally so IDs remain contiguous after splicing
    doc_text = _renumber_tables(doc_text)

    # Back up original and write corrected text
    shutil.copy2(text_filepath, text_filepath + ".pre_reocr.bak")
    with open(text_filepath, "w", encoding="utf-8") as f:
        f.write(doc_text)

    print(f"  Saved → {text_filepath}")

print("\nRe-OCR complete.")
print("Next step: run table_correction_vllm.py")
