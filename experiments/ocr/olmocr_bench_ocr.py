import os
import glob
import json
import base64
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

from PIL import Image

# OlmOCR specific prompt
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)

from vllm import LLM, SamplingParams

####################################################################################################
# Configuration

OUTPUT_DIR = "data/ocr/olmocr_bench_ocr_output"
IMAGE_DIR = "data/ocr/olmocr_bench_images"
PDF_DIR = "data/ocr/olmocr_bench_dataset/pdfs"
NUM_PAGES = 5000  # Process up to this many pages
BATCH_SIZE = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

OCR_PROMPT_TEXT = olmocr_prompt()

####################################################################################################
# Step 1: Render PDF pages to images

def render_pdf_to_image(pdf_path, dpi=144):
    """Render a single-page PDF to a PIL Image using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        doc.close()
        return None
    page = doc[0]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    doc.close()
    return img


def render_and_save_pages():
    """
    Load PDFs directly from PDF_DIR, render each as a PNG image,
    and save to IMAGE_DIR. Returns list of basenames that were rendered.
    """
    print(f"Loading PDFs from {PDF_DIR}...")
    if not os.path.isdir(PDF_DIR):
        raise FileNotFoundError(
            f"PDF directory not found at '{PDF_DIR}'. "
            f"Download the dataset first with:\n"
            f"  huggingface-cli download allenai/olmOCR-mix-0225-benchmarkset "
            f"--repo-type dataset --local-dir data/ocr/olmocr_bench_dataset"
        )

    # Find all PDF files
    pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "**", "*.pdf"), recursive=True))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDF files found in '{PDF_DIR}'")

    print(f"  Found {len(pdf_paths)} PDF files")

    # Limit to NUM_PAGES
    pdf_paths = pdf_paths[:NUM_PAGES]

    rendered = []
    skipped = 0

    for idx, pdf_path in enumerate(pdf_paths):
        try:
            # Use the PDF filename (without extension) as the basename
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            basename = f"olmocr_bench_{pdf_name}"

            image_path = os.path.join(IMAGE_DIR, f"{basename}.png")
            ocr_path = os.path.join(OUTPUT_DIR, f"{basename}.txt")

            # Skip if already rendered and OCR'd
            if os.path.exists(image_path) and os.path.exists(ocr_path):
                rendered.append(basename)
                continue

            # Skip if already rendered (but not yet OCR'd)
            if os.path.exists(image_path):
                rendered.append(basename)
                if (idx + 1) % 200 == 0:
                    print(f"  Processed {idx + 1}/{len(pdf_paths)} PDFs ({len(rendered)} rendered)...")
                continue

            # Render the PDF page
            img = render_pdf_to_image(pdf_path)
            if img is None:
                skipped += 1
                continue

            # Save the image
            img.save(image_path, format='PNG')
            rendered.append(basename)

            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(pdf_paths)} PDFs ({len(rendered)} rendered)...")

        except Exception as e:
            print(f"  Skipping {pdf_path}: {e}")
            skipped += 1
            continue

    print(f"  Total rendered: {len(rendered)}, skipped: {skipped}")
    return rendered


####################################################################################################
# Step 2: Run olmOCR on the rendered images

def run_ocr(basenames):
    """Run olmOCR on all rendered images that don't already have cached output."""

    # Identify which ones still need OCR
    to_process = []
    for basename in basenames:
        ocr_path = os.path.join(OUTPUT_DIR, f"{basename}.txt")
        if os.path.exists(ocr_path):
            continue
        image_path = os.path.join(IMAGE_DIR, f"{basename}.png")
        if not os.path.exists(image_path):
            continue
        to_process.append(basename)

    if len(to_process) == 0:
        print("All pages already have OCR output. Nothing to do.")
        return

    print(f"Running olmOCR on {len(to_process)} pages...")

    vlm = LLM("allenai/olmOCR-2-7B-1025-FP8")
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=8192,
        seed=342,
    )

    # Process in batches
    total_success = 0
    total_failed = 0

    for batch_start in range(0, len(to_process), BATCH_SIZE):
        batch_end = min(batch_start + BATCH_SIZE, len(to_process))
        batch_names = to_process[batch_start:batch_end]

        print(f"  Batch {batch_start}-{batch_end} ({len(batch_names)} pages)...")

        # Build messages
        messages = []
        valid_names = []
        for basename in batch_names:
            try:
                image_path = os.path.join(IMAGE_DIR, f"{basename}.png")
                with open(image_path, 'rb') as f:
                    image_bytes = f.read()

                b64_image = base64.b64encode(image_bytes).decode('utf-8')
                image_data_uri = f'data:image/png;base64,{b64_image}'

                msg = [
                    {"role": "system", "content": OCR_PROMPT_TEXT},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_uri}
                            }
                        ],
                    },
                ]
                messages.append(msg)
                valid_names.append(basename)
            except Exception as e:
                print(f"    Failed to load image for {basename}: {e}")
                total_failed += 1

        if len(messages) == 0:
            continue

        # Run inference
        try:
            responses = vlm.chat(messages=messages, sampling_params=sampling_params)
        except Exception as e:
            print(f"    Batch inference failed: {e}")
            total_failed += len(valid_names)
            continue

        # Save results
        for i, response in enumerate(responses):
            try:
                text = response.outputs[0].text
                if not text or len(text.strip()) == 0:
                    print(f"    Empty output for {valid_names[i]}, skipping.")
                    total_failed += 1
                    continue

                ocr_path = os.path.join(OUTPUT_DIR, f"{valid_names[i]}.txt")
                with open(ocr_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                total_success += 1

            except Exception as e:
                print(f"    Failed to save result for {valid_names[i]}: {e}")
                total_failed += 1

    print(f"\nOCR complete: {total_success} succeeded, {total_failed} failed")
    print(f"Results saved to {OUTPUT_DIR}/")


####################################################################################################
# Main

if __name__ == '__main__':
    basenames = render_and_save_pages()
    run_ocr(basenames)