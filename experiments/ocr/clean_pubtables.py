import os
import json
import tarfile
import base64
import re
from dotenv import load_dotenv
load_dotenv()

from scholarlm.instruction_prompts import CLEAN_TABLE_INSTRUCTIONS_V2

# (try to) set seeds for reproducibility
import random
import torch
random.seed(342)
torch.manual_seed(342)
torch.cuda.manual_seed(342)

from vllm import LLM, SamplingParams

####################################################################################################
# Configuration

TARBALL_PATH = "data/ocr/PubTables-1M-Detection_Images_Train_Part1.tar.gz"
OCR_INPUT_DIR = "data/ocr/pubtables_ocr_output"
CLEAN_OUTPUT_DIR = "data/ocr/pubtables_ocr_output_clean"
BATCH_SIZE = 500  # Process in batches to manage memory

os.makedirs(CLEAN_OUTPUT_DIR, exist_ok=True)

####################################################################################################
# Gather OCR output files that need cleaning

def get_ocr_files(ocr_dir):
    """Get list of (basename, filepath) tuples for all .txt files in ocr_dir."""
    results = []
    for fname in sorted(os.listdir(ocr_dir)):
        if fname.endswith('.txt'):
            basename = fname[:-4]  # strip .txt
            filepath = os.path.join(ocr_dir, fname)
            results.append((basename, filepath))
    return results


def extract_images_from_tar(tarball_path, basenames_needed):
    """
    Extract specific images from the tarball by matching basenames.
    Returns a dict mapping basename -> (b64_image, mime).
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    basenames_set = set(basenames_needed)
    results = {}

    with tarfile.open(tarball_path, 'r:gz') as tar:
        for member in tar:
            if not member.isfile():
                continue

            ext = os.path.splitext(member.name)[1].lower()
            if ext not in image_extensions:
                continue

            basename = os.path.splitext(os.path.basename(member.name))[0]
            if basename not in basenames_set:
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                image_bytes = f.read()
                f.close()

                b64_image = base64.b64encode(image_bytes).decode('utf-8')

                if ext in {'.jpg', '.jpeg'}:
                    mime = 'image/jpeg'
                elif ext == '.png':
                    mime = 'image/png'
                elif ext in {'.tiff', '.tif'}:
                    mime = 'image/tiff'
                elif ext == '.bmp':
                    mime = 'image/bmp'
                else:
                    mime = 'image/png'

                results[basename] = (b64_image, mime)

                # Stop early if we've found everything
                if len(results) == len(basenames_set):
                    break

            except Exception as e:
                print(f"Skipping {member.name}: {e}")
                continue

    print(f"Extracted {len(results)} images from {tarball_path}")
    return results


####################################################################################################
# Build work items: pair each OCR text with its source image

print("Scanning OCR output files...")
ocr_files = get_ocr_files(OCR_INPUT_DIR)

# Filter to files that haven't been cleaned yet
work_items = []
for basename, filepath in ocr_files:
    clean_output_path = os.path.join(CLEAN_OUTPUT_DIR, f"{basename}.txt")
    if os.path.exists(clean_output_path):
        continue  # already cleaned

    with open(filepath, 'r', encoding='utf-8') as f:
        text = f.read()

    work_items.append((basename, text))

print(f"Found {len(work_items)} files to clean")

if len(work_items) == 0:
    print("Nothing to do. Exiting.")
    exit(0)

# Extract the corresponding images from the tarball
basenames_needed = [basename for basename, _ in work_items]
print(f"Extracting {len(basenames_needed)} images from tarball...")
image_map = extract_images_from_tar(TARBALL_PATH, basenames_needed)

# Filter out work items where we couldn't find the source image
work_items_with_images = []
for basename, text in work_items:
    if basename in image_map:
        work_items_with_images.append((basename, text, image_map[basename]))
    else:
        print(f"Warning: no source image found for {basename}, skipping cleaning")

work_items = work_items_with_images
print(f"{len(work_items)} items ready for table cleaning")

####################################################################################################
# Set up the VLM for table cleaning

print("Loading Gemma model...")
vlm = LLM("gaunernst/gemma-3-27b-it-qat-autoawq")

sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=16384,
    seed=342,
)

####################################################################################################
# Process in batches

def build_clean_message(ocr_text, b64_image, mime):
    """
    Build a chat message that provides the CLEAN_TABLE_INSTRUCTIONS prompt,
    the source page image, and the OCR-parsed text with tables.
    """
    image_data_uri = f'data:{mime};base64,{b64_image}'
    message = [
        {"role": "system", "content": CLEAN_TABLE_INSTRUCTIONS_V2},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri
                    }
                },
                {
                    "type": "text",
                    "text": (
                        "Please follow the instructions to carefully clean the HTML table(s) found within the given OCR text.\n\n"
                        f"--- OCR Text ---\n{ocr_text}\n--- End OCR Text ---"
                    )
                }
            ],
        },
    ]
    return message


def process_batch(names, messages):
    """Run a batch through the VLM and save results. Returns (num_success, num_failed)."""
    success = 0
    failed = 0

    try:
        responses = vlm.chat(messages=messages, sampling_params=sampling_params)
    except Exception as e:
        print(f"Batch inference failed: {e}")
        return 0, len(names)

    for i, response in enumerate(responses):
        try:
            text = response.outputs[0].text
            if not text or len(text.strip()) == 0:
                print(f"Empty output for {names[i]}, skipping.")
                failed += 1
                continue

            output_path = os.path.join(CLEAN_OUTPUT_DIR, f"{names[i]}.txt")
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(text)
            success += 1

        except Exception as e:
            print(f"Failed to process result for {names[i]}: {e}")
            failed += 1

    return success, failed


total_processed = 0
total_failed = 0
batch_names = []
batch_messages = []

for basename, ocr_text, (b64_image, mime) in work_items:
    message = build_clean_message(ocr_text, b64_image, mime)
    batch_names.append(basename)
    batch_messages.append(message)

    if len(batch_names) >= BATCH_SIZE:
        print(f"Processing batch of {len(batch_names)} items (total so far: {total_processed})...")
        n_success, n_failed = process_batch(batch_names, batch_messages)
        total_processed += n_success
        total_failed += n_failed
        print(f"  Batch done: {n_success} succeeded, {n_failed} failed")
        batch_names = []
        batch_messages = []

# Process remaining items
if len(batch_names) > 0:
    print(f"Processing final batch of {len(batch_names)} items...")
    n_success, n_failed = process_batch(batch_names, batch_messages)
    total_processed += n_success
    total_failed += n_failed
    print(f"  Batch done: {n_success} succeeded, {n_failed} failed")

print(f"\nDone! Cleaned {total_processed} tables, {total_failed} failures.")
print(f"Results saved to {CLEAN_OUTPUT_DIR}/")