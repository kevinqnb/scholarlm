import os
import json
import tarfile
import base64
import io
from dotenv import load_dotenv
load_dotenv()
from scholarlm.utils import get_filenames_in_directory

# OlmOCR specfic prompt:
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

TARBALL_PATH = "data/ocr/PubTables-1M-Detection_Images_Train_Part1.tar.gz"
OUTPUT_DIR = "data/ocr/pubtables_ocr_output"
NUM_IMAGES = 10000
BATCH_SIZE = 500  # Process in batches to manage memory

os.makedirs(OUTPUT_DIR, exist_ok=True)

####################################################################################################
# Stream images from tar.gz and collect up to NUM_IMAGES

def stream_images_from_tar(tarball_path, max_images):
    """
    Stream image entries from a tar.gz file, yielding (filename, base64_encoded_image) tuples.
    Only yields files that look like images (png, jpg, jpeg, bmp, tiff).
    """
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    count = 0

    with tarfile.open(tarball_path, 'r:gz') as tar:
        for member in tar:
            if count >= max_images:
                break
            if not member.isfile():
                continue

            ext = os.path.splitext(member.name)[1].lower()
            if ext not in image_extensions:
                continue

            try:
                f = tar.extractfile(member)
                if f is None:
                    continue
                image_bytes = f.read()
                f.close()

                b64_image = base64.b64encode(image_bytes).decode('utf-8')

                # Determine MIME type
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

                # Use the basename without extension as the identifier
                basename = os.path.splitext(os.path.basename(member.name))[0]

                yield basename, b64_image, mime
                count += 1

            except Exception as e:
                print(f"Skipping {member.name}: {e}")
                continue

    print(f"Streamed {count} images from {tarball_path}")


####################################################################################################
# Set up the VLM

print("Loading olmOCR model...")
vlm = LLM("allenai/olmOCR-2-7B-1025-FP8")

sampling_params = SamplingParams(
    temperature=0.1,
    max_tokens=16384,
    seed=342,
)

ocr_prompt_text = olmocr_prompt()

####################################################################################################
# Process images in batches

print(f"Streaming up to {NUM_IMAGES} images from {TARBALL_PATH}...")

batch_names = []
batch_messages = []
total_processed = 0
total_failed = 0

image_stream = stream_images_from_tar(TARBALL_PATH, NUM_IMAGES)

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

            output_path = os.path.join(OUTPUT_DIR, f"{names[i]}.txt")
            with open(output_path, 'w', encoding='utf-8') as out_f:
                out_f.write(text)
            success += 1

        except Exception as e:
            print(f"Failed to process result for {names[i]}: {e}")
            failed += 1

    return success, failed


for name, b64_image, mime in image_stream:
    # Skip if already processed
    output_path = os.path.join(OUTPUT_DIR, f"{name}.txt")
    if os.path.exists(output_path):
        total_processed += 1
        continue

    image_data_uri = f'data:{mime};base64,{b64_image}'
    message = [
        {"role": "system", "content": ocr_prompt_text},
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

    batch_names.append(name)
    batch_messages.append(message)

    # Process when batch is full
    if len(batch_names) >= BATCH_SIZE:
        print(f"Processing batch of {len(batch_names)} images (total so far: {total_processed})...")
        n_success, n_failed = process_batch(batch_names, batch_messages)
        total_processed += n_success
        total_failed += n_failed
        print(f"  Batch done: {n_success} succeeded, {n_failed} failed")

        # Clear batch
        batch_names = []
        batch_messages = []

# Process any remaining images in the final partial batch
if len(batch_names) > 0:
    print(f"Processing final batch of {len(batch_names)} images...")
    n_success, n_failed = process_batch(batch_names, batch_messages)
    total_processed += n_success
    total_failed += n_failed
    print(f"  Batch done: {n_success} succeeded, {n_failed} failed")

print(f"\nDone! Processed {total_processed} images, {total_failed} failures.")
print(f"Results saved to {OUTPUT_DIR}/")


