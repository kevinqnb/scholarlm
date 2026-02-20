import os
import re
import json
import tarfile
import base64
import random
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset, DatasetDict, Image as HFImage
from PIL import Image

# OlmOCR prompt
from olmocr.prompts import build_no_anchoring_v4_yaml_prompt as olmocr_prompt

random.seed(342)

####################################################################################################
# Configuration

TARBALL_PATH = "data/ocr/PubTables-1M-Detection_Images_Train_Part1.tar.gz"
CLEAN_OUTPUT_DIR = "data/ocr/pubtables_ocr_output_clean"
BENCH_OCR_DIR = "data/ocr/olmocr_bench_ocr_output"
BENCH_IMAGE_DIR = "data/ocr/olmocr_bench_images"
DATASET_OUTPUT_DIR = "data/ocr/finetune_dataset"

# Ratio: 80% table examples, 20% non-table examples
TABLE_FRACTION = 0.80
NON_TABLE_FRACTION = 0.20
VAL_SPLIT = 0.05  # 5% held out for validation

os.makedirs(DATASET_OUTPUT_DIR, exist_ok=True)

OCR_PROMPT_TEXT = olmocr_prompt()

####################################################################################################
# Quality filters

def has_valid_table(text):
    """Check if text contains valid HTML table tags."""
    has_open = bool(re.search(r'<table', text, re.IGNORECASE))
    has_close = bool(re.search(r'</table>', text, re.IGNORECASE))
    return has_open and has_close


def is_truncated(text, max_tokens_approx=16000):
    """Heuristic check for truncated output."""
    open_count = len(re.findall(r'<table', text, re.IGNORECASE))
    close_count = len(re.findall(r'</table>', text, re.IGNORECASE))
    if open_count > close_count:
        return True
    if len(text) > max_tokens_approx * 4 * 0.95:
        return True
    return False


####################################################################################################
# Part 1: Load table examples (pubtables images + Gemma-cleaned text)

def load_table_examples():
    """
    Load (image_bytes, cleaned_text) pairs from the pubtables tarball + cleaned output.
    Filters for quality: valid HTML tables, not truncated.
    """
    print("Loading table examples...")

    # Get all cleaned text files
    cleaned_files = {}
    for fname in sorted(os.listdir(CLEAN_OUTPUT_DIR)):
        if fname.endswith('.txt'):
            basename = fname[:-4]
            filepath = os.path.join(CLEAN_OUTPUT_DIR, fname)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            cleaned_files[basename] = text

    print(f"  Found {len(cleaned_files)} cleaned text files")

    # Filter for quality
    valid_basenames = {}
    for basename, text in cleaned_files.items():
        if not has_valid_table(text):
            continue
        if is_truncated(text):
            continue
        if len(text.strip()) < 50:
            continue
        valid_basenames[basename] = text

    print(f"  {len(valid_basenames)} passed quality filters")

    # Extract corresponding images from tarball
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    basenames_set = set(valid_basenames.keys())
    examples = []

    with tarfile.open(TARBALL_PATH, 'r:gz') as tar:
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

                # Verify the image is valid
                img = Image.open(BytesIO(image_bytes))
                img.verify()

                examples.append({
                    'basename': basename,
                    'image_bytes': image_bytes,
                    'text': valid_basenames[basename],
                    'source': 'pubtables',
                    'has_table': True,
                })

                if len(examples) % 500 == 0:
                    print(f"  Collected {len(examples)} table examples...")

            except Exception as e:
                print(f"  Skipping {member.name}: {e}")
                continue

    print(f"  Total table examples: {len(examples)}")
    return examples


####################################################################################################
# Part 2: Load non-table examples (olmOCR benchmark images + OCR text)

def load_non_table_examples(max_count):
    """
    Load non-table examples from pre-computed olmOCR benchmark results.
    Reads rendered images from BENCH_IMAGE_DIR and OCR text from BENCH_OCR_DIR.
    Filters OUT any pages that contain tables.
    """
    print(f"Loading non-table examples (target: {max_count})...")

    # Find all OCR output files that have a corresponding image
    examples = []
    skipped_table = 0
    skipped_empty = 0
    skipped_no_image = 0

    ocr_files = sorted(os.listdir(BENCH_OCR_DIR))
    for fname in ocr_files:
        if len(examples) >= max_count:
            break

        if not fname.endswith('.txt'):
            continue

        basename = fname[:-4]

        # Check for corresponding image
        image_path = os.path.join(BENCH_IMAGE_DIR, f"{basename}.png")
        if not os.path.exists(image_path):
            skipped_no_image += 1
            continue

        # Read OCR text
        ocr_path = os.path.join(BENCH_OCR_DIR, fname)
        with open(ocr_path, 'r', encoding='utf-8') as f:
            text = f.read()

        # Filter: skip empty/short
        if not text or len(text.strip()) < 20:
            skipped_empty += 1
            continue

        # Filter: skip pages that contain tables
        if has_valid_table(text):
            skipped_table += 1
            continue

        # Read the image
        try:
            with open(image_path, 'rb') as f:
                image_bytes = f.read()

            # Verify
            img = Image.open(BytesIO(image_bytes))
            img.verify()

            examples.append({
                'basename': basename,
                'image_bytes': image_bytes,
                'text': text,
                'source': 'olmocr_benchmark',
                'has_table': False,
            })

        except Exception as e:
            print(f"  Skipping {basename}: {e}")
            continue

    print(f"  Total non-table examples: {len(examples)}")
    print(f"  Skipped (had tables): {skipped_table}, "
          f"Skipped (empty): {skipped_empty}, "
          f"Skipped (no image): {skipped_no_image}")
    return examples


####################################################################################################
# Part 3: Combine, format, and save as HuggingFace Dataset

def build_chat_messages(ocr_prompt, image_bytes):
    """
    Build the chat-format messages that olmOCR expects (system + user with image).
    The assistant response (target) is added separately.
    """
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    image_data_uri = f'data:image/png;base64,{b64_image}'

    messages = [
        {"role": "system", "content": ocr_prompt},
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
    return messages


def main():
    # Load table examples
    table_examples = load_table_examples()

    if len(table_examples) == 0:
        print("ERROR: No valid table examples found. Check CLEAN_OUTPUT_DIR and TARBALL_PATH.")
        return

    # Calculate how many non-table examples we need for the 80/20 ratio
    n_table = len(table_examples)
    n_non_table = int(n_table * NON_TABLE_FRACTION / TABLE_FRACTION)
    print(f"\nTarget ratio: {n_table} table ({TABLE_FRACTION*100:.0f}%) + "
          f"{n_non_table} non-table ({NON_TABLE_FRACTION*100:.0f}%)")

    # Load non-table examples
    non_table_examples = load_non_table_examples(max_count=n_non_table)

    # Combine all examples
    all_examples = table_examples + non_table_examples
    random.shuffle(all_examples)
    print(f"\nTotal combined examples: {len(all_examples)}")

    # Format into training records
    print("Formatting into training records...")
    records = []
    for i, ex in enumerate(all_examples):
        try:
            messages = build_chat_messages(OCR_PROMPT_TEXT, ex['image_bytes'])

            # Add the assistant response (the target output)
            messages.append({
                "role": "assistant",
                "content": ex['text'],
            })

            record = {
                'id': ex['basename'],
                'messages': json.dumps(messages),
                'source': ex['source'],
                'has_table': ex['has_table'],
                'image': ex['image_bytes'],
            }
            records.append(record)

            if (i + 1) % 500 == 0:
                print(f"  Formatted {i + 1}/{len(all_examples)}")

        except Exception as e:
            print(f"  Failed to format {ex['basename']}: {e}")
            continue

    print(f"  Total formatted records: {len(records)}")

    # Split into train/val
    n_val = max(1, int(len(records) * VAL_SPLIT))
    val_records = records[:n_val]
    train_records = records[n_val:]

    print(f"  Train: {len(train_records)}, Val: {len(val_records)}")

    # Count table vs non-table in each split
    train_table = sum(1 for r in train_records if r['has_table'])
    val_table = sum(1 for r in val_records if r['has_table'])
    print(f"  Train tables: {train_table}, Train non-tables: {len(train_records) - train_table}")
    print(f"  Val tables: {val_table}, Val non-tables: {len(val_records) - val_table}")

    # Create HuggingFace Dataset
    def records_to_dataset(recs):
        return Dataset.from_dict({
            'id': [r['id'] for r in recs],
            'messages': [r['messages'] for r in recs],
            'source': [r['source'] for r in recs],
            'has_table': [r['has_table'] for r in recs],
            'image': [r['image'] for r in recs],
        })

    dataset = DatasetDict({
        'train': records_to_dataset(train_records),
        'validation': records_to_dataset(val_records),
    })

    # Cast the image column
    dataset = dataset.cast_column('image', HFImage())

    # Save
    dataset.save_to_disk(DATASET_OUTPUT_DIR)
    print(f"\nDataset saved to {DATASET_OUTPUT_DIR}/")

    # Save metadata summary
    metadata = {
        'total_examples': len(records),
        'train_size': len(train_records),
        'val_size': len(val_records),
        'train_table_count': train_table,
        'train_non_table_count': len(train_records) - train_table,
        'val_table_count': val_table,
        'val_non_table_count': len(val_records) - val_table,
        'table_fraction': TABLE_FRACTION,
        'non_table_fraction': NON_TABLE_FRACTION,
        'ocr_prompt': OCR_PROMPT_TEXT,
    }
    with open(os.path.join(DATASET_OUTPUT_DIR, 'metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved.")


if __name__ == '__main__':
    main()