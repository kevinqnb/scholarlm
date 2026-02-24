"""
Backfill `page_number` for table-sourced entries in an existing extraction file.

For each entry that has a `table_number` but no `page_number`, the script finds
the table tag inside the corresponding full document and walks back to the nearest
enclosing `<page number="N">` tag to determine the correct page.

Usage:
    uv run python experiments/pond/backfill_page_numbers.py
"""

import os
import re
import json

from scholarlm.utils import get_filenames_in_directory

main_directory = "data/pond"
ocr_directory = os.path.join(main_directory, "ocr_output_cleaned_openai")

input_file = "data/experiments/2026_02_25/pond_openai.json"
output_file = "data/experiments/2026_02_25/pond_openai.json"


def find_table_page(document: str, table_number: int) -> int | None:
    """Return the page number that contains the given table in *document*.

    Scans all <page number="N"> tag positions and returns the page number of
    the last such tag that precedes the <table number="table_number"> tag.
    """
    table_tag = f'<table number="{table_number}">'
    table_pos = document.find(table_tag)
    if table_pos == -1:
        return None

    page_number = None
    for m in re.finditer(r'<page number="(\d+)">', document):
        if m.start() > table_pos:
            break
        page_number = int(m.group(1))

    return page_number


if __name__ == "__main__":
    # Load full documents in the same sorted order used during extraction.
    text_files = get_filenames_in_directory(ocr_directory, ignore=[".DS_Store", ".gitkeep"])
    text_files.sort()
    documents: list[str] = []
    for fname in text_files:
        with open(os.path.join(ocr_directory, fname), "r", encoding="utf-8") as f:
            documents.append(f.read())

    with open(input_file, "r") as f:
        data = json.load(f)

    # Cache (doc_id, table_number) -> page_number to avoid redundant searches.
    cache: dict[tuple[int, int], int | None] = {}

    n_filled = 0
    n_missing = 0
    for entry in data:
        if entry.get("page_number") is not None:
            continue
        table_number = entry.get("table_number")
        if table_number is None:
            continue

        doc_id = int(entry["document_id"])
        key = (doc_id, int(table_number))

        if key not in cache:
            cache[key] = find_table_page(documents[doc_id], int(table_number))

        page = cache[key]
        if page is not None:
            entry["page_number"] = page
            n_filled += 1
        else:
            n_missing += 1

    print(f"Filled page_number for {n_filled} entries.")
    if n_missing:
        print(f"Could not resolve page for {n_missing} entries (table tag not found in document).")

    with open(output_file, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"Saved to {output_file}.")
