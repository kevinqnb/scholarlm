"""
One-off script to build directory.json for the supermat dataset.

Pulls bibliographic data (title, authors, year) for each of the 142 local PDFs
from the SuperMat repo's biblio metadata files
(``/Users/quinn/research/coastal/SuperMat/data/biblio/batch-*/``), preferring
the richer plain ``*.json`` files (``title_main_a``) and falling back to the
``*.tei.json`` files (``title_a``) where the plain json is absent. Documents
with neither source get ``title``/``author``/``year`` set to ``None``.

Not part of the permanent pipeline -- run once to (re)generate directory.json.

Usage
-----
    python data/supermat/build_directory.py
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

BASE = Path(__file__).parent  # data/supermat/
BIBLIO_ROOT = Path("/Users/quinn/research/coastal/SuperMat/data/biblio")


def _load_bib(document_id: str) -> dict | None:
    """Return {"title": ..., "author": ..., "year": ...} for document_id, or None."""
    plain_matches = glob.glob(str(BIBLIO_ROOT / "batch-*" / f"{document_id}.json"))
    tei_matches = glob.glob(str(BIBLIO_ROOT / "batch-*" / f"{document_id}.tei.json"))

    data = None
    title_key = None
    if plain_matches:
        with open(plain_matches[0]) as f:
            data = json.load(f)
        title_key = "title_main_a"
    elif tei_matches:
        with open(tei_matches[0]) as f:
            data = json.load(f)
        title_key = "title_a"

    if data is None:
        return None

    title = data.get(title_key)
    authors = data.get("authors") or []
    author = "; ".join(
        a["surname"].lower() for a in authors if a.get("surname")
    ) or None
    date = data.get("date")
    year = None
    if date:
        try:
            year = int(str(date)[:4])
        except ValueError:
            year = None

    return {
        "title": title.lower() if title else None,
        "author": author,
        "year": year,
    }


def main() -> None:
    pdf_dir = BASE / "pdfs"
    document_ids = sorted(f[:-4] for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf"))
    print(f"Found {len(document_ids):,} local PDFs")

    directory: dict[str, dict] = {}
    n_with_bib = 0
    for doc_id in document_ids:
        bib = _load_bib(doc_id)
        if bib is None:
            directory[doc_id] = {"title": None, "author": None, "year": None}
        else:
            directory[doc_id] = bib
            n_with_bib += 1

    print(f"  {n_with_bib:,}/{len(document_ids):,} documents have bibliographic data")

    out_path = BASE / "directory.json"
    with open(out_path, "w") as f:
        json.dump(directory, f, indent=4, ensure_ascii=False, sort_keys=True)
    print(f"Saved directory.json -> {out_path}")


if __name__ == "__main__":
    main()
