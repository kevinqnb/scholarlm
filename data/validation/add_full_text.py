"""Backfill ``documents[doc]["full_text"]`` into an existing validation_set.json.

``create_validation_set.py`` now writes the whole OCR document alongside the
per-page bodies, but the committed ``data/{pond,nfix,supermat}/validation_set.json``
files predate that and the manual-validation site's live DB keys judgements on
``(dataset, measurement_id)`` -- re-running ``create_validation_set.py`` would
redraw ``rng.sample`` and point existing judgements at the wrong rows.

This script is the safe path: it loads a validation_set.json, reads its own
recorded ``ocr_dir``, injects ``full_text`` for every document already present,
and writes the file back. Nothing re-samples; ``measurements`` is untouched.

Usage::

    python data/validation/add_full_text.py data/pond/validation_set.json
    python data/validation/add_full_text.py data/{pond,nfix,supermat}/validation_set.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def enrich(path: Path, *, repo_root: Path = _REPO_ROOT) -> tuple[int, int]:
    """Add full_text to every document in ``path``. Returns (n_docs, n_added)."""
    data = json.loads(path.read_text())
    ocr_dir = repo_root / data["ocr_dir"]
    assert ocr_dir.is_dir(), f"{path.name}: ocr_dir does not exist: {ocr_dir}"

    documents: dict = data["documents"]
    added = 0
    for doc_id, doc in sorted(documents.items()):
        txt_path = ocr_dir / f"{doc_id}.txt"
        assert txt_path.is_file(), f"{path.name}: no OCR file {txt_path}"
        full_text = txt_path.read_text()

        # Sanity: the per-page bodies already in the file must be substrings of
        # the whole document, or ocr_dir is pointing at the wrong OCR variant.
        for page_str, body in doc.get("pages", {}).items():
            assert body.strip() and body.strip() in full_text, (
                f"{path.name}: {doc_id} page {page_str} body not found in "
                f"{txt_path} -- wrong OCR dir?"
            )

        if doc.get("full_text") != full_text:
            added += 1
        doc["full_text"] = full_text

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    )
    print(
        f"{path.name}: {len(documents)} documents, full_text set "
        f"({added} changed), {path.stat().st_size / 1e6:.2f} MB",
        file=sys.stderr,
    )
    return len(documents), added


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", type=Path, help="validation_set.json file(s)")
    args = ap.parse_args(argv)
    for p in args.paths:
        enrich(p)


if __name__ == "__main__":
    main()
