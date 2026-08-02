"""
Download the MeasEval corpus (SemEval-2021 Task 8) into data/measeval/raw/.

MeasEval (https://github.com/harperco/MeasEval) ships plain text and TSV
annotations directly -- unlike pond/nfix/supermat there is no PDF/OCR step,
so this script is the entire "get the data" phase for this dataset.

Usage
-----
Run from the repo root:

    python data/measeval/download_measeval.py

Or from data/measeval/:

    python download_measeval.py

Pass --force to re-clone even if data/measeval/raw/ already exists.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

BASE = Path(__file__).parent  # data/measeval/
RAW_DIR = BASE / "raw"
REPO_URL = "https://github.com/harperco/MeasEval"


def download(force: bool = False) -> None:
    if RAW_DIR.exists():
        if not force:
            print(f"  {RAW_DIR} already exists -- skipping clone (use --force to re-clone).")
            return
        print(f"  Removing existing {RAW_DIR} (--force)...")
        shutil.rmtree(RAW_DIR)

    print(f"  Cloning {REPO_URL} -> {RAW_DIR} ...")
    subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(RAW_DIR)],
        check=True,
    )
    # Strip the nested .git so raw/ isn't treated as an embedded repo/submodule
    # by the outer repo (its contents are gitignored regardless).
    shutil.rmtree(RAW_DIR / ".git")
    print("  Done.")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Re-clone even if raw/ already exists.")
    args = parser.parse_args(argv)
    download(force=args.force)


if __name__ == "__main__":
    main()
