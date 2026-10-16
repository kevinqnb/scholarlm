"""
Submit judge_local experiments in a controlled batch.

Takes an explicit list of judge_local ids (no globbing -- a batch is whatever
you name). Before submitting ANYTHING it asserts, for every id, that:
  - exactly one judge_local config exists for it,
  - its extraction_id has a non-empty final.json,
  - it has no results dir yet (never rerun over existing output),
  - it is not already in the SGE queue (qstat job name ``x<id>``).
Any violation exits non-zero listing all problems; nothing is submitted.
Submission goes through ``experiments/submit.sh`` one id at a time.

Usage
-----
    python experiments/submit_judges.py [--dry-run] <judge-id> [<judge-id> ...]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import utils as paths


def _queued_job_names() -> set[str]:
    xml = subprocess.run(["qstat", "-xml"], check=True, capture_output=True, text=True).stdout
    return {e.text for e in ET.fromstring(xml).iter("JB_name")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    assert len(set(args.ids)) == len(args.ids), "duplicate ids in batch"

    cfg_root = _REPO_ROOT / "experiments" / "experiment-configs"
    queued = _queued_job_names()
    problems: list[str] = []
    for jid in args.ids:
        matches = sorted(cfg_root.glob(f"*/judge_local/{jid}/{jid}.yaml"))
        if len(matches) != 1:
            problems.append(f"{jid}: expected exactly one judge_local config, found {matches}")
            continue
        cfg = paths.load_experiment_config(matches[0])
        ext_id = cfg["params"]["extraction_id"]
        try:
            final = paths.find_result_dir(ext_id) / "final.json"
        except FileNotFoundError:
            problems.append(f"{jid}: extraction {ext_id} has no results dir")
        else:
            if not final.exists() or final.stat().st_size == 0:
                problems.append(f"{jid}: extraction {ext_id} has no final.json")
        try:
            paths.find_result_dir(jid)
            problems.append(f"{jid}: results dir already exists")
        except FileNotFoundError:
            pass
        if f"x{jid}" in queued:
            problems.append(f"{jid}: already in the queue")
    if problems:
        print(f"{len(problems)} problem(s); nothing submitted:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    for jid in args.ids:
        cmd = ["bash", str(_REPO_ROOT / "experiments" / "submit.sh"), jid]
        if args.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
