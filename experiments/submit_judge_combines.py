"""
Submit judge_combine experiments once all of their judges have finished.

For every named judge_combine id (or every one matching --glob), reads the
combine config's own ``params.judge_ids`` and asserts, BEFORE submitting
anything, that each listed judge run has a ``responses.json`` and a
``run_metadata.json`` whose ``judge_model`` is one of the three voting judges,
with all three present exactly once. If any combine id is not ready the script
exits non-zero listing everything missing and submits nothing -- no partial
batches, no skipping.

Submission goes through the normal wrapper (``experiments/submit.sh <id>``).

Usage
-----
    python experiments/submit_judge_combines.py --dry-run <combine-id> [<combine-id> ...]
    python experiments/submit_judge_combines.py --glob '2026-10-04-*-ablation[2-5]-*-judge-combine-01' [--dry-run]

``--dry-run`` runs the readiness checks and prints the qsub command via
``submit.sh --dry-run``; it submits nothing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import utils as paths

EXPECTED_JUDGES = {"gpt-oss-120b", "llama-3.3-70b", "qwen-2.5-72b"}


def _combine_config_paths(ids: list[str], glob: str | None) -> list[Path]:
    root = _REPO_ROOT / "experiments" / "experiment-configs"
    if glob is not None:
        found = sorted(root.glob(f"*/judge_combine/{glob}/{glob}.yaml"))
        if not found:
            raise FileNotFoundError(f"--glob {glob!r} matched no judge_combine config")
        return found
    out = []
    for cid in ids:
        matches = sorted(root.glob(f"*/judge_combine/{cid}/{cid}.yaml"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected exactly one judge_combine config for {cid!r}, found {matches}"
            )
        out.append(matches[0])
    return out


def _check_ready(config_path: Path) -> list[str]:
    """Return a list of problems for this combine config (empty == ready)."""
    cfg = paths.load_experiment_config(config_path)
    judge_ids = cfg["params"]["judge_ids"]
    problems: list[str] = []
    seen_models: list[str] = []
    for jid in judge_ids:
        try:
            judge_dir = paths.find_result_dir(jid)
        except FileNotFoundError:
            problems.append(f"{cfg['id']}: judge {jid} has no results dir")
            continue
        if not (judge_dir / "responses.json").exists():
            problems.append(f"{cfg['id']}: judge {jid} has no responses.json")
            continue
        meta = paths.load_run_metadata(judge_dir)
        if meta is None or "judge_model" not in meta:
            problems.append(f"{cfg['id']}: judge {jid} has no run_metadata judge_model")
            continue
        seen_models.append(meta["judge_model"])
    if not problems and sorted(seen_models) != sorted(EXPECTED_JUDGES):
        problems.append(
            f"{cfg['id']}: judge models {sorted(seen_models)} != expected {sorted(EXPECTED_JUDGES)}"
        )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="judge_combine experiment ids")
    ap.add_argument("--glob", help="id glob (matched against judge_combine ids) instead of explicit ids")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if bool(args.ids) == (args.glob is not None):
        ap.error("give either explicit ids or --glob, not both/neither")

    configs = _combine_config_paths(args.ids, args.glob)
    problems = [p for c in configs for p in _check_ready(c)]
    if problems:
        print(f"{len(problems)} problem(s); nothing submitted:", file=sys.stderr)
        for p in problems:
            print(f"  {p}", file=sys.stderr)
        sys.exit(1)

    for c in configs:
        cmd = ["bash", str(_REPO_ROOT / "experiments" / "submit.sh"), c.parent.name]
        if args.dry_run:
            cmd.append("--dry-run")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
