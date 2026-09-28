"""Status report for the 2026-09-27 judge batch across pond/nfix/supermat.

Standalone diagnostic, NOT part of the contract-standard runner/submit.sh
path -- reads existing experiment-configs and experiments/results/, submits
nothing, modifies nothing. For each extraction id covered by the batch's
judge_local/judge_interp/judge_combine configs, reports:
  - whether the underlying extraction/ablation result exists yet (the two
    pond/nfix ablation1-llama8b-reppen-noqualifiers ids are the known
    still-pending case as of 2026-09-27)
  - per-judge status: NOT_STARTED / STALE (responses.json predates the
    JUDGE_INSTRUCTIONS value-matching fix, commit db9b97d) / DONE (at or
    after db9b97d)
  - combine status: NOT_STARTED / DONE / "STALE (inputs not all fresh)" --
    derived from its 6 input judges' freshness, since run_judge_combine.py
    never writes run_metadata.json itself (observed, not fixed here, same
    reason judge_prompts.py's silent document_id skip wasn't touched --
    see notes/scholarlm/experiments/2026-09-27-judge-batch-01.md).

"Stale" is determined by comparing run_metadata.json's recorded commit
against db9b97d via `git merge-base --is-ancestor`. Bump PROMPT_FIX_COMMIT
here if a later commit changes judge-scoring logic again in a way that
should invalidate prior runs.

Usage: python experiments/judge_batch_status.py [--json]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
import utils as paths  # noqa: E402

PROMPT_FIX_COMMIT = "db9b97d"
DATASETS = ["pond", "nfix", "supermat"]


def is_at_or_after_fix(commit: str) -> bool:
    if commit == PROMPT_FIX_COMMIT:
        return True
    r = subprocess.run(
        ["git", "-C", str(_REPO_ROOT), "merge-base", "--is-ancestor", PROMPT_FIX_COMMIT, commit],
        capture_output=True,
    )
    return r.returncode == 0


def judge_status(dataset: str, experiment_type: str, config_id: str) -> str:
    try:
        result_dir = paths.result_dir(dataset, experiment_type, config_id)
    except Exception:
        return "NOT_STARTED"
    if experiment_type == "judge_combine":
        return "EXISTS" if (result_dir / "combined.json").exists() else "NOT_STARTED"
    resp = result_dir / "responses.json"
    meta = result_dir / "run_metadata.json"
    if not resp.exists() or not meta.exists():
        return "NOT_STARTED"
    with open(meta) as f:
        commit = json.load(f).get("commit")
    if commit is None:
        return "UNKNOWN_COMMIT"
    return "DONE" if is_at_or_after_fix(commit) else "STALE"


def discover(dataset: str) -> dict[str, dict]:
    """{extraction_id: {"judges": [(type, config_id), ...], "combine": config_id, "extraction_exists": bool}}"""
    judge_local_dir = _REPO_ROOT / "experiments/experiment-configs" / dataset / "judge_local"
    judge_interp_dir = _REPO_ROOT / "experiments/experiment-configs" / dataset / "judge_interp"
    combine_dir = _REPO_ROOT / "experiments/experiment-configs" / dataset / "judge_combine"

    by_extraction: dict[str, list[tuple[str, str]]] = {}
    for etype, d in (("judge_local", judge_local_dir), ("judge_interp", judge_interp_dir)):
        for cfg_path in sorted(d.glob("2026-09-27-*/2026-09-27-*.yaml")):
            cfg = yaml.safe_load(cfg_path.read_text())
            eid = cfg["params"].get("extraction_id")
            if eid is None:
                # Phase 1/2 smoke-test configs (synthetic_file mode, no
                # extraction_id) share the same 2026-09-27-*/ glob -- not
                # part of the real batch, skip them.
                continue
            by_extraction.setdefault(eid, []).append((etype, cfg["id"]))

    combine_by_extraction: dict[str, str] = {}
    for cfg_path in sorted(combine_dir.glob("2026-09-27-*/2026-09-27-*.yaml")):
        cfg = yaml.safe_load(cfg_path.read_text())
        judge_ids = set(cfg["params"]["judge_ids"])
        for eid, entries in by_extraction.items():
            if judge_ids == {cid for _, cid in entries}:
                combine_by_extraction[eid] = cfg["id"]
                break

    result = {}
    for eid, entries in by_extraction.items():
        try:
            paths.find_result_dir(eid)
            extraction_exists = True
        except FileNotFoundError:
            extraction_exists = False
        result[eid] = {
            "judges": sorted(entries),
            "combine": combine_by_extraction.get(eid),
            "extraction_exists": extraction_exists,
        }
    return result


def build_report() -> dict[str, dict]:
    report: dict[str, dict] = {}
    for dataset in DATASETS:
        plan = discover(dataset)
        report[dataset] = {}
        for eid, spec in sorted(plan.items()):
            judge_statuses = {cid: judge_status(dataset, etype, cid) for etype, cid in spec["judges"]}
            combine_exists = (
                judge_status(dataset, "judge_combine", spec["combine"]) == "EXISTS"
                if spec["combine"] else False
            )
            if not spec["combine"]:
                combine_status = "NO_COMBINE_CONFIG"
            elif not combine_exists:
                combine_status = "NOT_STARTED"
            elif all(v == "DONE" for v in judge_statuses.values()):
                combine_status = "DONE"
            else:
                combine_status = "STALE (inputs not all fresh)"
            report[dataset][eid] = {
                "extraction_exists": spec["extraction_exists"],
                "judges": judge_statuses,
                "combine": combine_status,
            }
    return report


def main() -> None:
    report = build_report()
    if "--json" in sys.argv:
        print(json.dumps(report, indent=2))
        return

    for dataset, ids in report.items():
        print(f"\n## {dataset}")
        print("| extraction id | extraction | judges (done/stale/not-started) | combine |")
        print("|---|---|---|---|")
        for eid, s in ids.items():
            counts: dict[str, int] = {}
            for st in s["judges"].values():
                counts[st] = counts.get(st, 0) + 1
            judge_summary = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
            ext_mark = "yes" if s["extraction_exists"] else "**NOT FINISHED**"
            print(f"| {eid} | {ext_mark} | {judge_summary} | {s['combine']} |")


if __name__ == "__main__":
    main()
