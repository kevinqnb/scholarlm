"""
Combine judge results across multiple models into a single ground-truth file.

Reads individual judge ``responses.json`` files for an explicit list of judge
experiment ids (each produced by run_judge_interp.py or run_judge_local.py),
merges them by ``measurement_id``, computes a majority-vote ground truth
label from the voting judges, and writes a combined output file.

Output path (id-addressed, like every other Tier-1 type):
    experiments/results/{dataset}/judge_combine/{experiment_id}/combined.json

Usage
-----
    python experiments/run_judge_combine.py experiments/experiment-configs/pond/judge_combine/<id>/<id>.yaml

Required params: dataset, judge_ids (list of judge_interp/judge_local
experiment ids -- explicit, no auto-discovery: under id-based addressing
there is no shared directory tree to scan, the config just names which
judge runs to combine).
Optional params: voting_threshold (default: majority of voting judges).

The combined JSON has one record per measurement with all individual judge
fields merged in (``judgement_{judge_key}``, ``judgement_prob_{judge_key}``,
etc.) plus a ``judgement_combined`` boolean ground truth field.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

# ---------------------------------------------------------------------------
# Voting judge keys — only these votes count toward ground truth.
# "human" is included so human-validation responses.json files (produced
# historically by experiments/validation.py, a Streamlit app, since removed)
# can participate in majority voting (use --voting-threshold 1 for standalone use).
# ---------------------------------------------------------------------------

VOTING_JUDGE_KEYS = {"gpt-oss-120b", "llama-3.3-70b", "qwen-2.5-72b"}

import utils as paths


def _load_judge_result(judge_id: str) -> tuple[str, Path]:
    """Resolve one judge experiment's responses.json and its judge_model key.

    Returns:
        (judge_model_key, path_to_responses_json).

    Raises:
        FileNotFoundError: If judge_id has no results directory or no
            responses.json.
        ValueError: If run_metadata.json is missing or has no judge_model
            field -- run_judge_interp.py/run_judge_local.py always write one,
            so this means judge_id doesn't point at a real judge run.
    """
    judge_dir = paths.find_result_dir(judge_id)
    responses_path = judge_dir / "responses.json"
    if not responses_path.exists():
        raise FileNotFoundError(f"No responses.json for judge_id={judge_id!r} under {judge_dir}")
    meta = paths.load_run_metadata(judge_dir)
    if meta is None or "judge_model" not in meta:
        raise ValueError(
            f"judge_id={judge_id!r}: no run_metadata.json (or no judge_model field in "
            f"it) under {judge_dir} -- cannot determine which judge model this run used."
        )
    return meta["judge_model"], responses_path


# ---------------------------------------------------------------------------
# Core combination logic
# ---------------------------------------------------------------------------


def combine_judge_results(
    judge_files: dict[str, Path],
    voting_judges: set[str],
    voting_threshold: int,
) -> list[dict]:
    """Merge judge result files and compute a majority-vote ground truth label.

    Args:
        judge_files: Mapping ``{judge_key: path_to_responses_json}``.
        voting_judges: Subset of judge keys whose votes count toward ground truth.
        voting_threshold: Minimum number of affirmative votes for
            ``judgement_combined`` to be ``True``.

    Returns:
        List of combined records, one per ``measurement_id``.
    """
    _JUDGE_FIELDS = {
        "judgement",
        "judgement_model",
        "judgement_prob",
        "judgement_p_true",
        "judgement_p_false",
        "judgement_logit_p_true",
        "judgement_logit_p_false",
        "judgement_raw_text",
    }

    combined: dict[int, dict] = {}

    for judge_key, filepath in judge_files.items():
        with open(filepath) as f:
            records: list[dict] = json.load(f)

        for entry in records:
            eid: int = entry["measurement_id"]
            # Base record (non-judge fields) — written on first encounter
            base = {k: v for k, v in entry.items() if k not in _JUDGE_FIELDS}
            # Judge-specific fields, namespaced by judge key
            judge_specific = {
                f"{k}_{judge_key}": v
                for k, v in entry.items()
                if k in _JUDGE_FIELDS and k != "judgement_model"
            }
            if eid not in combined:
                combined[eid] = base
            combined[eid].update(judge_specific)

    # Majority vote over voting judges
    result: list[dict] = []
    for record in combined.values():
        affirmative = sum(
            1 for j in voting_judges
            if record.get(f"judgement_{j}") is True
        )
        record["judgement_combined"] = affirmative >= voting_threshold
        result.append(record)

    return result


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_combine(
    judge_ids: list[str],
    output_dir: Path,
    voting_threshold: int | None = None,
) -> Path:
    """Combine multiple judge runs (named by experiment id) into one ground-truth file.

    Args:
        judge_ids: Explicit list of judge_interp/judge_local experiment ids to combine.
        output_dir: Directory to write ``combined.json`` to.
        voting_threshold: Minimum votes for a positive label.
            Defaults to majority of available voting judges (ceil(n/2)).

    Returns:
        Path to the written ``combined.json`` file.
    """
    judge_files: dict[str, Path] = {}
    for judge_id in judge_ids:
        judge_key, responses_path = _load_judge_result(judge_id)
        if judge_key in judge_files:
            raise ValueError(
                f"judge_ids {judge_ids} includes two runs of the same judge model "
                f"{judge_key!r} -- combining two runs of the same judge would let "
                "one model's vote count twice."
            )
        judge_files[judge_key] = responses_path
        print(f"  {judge_id} ({judge_key}): {responses_path}")

    voting_judges = VOTING_JUDGE_KEYS & set(judge_files.keys())
    if not voting_judges:
        raise ValueError(
            f"None of the given judge_ids resolved to a voting judge "
            f"({VOTING_JUDGE_KEYS}). Cannot compute ground truth."
        )

    if voting_threshold is None:
        import math
        voting_threshold = math.ceil(len(voting_judges) / 2)
    print(f"\nVoting judges: {voting_judges}")
    print(f"Voting threshold: {voting_threshold} / {len(voting_judges)}")

    combined = combine_judge_results(judge_files, voting_judges, voting_threshold)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "combined.json"
    with open(output_file, "w") as f:
        json.dump(combined, f, indent=4, ensure_ascii=False)

    n_total = len(combined)
    n_valid = sum(1 for r in combined if r.get("judgement_combined"))
    print(f"\nCombined {n_total} records ({n_valid} valid, {n_total - n_valid} invalid).")
    print(f"Output: {output_file}")
    return output_file


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Combine judge results into a single ground-truth file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(params, "dataset", "judge_ids", config_path=config_path)

    dataset = params["dataset"]
    judge_ids = params["judge_ids"]
    if not isinstance(judge_ids, list) or not judge_ids:
        raise ValueError(f"{config_path}: params.judge_ids must be a non-empty list")

    for judge_id in judge_ids:
        resolved_dataset = paths.find_result_dir(judge_id).parts[-3]
        if resolved_dataset != dataset:
            raise ValueError(
                f"{config_path}: params.dataset {dataset!r} does not match the dataset "
                f"of judge_id {judge_id!r} ({resolved_dataset!r})"
            )

    output_dir = paths.result_dir(dataset, "judge_combine", cfg["id"])
    run_combine(
        judge_ids=judge_ids,
        output_dir=output_dir,
        voting_threshold=params.get("voting_threshold"),
    )


if __name__ == "__main__":
    main()
