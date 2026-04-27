"""
Combine judge results across multiple models into a single ground-truth file.

Reads individual judge ``responses.json`` files for a given
(dataset, extraction_model) pair, merges them by ``measurement_id``, computes
a majority-vote ground truth label from the **frontier judges only**, and writes
a combined output file.

Output path:
    data/experiments/{dataset}/judge/{extraction_model}/combined/combined.json

Usage
-----
    # Auto-discover all judge results under the extraction model's directory:
    python experiments/run_judge_combine.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b

    # Specify judge model names explicitly (useful when multiple date dirs exist):
    python experiments/run_judge_combine.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --judges openai anthropic gemini llama-3.1-8b

    # Override the voting threshold (default: majority of frontier judges):
    python experiments/run_judge_combine.py \\
        --dataset pond \\
        --extraction-model gemma-3-27b \\
        --voting-threshold 2

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
# "human" is included so human-validation runs produced by validation.py
# can participate in majority voting (use --voting-threshold 1 for standalone use).
# ---------------------------------------------------------------------------

FRONTIER_JUDGE_KEYS = {"openai", "anthropic", "gemini", "llama-3.1-8b"}

import paths


def _find_judge_result(
    dataset_name: str,
    extraction_model: str,
    extraction_date: str,
    judge_key: str,
    ablation: str | None = None,
) -> Path:
    """Locate the most recent ``responses.json`` for a given judge key.

    Searches ``data/experiments/{dataset}/judge/{extraction_model}/{extraction_date}/{judge_key}/*/responses.json``
    and returns the path in the most recently dated directory.

    Args:
        dataset_name: Dataset identifier.
        extraction_model: Extraction model short name.
        extraction_date: Date tag of the extraction run (``YYYY_mm_dd``).
        judge_key: Judge model key (e.g. ``"llama-3.1-8b"``, ``"openai"``).

    Returns:
        Path to ``responses.json``.

    Raises:
        FileNotFoundError: If no result is found.
    """
    judge_dir = paths.judge_base(dataset_name, extraction_model, extraction_date, ablation) / judge_key
    if not judge_dir.exists():
        raise FileNotFoundError(f"No judge directory found: {judge_dir}")
    date_dirs = sorted(judge_dir.iterdir(), reverse=True)
    for d in date_dirs:
        candidate = d / "responses.json"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No responses.json found for judge '{judge_key}' under {judge_dir}"
    )


def _discover_judge_keys(
    dataset_name: str,
    extraction_model: str,
    extraction_date: str,
    ablation: str | None = None,
) -> list[str]:
    """Return all judge keys that have a responses.json under the extraction date dir."""
    base = paths.judge_base(dataset_name, extraction_model, extraction_date, ablation)
    if not base.exists():
        return []
    keys = []
    for judge_dir in sorted(base.iterdir()):
        print(f"Checking for judge results in: {judge_dir}")
        if judge_dir.name == "combined":
            continue
        for date_dir in judge_dir.iterdir():
            if (date_dir / "responses.json").exists():
                keys.append(judge_dir.name)
                break
    return keys


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
        voting_judges: Subset of judge keys whose votes count toward ground truth
            (typically frontier providers only).
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

    # Majority vote over frontier (voting) judges
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
    dataset_name: str,
    extraction_model: str,
    extraction_date: str,
    judge_keys: list[str] | None = None,
    voting_threshold: int | None = None,
    ablation: str | None = None,
) -> Path:
    """Discover judge results, combine them, and write the combined output file.

    Args:
        dataset_name: Dataset identifier.
        extraction_model: Extraction model short name.
        extraction_date: Date tag of the extraction run (``YYYY_mm_dd``).
        judge_keys: Explicit list of judge keys to combine. If ``None``,
            auto-discovers all judges with results under the extraction date dir.
        voting_threshold: Minimum frontier-judge votes for a positive label.
            Defaults to majority of available frontier judges (ceil(n/2)).

    Returns:
        Path to the written ``combined.json`` file.
    """
    if judge_keys is None:
        judge_keys = _discover_judge_keys(dataset_name, extraction_model, extraction_date, ablation)
        if not judge_keys:
            raise FileNotFoundError(
                f"No judge results found for dataset='{dataset_name}' "
                f"extraction_model='{extraction_model}' extraction_date='{extraction_date}'."
            )
        print(f"Auto-discovered judges: {judge_keys}")

    judge_files: dict[str, Path] = {}
    for key in judge_keys:
        judge_files[key] = _find_judge_result(dataset_name, extraction_model, extraction_date, key, ablation)
        print(f"  {key}: {judge_files[key]}")

    voting_judges = FRONTIER_JUDGE_KEYS & set(judge_keys)
    if not voting_judges:
        raise ValueError(
            f"None of the provided judge keys are frontier providers "
            f"({FRONTIER_JUDGE_KEYS}). Cannot compute ground truth."
        )

    if voting_threshold is None:
        import math
        voting_threshold = math.ceil(len(voting_judges) / 2)
    print(f"\nFrontier judges for voting: {voting_judges}")
    print(f"Voting threshold: {voting_threshold} / {len(voting_judges)}")

    combined = combine_judge_results(judge_files, voting_judges, voting_threshold)

    output_dir = paths.judge_combined(dataset_name, extraction_model, extraction_date, ablation)
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
    p.add_argument("--dataset", required=True, help="Dataset name (e.g. 'pond', 'nfix').")
    p.add_argument("--extraction-model", required=True, help="Extraction model short name.")
    p.add_argument("--extraction-date", required=True, help="Date tag YYYY_mm_dd of the extraction run.")
    p.add_argument(
        "--ablation", default=None, metavar="N",
        help="Ablation number (e.g. 2). If set, reads from and writes to ablations/ablation{N}/.",
    )
    p.add_argument(
        "--judges", nargs="+", default=None,
        help="Judge keys to combine. Default: auto-discover from directory.",
    )
    p.add_argument(
        "--voting-threshold", type=int, default=None,
        help="Min frontier votes for a positive label. Default: majority of frontier judges.",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    run_combine(
        dataset_name=args.dataset,
        extraction_model=args.extraction_model,
        extraction_date=args.extraction_date,
        judge_keys=args.judges,
        voting_threshold=args.voting_threshold,
        ablation=args.ablation,
    )


if __name__ == "__main__":
    main()
