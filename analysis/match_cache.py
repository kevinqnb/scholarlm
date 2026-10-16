"""Build the ground-truth <-> extraction match cache for each experiment id.

Matches each run's extraction (postprocessed.json, else final.json) against an
explicitly given ground-truth file, using the dataset's DatasetConfig matching
rules. Writes analysis/results/match-cache/<id>/match_cache.pkl plus a
match_cache.meta.json sidecar with both input files' paths and hashes. Always
rebuilds; readers (common/matching.py, metrics.py) never build.

- The ground-truth file is always explicit, so a repointed DatasetConfig can't
  silently change results.
- The cache is built at fuzzy_threshold 0.0 so any threshold can be applied later
  without rescanning.

Usage
-----
    python analysis/match_cache.py <experiment_id> [...] --ground-truth-file <path>
    python analysis/match_cache.py --config analysis/analysis-configs/recovery-validity/<id>.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from scholarlm.utils.data import match_datasets
from analysis.common.config import get_ground_truth_path, load_analysis_config
from analysis.common.loaders import load_ground_truth_file
from experiments.run_extraction import load_dataset_config
import utils as paths

from analysis.common.matching import (
    edges_above_threshold, extraction_path, get_matching_config, match_cache_meta_path, match_cache_path, parse_numeric,
)
from analysis.common.provenance import repo_relative, sha256_file


def cached_match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None,
    cache_path: Path,
    fuzzy_normalizers: dict | None = None,
) -> tuple:
    """Run match_datasets at threshold 0.0 and overwrite the pickle at ``cache_path``.

    Takes no threshold on purpose; apply one later with ``edges_above_threshold``.

    Args:
        df_left: Ground-truth frame.
        df_right: Extraction frame.
        strict_matching: Exact-match column mapping.
        fuzzy_matching: Fuzzy-match column mapping.
        cache_path: Pickle to write.
        fuzzy_normalizers: Per-column normalizers for fuzzy matching.

    Returns:
        ``(matching, edges, edge_weights)``.
    """
    result = match_datasets(
        df_left,
        df_right,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching or {},
        fuzzy_threshold=0.0,
        fuzzy_normalizers=fuzzy_normalizers,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def build_match_cache(experiment_id: str, ground_truth_path: Path) -> Path:
    """Build and write the match cache and sidecar for one experiment.

    Columns in the dataset's ``numeric_coerce`` are parsed to float first;
    unparseable values become NaN and are printed.

    Args:
        experiment_id: Extraction experiment id.
        ground_truth_path: Ground-truth file (no default; it changes the numbers).

    Returns:
        Path of the written match_cache.pkl.

    Raises:
        FileNotFoundError: No postprocessed.json or final.json.
        ValueError: Ground truth and extraction share no document_id (almost
            certainly the wrong ground-truth file).
    """
    result_dir = paths.find_result_dir(experiment_id)
    dataset = result_dir.relative_to(paths.RESULTS_ROOT).parts[0]

    extraction_file_path, _used_fallback = extraction_path(experiment_id)

    dataset_config = load_dataset_config(dataset)
    cfg = get_matching_config(dataset_config)

    with open(extraction_file_path) as f:
        extraction_records = json.load(f)
    extraction_df = pd.DataFrame(extraction_records).reset_index(drop=True)

    ground_truth_df = load_ground_truth_file(ground_truth_path).reset_index(drop=True)

    if "document_id" in ground_truth_df.columns and "document_id" in extraction_df.columns:
        shared_documents = set(ground_truth_df["document_id"]) & set(extraction_df["document_id"])
        if not shared_documents:
            raise ValueError(
                f"{experiment_id}: ground truth {ground_truth_path} and this run's "
                f"{extraction_file_path.name} share zero document_id values -- almost "
                f"certainly the wrong ground_truth_file for this experiment's dataset "
                f"({dataset!r})"
            )

    for gt_col in cfg.get("numeric_coerce", []):
        ex_col = cfg["strict"][gt_col]
        for df, col in ((ground_truth_df, gt_col), (extraction_df, ex_col)):
            original = df[col]
            parsed = original.apply(parse_numeric)
            unparseable = sorted(set(
                original[parsed.isna() & original.notna()].astype(str)
            ))
            if unparseable:
                print(
                    f"{experiment_id}: {len(unparseable)} distinct unparseable "
                    f"{col!r} value(s) coerced to NaN (will never strict-match): "
                    f"{unparseable[:10]}"
                    + (" ..." if len(unparseable) > 10 else "")
                )
            df[col] = parsed

    cache_path = match_cache_path(experiment_id)
    # Remove the old sidecar first so an interrupted build can't leave a stale
    # sidecar vouching for the new pkl.
    match_cache_meta_path(experiment_id).unlink(missing_ok=True)
    matching, edges, edge_weights = cached_match(
        ground_truth_df,
        extraction_df,
        strict_matching=cfg["strict"],
        fuzzy_matching=cfg["fuzzy"],
        cache_path=cache_path,
        fuzzy_normalizers=cfg["fuzzy_normalizers"],
    )

    meta = {
        "ground_truth_file": repo_relative(ground_truth_path),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "n_gt": len(ground_truth_df),
        "extraction_file": repo_relative(extraction_file_path),
        "extraction_sha256": sha256_file(extraction_file_path),
    }
    with open(match_cache_meta_path(experiment_id), "w") as f:
        json.dump(meta, f, indent=2)

    selected = edges_above_threshold(edges, edge_weights, cfg["fuzzy_threshold"])
    n_gt_recovered = len({gt_idx for gt_idx, _ in selected})
    n_ex_matched = len({ex_idx for _, ex_idx in selected})

    print(
        f"{experiment_id}: ground truth {meta['ground_truth_file']} "
        f"({len(ground_truth_df)} rows), {meta['extraction_file']} "
        f"({len(extraction_df)} rows), {len(edges)} candidate edges cached at "
        f"threshold=0.0 -> {cache_path}\n"
        f"{experiment_id}: at this dataset's selected threshold="
        f"{cfg['fuzzy_threshold']:.4f}: {len(selected)} edges, "
        f"{n_gt_recovered}/{len(ground_truth_df)} ground-truth rows recovered, "
        f"{n_ex_matched}/{len(extraction_df)} extraction rows matched"
    )
    return cache_path


def main() -> None:
    """CLI: build caches for ids given directly or via ``--config`` (exactly one)."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_ids", nargs="*", help="Experiment ids to compute and cache matches for.")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="analysis-configs/recovery-validity/<id>.yaml providing params.experiment_ids and "
             "params.ground_truth_file -- mutually exclusive with passing "
             "experiment_ids/--ground-truth-file directly.",
    )
    parser.add_argument(
        "--ground-truth-file", type=Path, default=None,
        help="Ground-truth CSV/JSON to match experiment_ids against. Required "
             "when passing experiment_ids directly; ignored (read from "
             "params.ground_truth_file instead) with --config.",
    )
    args = parser.parse_args()

    if bool(args.config) == bool(args.experiment_ids):
        parser.error("pass experiment_ids directly, or --config, not both/neither")
    if args.config and args.ground_truth_file is not None:
        parser.error(
            "--ground-truth-file is ignored with --config -- set "
            "params.ground_truth_file in the analysis config instead"
        )
    if args.experiment_ids and args.ground_truth_file is None:
        parser.error("--ground-truth-file is required when passing experiment_ids directly")

    if args.config:
        cfg = load_analysis_config(args.config, "recovery-validity")
        experiment_ids = cfg["params"]["experiment_ids"]
        ground_truth_path = get_ground_truth_path(cfg)
    else:
        experiment_ids = args.experiment_ids
        ground_truth_path = args.ground_truth_file
        if not ground_truth_path.is_absolute():
            ground_truth_path = _REPO_ROOT / ground_truth_path
        if not ground_truth_path.exists():
            parser.error(f"--ground-truth-file {ground_truth_path} does not exist")

    for experiment_id in experiment_ids:
        build_match_cache(experiment_id, ground_truth_path)


if __name__ == "__main__":
    main()
