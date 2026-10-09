"""Compute and cache extraction <-> ground-truth matches, by experiment id.

The only job of this script: given a list of experiment ids and an explicit
ground-truth file, load each run's extraction file, that ground truth, and
the run's dataset's matching rules (``strict_matching``/``fuzzy_matching``/
``fuzzy_threshold``/``numeric_coerce`` on its ``DatasetConfig``, in
``experiments/dataset-configs/{dataset}.py`` -- see ``get_matching_config``
below and ``DatasetConfig``'s own docstring), run match_datasets, and write
the result to analysis/results/match_cache/<id>/match_cache.pkl (see
``MATCH_CACHE_ROOT`` / ``match_cache_path`` -- one flat directory per
experiment id, deliberately NOT inside the run's own experiments/results/
directory: a cache is an analysis artifact derived from a run plus a ground
truth file, not run output), plus a match_cache.meta.json sidecar next to it
recording exactly which ground truth file and which extraction file it was
built against (repo-relative path + sha256 for each, plus the ground truth's
row count) -- see ``build_match_cache``. Caches written to the old per-run
location (experiments/results/.../<id>/match_cache.pkl) are never read; there
is no fallback to it. The
extraction file is ``postprocessed.json`` (analysis/postprocessing.py's
qualifier-fill/unit-standardization output) when it exists, else
``final.json`` with a printed warning -- see ``extraction_path``. Always
recomputes and overwrites -- this is the point where a fresh, authoritative
cache gets built, not a read-through cache that might silently keep serving a
match computed under an older matching configuration or a different ground
truth/extraction file.
analysis/common/metrics.py's recovery_rate/validity_rate (via analysis/common/loaders.py's
cached_match) read the file this writes; they never write it themselves.

The ground truth file is never inferred from the run's dataset's own
DatasetConfig.ground_truth_file -- it is always given explicitly, either via
an analysis config's ``params.ground_truth_file`` (see
analysis/common/config.py) or, in ad-hoc CLI mode, ``--ground-truth-file``.
This is deliberate: reading it implicitly off the DatasetConfig would let a
cache (and everything scored against it) silently start using a different
file if that config is later repointed (e.g. a revised ground-truth review
pass), with no record of which file actually produced a given number.

The cache is always built with fuzzy_threshold=0.0, regardless of the
dataset's configured "selected" threshold -- this is the same convention
every existing caller of cached_match already uses (recovery_rate,
validity_rate, per_paper_metrics, calibration*.py, probe_pca.py: every one
of them hardcodes fuzzy_threshold=0.0 in its own cached_match call and then
filters the returned edges/edge_weights by `w >= threshold` itself; before
2026-10-03 that compare was a strict `w > threshold`). A
threshold only decides which strict-matched candidate edges count as a
match; it never changes which candidates exist. Baking a non-zero threshold
into match_datasets' own edge construction would permanently discard every
below-threshold edge from the pickle, so a smaller/different threshold
later could never be recovered from that cache without redoing the full
O(n_gt * n_extraction) strict+fuzzy scan. Caching at 0.0 stores every
strict-matched candidate once; edges_above_threshold (analysis/common/matching.py) applies
whatever threshold is wanted on top of that, for free.

This is meant to be the one centralized place matching *runs* live, reading
matching *rules* from each dataset's own committed config (not a copy kept
here) -- add a new dataset's rules to its DatasetConfig in
experiments/dataset-configs/{dataset}.py rather than growing another copy of
get_matching_rules-style logic elsewhere. This deliberately does not touch
the legacy analysis/ablation.py/baselines.py's own get_matching_rules, which
predates this and scores a different column shape (``converted_value``, not
``point_value``) against an earlier extraction/judge era -- the two are
allowed to diverge (see DatasetConfig's ``strict_matching`` docstring).

Usage
-----
    python analysis/match_cache.py <experiment_id> [<experiment_id> ...] \\
        --ground-truth-file <path>
    python analysis/match_cache.py --config analysis/analysis-configs/<id>.yaml
    bash analysis/match_cache.sh

``--config`` reads ``params.experiment_ids`` and ``params.ground_truth_file``
from an analysis-configs/<id>.yaml (see analysis/common/config.py) instead
of taking them as flags -- mutually exclusive with passing ids/
``--ground-truth-file`` directly.
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
    """Run match_datasets at fuzzy_threshold=0.0 and write the result to
    cache_path, overwriting any existing cache there.

    No fuzzy_threshold parameter on purpose -- this always computes and
    caches the full strict-matched candidate graph (see module docstring).
    Use edges_above_threshold on the returned edges/edge_weights to apply an
    actual threshold.
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
    """Compute and cache the match for one experiment id against an
    explicitly given ground truth file. Returns the cache path.

    ``ground_truth_path`` has no default (CLAUDE.md: no inferred default for
    a value that changes the reported numbers) -- callers get it from an
    analysis config's ``params.ground_truth_file``
    (analysis.analysis_config.get_ground_truth_path) or, in ad-hoc CLI mode,
    ``--ground-truth-file``. Matching *rules* (strict/fuzzy/threshold/
    numeric_coerce) still come from the experiment's own dataset's
    DatasetConfig -- only which ground truth *rows* to match against is
    pinned explicitly now.

    The extraction file itself is resolved via extraction_path() --
    postprocessed.json when it exists, else final.json with a warning. Which
    one was actually used (path + sha256) is recorded in the
    match_cache.meta.json sidecar alongside the ground truth's own, so a
    cache built against final.json can never be silently scored later as if
    it reflected a postprocessed.json that didn't exist yet -- see
    analysis/recovery_validity.py's assert_extraction_matches_cache.

    Raises:
        FileNotFoundError: no postprocessed.json or final.json for this id.
        ValueError: the ground truth and extraction frames share no
            document_id at all -- almost certainly the wrong ground truth
            file for this experiment's dataset, not a real zero-overlap
            result.
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
    # Delete any stale sidecar before writing a new pkl -- if this call is
    # interrupted between the pkl write and the sidecar write below, a leftover
    # sidecar from a PREVIOUS (different) ground truth file would otherwise
    # sit next to the new pkl and make assert_ground_truth_matches_cache
    # wrongly pass on the next run.
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_ids", nargs="*", help="Experiment ids to compute and cache matches for.")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="analysis-configs/<id>.yaml providing params.experiment_ids and "
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
        cfg = load_analysis_config(args.config)
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
