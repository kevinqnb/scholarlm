"""Compute and cache extraction <-> ground-truth matches, by experiment id.

The only job of this script: given a list of experiment ids and this file's
MATCHING_CONFIGS (one entry per dataset), load each run's final.json and its
dataset's ground truth, run match_datasets, and write the result to
match_cache.pkl inside that run's own results directory
(experiments/results/{dataset}/{experiment-type}/<id>/match_cache.pkl).
Always recomputes and overwrites -- this is the point where a fresh,
authoritative cache gets built, not a read-through cache that might silently
keep serving a match computed under an older matching configuration.
analysis/metrics.py's recovery_rate/validity_rate (via analysis/loaders.py's
cached_match) read the file this writes; they never write it themselves.

The cache is always built with fuzzy_threshold=0.0, regardless of the
dataset's configured "selected" threshold -- this is the same convention
every existing caller of cached_match already uses (recovery_rate,
validity_rate, per_paper_metrics, calibration*.py, probe_pca.py: every one
of them hardcodes fuzzy_threshold=0.0 in its own cached_match call and then
filters the returned edges/edge_weights by `w > threshold` itself). A
threshold only decides which strict-matched candidate edges count as a
match; it never changes which candidates exist. Baking a non-zero threshold
into match_datasets' own edge construction would permanently discard every
below-threshold edge from the pickle, so a smaller/different threshold
later could never be recovered from that cache without redoing the full
O(n_gt * n_extraction) strict+fuzzy scan. Caching at 0.0 stores every
strict-matched candidate once; edges_above_threshold (below) applies
whatever threshold is wanted on top of that, for free.

This is meant to be the one centralized place matching configuration and
matching runs live -- add a new dataset's rules to MATCHING_CONFIGS here
rather than growing another copy of get_matching_rules-style logic elsewhere.

Usage
-----
    python analysis/match_cache.py <experiment_id> [<experiment_id> ...]
    bash analysis/match_cache.sh
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from scholarlm.utils.data import match_datasets
from analysis.loaders import load_ground_truth
from experiments.run_extraction import load_dataset_config
import utils as paths

# ---------------------------------------------------------------------------
# Per-dataset matching configuration
#
# strict / fuzzy: column-name mapping from ground-truth df -> extraction df,
# passed straight through to match_datasets (df_left=ground_truth,
# df_right=extraction). "units" below is the real column name on both sides
# (there is no separate singular "unit" column).
#
# numeric_coerce: subset of strict's keys (ground-truth column names) that
# must be forced to float on both sides before matching. Needed because
# match_datasets' strict equality only takes the numeric (np.isclose) path
# when BOTH sides are already a numeric Python type -- point_value is
# numeric in ground truth but a raw LLM-output string (e.g. "0.26") in
# final.json (ParseQuantityResponse types it str | None), so left uncoerced
# it silently falls through to string comparison and never matches.
#
# Coercion goes through _parse_numeric (below), not bare pd.to_numeric:
# real point_value output includes unicode scientific notation ("1.5 ×
# 10^8", observed in pond's 2026-09-21-pond-extraction-gemma27b-full-01)
# that float() can't parse on its own. Values _parse_numeric can't make
# sense of at all (label-echo garbage like "pH", "TN") become NaN --
# match_datasets treats NaN as null on both sides, so those rows simply
# never strict-match, which is correct: they have no valid value to match
# on. build_match_cache prints every value that fell through to NaN, so a
# genuinely new garbage pattern doesn't disappear silently.
# ---------------------------------------------------------------------------

# Matches "<mantissa> × 10^<exponent>" (also accepts x/X for ×, and a
# missing ^) -- the one non-plain-float numeric form observed in real
# point_value output so far.
_SCI_NOTATION_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[×xX]\s*10\s*\^?\s*([-+]?\d+)\s*$")


def _parse_numeric(x):
    """Parse a value into a float for strict-match coercion.

    Handles plain numbers/numeric strings and "<mantissa> × 10^<exponent>"
    scientific notation. Anything else (None, or a string that is neither)
    returns NaN.
    """
    if x is None:
        return np.nan
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if not isinstance(x, str):
        return np.nan
    s = x.strip()
    try:
        return float(s)
    except ValueError:
        pass
    m = _SCI_NOTATION_RE.match(s)
    if m:
        mantissa, exponent = m.groups()
        return float(mantissa) * (10.0 ** int(exponent))
    return np.nan

MATCHING_CONFIGS: dict[str, dict] = {
    "pond": {
        "strict": {
            "document_id": "document_id",
            "attribute": "attribute",
            "point_value": "point_value",
            "units": "units",
        },
        "fuzzy": {
            "name": "name",
            "ecosystem": "ecosystem",
        },
        # Selected/default operating threshold for this dataset -- applied
        # on top of the 0.0-threshold cache via edges_above_threshold, NOT
        # passed to match_datasets when the cache itself is built (see
        # module docstring).
        "fuzzy_threshold": 1 / 3,
        "numeric_coerce": ["point_value"],
    },
}


def cached_match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    strict_matching: dict,
    fuzzy_matching: dict | None,
    cache_path: Path,
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
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(result, f)
    return result


def edges_above_threshold(
    edges: list[tuple[int, int]], edge_weights: list[float], fuzzy_threshold: float,
) -> list[tuple[int, int]]:
    """Filter a 0.0-threshold-cached edge list down to the edges some
    fuzzy_threshold selects, mirroring the `w > fuzzy_threshold` convention
    every existing cached_match caller applies post-hoc (analysis/metrics.py,
    calibration*.py, probe_pca.py) -- strictly greater-than, so an edge whose
    score exactly equals fuzzy_threshold is excluded here even though
    match_datasets' own `score < fuzzy_threshold` construction-time filter
    would have kept it. That boundary inconsistency predates this script and
    isn't this function's to fix (see CLAUDE.md's eval-code guard); this
    function only reproduces the existing convention, not a new one.
    """
    return [
        (gt_idx, ex_idx)
        for (gt_idx, ex_idx), w in zip(edges, edge_weights)
        if w > fuzzy_threshold
    ]


def match_cache_path(experiment_id: str) -> Path:
    """The match_cache.pkl path for an experiment id, whether or not it has
    been built yet. The one place this path gets constructed -- other
    scripts should call this rather than hand-building result_dir / 'match_cache.pkl'.
    """
    return paths.find_result_dir(experiment_id) / "match_cache.pkl"


def load_match_cache(
    experiment_id: str, fuzzy_threshold: float | None = None,
) -> tuple | list[tuple[int, int]]:
    """Load an already-built match_cache.pkl for use in another script.

    Never computes anything -- raises if build_match_cache hasn't been run
    for this experiment_id yet, rather than silently building one inline (a
    fresh build takes O(n_gt * n_extraction) strict+fuzzy scoring, tens of
    minutes for a real run; see analysis/match_cache.sh).

    Args:
        experiment_id: The experiment id whose match_cache.pkl to load.
        fuzzy_threshold: If given, returns just the edges surviving this
            threshold (via edges_above_threshold) -- what recovery/validity-
            style code wants. If None (default), returns the raw cached
            (matching, edges, edge_weights) tuple as written by cached_match.

    Raises:
        FileNotFoundError: If no match_cache.pkl exists for this id.
    """
    cache_path = match_cache_path(experiment_id)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No match cache for {experiment_id!r} at {cache_path}. "
            f"Run `python analysis/match_cache.py {experiment_id}` first."
        )
    with open(cache_path, "rb") as f:
        matching, edges, edge_weights = pickle.load(f)
    if fuzzy_threshold is None:
        return matching, edges, edge_weights
    return edges_above_threshold(edges, edge_weights, fuzzy_threshold)


def build_match_cache(experiment_id: str) -> Path:
    """Compute and cache the match for one experiment id. Returns the cache path."""
    result_dir = paths.find_result_dir(experiment_id)
    dataset = result_dir.relative_to(paths.RESULTS_ROOT).parts[0]

    final_path = result_dir / "final.json"
    if not final_path.exists():
        raise FileNotFoundError(f"{experiment_id}: no final.json at {final_path}")

    if dataset not in MATCHING_CONFIGS:
        raise KeyError(
            f"{experiment_id}: no matching configuration for dataset {dataset!r} "
            f"in MATCHING_CONFIGS. Configured datasets: {sorted(MATCHING_CONFIGS)}"
        )
    cfg = MATCHING_CONFIGS[dataset]

    with open(final_path) as f:
        extraction_records = json.load(f)
    extraction_df = pd.DataFrame(extraction_records).reset_index(drop=True)

    dataset_config = load_dataset_config(dataset)
    ground_truth_df = load_ground_truth(dataset_config).reset_index(drop=True)

    for gt_col in cfg.get("numeric_coerce", []):
        ex_col = cfg["strict"][gt_col]
        for df, col in ((ground_truth_df, gt_col), (extraction_df, ex_col)):
            original = df[col]
            parsed = original.apply(_parse_numeric)
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

    cache_path = result_dir / "match_cache.pkl"  # == match_cache_path(experiment_id); result_dir already resolved above
    matching, edges, edge_weights = cached_match(
        ground_truth_df,
        extraction_df,
        strict_matching=cfg["strict"],
        fuzzy_matching=cfg["fuzzy"],
        cache_path=cache_path,
    )

    selected = edges_above_threshold(edges, edge_weights, cfg["fuzzy_threshold"])
    n_gt_recovered = len({gt_idx for gt_idx, _ in selected})
    n_ex_matched = len({ex_idx for _, ex_idx in selected})

    print(
        f"{experiment_id}: {len(ground_truth_df)} ground-truth rows, "
        f"{len(extraction_df)} extraction rows, {len(edges)} candidate edges "
        f"cached at threshold=0.0 -> {cache_path}\n"
        f"{experiment_id}: at this dataset's selected threshold="
        f"{cfg['fuzzy_threshold']:.4f}: {len(selected)} edges, "
        f"{n_gt_recovered}/{len(ground_truth_df)} ground-truth rows recovered, "
        f"{n_ex_matched}/{len(extraction_df)} extraction rows matched"
    )
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment_ids", nargs="+", help="Experiment ids to compute and cache matches for.")
    args = parser.parse_args()

    for experiment_id in args.experiment_ids:
        build_match_cache(experiment_id)


if __name__ == "__main__":
    main()
