"""Match-cache access and the per-dataset matching rules: where analysis/match_cache.py
writes each experiment's ground-truth <-> extraction candidate graph, how other analysis
code loads and thresholds it, and which extraction file it is built from.

Building a cache (match_datasets over every candidate pair) stays in
analysis/match_cache.py; nothing here ever computes a matching.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "experiments", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scholarlm.utils.parsing import SCI_NOTATION_RE  # noqa: E402
import utils as paths  # noqa: E402

# Where every match cache lives: MATCH_CACHE_ROOT/<experiment id>/{match_cache.pkl,
# match_cache.meta.json}. A per-id directory (not a flat <id>.pkl) because
# analysis/recovery_validity.py locates the sidecar as
# ``cache_path.with_name("match_cache.meta.json")``.
MATCH_CACHE_ROOT = _REPO_ROOT / "analysis" / "results" / "match_cache"

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
# Coercion goes through parse_numeric (below), not bare pd.to_numeric:
# real point_value output includes unicode scientific notation ("1.5 ×
# 10^8", observed in pond's 2026-09-21-pond-extraction-gemma27b-full-01)
# that float() can't parse on its own. Values parse_numeric can't make
# sense of at all (label-echo garbage like "pH", "TN") become NaN, and
# match_datasets treats NaN as null. Strict null semantics are: null on one
# side only never matches, but null on BOTH sides matches. So a garbage
# extraction value never strict-matches a ground-truth row that has a real
# value in that column, but it DOES strict-match a ground-truth row whose
# value in that column is itself null/NaN (the other strict fields must then
# carry the match). build_match_cache prints every value that fell through to
# NaN, so a genuinely new garbage pattern doesn't disappear silently.
# ---------------------------------------------------------------------------

def parse_numeric(x):
    """Parse a value into a float for strict-match coercion.

    Handles plain numbers/numeric strings and "<mantissa> × 10^<exponent>"
    scientific notation (SCI_NOTATION_RE, shared with
    scholarlm.utils.parsing.parse_quantity_shape's own plain-value check).
    Anything else (None, or a string that is neither) returns NaN.
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
    m = SCI_NOTATION_RE.match(s)
    if m:
        mantissa, exponent = m.groups()
        return float(mantissa) * (10.0 ** int(exponent))
    return np.nan

def get_matching_config(dataset_config) -> dict:
    """Read this dataset's matching rules off its own DatasetConfig, in the
    ``{"strict": ..., "fuzzy": ..., "fuzzy_threshold": ..., "numeric_coerce": ...}``
    shape the rest of this module (and analysis/recovery_validity.py) expects.

    The single place that bridges DatasetConfig's ``strict_matching``/
    ``fuzzy_matching``/``fuzzy_threshold``/``numeric_coerce`` fields (see
    their docstrings in ``scholarlm.config.DatasetConfig``) into this
    module's own dict shape -- callers should use this rather than reading
    those fields off a DatasetConfig directly, so there is one place to
    change if that shape ever does.

    Raises:
        KeyError: ``strict_matching`` or ``fuzzy_matching``/``fuzzy_threshold``
            is unset on this dataset's config -- add them to
            experiments/dataset-configs/{dataset}.py before using this
            dataset through match_cache.py/recovery_validity.py.
    """
    missing = [
        field for field in ("strict_matching", "fuzzy_matching", "fuzzy_threshold")
        if getattr(dataset_config, field, None) is None
    ]
    if missing:
        raise KeyError(
            f"DatasetConfig for {dataset_config.name!r} has no {missing} set -- "
            f"add them to experiments/dataset-configs/{dataset_config.name}.py "
            f"before using match_cache.py/recovery_validity.py for this dataset "
            f"(see DatasetConfig's docstring)."
        )
    return {
        "strict": dataset_config.strict_matching,
        "fuzzy": dataset_config.fuzzy_matching,
        "fuzzy_threshold": dataset_config.fuzzy_threshold,
        "numeric_coerce": dataset_config.numeric_coerce or [],
        "fuzzy_normalizers": dataset_config.fuzzy_normalizers or {},
    }


def edges_above_threshold(
    edges: list[tuple[int, int]], edge_weights: list[float], fuzzy_threshold: float,
) -> list[tuple[int, int]]:
    """Filter a 0.0-threshold-cached edge list down to the edges some
    fuzzy_threshold selects: ``w >= fuzzy_threshold`` (inclusive), the same
    boundary as match_datasets' own construction-time filter (it drops
    ``score < fuzzy_threshold``) and as every post-hoc caller (analysis/
    metrics.py, calibration*.py, probe_pca.py, validity_evaluation.py). Before
    2026-10-03 the post-hoc callers used strict ``>`` while match_datasets
    kept ``>=``, so an edge scoring exactly the threshold was dropped by them
    but not by it; that is fixed, and it invalidates any recovery/validity
    number computed with the old strict-greater-than compare.
    """
    return [
        (gt_idx, ex_idx)
        for (gt_idx, ex_idx), w in zip(edges, edge_weights)
        if w >= fuzzy_threshold
    ]


def match_cache_path(experiment_id: str) -> Path:
    """The match_cache.pkl path for an experiment id
    (``MATCH_CACHE_ROOT/<id>/match_cache.pkl``), whether or not it has been
    built yet. The one place this path gets constructed -- other scripts should
    call this rather than hand-building it. Does not check that the id names a
    real run; the build step (``build_match_cache``) does that.
    """
    return MATCH_CACHE_ROOT / experiment_id / "match_cache.pkl"


def match_cache_meta_path(experiment_id: str) -> Path:
    """The match_cache.meta.json sidecar path for an experiment id -- records
    which ground truth file (repo-relative path, sha256, row count) the
    match_cache.pkl at match_cache_path(experiment_id) was built against. See
    build_match_cache / recovery_validity._assert_ground_truth_matches_cache.
    """
    return match_cache_path(experiment_id).with_name("match_cache.meta.json")


def load_match_cache(
    experiment_id: str, fuzzy_threshold: float | None = None,
) -> tuple | list[tuple[int, int]]:
    """Load an already-built match_cache.pkl for use in another script.

    Reads only from ``match_cache_path(experiment_id)`` under MATCH_CACHE_ROOT --
    never from the old per-run location. Never computes anything -- raises if
    build_match_cache hasn't been run for this experiment_id yet, rather than silently building one inline (a
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


def extraction_path(experiment_id: str) -> tuple:
    """Resolve the extraction file to match experiment_id against:
    postprocessed.json (analysis/postprocessing.py's qualifier-fill/unit-
    standardization output) if it exists, else final.json, with a printed
    warning -- the one place this preference is decided, so
    build_match_cache and analysis/recovery_validity.py's load_frames can't
    drift apart on it.

    Returns:
        (path, used_fallback). used_fallback=True means postprocessed.json
        didn't exist and final.json was used instead (analysis/
        postprocessing.py hasn't been run for this id yet).

    Raises:
        FileNotFoundError: neither file exists for this id.
    """
    result_dir = paths.find_result_dir(experiment_id)
    postprocessed_path = result_dir / "postprocessed.json"
    if postprocessed_path.exists():
        return postprocessed_path, False

    final_path = result_dir / "final.json"
    if not final_path.exists():
        raise FileNotFoundError(
            f"{experiment_id}: no postprocessed.json or final.json at {result_dir}"
        )
    print(
        f"{experiment_id}: no postprocessed.json at {postprocessed_path} -- falling "
        f"back to final.json (run `python analysis/postprocessing.py {experiment_id} "
        f"--ground-truth-file <path>` first for qualifier-fill/unit standardization)"
    )
    return final_path, True
