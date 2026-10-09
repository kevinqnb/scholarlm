"""Match-cache access and the per-dataset matching rules: where analysis/match_cache.py
writes each experiment's ground-truth <-> extraction candidate graph, how other analysis
code loads and thresholds it, and which extraction file it is built from.

Building a cache (match_datasets over every candidate pair) stays in
analysis/match_cache.py; nothing here ever computes a matching.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "experiments", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from scholarlm.utils.parsing import SCI_NOTATION_RE  # noqa: E402
import utils as paths  # noqa: E402
from analysis.common.loaders import load_ground_truth_file  # noqa: E402
from analysis.common.provenance import repo_relative, sha256_file  # noqa: E402
from experiments.run_extraction import load_dataset_config  # noqa: E402

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
    build_match_cache / recovery_validity.assert_ground_truth_matches_cache.
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


def load_frames(experiment_id: str, ground_truth_path: Path):
    """Load an experiment's ground truth + extraction frames exactly as
    ````match_cache.build_match_cache`` did when it built this id's cache
    (reset_index, no unit conversion, no row filtering) -- the cached
    (gt_idx, ex_idx) edges are positions into frames loaded this same way,
    so loading them any other way would silently misalign the cache. The
    extraction file itself is resolved via ``extraction_path``,
    the same postprocessed.json-else-final.json preference
    ``build_match_cache`` used.

    ``ground_truth_path`` has no default -- see module docstring for why the
    ground truth file is always given explicitly rather than read off the
    dataset's own DatasetConfig.

    Returns:
        (dataset, dataset_config, ground_truth_df, extraction_df,
        extraction_file_path, ground_truth_path).

    Raises:
        FileNotFoundError: no results directory, or no postprocessed.json/
            final.json, for this id.
        ValueError: run_metadata.json's own dataset field disagrees with the
            dataset the id's own results-directory path resolves to.
    """
    result_dir = paths.find_result_dir(experiment_id)
    dataset = result_dir.parts[-3]

    meta = paths.load_run_metadata(result_dir)
    if meta is not None and meta.get("dataset") not in (None, dataset):
        raise ValueError(
            f"{experiment_id}: run_metadata.json dataset={meta.get('dataset')!r} "
            f"disagrees with the dataset its own path resolves to ({dataset!r})"
        )

    extraction_file_path, _used_fallback = extraction_path(experiment_id)
    with open(extraction_file_path) as f:
        extraction_df = pd.DataFrame(json.load(f)).reset_index(drop=True)

    dataset_config = load_dataset_config(dataset)
    ground_truth_df = load_ground_truth_file(ground_truth_path).reset_index(drop=True)

    return dataset, dataset_config, ground_truth_df, extraction_df, extraction_file_path, ground_truth_path

def assert_cache_fresh(cache_path: Path, *source_paths: Path) -> None:
    """Raise if cache_path is older than any of source_paths.

    A match cache's (gt_idx, ex_idx) edges are only meaningful against the
    exact frames that were loaded when it was built -- if the ground truth
    or the extraction file (postprocessed.json/final.json) has changed
    since, those indices may now point at different rows. Refuse rather than
    guess.
    """
    cache_mtime = cache_path.stat().st_mtime
    stale = [p for p in source_paths if p.stat().st_mtime > cache_mtime]
    if stale:
        raise RuntimeError(
            f"{cache_path} is older than {[str(p) for p in stale]} -- its cached "
            f"match indices may no longer line up with the current rows. Rerun "
            f"`python analysis/match_cache.py <experiment_id>` before using it."
        )

def assert_ground_truth_matches_cache(
    experiment_id: str, cache_path: Path, ground_truth_path: Path, ground_truth_df: pd.DataFrame,
) -> None:
    """Raise unless match_cache.py's own match_cache.meta.json sidecar
    confirms cache_path was built against this exact ground_truth_path.

    The mtime check in assert_cache_fresh only catches a cache older than
    its sources -- it says nothing about WHICH ground truth file a cache was
    built against, now that the file is an explicit, selectable parameter
    rather than always the one true value on the run's DatasetConfig. A cache
    built against ground truth A and then scored here against ground truth B
    would otherwise pass every existing guard (both older than the cache,
    edges in-range for n_gt/n_ext) while silently misaligning every cached
    (gt_idx, ex_idx) pair -- exactly the "runs cleanly, produces a wrong
    number" failure mode this repo is built to avoid. Checked by content hash,
    not just path string, so an edited-in-place ground truth file is also
    caught.

    Raises:
        FileNotFoundError: no match_cache.meta.json sidecar (a cache built
            before this check existed) -- rebuild it rather than trust it.
        RuntimeError: the sidecar's ground_truth_file/sha256/n_gt disagree
            with ground_truth_path/ground_truth_df.
    """
    meta_path = cache_path.with_name("match_cache.meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{experiment_id}: no {meta_path} sidecar for this match cache -- it "
            f"predates explicit ground-truth-file tracking and cannot be trusted "
            f"to have been built against {ground_truth_path}. Rerun "
            f"`python analysis/match_cache.py {experiment_id} "
            f"--ground-truth-file {ground_truth_path}` (or --config) to rebuild it."
        )
    with open(meta_path) as f:
        cache_meta = json.load(f)

    actual = {
        "ground_truth_file": repo_relative(ground_truth_path),
        "ground_truth_sha256": sha256_file(ground_truth_path),
        "n_gt": len(ground_truth_df),
    }
    mismatches = [
        f"{key}: cache={cache_meta.get(key)!r} vs current={actual[key]!r}"
        for key in actual if cache_meta.get(key) != actual[key]
    ]
    if mismatches:
        raise RuntimeError(
            f"{experiment_id}: {cache_path} was built against a different ground "
            f"truth than {ground_truth_path} ({'; '.join(mismatches)}). Rerun "
            f"`python analysis/match_cache.py {experiment_id} "
            f"--ground-truth-file {ground_truth_path}` (or --config) against the "
            f"ground truth this analysis actually wants."
        )

def assert_extraction_matches_cache(
    experiment_id: str, cache_path: Path, extraction_file_path: Path,
) -> None:
    """Raise unless match_cache.py's own match_cache.meta.json sidecar
    confirms cache_path was built against this exact extraction_file_path
    (postprocessed.json, or final.json on fallback -- see
    extraction_path).

    Mirrors assert_ground_truth_matches_cache, for the other file a cache's
    (gt_idx, ex_idx) edges are positions into. Without this, a cache built
    from final.json (before analysis/postprocessing.py had ever run for this
    id) would look perfectly valid by every other guard -- same row count,
    fresher mtime -- if later scored against a postprocessed.json with
    different point_value/units values for the same rows: the cached edges
    would silently no longer reflect what ext_matched_mask/validity actually
    read, exactly the "runs clean, wrong number" failure this repo is built
    to avoid. Checked by content hash, not just which filename was used, so
    an edited-in-place postprocessed.json (e.g. after widening
    parsing.py's unit-variant table and rerunning analysis/postprocessing.py)
    is also caught.

    Raises:
        FileNotFoundError: no match_cache.meta.json sidecar, or one that
            predates extraction-file tracking (built before this check
            existed).
        RuntimeError: the sidecar's extraction_file/sha256 disagree with
            extraction_file_path.
    """
    meta_path = cache_path.with_name("match_cache.meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{experiment_id}: no {meta_path} sidecar for this match cache. Rerun "
            f"`python analysis/match_cache.py {experiment_id} "
            f"--ground-truth-file <path>` (or --config) to rebuild it."
        )
    with open(meta_path) as f:
        cache_meta = json.load(f)

    if "extraction_file" not in cache_meta:
        raise FileNotFoundError(
            f"{experiment_id}: {meta_path} predates extraction-file tracking "
            f"(postprocessed.json-vs-final.json fallback) and cannot be trusted to "
            f"have been built against {extraction_file_path}. Rerun "
            f"`python analysis/match_cache.py {experiment_id} "
            f"--ground-truth-file <path>` (or --config) to rebuild it."
        )

    actual = {
        "extraction_file": repo_relative(extraction_file_path),
        "extraction_sha256": sha256_file(extraction_file_path),
    }
    mismatches = [
        f"{key}: cache={cache_meta.get(key)!r} vs current={actual[key]!r}"
        for key in actual if cache_meta.get(key) != actual[key]
    ]
    if mismatches:
        raise RuntimeError(
            f"{experiment_id}: {cache_path} was built against a different extraction "
            f"file than {extraction_file_path} ({'; '.join(mismatches)}). Rerun "
            f"`python analysis/match_cache.py {experiment_id} "
            f"--ground-truth-file <path>` (or --config) to rebuild it against the "
            f"extraction file this analysis actually wants."
        )

def assert_matching_columns_present(
    ground_truth_df: pd.DataFrame, extraction_df: pd.DataFrame, cfg: dict,
) -> None:
    """Raise if a column the dataset's matching config names isn't in the
    current frames.

    The mtime check above only catches a cache that predates its OWN source
    files -- it says nothing about whether that cache was ever built under
    the CURRENT matching config in the first place. A match_cache.pkl left
    over from a different matching-rules era (different strict/fuzzy column
    names, e.g. a legacy ``value``/``converted_value`` cache sitting where a
    ``point_value``-keyed one is now expected) can still look "fresh" by
    mtime alone. This is a cheap, independent guard: match_datasets can't
    have produced today's cache from a frame that doesn't even have the
    columns today's config asks it to match on.
    """
    missing_gt = sorted({
        col for col in list(cfg["strict"]) + list(cfg.get("fuzzy") or {})
        if col not in ground_truth_df.columns
    })
    missing_ext = sorted({
        col for col in list(cfg["strict"].values()) + list((cfg.get("fuzzy") or {}).values())
        if col not in extraction_df.columns
    })
    if missing_gt or missing_ext:
        raise KeyError(
            f"matching config column(s) missing from the current data -- "
            f"ground truth missing {missing_gt}, extraction missing "
            f"{missing_ext}. A match_cache.pkl built under different matching "
            f"rules (or against an extraction/ground-truth schema from before "
            f"a later column rename) cannot be trusted here even if its mtime "
            f"looks fresh -- rerun `python analysis/match_cache.py <experiment_id>` "
            f"only after confirming these columns exist in the current data."
        )


def load_cached_matching(extraction_id: str, ground_truth_path: Path):
    """Frames + thresholded match edges for one extraction, from the
    match_cache.pkl analysis/match_cache.py already built. Never computes a
    matching: a missing, stale, or wrong-provenance cache is a hard error.

    Reuses analysis/recovery_validity.py's own loading and guards, so
    calibration reads the exact frames (postprocessed.json if present, else
    final.json -- no unit conversion, no row filtering) and the exact
    dataset-config matching rules/threshold recovery_validity.py does:

      - the cache's match_cache.meta.json must record this ground_truth_path
        (sha256 + n_gt) and this extraction file (sha256);
      - the cache must be newer than both source files;
      - the dataset config's strict/fuzzy columns must exist in both frames;
      - every selected edge must index inside the current frames.

    The cache holds every strict-matched candidate (built at threshold 0.0);
    the dataset config's own fuzzy_threshold is applied here via
    load_match_cache(..., fuzzy_threshold=...).

    Returns:
        (ground_truth_df, extraction_df, edges) -- edges are (gt_idx, ex_idx)
        positions into those two frames.
    """
    _dataset, dataset_config, gt_df, ext_df, ext_path, gt_path = load_frames(
        extraction_id, ground_truth_path,
    )
    cfg = get_matching_config(dataset_config)
    assert_matching_columns_present(gt_df, ext_df, cfg)

    cache_path = match_cache_path(extraction_id)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{extraction_id}: no match_cache.pkl at {cache_path}. Run "
            f"`python analysis/match_cache.py {extraction_id} "
            f"--ground-truth-file {gt_path}` first -- calibration never builds matchings."
        )
    assert_cache_fresh(cache_path, ext_path, gt_path)
    assert_ground_truth_matches_cache(extraction_id, cache_path, gt_path, gt_df)
    assert_extraction_matches_cache(extraction_id, cache_path, ext_path)

    edges = load_match_cache(extraction_id, fuzzy_threshold=cfg["fuzzy_threshold"])
    n_gt, n_ext = len(gt_df), len(ext_df)
    for gt_idx, ex_idx in edges:
        if not (0 <= gt_idx < n_gt and 0 <= ex_idx < n_ext):
            raise RuntimeError(
                f"{extraction_id}: cached edge ({gt_idx}, {ex_idx}) out of range for "
                f"n_gt={n_gt}, n_ext={n_ext} -- rerun analysis/match_cache.py"
            )
    return gt_df, ext_df, edges

def edges_to_judged_rows(edges, ext_df, judged_df):
    """Re-index cached (gt_idx, ex_idx) edges from the match cache's extraction
    frame onto the judged rows (combined.json / the judge's activations).

    match_cache.py matches against postprocessed.json, which list-expands a row
    into several rows sharing one ``measurement_id`` (see analysis/
    postprocessing.py); the judge, its activations and combined.json are per
    original row. Each edge's ex_idx is mapped through ``measurement_id`` to the
    judged row's position, and duplicates collapse -- so a judged row "has an
    edge" if any of its expanded rows matched, and a GT row is recovered by a
    judged row if any of that row's expanded rows matched it.

    Position-only alignment is never assumed: judged_df's measurement_ids must
    be exactly range(n) in row order, every ext_df measurement_id must be one of
    them, and ext_df/judged_df must agree on document_id and attribute at every
    shared measurement_id (same check as recovery_validity.load_validity_labels).

    Returns:
        Sorted list of unique (gt_idx, judged_row_idx).

    Raises:
        ValueError: any alignment condition above fails.
    """
    n = len(judged_df)
    judged_mids = judged_df["measurement_id"].tolist()
    if judged_mids != list(range(n)):
        raise ValueError("judged rows' measurement_id is not range(n) in row order")
    ext_mids = ext_df["measurement_id"].to_numpy()
    if not ((ext_mids >= 0) & (ext_mids < n)).all():
        raise ValueError("extraction has measurement_id(s) outside the judged rows' range")
    for col in ("document_id", "attribute"):
        if not (ext_df[col].to_numpy() == judged_df[col].to_numpy()[ext_mids]).all():
            raise ValueError(f"extraction and judged rows disagree on {col} at some measurement_id")
    n_ext = len(ext_df)
    mapped = set()
    for gt_idx, ex_idx in edges:
        if not 0 <= ex_idx < n_ext:
            raise ValueError(f"edge ex_idx {ex_idx} out of range for {n_ext} extraction rows")
        mapped.add((int(gt_idx), int(ext_mids[ex_idx])))
    return sorted(mapped)
