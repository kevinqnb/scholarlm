"""Read match caches, apply matching rules, and guard against stale or mismatched caches.

analysis/match_cache.py builds each experiment's ground-truth <-> extraction candidate
graph. This module locates, loads, thresholds and verifies it, and never computes a
matching itself.
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
from analysis.common.config import analysis_results_dir  # noqa: E402

# Each cache lives in MATCH_CACHE_ROOT/<experiment id>/ as match_cache.pkl plus a
# match_cache.meta.json sidecar recording the input files it was built from.
MATCH_CACHE_ROOT = analysis_results_dir("match-cache")

# Matching rules come from each DatasetConfig:
#   strict / fuzzy: ground-truth column -> extraction column, passed to match_datasets.
#   numeric_coerce: strict columns forced to float on both sides first. Extracted
#     point_value is an LLM string ("0.26"), and match_datasets only compares
#     numerically when both sides are numbers, so uncoerced values never match.
# Unparseable values become NaN. match_datasets treats NaN as null, and null matches
# null, so such a row can still strict-match a ground-truth row that is null there.

def parse_numeric(x):
    """Parse a value to float for strict-match coercion.

    Args:
        x: Number, numeric string, or ``"<mantissa> × 10^<exponent>"`` string.

    Returns:
        The float, or NaN for None, bools, other types and unparseable strings.
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
    """Read a dataset's matching rules from its DatasetConfig.

    Args:
        dataset_config: The dataset's ``DatasetConfig``.

    Returns:
        Dict with ``strict``, ``fuzzy``, ``fuzzy_threshold``, ``numeric_coerce``
        (default []) and ``fuzzy_normalizers`` (default {}).

    Raises:
        KeyError: ``strict_matching``, ``fuzzy_matching`` or ``fuzzy_threshold`` is unset.
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
    """Keep edges with weight >= ``fuzzy_threshold`` (same boundary as match_datasets).

    Args:
        edges: (gt_idx, ex_idx) pairs from a threshold-0.0 cache.
        edge_weights: Weight per edge.
        fuzzy_threshold: Minimum weight to keep, inclusive.

    Returns:
        The kept (gt_idx, ex_idx) pairs, in input order.
    """
    return [
        (gt_idx, ex_idx)
        for (gt_idx, ex_idx), w in zip(edges, edge_weights)
        if w >= fuzzy_threshold
    ]


def match_cache_path(experiment_id: str) -> Path:
    """Path of an experiment's match cache (whether or not it exists).

    Args:
        experiment_id: Extraction experiment id.

    Returns:
        ``MATCH_CACHE_ROOT/<experiment_id>/match_cache.pkl``.
    """
    return MATCH_CACHE_ROOT / experiment_id / "match_cache.pkl"


def match_cache_meta_path(experiment_id: str) -> Path:
    """Path of the sidecar recording which input files a match cache was built from.

    Args:
        experiment_id: Extraction experiment id.

    Returns:
        ``MATCH_CACHE_ROOT/<experiment_id>/match_cache.meta.json``.
    """
    return match_cache_path(experiment_id).with_name("match_cache.meta.json")


def load_match_cache(
    experiment_id: str, fuzzy_threshold: float | None = None,
) -> tuple | list[tuple[int, int]]:
    """Load a built match cache. Never builds one (a build takes tens of minutes).

    Args:
        experiment_id: Extraction experiment id.
        fuzzy_threshold: If given, return only edges with weight >= this.

    Returns:
        ``(matching, edges, edge_weights)``, or the filtered edge list when
        ``fuzzy_threshold`` is set.

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
    """Pick the extraction file to match: postprocessed.json, else final.json with a warning.

    Defined once so cache building and cache reading always agree.

    Args:
        experiment_id: Extraction experiment id.

    Returns:
        ``(path, used_fallback)``; ``used_fallback`` is True when final.json was used.

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
    """Load ground-truth and extraction frames exactly as match_cache.py did.

    Cached edges are row positions, so the frames must be loaded the same way
    (reset_index, no conversion, no filtering) or the cache silently misaligns.

    Args:
        experiment_id: Extraction experiment id.
        ground_truth_path: Ground-truth file, given explicitly.

    Returns:
        Tuple of ``(dataset, dataset_config, ground_truth_df, extraction_df,
        extraction_file_path, ground_truth_path)``.

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
    """Fail if a cache is older than any of its source files.

    Args:
        cache_path: The cache file.
        *source_paths: Files the cache was built from.

    Raises:
        RuntimeError: A source file is newer than the cache.
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
    """Fail unless the cache sidecar records this exact ground-truth file (path, hash, rows).

    The mtime check cannot tell which ground truth a cache used; a different one
    would silently misalign every cached edge.

    Args:
        experiment_id: Extraction experiment id.
        cache_path: The match cache.
        ground_truth_path: Ground-truth file in use now.
        ground_truth_df: Its loaded frame (for the row count).

    Raises:
        FileNotFoundError: No sidecar.
        RuntimeError: Sidecar path, sha256 or row count differs.
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
    """Fail unless the cache sidecar records this exact extraction file (path and hash).

    Catches a cache built from final.json (or an older postprocessed.json) being
    scored against different extraction rows.

    Args:
        experiment_id: Extraction experiment id.
        cache_path: The match cache.
        extraction_file_path: Extraction file in use now.

    Raises:
        FileNotFoundError: No sidecar, or one without extraction-file fields.
        RuntimeError: Sidecar path or sha256 differs.
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
    """Fail if a column named by the matching config is missing from either frame.

    A cheap check that the cache could have been built under the current rules.

    Args:
        ground_truth_df: Ground-truth frame.
        extraction_df: Extraction frame.
        cfg: Output of ``get_matching_config``.

    Raises:
        KeyError: Any strict/fuzzy column is missing.
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
    """Load frames and verified, thresholded match edges for one extraction.

    Runs every cache guard (columns present, freshness, ground-truth and extraction
    provenance, edges in range) and applies the dataset's fuzzy_threshold.

    Args:
        extraction_id: Extraction experiment id.
        ground_truth_path: Ground-truth file.

    Returns:
        Tuple of ``(ground_truth_df, extraction_df, edges)``; edges are (gt_idx, ex_idx)
        positions into the two frames.

    Raises:
        FileNotFoundError: No cache or sidecar.
        KeyError, RuntimeError: A guard fails.
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
    """Map cached edges from postprocessed extraction rows onto judged rows.

    postprocessed.json may split one judged row into several sharing a
    ``measurement_id``; edges are mapped back through that id and deduplicated.

    Args:
        edges: (gt_idx, ex_idx) pairs into ``ext_df``.
        ext_df: Extraction frame the cache was built on.
        judged_df: Judged rows, with ``measurement_id`` equal to range(n) in order.

    Returns:
        Sorted unique (gt_idx, judged_row_idx) pairs.

    Raises:
        ValueError: Ids are not aligned, document_id/attribute disagree, or an edge
            is out of range.
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
