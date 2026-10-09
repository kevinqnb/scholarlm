"""Recovery / validity building blocks shared by recovery_validity.py and decision_threshold.py.

Loads an extraction's checked match-cache inputs, finds and verifies its
judge_combine run and labels, and builds recovery / validity masks, cross-checked
against analysis/common/metrics.py. Recovery = ground-truth row has any surviving
edge; validity = judged valid OR matched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "experiments", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import utils as paths  # noqa: E402
from analysis.common import matching  # noqa: E402
from analysis.common.matching import (  # noqa: E402
    assert_cache_fresh, assert_extraction_matches_cache, assert_ground_truth_matches_cache,
    assert_matching_columns_present, load_frames,
)
from analysis.common.metrics import recovery_rate as _recovery_rate, validity_rate as _validity_rate  # noqa: E402


def _resolve_judged_extraction_ids(judge_ids: list[str]) -> tuple[set[str], str | None]:
    """Extraction ids that the judge runs' committed configs point at.

    Args:
        judge_ids: Judge experiment ids from a judge_combine config.

    Returns:
        ``(judged_extraction_ids, skip_reason)``. ``skip_reason`` is None on success;
        otherwise the set is incomplete and the caller decides whether to skip or fail.
    """
    judged_extraction_ids: set[str] = set()
    for judge_id in judge_ids:
        try:
            judge_config_path = paths.find_experiment_config(judge_id)
        except FileNotFoundError:
            return judged_extraction_ids, f"judge_id {judge_id!r} has no committed experiment-config"
        judge_extraction_id = paths.load_experiment_config(judge_config_path)["params"].get("extraction_id")
        if judge_extraction_id is None:
            return judged_extraction_ids, f"judge_id {judge_id!r} has no params.extraction_id (synthetic judge?)"
        judged_extraction_ids.add(judge_extraction_id)
    return judged_extraction_ids, None

def _cross_check_judge_run_metadata(combine_id: str, judge_ids: list[str], extraction_id: str) -> None:
    """Fail if a finished judge run's run_metadata.json names a different extraction.

    Catches a judge run executed before its config was edited.

    Args:
        combine_id: judge_combine id (for error messages).
        judge_ids: Judge experiment ids.
        extraction_id: Extraction the configs point at.

    Raises:
        ValueError: A run_metadata.json disagrees with its config.
    """
    for judge_id in judge_ids:
        try:
            judge_dir = paths.find_result_dir(judge_id)
        except FileNotFoundError:
            continue
        run_meta = paths.load_run_metadata(judge_dir)
        if run_meta is not None and run_meta.get("extraction_id") not in (None, extraction_id):
            raise ValueError(
                f"{combine_id}: judge_id {judge_id!r}'s committed config points at "
                f"extraction_id={extraction_id!r}, but its own run_metadata.json "
                f"recorded extraction_id={run_meta.get('extraction_id')!r} -- it was "
                f"not run against the config it currently has"
            )

def verify_judge_combine_id(dataset: str, judge_combine_id: str, extraction_id: str) -> list[str]:
    """Check a judge_combine id declared in an analysis config judges this extraction.

    Same check ``find_judge_combine_id`` applies per candidate, without the scan.
    Use it to pick one when several combines judge the same extraction.

    Args:
        dataset: Dataset name.
        judge_combine_id: Declared judge_combine id.
        extraction_id: Extraction it should judge.

    Returns:
        The combine's judge_ids.

    Raises:
        FileNotFoundError: Missing config for the combine or one of its judges.
        ValueError: Wrong dataset/type, judges resolve to another extraction, or a
            run_metadata.json disagrees with its config.
    """
    config_path = paths.find_experiment_config(judge_combine_id)
    if config_path.parts[-4] != dataset or config_path.parts[-3] != "judge_combine":
        raise ValueError(
            f"judge_combine_id={judge_combine_id!r} resolves to {config_path}, "
            f"not under experiment-configs/{dataset}/judge_combine/"
        )
    cfg = paths.load_experiment_config(config_path)
    judge_ids = cfg["params"]["judge_ids"]

    judged_extraction_ids, skip_reason = _resolve_judged_extraction_ids(judge_ids)
    if skip_reason is not None:
        raise FileNotFoundError(f"{config_path}: {skip_reason}")
    if judged_extraction_ids != {extraction_id}:
        raise ValueError(
            f"{config_path}: its own judge_ids {judge_ids} resolve to "
            f"extraction_id(s) {sorted(judged_extraction_ids)}, not the "
            f"declared extraction_id={extraction_id!r}"
        )

    _cross_check_judge_run_metadata(judge_combine_id, judge_ids, extraction_id)
    return judge_ids

def find_judge_combine_id(dataset: str, extraction_id: str) -> tuple[str, list[str]]:
    """Find the one judge_combine run whose judges all judged this extraction.

    Scans the dataset's judge_combine configs and follows each judge's committed
    ``params.extraction_id`` (ids share no naming convention, so they are never
    string-matched). Candidates that can't be resolved, such as synthetic judges,
    are skipped and reported.

    Args:
        dataset: Dataset name.
        extraction_id: Extraction experiment id.

    Returns:
        ``(judge_combine_id, judge_ids)``.

    Raises:
        FileNotFoundError: No combine matches.
        ValueError: Several match, a combine's judges disagree on the extraction,
            or the winner's run_metadata.json disagrees with its config.
    """
    combine_dir = paths.EXPERIMENT_CONFIGS_ROOT / dataset / "judge_combine"
    matches: list[tuple[str, list[str]]] = []
    skipped: list[tuple[str, str]] = []

    for config_path in sorted(combine_dir.glob("*/*.yaml")):
        cfg = paths.load_experiment_config(config_path)
        judge_ids = cfg["params"]["judge_ids"]

        judged_extraction_ids, skip_reason = _resolve_judged_extraction_ids(judge_ids)
        if skip_reason is not None:
            skipped.append((cfg["id"], skip_reason))
            continue

        if len(judged_extraction_ids) != 1:
            raise ValueError(
                f"{config_path}: its own judge_ids {judge_ids} disagree about "
                f"which extraction they judged: {sorted(judged_extraction_ids)}"
            )
        if judged_extraction_ids == {extraction_id}:
            matches.append((cfg["id"], judge_ids))

    if not matches:
        skip_note = f" Skipped candidates: {skipped}." if skipped else ""
        raise FileNotFoundError(
            f"No judge_combine experiment under "
            f"experiment-configs/{dataset}/judge_combine/ has judge_ids that all "
            f"trace back to extraction_id={extraction_id!r}.{skip_note} Run "
            f"run_judge_local.py/run_judge_interp.py + run_judge_combine.py for "
            f"it first."
        )
    if len(matches) > 1:
        raise ValueError(
            f"extraction_id={extraction_id!r} is judged by more than one "
            f"judge_combine experiment (ambiguous which ground truth to use): "
            f"{sorted(m[0] for m in matches)}"
        )
    if skipped:
        print(f"  find_judge_combine_id: skipped {len(skipped)} unrelated candidate(s): {skipped}")

    winning_id, winning_judge_ids = matches[0]
    _cross_check_judge_run_metadata(winning_id, winning_judge_ids, extraction_id)
    return winning_id, winning_judge_ids

def load_validity_labels(judge_combine_id: str, extraction_df: pd.DataFrame) -> np.ndarray:
    """Per-row judge labels from combined.json, joined on (document_id, measurement_id).

    Rows split by postprocessing inherit their parent's label, even though the judge
    never saw them individually (see ``count_split_rows`` in recovery_validity.py).
    The key sets must match exactly and attributes must agree.

    Args:
        judge_combine_id: judge_combine experiment id.
        extraction_df: Extraction rows to label.

    Returns:
        Boolean array, one label per ``extraction_df`` row.

    Raises:
        FileNotFoundError: No combined.json.
        ValueError: Non-bool labels, duplicate keys, key-set mismatch, attribute
            mismatch, or no measurement_id column.
    """
    combined_path = paths.find_result_dir(judge_combine_id) / "combined.json"
    if not combined_path.exists():
        raise FileNotFoundError(f"No combined.json for {judge_combine_id!r} at {combined_path}")
    with open(combined_path) as f:
        combined = json.load(f)

    if "measurement_id" not in extraction_df.columns:
        raise ValueError(f"{judge_combine_id}: extraction data has no measurement_id column")

    non_bool = [r["measurement_id"] for r in combined if not isinstance(r["judgement_combined"], bool)]
    if non_bool:
        raise ValueError(
            f"{judge_combine_id}: judgement_combined is not a bool for "
            f"measurement_id(s) {non_bool[:10]}"
        )

    by_key = {(r["document_id"], r["measurement_id"]): r for r in combined}
    if len(by_key) != len(combined):
        raise ValueError(
            f"{judge_combine_id}: combined.json has {len(combined)} record(s) but only "
            f"{len(by_key)} distinct (document_id, measurement_id) key(s)"
        )

    ext_keys = list(zip(extraction_df["document_id"], extraction_df["measurement_id"]))
    ext_key_set = set(ext_keys)
    unjudged = ext_key_set - set(by_key)
    judged_but_absent = set(by_key) - ext_key_set
    if unjudged or judged_but_absent:
        raise ValueError(
            f"{judge_combine_id}: the judged and extracted (document_id, measurement_id) "
            f"keys disagree -- not the same run: {len(unjudged)} extracted key(s) have no "
            f"judgement (e.g. {sorted(unjudged, key=str)[:3]}), {len(judged_but_absent)} judged "
            f"key(s) are absent from the extraction (e.g. {sorted(judged_but_absent, key=str)[:3]}); "
            f"extraction likely re-run, or postprocessed, after judging in a way that is not "
            f"a pure split of judged rows"
        )

    mismatched = [
        i for i, (key, attribute) in enumerate(zip(ext_keys, extraction_df["attribute"]))
        if by_key[key]["attribute"] != attribute
    ]
    if mismatched:
        raise ValueError(
            f"{judge_combine_id}: {len(mismatched)} row(s) disagree with the judged data on "
            f"attribute at the same (document_id, measurement_id) -- first mismatch at "
            f"extraction row {mismatched[0]}"
        )

    return np.array([by_key[key]["judgement_combined"] for key in ext_keys], dtype=bool)

def gt_recovered_mask(n_gt: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Mark ground-truth rows that have at least one edge.

    Args:
        n_gt: Number of ground-truth rows.
        edges: (gt_idx, ex_idx) pairs.

    Returns:
        Boolean array of length ``n_gt``.
    """
    mask = np.zeros(n_gt, dtype=bool)
    for gt_idx, _ex_idx in edges:
        mask[gt_idx] = True
    return mask

def ext_matched_mask(n_ext: int, edges: list[tuple[int, int]]) -> np.ndarray:
    """Mark extraction rows that have at least one edge.

    Args:
        n_ext: Number of extraction rows.
        edges: (gt_idx, ex_idx) pairs.

    Returns:
        Boolean array of length ``n_ext``.
    """
    mask = np.zeros(n_ext, dtype=bool)
    for _gt_idx, ex_idx in edges:
        mask[ex_idx] = True
    return mask

def filter_edges_by_threshold(
    edges: list[tuple[int, int]], edge_weights: list[float], fuzzy_threshold: float,
) -> tuple[list[tuple[int, int]], list[float]]:
    """Like ``matching.edges_above_threshold``, but also returns the kept weights.

    Args:
        edges: (gt_idx, ex_idx) pairs.
        edge_weights: Weight per edge.
        fuzzy_threshold: Minimum weight to keep, inclusive.

    Returns:
        ``(kept_edges, kept_weights)``, aligned.

    Raises:
        ValueError: ``edges`` and ``edge_weights`` differ in length.
    """
    if len(edges) != len(edge_weights):
        raise ValueError(f"{len(edges)} edges vs {len(edge_weights)} weights")
    kept = [(e, w) for e, w in zip(edges, edge_weights) if w >= fuzzy_threshold]
    return [e for e, _ in kept], [w for _, w in kept]

def verify_recovery(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    cfg: dict,
    threshold: float,
    cache_path: Path,
    recovered: np.ndarray,
) -> None:
    """Check the recovered mask against ``metrics.recovery_rate`` on the same cache.

    The masks here duplicate metrics.py's loop (which doesn't expose them), so this
    guards against the two silently diverging.

    Args:
        ground_truth_df: Ground-truth frame.
        extraction_df: Extraction frame.
        cfg: Output of ``get_matching_config``.
        threshold: Fuzzy threshold applied.
        cache_path: Match cache.
        recovered: Boolean mask per ground-truth row.

    Raises:
        AssertionError: The two recovery rates differ.
    """
    ref_recovery = _recovery_rate(
        ground_truth_df, extraction_df,
        strict_matching=cfg["strict"], fuzzy_matching=cfg["fuzzy"],
        fuzzy_threshold=threshold, cache_path=cache_path,
    )
    if not np.isclose(ref_recovery, recovered.mean()):
        raise AssertionError(
            f"recovery mismatch: {recovered.mean()} (this script) vs "
            f"{ref_recovery} (analysis.metrics.recovery_rate) -- the two "
            f"recovered-row computations have diverged"
        )

def verify_validity(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    cfg: dict,
    threshold: float,
    cache_path: Path,
    matched: np.ndarray,
    judged_labels: np.ndarray,
) -> np.ndarray:
    """Check matched-OR-judged labels against ``metrics.validity_rate``, then return them.

    Args:
        ground_truth_df: Ground-truth frame.
        extraction_df: Extraction frame.
        cfg: Output of ``get_matching_config``.
        threshold: Fuzzy threshold applied.
        cache_path: Match cache.
        matched: Boolean mask per extraction row.
        judged_labels: Boolean judge label per extraction row.

    Returns:
        Validity labels, ``judged_labels | matched``.

    Raises:
        AssertionError: The two validity rates differ.
    """
    judged_df = pd.DataFrame({"judgement_combined": judged_labels})
    ref_validity = _validity_rate(
        ground_truth_df, extraction_df,
        strict_matching=cfg["strict"], fuzzy_matching=cfg["fuzzy"],
        fuzzy_threshold=threshold, judged_df=judged_df, cache_path=cache_path,
    )
    validity_labels = judged_labels | matched
    if not np.isclose(ref_validity, validity_labels.mean()):
        raise AssertionError(
            f"validity mismatch: {validity_labels.mean()} (this script) vs "
            f"{ref_validity} (analysis.metrics.validity_rate) -- the two "
            f"matched/judged-row computations have diverged"
        )
    return validity_labels

def load_checked_inputs(experiment_id: str, ground_truth_path: Path) -> dict:
    """Load one extraction's frames, matching rules and unthresholded cache edges.

    Runs every cache guard (columns, freshness, provenance, edge range).

    Args:
        experiment_id: Extraction experiment id.
        ground_truth_path: Ground-truth file.

    Returns:
        Dict with ``dataset``, ``ground_truth_df``, ``extraction_df``,
        ``extraction_file_path``, ``ground_truth_path``, ``cfg``, ``cache_path``,
        ``raw_edges``, ``raw_weights``.

    Raises:
        FileNotFoundError: No cache or sidecar.
        KeyError, RuntimeError: A guard fails.
    """
    dataset, dataset_config, ground_truth_df, extraction_df, extraction_file_path, ground_truth_path = load_frames(
        experiment_id, ground_truth_path,
    )

    cfg = matching.get_matching_config(dataset_config)
    assert_matching_columns_present(ground_truth_df, extraction_df, cfg)

    cache_path = matching.match_cache_path(experiment_id)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{experiment_id}: no match_cache.pkl at {cache_path}. Run "
            f"`python analysis/match_cache.py {experiment_id} "
            f"--ground-truth-file {ground_truth_path}` first."
        )
    assert_cache_fresh(cache_path, extraction_file_path, ground_truth_path)
    assert_ground_truth_matches_cache(experiment_id, cache_path, ground_truth_path, ground_truth_df)
    assert_extraction_matches_cache(experiment_id, cache_path, extraction_file_path)

    n_gt, n_ext = len(ground_truth_df), len(extraction_df)
    # Cache is built at threshold 0.0, so these are all strict-matched candidates.
    _matching, raw_edges, raw_weights = matching.load_match_cache(experiment_id)
    for gt_idx, ex_idx in raw_edges:
        if not (0 <= gt_idx < n_gt and 0 <= ex_idx < n_ext):
            raise RuntimeError(
                f"{experiment_id}: cached edge (gt_idx={gt_idx}, ex_idx={ex_idx}) out "
                f"of range for n_gt={n_gt}, n_ext={n_ext} -- the cache no longer "
                f"matches the current ground truth/extraction rows. Rerun "
                f"`python analysis/match_cache.py {experiment_id}`."
            )

    return {
        "dataset": dataset,
        "ground_truth_df": ground_truth_df,
        "extraction_df": extraction_df,
        "extraction_file_path": extraction_file_path,
        "ground_truth_path": ground_truth_path,
        "cfg": cfg,
        "cache_path": cache_path,
        "raw_edges": raw_edges,
        "raw_weights": raw_weights,
    }

def resolve_judged_labels(
    dataset: str, experiment_id: str, judge_combine_id: str | None, extraction_df: pd.DataFrame,
) -> tuple[str, list[str], np.ndarray]:
    """Resolve the judge_combine run for an extraction and load its per-row labels.

    Args:
        dataset: Dataset name.
        experiment_id: Extraction experiment id.
        judge_combine_id: Declared combine id to verify, or None to search.
        extraction_df: Extraction rows to label.

    Returns:
        ``(judge_combine_id, judge_ids, labels)``.
    """
    if judge_combine_id is not None:
        judge_ids = verify_judge_combine_id(dataset, judge_combine_id, experiment_id)
    else:
        judge_combine_id, judge_ids = find_judge_combine_id(dataset, experiment_id)
    return judge_combine_id, judge_ids, load_validity_labels(judge_combine_id, extraction_df)
