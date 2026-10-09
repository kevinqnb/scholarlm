"""Recovery / validity building blocks shared by analysis/recovery_validity.py (the
paper-clustered bootstrap report) and analysis/decision_threshold.py: loading one
extraction's checked inputs from its match cache, resolving and verifying its
judge_combine run and validity labels, and the recovery / validity masks with their
independent re-verification.

Moved verbatim out of recovery_validity.py; see that module's docstring for the
definitions (recovery = any surviving edge; validity = judged valid OR matched).
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
    """The set of extraction_id(s) that judge_ids' own committed configs
    (``params.extraction_id``) resolve to -- the per-candidate check shared
    by find_judge_combine_id's scan and verify_judge_combine_id's single-
    candidate check, factored out so the two never drift apart.

    Returns:
        (judged_extraction_ids, skip_reason). skip_reason is None on full
        success; otherwise judged_extraction_ids is incomplete and the
        caller decides whether that's a skip (scanning) or a hard error
        (a config-declared id, which has no other candidate to fall back to).
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
    """Raise if a judge_id has actually run (has a run_metadata.json) and its
    recorded extraction_id disagrees with its own committed config -- shared
    tail check for find_judge_combine_id and verify_judge_combine_id.

    Raises:
        ValueError: a judge_id's run_metadata.json disagrees with its config.
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
    """Validate a config-DECLARED judge_combine_id against extraction_id, for
    an analysis config's optional ``params.recovery_validity.judge_combine_ids``
    override (see analysis/common/config.py) -- so a wrong declared id
    still fails loud rather than being trusted blindly.

    Runs the exact same per-candidate check find_judge_combine_id applies to
    every candidate it finds by scanning, just against this one id instead of
    all of experiment-configs/{dataset}/judge_combine/*/*.yaml -- it never
    replaces or loosens that check, only skips the scan. This also means it
    can disambiguate a case where more than one judge_combine run judges the
    same extraction, which find_judge_combine_id itself would refuse
    (ValueError: ambiguous).

    Returns:
        judge_ids -- same shape as find_judge_combine_id's second return value.

    Raises:
        FileNotFoundError: no committed config for judge_combine_id, or one
            of its own judge_ids has no committed config.
        ValueError: judge_combine_id isn't under
            experiment-configs/{dataset}/judge_combine/, its judge_ids
            disagree with each other or don't resolve to extraction_id, or a
            judge_id's run_metadata.json disagrees with its committed config.
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
    """Find the judge_combine experiment whose judge_ids all judged extraction_id.

    Scans ``experiments/experiment-configs/{dataset}/judge_combine/*/*.yaml``
    and, for each one, resolves its ``params.judge_ids`` back to the
    extraction/ablation id each judge run judges, by reading each judge_id's
    own COMMITTED config (``params.extraction_id``) -- not that judge run's
    ``run_metadata.json``, which only exists after the run has finished and
    would otherwise force an unrelated, not-yet-run candidate to be treated
    as unresolvable. Never inferred from the id strings themselves -- an
    extraction id and the judge/combine ids that judge it are minted
    independently and share no literal substring convention (e.g.
    ``2026-05-05-pond-gemma-3-27b-extraction-01`` is judged by
    ``2026-09-13-pond-gemma3-27b-extraction-judge-combine-01``).

    Once exactly one combine matches, its judge_ids' own run_metadata.json
    (where a run has actually completed) is cross-checked against their
    committed extraction_id, so a judge run executed against a since-edited
    config still gets caught rather than silently trusted.

    A candidate combine whose own judge_ids can't all be resolved this way
    (missing config, or a judge config with no ``params.extraction_id`` --
    e.g. a synthetic-probe judge) is not a hard error by itself, since it can
    never be the id being searched for either way, but it is never silent:
    every skip is collected and named in the eventual no-match error, and
    printed even when a match is found.

    Returns:
        (judge_combine_id, judge_ids).

    Raises:
        FileNotFoundError: no judge_combine config's judge_ids all resolve to
            extraction_id.
        ValueError: more than one does (ambiguous which ground truth to use),
            a single combine's own judge_ids disagree with each other about
            which extraction they judged (a malformed combine config), or
            the winning combine's judge_ids' run_metadata.json disagrees with
            their own committed config.
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
    """Load combined.json for judge_combine_id and return one bool judgement
    per extraction_df row, joined on ``(document_id, measurement_id)``.

    ``measurement_id`` is unique per row in final.json, but
    analysis/postprocessing.py can split one multi-value row (e.g. ``"3,4"``)
    into several postprocessed.json rows that all keep the parent's
    measurement_id. The judges only ever saw the parent, so every split child
    INHERITS its parent's judgement -- including a child the judges never saw
    on its own (a decimal-comma value split into two bogus values inherits the
    parent's label). Use ``count_split_rows`` to report how many rows that is.

    The join is exact: combined.json's keys must be unique and must equal
    extraction_df's key set (no judged measurement missing from the extraction,
    no extracted measurement unjudged), and ``attribute`` must agree between the
    two sides at every key. Row order in either file is irrelevant.

    Raises:
        FileNotFoundError: no combined.json for judge_combine_id.
        ValueError: duplicate keys in combined.json, a missing measurement_id
            column, a key set mismatch between the two sides, an attribute
            disagreement at a shared key, or a non-bool judgement_combined.
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
    mask = np.zeros(n_gt, dtype=bool)
    for gt_idx, _ex_idx in edges:
        mask[gt_idx] = True
    return mask

def ext_matched_mask(n_ext: int, edges: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(n_ext, dtype=bool)
    for _gt_idx, ex_idx in edges:
        mask[ex_idx] = True
    return mask

def filter_edges_by_threshold(
    edges: list[tuple[int, int]], edge_weights: list[float], fuzzy_threshold: float,
) -> tuple[list[tuple[int, int]], list[float]]:
    """(edges, weights) with ``w >= fuzzy_threshold`` -- the same inclusive
    boundary as ``matching.edges_above_threshold``, but keeping the weights
    (which the matching needs) aligned with the surviving edges.
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
    """Assert this script's threshold-filtered any-edge recovered mask -- the
    reported ``recovery`` -- agrees with ``analysis.metrics.recovery_rate``
    computed against the exact same cache. (The secondary
    ``recovery_max_weight_matching`` is guarded separately, by
    ``_verify_matching``.)

    ``gt_recovered_mask``/``ext_matched_mask`` above are a near-duplicate of
    metrics.py's own internal edge-filtering loop (kept separate only
    because metrics.py doesn't expose the boolean arrays) -- this guards
    against that duplication silently diverging from the reviewed eval code
    it mirrors, per CLAUDE.md's rule against re-deriving evaluation logic
    unchecked.

    Raises:
        AssertionError: the two recovery computations disagree.
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
    """Assert this script's matched/judged masks agree with
    ``analysis.metrics.validity_rate`` computed against the exact same
    cache, then return the OR'd validity-label array. See ``verify_recovery``.

    Raises:
        AssertionError: the two validity computations disagree.
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
    """Load one id's frames, matching config and raw threshold-0 cache edges,
    running every cache guard -- the shared front half of
    ``compute_metrics_for_id`` and ``fuzzy_threshold_curve``.

    Returns a dict with keys dataset, ground_truth_df, extraction_df,
    extraction_file_path, ground_truth_path, cfg, cache_path, raw_edges,
    raw_weights. Raises loud on everything ``compute_metrics_for_id``'s
    docstring lists for the cache.
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
    # Cache is always built at fuzzy_threshold=0.0 (see match_cache.py's
    # module docstring): the raw edge list is every strict-matched candidate.
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
    """(judge_combine_id, judge_ids, per-row judged labels) for experiment_id:
    a declared judge_combine_id is verified (``verify_judge_combine_id``),
    otherwise one is found by scanning (``find_judge_combine_id``)."""
    if judge_combine_id is not None:
        judge_ids = verify_judge_combine_id(dataset, judge_combine_id, experiment_id)
    else:
        judge_combine_id, judge_ids = find_judge_combine_id(dataset, experiment_id)
    return judge_combine_id, judge_ids, load_validity_labels(judge_combine_id, extraction_df)
