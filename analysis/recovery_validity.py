"""Compute recovery and validity for a list of experiment ids, from pre-built
match caches (``analysis/match_cache.py``) and judge_combine judgements, with
paper-clustered bootstrap confidence intervals.

This is the centralized replacement for the recovery/validity halves of
``analysis/ablation.py`` and ``analysis/baselines.py`` -- same underlying
``analysis.metrics.recovery_rate``/``validity_rate`` semantics, but:

  - reads matching rules (``strict_matching``/``fuzzy_matching``/
    ``fuzzy_threshold``/``numeric_coerce``) off each dataset's own
    ``DatasetConfig`` in ``experiments/dataset-configs/{dataset}.py`` -- the
    same source ``analysis/match_cache.py`` reads via
    ``get_matching_config`` -- rather than keeping a second copy of those
    rules here;
  - reads a match cache instead of computing one (never recomputes -- run
    ``python analysis/match_cache.py <id>`` first for any id that doesn't
    have one yet, and refuses a cache that predates its own final.json or
    the dataset's ground truth file rather than silently matching against
    row positions that have since shifted underneath it). Every cache is
    built at ``fuzzy_threshold=0.0`` regardless of the dataset's configured
    threshold (see ``analysis/match_cache.py``'s module docstring), so this
    script always loads it back via
    ``match_cache.load_match_cache(id, fuzzy_threshold=<the dataset's own
    threshold>)`` -- never the raw 0.0 cache -- so the edges it works with
    are already the ones that threshold selects, not a second manual filter
    reimplementing the same cutoff;
  - resolves judgements by tracing each judge_combine experiment's own
    ``judge_ids`` back to the extraction/ablation id they judged (via each
    judge_id's own committed config), rather than a legacy
    ``(dataset, model, date)`` "most recent" lookup -- experiment ids for an
    extraction and its judge runs are minted independently and share no
    literal naming convention, so this is a real data join, not string
    matching. Optional: pass ``compute_validity=False``
    (``--skip-validity`` on the CLI) to report recovery only, for an id that
    has no judge_combine coverage yet;
  - reports a percentile bootstrap CI resampled over whole papers
    (document_id clusters), not the analytic Wilson interval
    ``recovery_rate``/``validity_rate`` return -- rows from the same paper
    are correlated, so a row-level interval understates uncertainty. The
    Wilson point estimate is still cross-checked against this script's own
    recovered/matched masks on every call (see ``_verify_recovery``/
    ``_verify_validity``) as a guard against the two silently diverging.

Only datasets whose ``DatasetConfig`` sets ``strict_matching``/
``fuzzy_matching``/``fuzzy_threshold`` are supported (pond, as of this
writing) -- the same restriction match_cache.py itself has.

Usage
-----
    python analysis/recovery_validity.py <id> [<id> ...] \\
        --n-resamples 2000 --seed 0 [--alpha 0.05] [--skip-validity] [--output PATH]

``--n-resamples`` and ``--seed`` are required, not defaulted (CLAUDE.md: no
inferred defaults for a value that changes the reported numbers) -- pass the
repo's own ``experiments/config.yaml`` ``defaults.seed`` for ``--seed`` to
keep it consistent with the rest of the repo's seeding.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from analysis import match_cache
from analysis.loaders import load_ground_truth
from analysis.metrics import recovery_rate as _recovery_rate, validity_rate as _validity_rate
from experiments.run_extraction import load_dataset_config
import utils as paths


# ---------------------------------------------------------------------------
# Loading + cache-freshness guard
# ---------------------------------------------------------------------------


def _ground_truth_path(dataset_config) -> Path:
    """Resolve a DatasetConfig's ground_truth_file the same way
    analysis.loaders.load_ground_truth does, without loading it -- needed
    here only for its mtime.
    """
    path = Path(dataset_config.ground_truth_file)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def load_frames(experiment_id: str):
    """Load an experiment's ground truth + extraction frames exactly as
    ``match_cache.build_match_cache`` did when it built this id's cache
    (reset_index, no unit conversion, no row filtering) -- the cached
    (gt_idx, ex_idx) edges are positions into frames loaded this same way,
    so loading them any other way would silently misalign the cache.

    Returns:
        (dataset, dataset_config, ground_truth_df, extraction_df, final_path,
        ground_truth_path).

    Raises:
        FileNotFoundError: no results directory or no final.json for this id.
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

    final_path = result_dir / "final.json"
    if not final_path.exists():
        raise FileNotFoundError(f"{experiment_id}: no final.json at {final_path}")
    with open(final_path) as f:
        extraction_df = pd.DataFrame(json.load(f)).reset_index(drop=True)

    dataset_config = load_dataset_config(dataset)
    ground_truth_path = _ground_truth_path(dataset_config)
    ground_truth_df = load_ground_truth(dataset_config).reset_index(drop=True)

    return dataset, dataset_config, ground_truth_df, extraction_df, final_path, ground_truth_path


def _assert_cache_fresh(cache_path: Path, *source_paths: Path) -> None:
    """Raise if cache_path is older than any of source_paths.

    A match cache's (gt_idx, ex_idx) edges are only meaningful against the
    exact frames that were loaded when it was built -- if the ground truth
    or the extraction's final.json has changed since, those indices may now
    point at different rows. Refuse rather than guess.
    """
    cache_mtime = cache_path.stat().st_mtime
    stale = [p for p in source_paths if p.stat().st_mtime > cache_mtime]
    if stale:
        raise RuntimeError(
            f"{cache_path} is older than {[str(p) for p in stale]} -- its cached "
            f"match indices may no longer line up with the current rows. Rerun "
            f"`python analysis/match_cache.py <experiment_id>` before using it."
        )


def _assert_matching_columns_present(
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


# ---------------------------------------------------------------------------
# Judgement resolution: extraction id -> judge_combine id -> per-row labels
# ---------------------------------------------------------------------------


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

        judged_extraction_ids = set()
        skip_reason = None
        for judge_id in judge_ids:
            try:
                judge_config_path = paths.find_experiment_config(judge_id)
            except FileNotFoundError:
                skip_reason = f"judge_id {judge_id!r} has no committed experiment-config"
                break
            judge_extraction_id = paths.load_experiment_config(judge_config_path)["params"].get("extraction_id")
            if judge_extraction_id is None:
                skip_reason = f"judge_id {judge_id!r} has no params.extraction_id (synthetic judge?)"
                break
            judged_extraction_ids.add(judge_extraction_id)

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
    for judge_id in winning_judge_ids:
        try:
            judge_dir = paths.find_result_dir(judge_id)
        except FileNotFoundError:
            continue
        run_meta = paths.load_run_metadata(judge_dir)
        if run_meta is not None and run_meta.get("extraction_id") not in (None, extraction_id):
            raise ValueError(
                f"{winning_id}: judge_id {judge_id!r}'s committed config points at "
                f"extraction_id={extraction_id!r}, but its own run_metadata.json "
                f"recorded extraction_id={run_meta.get('extraction_id')!r} -- it was "
                f"not run against the config it currently has"
            )
    return winning_id, winning_judge_ids


def load_validity_labels(judge_combine_id: str, extraction_df: pd.DataFrame) -> np.ndarray:
    """Load combined.json for judge_combine_id, joined by measurement_id and
    aligned positionally to extraction_df.

    Never assumes combined.json's row order matches extraction_df's -- joins
    explicitly by measurement_id and asserts the join is exact (a permutation
    of range(n), both sides agreeing on document_id/attribute at each id) so
    an extraction re-run after judging is caught rather than silently
    mislabeling rows.

    Raises:
        FileNotFoundError: no combined.json for judge_combine_id.
        ValueError: row-count mismatch, a non-permutation measurement_id set
            on either side, a document_id/attribute disagreement at some
            shared measurement_id, or a non-bool judgement_combined value.
    """
    combined_path = paths.find_result_dir(judge_combine_id) / "combined.json"
    if not combined_path.exists():
        raise FileNotFoundError(f"No combined.json for {judge_combine_id!r} at {combined_path}")
    with open(combined_path) as f:
        combined = json.load(f)

    n_ext = len(extraction_df)
    if len(combined) != n_ext:
        raise ValueError(
            f"{judge_combine_id}: combined.json has {len(combined)} record(s), "
            f"extraction has {n_ext} row(s) -- not the same run"
        )

    combined_mids = sorted(r["measurement_id"] for r in combined)
    if combined_mids != list(range(n_ext)):
        raise ValueError(
            f"{judge_combine_id}: combined.json measurement_id set is not "
            f"exactly range({n_ext}) ({len(set(combined_mids))} distinct of "
            f"{len(combined_mids)} records) -- cannot align positionally"
        )

    if "measurement_id" not in extraction_df.columns:
        raise ValueError(f"{judge_combine_id}: extraction final.json has no measurement_id column")
    ext_mids = extraction_df["measurement_id"].tolist()
    if ext_mids != list(range(n_ext)):
        raise ValueError(
            f"{judge_combine_id}: extraction's own measurement_id is not "
            f"range({n_ext}) in row order -- cannot align positionally"
        )

    by_mid = {r["measurement_id"]: r for r in combined}
    ordered = [by_mid[i] for i in range(n_ext)]

    mismatched = [
        i for i, (ext_row, judge_row) in enumerate(zip(extraction_df.itertuples(), ordered))
        if ext_row.document_id != judge_row["document_id"] or ext_row.attribute != judge_row["attribute"]
    ]
    if mismatched:
        raise ValueError(
            f"{judge_combine_id}: {len(mismatched)} row(s) disagree with the "
            f"extraction's final.json on document_id/attribute at the same "
            f"measurement_id (extraction likely re-run after judging) -- first "
            f"mismatch at measurement_id={mismatched[0]}"
        )

    non_bool = [r["measurement_id"] for r in ordered if not isinstance(r["judgement_combined"], bool)]
    if non_bool:
        raise ValueError(
            f"{judge_combine_id}: judgement_combined is not a bool for "
            f"measurement_id(s) {non_bool[:10]}"
        )

    return np.array([r["judgement_combined"] for r in ordered], dtype=bool)


# ---------------------------------------------------------------------------
# Recovered / matched masks, from an ALREADY-threshold-filtered edge list
# (i.e. match_cache.load_match_cache(id, fuzzy_threshold=...) -- see module
# docstring) + a cross-check against analysis.metrics' own rates
# ---------------------------------------------------------------------------


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


def _verify_recovery(
    ground_truth_df: pd.DataFrame,
    extraction_df: pd.DataFrame,
    cfg: dict,
    threshold: float,
    cache_path: Path,
    recovered: np.ndarray,
) -> None:
    """Assert this script's recovered mask agrees with
    ``analysis.metrics.recovery_rate`` computed against the exact same cache.

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


def _verify_validity(
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
    cache, then return the OR'd validity-label array. See ``_verify_recovery``.

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


# ---------------------------------------------------------------------------
# Paper-clustered percentile bootstrap
# ---------------------------------------------------------------------------


def bootstrap_cluster_rate(
    labels: np.ndarray,
    clusters: np.ndarray,
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap of a proportion, resampling whole clusters.

    Rows sharing a cluster id (a paper's document_id) are not independent, so
    resampling individual rows understates uncertainty -- this resamples the
    set of clusters with replacement (each draw takes ALL of that cluster's
    rows), which is the standard case-resampling cluster bootstrap.

    Args:
        labels: Boolean array, one entry per row.
        clusters: Parallel array of cluster ids, same length as labels.
        n_resamples: Number of bootstrap resamples. No default (CLAUDE.md: a
            value like this that trades off precision for runtime doesn't
            get an inferred default).
        seed: RNG seed. No default, for the same reason.
        alpha: CI significance level; returns the (alpha/2, 1-alpha/2)
            percentiles of the bootstrap distribution.

    Returns:
        (point_estimate, ci_lo, ci_hi). point_estimate is the plain
        (unresampled) rate over all rows.

    Raises:
        ValueError: labels/clusters length mismatch, or zero rows.
    """
    labels = np.asarray(labels, dtype=bool)
    clusters = np.asarray(clusters)
    if len(labels) != len(clusters):
        raise ValueError(
            f"labels ({len(labels)}) and clusters ({len(clusters)}) length mismatch"
        )
    if len(labels) == 0:
        raise ValueError("cannot compute a rate over zero rows")

    point = float(labels.mean())

    unique_clusters = np.unique(clusters)
    n_clusters = len(unique_clusters)
    n_true_by_cluster = np.array([
        int(labels[clusters == c].sum()) for c in unique_clusters
    ])
    n_total_by_cluster = np.array([
        int((clusters == c).sum()) for c in unique_clusters
    ])

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))
    boot_true = n_true_by_cluster[draws].sum(axis=1)
    boot_total = n_total_by_cluster[draws].sum(axis=1)
    boot_rates = boot_true / boot_total

    lo, hi = np.percentile(boot_rates, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------------------
# Per-id orchestration
# ---------------------------------------------------------------------------


def compute_metrics_for_id(
    experiment_id: str,
    *,
    n_resamples: int,
    seed: int,
    alpha: float = 0.05,
    compute_validity: bool = True,
) -> dict:
    """Compute recovery (+ validity, unless ``compute_validity=False``) with
    paper-clustered bootstrap CIs for one experiment id.

    ``compute_validity=False`` skips judge_combine resolution entirely --
    for an id that has no judge coverage yet, not a way to suppress a real
    resolution failure. The returned row's validity-related fields are then
    ``None``, not 0.0 or some other placeholder (CLAUDE.md: no silent
    fallback in place of a value that was never computed).

    Raises loud on any of: unsupported dataset, missing/stale/schema-mismatched
    match cache, an out-of-range cached edge, no (or an ambiguous)
    judge_combine match, a combined.json/final.json row-alignment problem, or
    the recovered/matched masks disagreeing with analysis.metrics' own rates.
    No fallback path for any of these -- see module docstring.
    """
    dataset, dataset_config, ground_truth_df, extraction_df, final_path, ground_truth_path = load_frames(experiment_id)

    cfg = match_cache.get_matching_config(dataset_config)
    threshold = cfg["fuzzy_threshold"]
    _assert_matching_columns_present(ground_truth_df, extraction_df, cfg)

    cache_path = match_cache.match_cache_path(experiment_id)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{experiment_id}: no match_cache.pkl at {cache_path}. Run "
            f"`python analysis/match_cache.py {experiment_id}` first."
        )
    _assert_cache_fresh(cache_path, final_path, ground_truth_path)

    n_gt, n_ext = len(ground_truth_df), len(extraction_df)
    # Cache is always built at fuzzy_threshold=0.0 (see match_cache.py's
    # module docstring) -- passing the dataset's own threshold here returns
    # the already-selected edges directly, rather than a second manual
    # weight-filter reimplementing that same cutoff.
    selected_edges = match_cache.load_match_cache(experiment_id, fuzzy_threshold=threshold)
    for gt_idx, ex_idx in selected_edges:
        if not (0 <= gt_idx < n_gt and 0 <= ex_idx < n_ext):
            raise RuntimeError(
                f"{experiment_id}: cached edge (gt_idx={gt_idx}, ex_idx={ex_idx}) out "
                f"of range for n_gt={n_gt}, n_ext={n_ext} -- the cache no longer "
                f"matches the current ground truth/extraction rows. Rerun "
                f"`python analysis/match_cache.py {experiment_id}`."
            )

    recovered = gt_recovered_mask(n_gt, selected_edges)
    matched = ext_matched_mask(n_ext, selected_edges)
    _verify_recovery(ground_truth_df, extraction_df, cfg, threshold, cache_path, recovered)

    recovery_point, recovery_lo, recovery_hi = bootstrap_cluster_rate(
        recovered, ground_truth_df["document_id"].to_numpy(),
        n_resamples=n_resamples, seed=seed, alpha=alpha,
    )

    row = {
        "experiment_id": experiment_id,
        "dataset": dataset,
        "n_gt": n_gt,
        "n_ext": n_ext,
        "fuzzy_threshold": threshold,
        "n_resamples": n_resamples,
        "seed": seed,
        "bootstrap_unit": "paper",
        "recovery": recovery_point,
        "recovery_ci_lo": recovery_lo,
        "recovery_ci_hi": recovery_hi,
        "judge_combine_id": None,
        "judge_ids": None,
        "validity": None,
        "validity_ci_lo": None,
        "validity_ci_hi": None,
    }

    if not compute_validity:
        return row

    judge_combine_id, judge_ids = find_judge_combine_id(dataset, experiment_id)
    judged_labels = load_validity_labels(judge_combine_id, extraction_df)
    validity_labels = _verify_validity(
        ground_truth_df, extraction_df, cfg, threshold, cache_path, matched, judged_labels,
    )

    validity_point, validity_lo, validity_hi = bootstrap_cluster_rate(
        validity_labels, extraction_df["document_id"].to_numpy(),
        n_resamples=n_resamples, seed=seed, alpha=alpha,
    )

    row.update({
        "judge_combine_id": judge_combine_id,
        "judge_ids": ";".join(judge_ids),
        "validity": validity_point,
        "validity_ci_lo": validity_lo,
        "validity_ci_hi": validity_hi,
    })
    return row


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("experiment_ids", nargs="+", help="Extraction/ablation/baseline experiment ids to score.")
    p.add_argument("--n-resamples", type=int, required=True, help="Number of paper-cluster bootstrap resamples.")
    p.add_argument("--seed", type=int, required=True, help="Bootstrap RNG seed.")
    p.add_argument("--alpha", type=float, default=0.05, help="CI significance level (default: 0.05, a 95%% interval).")
    p.add_argument(
        "--skip-validity", action="store_true",
        help="Report recovery only -- skip judge_combine resolution and validity entirely "
             "(for an id with no judge coverage yet).",
    )
    p.add_argument(
        "--output", type=Path, default=Path("results/recovery_validity.csv"),
        help="CSV path to write (default: results/recovery_validity.csv, matching "
             "analysis/ablation.py's/baselines.py's own results/ output convention).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    rows = []
    for experiment_id in args.experiment_ids:
        print(f"Processing {experiment_id} ...")
        row = compute_metrics_for_id(
            experiment_id, n_resamples=args.n_resamples, seed=args.seed, alpha=args.alpha,
            compute_validity=not args.skip_validity,
        )
        rows.append(row)
        validity_str = (
            f"validity={row['validity']:.3f} [{row['validity_ci_lo']:.3f}, {row['validity_ci_hi']:.3f}] "
            f"(judge_combine={row['judge_combine_id']})"
            if row["validity"] is not None else "validity=skipped"
        )
        print(
            f"  recovery={row['recovery']:.3f} "
            f"[{row['recovery_ci_lo']:.3f}, {row['recovery_ci_hi']:.3f}]  {validity_str}"
        )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\nWrote {len(df)} row(s) to {args.output}")


if __name__ == "__main__":
    main()
