"""Id-addressed settings registry + resolution helpers for
analysis/calibration_updated.py, under the 2026-09-16 experiment-contract
migration (see notes/scholarlm/builds/2026-09-16-synthetic-probe-id-migration-01.md).

Split out from calibration_updated.py itself because that script does its real
data loading at import time -- these id-resolution helpers have no side
effects, so they're what can actually be unit tested (tests/test_calibration_ids.py).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import utils as paths  # noqa: E402

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.analysis_config import (  # noqa: E402
    ANALYSIS_CONFIGS_ROOT, _resolve_ground_truth_path, load_synthetic_probe_config,
)


ALL_DATASETS = ['pond', 'nfix', 'supermat']

# The only judge with an id-addressed trained probe (2026-09-16 migration).
# llama-3.1-8b's old-tree probe (data/experiments/*/synthetic_probe/llama-3.1-8b/)
# predates the 2026-09-08 full-paper-judge rewrite (commit e7da363) that every
# real judge_interp run below postdates -- mixing it in would combine
# invalidated and valid judge numbers in the same figure. It needs a fresh
# synthetic-judge train run + retrain before it can be added back.
JUDGE_MODEL = 'qwen-2.5-7b'

# Datasets with a migrated synthetic-probe train run -- each produces its own
# trained probe, applied to every dataset's real extractions below (not
# per-setting, since the judge/probe side is independent of which extraction
# pipeline is being scored).
#
# nfix/supermat entries added 2026-09-16 pointing at newly-minted
# judge_interp configs -- committed, but NOT YET RUN as of this edit (rung 4
# augmentation was mid-flight; see notes/scholarlm/builds/
# 2026-09-16-calibration-id-migration-01.md's second session for the prior
# state). This is intentionally safe to commit ahead of the data: every
# lookup below goes through pinned_run_dir/resolve_run, which fails loud
# (FileNotFoundError) against a run id whose experiments/results/ dir has no
# output yet, rather than silently reading nothing or someone else's run.
# Don't treat calibration_updated.py succeeding against these ids as
# evidence the underlying judge_interp runs exist -- confirm each run
# actually completed first.
TRAIN_DATASETS = ['pond', 'nfix', 'supermat']

SYN_TRAIN_IDS = {
    'pond': '2026-09-10-pond-qwen-2.5-7b-synthetic-judge-train-01',
    'nfix': '2026-09-16-nfix-qwen-2.5-7b-synthetic-judge-train-01',
    'supermat': '2026-09-16-supermat-qwen-2.5-7b-synthetic-judge-train-01',
}
SYN_TEST_IDS = {
    'pond': {
        'primary': '2026-09-10-pond-qwen-2.5-7b-synthetic-judge-test-primary-01',
        'diag': '2026-09-10-pond-qwen-2.5-7b-synthetic-judge-test-diag-01',
    },
    'nfix': {
        'primary': '2026-09-16-nfix-qwen-2.5-7b-synthetic-judge-test-primary-01',
        'diag': '2026-09-16-nfix-qwen-2.5-7b-synthetic-judge-test-diag-01',
    },
    'supermat': {
        'primary': '2026-09-16-supermat-qwen-2.5-7b-synthetic-judge-test-primary-01',
        'diag': '2026-09-16-supermat-qwen-2.5-7b-synthetic-judge-test-diag-01',
    },
}

# The two synthetic-test splits every TRAIN_DATASETS entry's SYN_TEST_IDS
# must provide (see test_syn_test_ids_cover_primary_and_diag) -- fixed by the
# split *naming* convention, independent of which/how many datasets have a
# trained probe, so --syn-split is validated against this, not against
# SYN_TEST_IDS's keys directly.
SYN_SPLITS = ('primary', 'diag')

# One entry per judged pipeline-variant. `result_type` names the
# experiments/results/{dataset}/{result_type}/ subtree the extraction output
# itself lives under; the three *_id maps are pinned experiment ids per
# dataset -- no "most recent date" resolution anywhere downstream of this file.
SETTINGS = {
    'gemma-3-27b-extraction': {
        'result_type': 'extraction',
        'extraction_id': {
            'pond': '2026-05-05-pond-gemma-3-27b-extraction-01',
            'nfix': '2026-05-06-nfix-gemma-3-27b-extraction-01',
            'supermat': '2026-07-09-supermat-gemma-3-27b-extraction-01',
        },
        'judge_interp_id': {
            'pond': '2026-09-13-pond-gemma3-27b-extraction-qwen7b-judge-interp-01',
            'nfix': '2026-09-13-nfix-gemma3-27b-extraction-qwen7b-judge-interp-01',
            'supermat': '2026-09-13-supermat-gemma3-27b-extraction-qwen7b-judge-interp-01',
        },
        'judge_combine_id': {
            'pond': '2026-09-13-pond-gemma3-27b-extraction-judge-combine-01',
            'nfix': '2026-09-13-nfix-gemma3-27b-extraction-judge-combine-01',
            'supermat': '2026-09-13-supermat-gemma3-27b-extraction-judge-combine-01',
        },
        'pi_te_estimate': 0.5,
    },
    'gpt-oss-120b-ablation1': {
        'result_type': 'ablation',
        'extraction_id': {
            'pond': '2026-05-03-pond-gpt-oss-120b-ablation1-01',
            'nfix': '2026-05-06-nfix-gpt-oss-120b-ablation1-01',
            'supermat': '2026-08-13-supermat-gpt-oss-120b-ablation1-01',
        },
        'judge_interp_id': {
            'pond': '2026-09-13-pond-gptoss-120b-ablation1-qwen7b-judge-interp-01',
            'nfix': '2026-09-13-nfix-gptoss-120b-ablation1-qwen7b-judge-interp-01',
            'supermat': '2026-09-13-supermat-gptoss-120b-ablation1-qwen7b-judge-interp-01',
        },
        'judge_combine_id': {
            'pond': '2026-09-13-pond-gptoss-120b-ablation1-judge-combine-01',
            'nfix': '2026-09-13-nfix-gptoss-120b-ablation1-judge-combine-01',
            'supermat': '2026-09-13-supermat-gptoss-120b-ablation1-judge-combine-01',
        },
        'pi_te_estimate': 0.90,  # carried over from the pre-migration gpt-oss-120b entry
    },
    'baseline-nuextract': {
        'result_type': 'baseline_nuextract',
        'extraction_id': {
            'pond': '2026-07-11-pond-baseline-nuextract-01',
            'nfix': '2026-07-11-nfix-baseline-nuextract-01',
            'supermat': '2026-07-13-supermat-baseline-nuextract-01',
        },
        'judge_interp_id': {
            'pond': '2026-09-13-pond-baseline-nuextract-qwen7b-judge-interp-01',
            'nfix': '2026-09-13-nfix-baseline-nuextract-qwen7b-judge-interp-01',
            'supermat': '2026-09-13-supermat-baseline-nuextract-qwen7b-judge-interp-01',
        },
        'judge_combine_id': {
            'pond': '2026-09-13-pond-baseline-nuextract-judge-combine-01',
            'nfix': '2026-09-13-nfix-baseline-nuextract-judge-combine-01',
            'supermat': '2026-09-13-supermat-baseline-nuextract-judge-combine-01',
        },
        'pi_te_estimate': 0.5,
    },
}

DEFAULT_SETTING = 'gemma-3-27b-extraction'


def resolve_run(run_id: str) -> tuple[str, str]:
    """(dataset, experiment_type) for a result-tree run id, read from its own
    committed config -- never trusted from a caller's registry entry, so a
    copy-paste id pasted under the wrong dataset/type fails immediately
    instead of quietly reading someone else's run.

    Raises:
        FileNotFoundError: If no config exists for this id.
        ValueError: If the id matches more than one config.
    """
    config_path = paths.find_experiment_config(run_id)
    return config_path.parts[-4], config_path.parts[-3]


def pinned_run_dir(run_id: str, expected_dataset: str, expected_type: str) -> Path:
    """experiments/results/{expected_dataset}/{expected_type}/{run_id}/, after
    verifying the id's own config actually lives there.

    Raises:
        FileNotFoundError: If no config exists for this id.
        ValueError: If the config's own (dataset, experiment_type) disagrees
            with what the caller expected.
    """
    resolved_dataset, resolved_type = resolve_run(run_id)
    if (resolved_dataset, resolved_type) != (expected_dataset, expected_type):
        raise ValueError(
            f"{run_id}: config lives at dataset={resolved_dataset!r} "
            f"type={resolved_type!r}, but the caller expected "
            f"dataset={expected_dataset!r} type={expected_type!r}"
        )
    return paths.result_dir(expected_dataset, expected_type, run_id)


def pinned_extraction_dir(dataset: str, result_type: str, run_id: str) -> Path:
    """experiments/results/{dataset}/{result_type}/{run_id}/, after verifying
    run_metadata.json's own 'dataset' field.

    The extraction/ablation/baseline runs SETTINGS points at predate the
    experiment-config-standard restructure (Phase B id migration -- see
    experiments/utils.py's _EXPERIMENT_ID_RE comment) and were never given a
    committed experiment-configs/ yaml, only id-addressed output -- so unlike
    pinned_run_dir (which every judge_interp/judge_combine id here does have a
    config for), the boundary check is against run_metadata.json instead of
    find_experiment_config.

    Raises:
        FileNotFoundError: If the result dir or its run_metadata.json is missing.
        ValueError: If run_metadata.json's own dataset disagrees with `dataset`.
    """
    run_dir = paths.result_dir(dataset, result_type, run_id)
    metadata = paths.load_run_metadata(run_dir)
    if metadata is None:
        raise FileNotFoundError(f"{run_dir}: no run_metadata.json")
    if metadata.get('dataset') != dataset:
        raise ValueError(
            f"{run_id}: run_metadata.json dataset={metadata.get('dataset')!r}, "
            f"expected {dataset!r}"
        )
    return run_dir


def select_setting(
    name: str | None, datasets: list[str] | None = None
) -> tuple[str, dict, list[str]]:
    """Resolve a setting name and optional dataset narrowing.

    Returns:
        (setting_name, registry_entry, datasets) -- datasets defaults to
        ALL_DATASETS when not narrowed.

    Raises:
        ValueError: Unknown setting name, or an unknown dataset in `datasets`.
    """
    setting = name or DEFAULT_SETTING
    if setting not in SETTINGS:
        raise ValueError(f"Unknown setting {setting!r}; known: {sorted(SETTINGS)}")
    if datasets:
        unknown = [d for d in datasets if d not in ALL_DATASETS]
        if unknown:
            raise ValueError(f"Unknown dataset(s) {unknown}; available: {ALL_DATASETS}")
        resolved_datasets = [d for d in ALL_DATASETS if d in datasets]
    else:
        resolved_datasets = list(ALL_DATASETS)
    return setting, SETTINGS[setting], resolved_datasets


# ── Config-driven resolution (analysis/calibration_updated.py <config>) ──────
# The SETTINGS/SYN_*_IDS registry above predates analysis-configs/ and is no
# longer read by calibration_updated.py; everything it needs now comes from a
# load_calibration_config() config via resolve_calibration_inputs below.

_SYNTHETIC_PROBE_RESULTS_ROOT = _REPO_ROOT / "analysis" / "results" / "synthetic_probe"

_SYN_FILE_VERSION_RE = re.compile(r"_(v\d+)(?:_diag)?\.json$")


def _synthetic_file_version(run_id: str, cfg: dict) -> str:
    synthetic_file = cfg["params"].get("synthetic_file")
    match = _SYN_FILE_VERSION_RE.search(str(synthetic_file))
    if match is None:
        raise ValueError(
            f"{run_id}: params.synthetic_file={synthetic_file!r} has no _v<N>[_diag].json "
            f"suffix, so its synthetic-corpus version can't be checked"
        )
    return match.group(1)


def _run_config(run_id: str, expected_dataset: str, expected_type: str) -> tuple[Path, dict]:
    """(run_dir, committed experiment config) for run_id, with its (dataset, type) verified."""
    run_dir = pinned_run_dir(run_id, expected_dataset, expected_type)
    return run_dir, paths.load_experiment_config(paths.find_experiment_config(run_id))


def resolve_calibration_inputs(cfg: dict) -> dict:
    """Resolve and cross-check every run a calibration config names.

    Pure id/config/path logic: touches no activations or pickles beyond the
    probe-training ``results.json`` existence/path check, so it can run (and be
    unit tested) without any heavy data. Each wrong-companion-run failure mode
    is a hard error here, because every downstream join is positional or
    keyed by measurement_id and would otherwise silently score the wrong run:

      - extraction/judge_interp/judge_combine/probe/test ids each live under
        the expected dataset and experiment type (read from their own config);
      - the real judge_interp run's params.extraction_id is the config's
        extraction_id, and the judge_combine run's params.judge_ids contains
        the judge_interp id (ties the judgement_p_true_<judge> column to the
        activations that go with it);
      - one judge model across every real/synthetic judge_interp run;
      - synthetic train and both test runs share the synthetic-corpus version
        (synthetic_file's _v<N> suffix), so a v3 probe is never evaluated on
        a v2 test set;
      - the synthetic-probe analysis config's params.dataset is this dataset,
        and its results.json points into the train run's trained_probe/.

    Returns:
        {'judge_model': str,
         'datasets': {ds: {'extraction_dir', 'judge_interp_dir',
                           'judge_combine_dir', 'ground_truth_path',
                           'syn_train_id', 'probe_dir', 'syn_test_id',
                           'syn_test_dir'}}}
    """
    params = cfg["params"]
    judge_models: dict[str, str] = {}
    out: dict[str, dict] = {}

    for ds, block in params["datasets"].items():
        ext_dataset, ext_type = resolve_run(block["extraction_id"])
        if ext_dataset != ds:
            raise ValueError(
                f"{block['extraction_id']}: config lives under dataset={ext_dataset!r}, "
                f"but it's listed under {ds!r}"
            )
        extraction_dir = pinned_run_dir(block["extraction_id"], ds, ext_type)

        interp_dir, interp_cfg = _run_config(block["judge_interp_id"], ds, "judge_interp")
        if interp_cfg["params"].get("extraction_id") != block["extraction_id"]:
            raise ValueError(
                f"{block['judge_interp_id']}: params.extraction_id="
                f"{interp_cfg['params'].get('extraction_id')!r} != {block['extraction_id']!r}"
            )
        judge_models[f"{ds} real judge_interp"] = interp_cfg["params"]["judge"]

        combine_dir, combine_cfg = _run_config(block["judge_combine_id"], ds, "judge_combine")
        if block["judge_interp_id"] not in combine_cfg["params"]["judge_ids"]:
            raise ValueError(
                f"{block['judge_combine_id']}: params.judge_ids does not contain "
                f"{block['judge_interp_id']!r}"
            )

        probe_cfg_path = ANALYSIS_CONFIGS_ROOT / f"{block['synthetic_probe_config']}.yaml"
        if not probe_cfg_path.exists():
            raise FileNotFoundError(f"synthetic_probe_config {probe_cfg_path} does not exist")
        probe_cfg = load_synthetic_probe_config(probe_cfg_path)
        if probe_cfg["params"]["dataset"] != ds:
            raise ValueError(
                f"{probe_cfg['id']}: params.dataset={probe_cfg['params']['dataset']!r}, "
                f"but it's listed under {ds!r}"
            )
        syn_train_id = probe_cfg["params"]["judge_interp_id"]
        train_dir, train_cfg = _run_config(syn_train_id, ds, "judge_interp")
        judge_models[f"{ds} synthetic train"] = train_cfg["params"]["judge"]
        corpus_version = _synthetic_file_version(syn_train_id, train_cfg)
        probe_dir = train_dir / "trained_probe"

        results_path = _SYNTHETIC_PROBE_RESULTS_ROOT / probe_cfg["id"] / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(
                f"{results_path} missing -- run analysis/synthetic_probe_train.py "
                f"analysis/analysis-configs/{probe_cfg['id']}.yaml first"
            )
        with open(results_path) as f:
            results = json.load(f)
        if Path(results["probe_path"]).parent.resolve() != probe_dir.resolve():
            raise ValueError(
                f"{results_path}: probe_path {results['probe_path']!r} is not inside {probe_dir}"
            )

        for split, test_id in block["syn_test_ids"].items():
            _, test_cfg = _run_config(test_id, ds, "judge_interp")
            judge_models[f"{ds} synthetic test {split}"] = test_cfg["params"]["judge"]
            test_version = _synthetic_file_version(test_id, test_cfg)
            if test_version != corpus_version:
                raise ValueError(
                    f"{test_id}: synthetic corpus {test_version!r} != train run "
                    f"{syn_train_id}'s {corpus_version!r}"
                )
        syn_test_id = block["syn_test_ids"][params["syn_split"]]

        out[ds] = {
            "extraction_dir": extraction_dir,
            "judge_interp_dir": interp_dir,
            "judge_combine_dir": combine_dir,
            "ground_truth_path": _resolve_ground_truth_path(block["ground_truth_file"]),
            "syn_train_id": syn_train_id,
            "probe_dir": probe_dir,
            "syn_test_id": syn_test_id,
            "syn_test_dir": pinned_run_dir(syn_test_id, ds, "judge_interp"),
        }

    if len(set(judge_models.values())) != 1:
        raise ValueError(f"judge model differs across runs: {judge_models}")
    return {"judge_model": next(iter(judge_models.values())), "datasets": out}


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
    match_cache.load_match_cache(..., fuzzy_threshold=...).

    Returns:
        (ground_truth_df, extraction_df, edges) -- edges are (gt_idx, ex_idx)
        positions into those two frames.
    """
    # Lazy: these pull in the extraction runner, which _resolve_job.py's
    # submit-time validation of a calibration config shouldn't pay for.
    from analysis import match_cache, recovery_validity as rv

    _dataset, dataset_config, gt_df, ext_df, ext_path, gt_path = rv.load_frames(
        extraction_id, ground_truth_path,
    )
    cfg = match_cache.get_matching_config(dataset_config)
    rv._assert_matching_columns_present(gt_df, ext_df, cfg)

    cache_path = match_cache.match_cache_path(extraction_id)
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{extraction_id}: no match_cache.pkl at {cache_path}. Run "
            f"`python analysis/match_cache.py {extraction_id} "
            f"--ground-truth-file {gt_path}` first -- calibration never builds matchings."
        )
    rv._assert_cache_fresh(cache_path, ext_path, gt_path)
    rv._assert_ground_truth_matches_cache(extraction_id, cache_path, gt_path, gt_df)
    rv._assert_extraction_matches_cache(extraction_id, cache_path, ext_path)

    edges = match_cache.load_match_cache(extraction_id, fuzzy_threshold=cfg["fuzzy_threshold"])
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
