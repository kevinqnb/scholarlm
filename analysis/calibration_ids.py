"""Id-addressed settings registry + resolution helpers for
analysis/calibration_updated.py, under the 2026-09-16 experiment-contract
migration (see notes/scholarlm/builds/2026-09-16-synthetic-probe-id-migration-01.md).

Split out from calibration_updated.py itself because that script does its real
data loading at import time -- these id-resolution helpers have no side
effects, so they're what can actually be unit tested (tests/test_calibration_ids.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import utils as paths  # noqa: E402


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
# pipeline is being scored). Currently pond only; add an entry to each of the
# three maps below (train id, and both test splits) once a dataset's
# synthetic judge_interp runs exist under the id-addressed contract -- see
# notes/scholarlm/builds/2026-09-16-calibration-id-migration-01.md's second
# 2026-09-16 session for what that requires (nfix/supermat currently have no
# synthetic judge run at all under the post-e7da363 full-paper judge, old
# tree or new -- this isn't just a backfill like pond's was).
TRAIN_DATASETS = ['pond']

SYN_TRAIN_IDS = {
    'pond': '2026-09-10-pond-qwen-2.5-7b-synthetic-judge-train-01',
}
SYN_TEST_IDS = {
    'pond': {
        'primary': '2026-09-10-pond-qwen-2.5-7b-synthetic-judge-test-primary-01',
        'diag': '2026-09-10-pond-qwen-2.5-7b-synthetic-judge-test-diag-01',
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
        'pi_te_estimate': None,
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
        'pi_te_estimate': 0.85,  # carried over from the pre-migration gpt-oss-120b entry
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
        'pi_te_estimate': None,
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
