"""Resolve and cross-check the run ids a calibration config names.

Used by calibration.py, calibration_validated.py, platt_scaling.py,
decision_threshold.py and meta_inputs.py. Side-effect free so it can be unit tested.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

_EXPERIMENTS_DIR = Path(__file__).parent.parent.parent / "experiments"
if str(_EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENTS_DIR))

import utils as paths  # noqa: E402

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from analysis.common.config import (  # noqa: E402
    _resolve_ground_truth_path, analysis_config_path, analysis_results_dir, load_synthetic_probe_config,
)


def resolve_run(run_id: str) -> tuple[str, str]:
    """Look up a run's dataset and experiment type from its own committed config.

    Reading these from the config (not the caller) makes a misfiled id fail loudly.

    Args:
        run_id: Experiment id.

    Returns:
        ``(dataset, experiment_type)``.

    Raises:
        FileNotFoundError: If no config exists for this id.
        ValueError: If the id matches more than one config.
    """
    config_path = paths.find_experiment_config(run_id)
    return config_path.parts[-4], config_path.parts[-3]


def pinned_run_dir(run_id: str, expected_dataset: str, expected_type: str) -> Path:
    """Result directory of a run, after checking its config matches the expected location.

    Args:
        run_id: Experiment id.
        expected_dataset: Dataset the caller expects the run to belong to.
        expected_type: Experiment type the caller expects.

    Returns:
        ``experiments/results/{expected_dataset}/{expected_type}/{run_id}``.

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


# ── Config-driven resolution ─────────────────────────────────────────────────

_SYNTHETIC_PROBE_RESULTS_ROOT = analysis_results_dir("synthetic-probe")

# _v<N> plus an optional letter (e.g. supermat's realigned split v3s, distinct from v3).
_SYN_FILE_VERSION_RE = re.compile(r"_(v\d+[a-z]?)(?:_diag)?\.json$")


def _synthetic_file_version(run_id: str, cfg: dict) -> str:
    """Synthetic-corpus version from a judge_interp config's ``params.synthetic_file``.

    Args:
        run_id: Experiment id (for error messages).
        cfg: The run's experiment config.

    Returns:
        Version string such as ``"v3"`` or ``"v3s"``.

    Raises:
        ValueError: The file name has no ``_v<N>[_diag].json`` suffix.
    """
    synthetic_file = cfg["params"].get("synthetic_file")
    match = _SYN_FILE_VERSION_RE.search(str(synthetic_file))
    if match is None:
        raise ValueError(
            f"{run_id}: params.synthetic_file={synthetic_file!r} has no _v<N>[_diag].json "
            f"suffix, so its synthetic-corpus version can't be checked"
        )
    return match.group(1)


def _run_config(run_id: str, expected_dataset: str, expected_type: str) -> tuple[Path, dict]:
    """Result directory and committed config of a run, with its location verified.

    Args:
        run_id: Experiment id.
        expected_dataset: Expected dataset.
        expected_type: Expected experiment type.

    Returns:
        ``(run_dir, experiment_config_dict)``.
    """
    run_dir = pinned_run_dir(run_id, expected_dataset, expected_type)
    return run_dir, paths.load_experiment_config(paths.find_experiment_config(run_id))


def resolve_calibration_inputs(cfg: dict) -> dict:
    """Resolve every run a calibration config names and check they belong together.

    Downstream joins are by position or measurement_id, so a mismatched companion
    run would be scored silently. Checks:
      - each id lives under the expected dataset and experiment type;
      - judge_interp was run on this extraction, and judge_combine includes it;
      - one judge model across all real and synthetic judge_interp runs;
      - synthetic train and test runs share a corpus version;
      - the synthetic-probe config is for this dataset and its trained probe exists.

    Args:
        cfg: Loaded calibration analysis config.

    Returns:
        Dict with:
            - ``judge_model``: the shared judge model name
            - ``datasets``: per dataset, ``extraction_dir``, ``judge_interp_dir``,
              ``judge_combine_dir``, ``ground_truth_path``, ``syn_train_id``,
              ``probe_dir``, ``syn_test_id``, ``syn_test_dir``

    Raises:
        FileNotFoundError: A config, probe config or probe results.json is missing.
        ValueError: Any of the checks above fails.
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

        probe_cfg_path = analysis_config_path("synthetic-probe", block["synthetic_probe_config"])
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
        probe_dir = _SYNTHETIC_PROBE_RESULTS_ROOT / probe_cfg["id"] / "trained_probe"

        results_path = _SYNTHETIC_PROBE_RESULTS_ROOT / probe_cfg["id"] / "results.json"
        if not results_path.exists():
            raise FileNotFoundError(
                f"{results_path} missing -- run analysis/synthetic_probe_train.py "
                f"{probe_cfg_path} first"
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


