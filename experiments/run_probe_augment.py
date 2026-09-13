"""
Probe-dataset augmentation runner.

Thin config-driven wrapper around data/{dataset}/create_probe_dataset.py
--augment (unchanged, out of scope for this restructure -- this runner never
touches its internals). Replaces experiments/gen_augment.sh's shell-script
mode dispatch (hardcoded pond/nfix/supermat/rung3/pond_rung3 case branches,
each with inline --augment-prompt-budget-multiple / --augment-valid-floor
constants) with params read from a config, so each augmentation run gets a
committed, reproducible record instead of a shell-script comment documenting
"what was passed this time".

The real output (probe_dataset<suffix>.json, probe_dataset_test<suffix>.json,
the diagnostic file, probe_augment_cache.json) is written by
create_probe_dataset.py itself to data/{dataset}/ -- unchanged, NOT under
experiments/results/. This runner additionally writes a small
run_metadata.json to experiments/results/{dataset}/probe_augment/{experiment_id}/
recording what params/seed produced that data/{dataset}/ output, since the
real artifacts don't carry an experiment id of their own.

Usage
-----
    python experiments/run_probe_augment.py experiments/experiment-configs/pond/probe_augment/<id>/<id>.yaml

Required params: dataset (pond | nfix | supermat), model (an
experiments/model-configs/extraction/ key -- gpt-oss-120b for every
augmentation run so far), reviewed (bool -- explicit per CLAUDE.md's
no-implicit-default rule; supermat has no ground_truth_review.json, so this
must be false there, but that's a fact about the dataset, not something this
runner should default silently), prompt_budget_multiple (int -- no shared
default across datasets; set from the rung-3 measured yield each time).

Optional params, passed through to create_probe_dataset.py's --augment-*
flags only when present (its own defaults apply otherwise -- this wrapper
never duplicates a default create_probe_dataset.py already declares, to
avoid the two silently drifting apart): pos_axes (list), valid_floor,
diag_valid_floor, sample_gt, out_suffix, rewrite_temperature,
prewarm_max_retries, prewarm_drop_ceiling, cache, stub (bool -- deterministic
stub client, no LLM, smoke-test only). Also: api_base.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
_EXPERIMENTS_DIR = Path(__file__).parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_EXPERIMENTS_DIR))

import utils as paths

# Only datasets with a data/{dataset}/create_probe_dataset.py --augment path.
_SUPPORTED_DATASETS = ("pond", "nfix", "supermat")

# params key -> create_probe_dataset.py CLI flag. Passed through only when the
# key is present in params, so create_probe_dataset.py's own defaults apply
# otherwise -- this wrapper never duplicates a default the wrapped script
# already declares (see module docstring).
_VALUE_FLAGS: dict[str, str] = {
    "valid_floor": "--augment-valid-floor",
    "diag_valid_floor": "--augment-diag-valid-floor",
    "sample_gt": "--augment-sample-gt",
    "out_suffix": "--augment-out-suffix",
    "rewrite_temperature": "--augment-rewrite-temperature",
    "prewarm_max_retries": "--augment-prewarm-max-retries",
    "prewarm_drop_ceiling": "--augment-prewarm-drop-ceiling",
    "cache": "--augment-cache",
}
_LIST_FLAGS: dict[str, str] = {
    "pos_axes": "--augment-pos-axes",
}
_BOOL_FLAGS: dict[str, str] = {
    "stub": "--augment-stub",
}


def build_augment_argv(
    dataset: str,
    seed: int,
    reviewed: bool,
    prompt_budget_multiple: int,
    api_base: str,
    params: dict,
) -> list[str]:
    """Build the create_probe_dataset.py --augment argv from params.

    Args:
        dataset: One of _SUPPORTED_DATASETS.
        seed: Passed through verbatim -- probe augmentation legitimately uses
            its own per-run seed (historically 42), not the repo-wide default;
            unlike every other Tier-1 runner, this one has no repo-seed
            consistency check.
        reviewed: Whether to read ground_truth_review.json instead of
            ground_truth.json.
        prompt_budget_multiple: Required -- no shared default across datasets.
        api_base: vLLM endpoint hosting the augmentation model.
        params: The experiment config's params mapping (for the optional
            pass-through flags).

    Returns:
        Full argv (script path first) for subprocess invocation.
    """
    script = _REPO_ROOT / "data" / dataset / "create_probe_dataset.py"
    argv = [
        str(script),
        "--augment",
        "--seed", str(seed),
        "--augment-prompt-budget-multiple", str(prompt_budget_multiple),
        "--gpt-oss-api-base", api_base,
    ]
    if reviewed:
        argv.append("--reviewed")

    for key, flag in _VALUE_FLAGS.items():
        if key in params:
            argv += [flag, str(params[key])]
    for key, flag in _LIST_FLAGS.items():
        if key in params:
            argv += [flag, *[str(v) for v in params[key]]]
    for key, flag in _BOOL_FLAGS.items():
        if params.get(key, False):
            argv.append(flag)

    return argv


def run_probe_augment(
    dataset: str,
    seed: int,
    reviewed: bool,
    prompt_budget_multiple: int,
    api_base: str,
    params: dict,
    output_dir: Path,
) -> None:
    """Run create_probe_dataset.py --augment as a subprocess and record metadata.

    Raises:
        subprocess.CalledProcessError: If create_probe_dataset.py exits non-zero
            (a partial/failed augmentation run must not look like a success).
    """
    argv = build_augment_argv(dataset, seed, reviewed, prompt_budget_multiple, api_base, params)
    print(f"Dataset          : {dataset}")
    print(f"Seed             : {seed}")
    print(f"Reviewed         : {reviewed}")
    print(f"Prompt budget x  : {prompt_budget_multiple}")
    print(f"API base         : {api_base}")
    print(f"Command          : {sys.executable} {' '.join(argv)}\n")

    start_time = time.time()
    subprocess.run([sys.executable, *argv], cwd=_REPO_ROOT, check=True)

    paths.write_run_metadata(
        output_dir,
        start_time=start_time,
        dataset=dataset,
        seed=seed,
        reviewed=reviewed,
        prompt_budget_multiple=prompt_budget_multiple,
        params=params,
        note=(
            "The real augmentation output (probe_dataset*.json, "
            f"probe_augment_cache.json) was written to data/{dataset}/ by "
            "create_probe_dataset.py itself, not here -- this file only "
            "records what produced it."
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run probe-dataset augmentation (create_probe_dataset.py --augment).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("config", help="Path to an experiment-configs/.../<id>.yaml.")
    p.add_argument(
        "--api-base", default=None, metavar="URL",
        help="Override params.api_base (submit.sh injects the compute node's vLLM endpoint here).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    config_path = Path(args.config)
    cfg = paths.load_experiment_config(config_path)
    params = cfg["params"]
    paths.require_params(
        params, "dataset", "model", "reviewed", "prompt_budget_multiple", config_path=config_path
    )

    dataset = params["dataset"]
    if dataset not in _SUPPORTED_DATASETS:
        raise ValueError(
            f"{config_path}: params.dataset {dataset!r} has no create_probe_dataset.py "
            f"--augment support (choices: {_SUPPORTED_DATASETS})"
        )

    output_dir = paths.result_dir(dataset, "probe_augment", cfg["id"])
    api_base = args.api_base or params.get("api_base") or "http://localhost:8081/v1"

    run_probe_augment(
        dataset=dataset,
        seed=cfg["seed"],
        reviewed=params["reviewed"],
        prompt_budget_multiple=params["prompt_budget_multiple"],
        api_base=api_base,
        params=params,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
