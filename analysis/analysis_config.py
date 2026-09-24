"""Shared loader for analysis/analysis-configs/<id>.yaml.

Reuses the harness's standard id/project/description/seed/params envelope
(see notes/hub/conventions.md and experiments/utils.load_experiment_config's
own copy of the same envelope check) rather than inventing a second config
shape -- the difference from an experiments/experiment-configs/ entry is only
where it lives (flat under analysis/analysis-configs/, not nested by
dataset/experiment-type, since an analysis config isn't routed through
submit.sh/_resolve_job.py) and what params it carries.

Every analysis-config consumer (analysis/match_cache.py,
analysis/recovery_validity.py, ...) shares one params.experiment_ids list --
the ids that feed the analysis -- and one params.ground_truth_file: the exact
ground-truth CSV/JSON that experiment_ids are scored against, declared
explicitly here rather than read implicitly off each id's dataset's own
DatasetConfig.ground_truth_file. That indirection would let an analysis
config's numbers silently drift if the dataset config's ground truth file is
later edited or repointed (a revised review pass, a new subset) -- this
config pins the exact file an analysis was run against, so re-running it
later reproduces the same comparison even if the dataset config has since
moved on. A script that needs its own parameters reads them from
params.<script_name> via get_section(), not the top level, so a stray or
misspelled key under one script's section can never be silently ignored by
another (or by no one).
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ANALYSIS_CONFIGS_ROOT = _REPO_ROOT / "analysis" / "analysis-configs"

# Every params section name a consumer script owns, so load_analysis_config
# can reject a stray top-level key (e.g. a value meant for
# params.recovery_validity dropped at the top level instead) instead of
# silently ignoring it. Add a script's section name here when it grows one.
KNOWN_PARAM_SECTIONS = {"recovery_validity"}


def _resolve_ground_truth_path(ground_truth_file: str) -> Path:
    path = Path(ground_truth_file)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def get_ground_truth_path(cfg: dict) -> Path:
    """Resolve an already-loaded analysis config's params.ground_truth_file
    to an absolute Path, repo-root-relative if it wasn't already absolute.

    load_analysis_config has already asserted this file exists, so callers
    that only ever hold a config loaded that way don't need to re-check --
    this is exposed separately only so it can be re-resolved (e.g. for a
    freshness/hash check) without re-parsing the YAML.
    """
    return _resolve_ground_truth_path(cfg["params"]["ground_truth_file"])


def load_analysis_config(path: Path) -> dict:
    """Load and validate an analysis-configs/<id>.yaml envelope.

    Enforces the same id/project/description/seed/params envelope as
    experiments/utils.load_experiment_config (id must match the filename
    stem, params must be a mapping), plus checks specific to this envelope's
    job: params.experiment_ids must be a non-empty list of strings, and
    params.ground_truth_file must be a non-empty string naming a CSV/JSON
    file that exists (repo-root-relative if not absolute) -- see
    get_ground_truth_path to resolve it to a Path.

    Unlike an experiments/experiment-configs/ entry, ``seed`` here is NOT
    checked against experiments/config.yaml's defaults.seed. That check
    exists there because an experiment config's seed feeds
    utils.set_seeds() to make a model run reproducible against the one
    canonical value the rest of the repo's pipeline runs share. This
    envelope's seed instead seeds recovery_validity.py's paper-clustered
    bootstrap resample -- an unrelated RNG stream with no reason to match
    model-generation seeding, and one an analyst may deliberately want to
    vary (e.g. checking a CI is stable across bootstrap seeds). Only
    required to be present and explicit (no inferred default), never
    required to equal defaults.seed.

    Raises:
        ValueError: malformed envelope, id/filename mismatch,
            missing/empty/non-list/duplicate experiment_ids, or a missing/
            empty/non-string/nonexistent ground_truth_file.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    missing = [k for k in ("id", "project", "description", "seed", "params") if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing required key(s): {missing}")
    if cfg["id"] != path.stem:
        raise ValueError(
            f"{path}: id {cfg['id']!r} does not match filename stem {path.stem!r}"
        )
    if not isinstance(cfg["params"], dict):
        raise ValueError(f"{path}: params must be a mapping")

    experiment_ids = cfg["params"].get("experiment_ids")
    if not isinstance(experiment_ids, list) or not experiment_ids or not all(
        isinstance(x, str) for x in experiment_ids
    ):
        raise ValueError(
            f"{path}: params.experiment_ids must be a non-empty list of strings, "
            f"got {experiment_ids!r}"
        )
    if len(set(experiment_ids)) != len(experiment_ids):
        dupes = sorted({x for x in experiment_ids if experiment_ids.count(x) > 1})
        raise ValueError(f"{path}: params.experiment_ids has duplicate id(s): {dupes}")

    ground_truth_file = cfg["params"].get("ground_truth_file")
    if not isinstance(ground_truth_file, str) or not ground_truth_file:
        raise ValueError(
            f"{path}: params.ground_truth_file must be a non-empty string, "
            f"got {ground_truth_file!r}"
        )
    if not _resolve_ground_truth_path(ground_truth_file).exists():
        raise ValueError(
            f"{path}: params.ground_truth_file {ground_truth_file!r} does not "
            f"exist at {_resolve_ground_truth_path(ground_truth_file)}"
        )

    unexpected = set(cfg["params"]) - {"experiment_ids", "ground_truth_file"} - KNOWN_PARAM_SECTIONS
    if unexpected:
        raise ValueError(
            f"{path}: params has unexpected top-level key(s) {sorted(unexpected)} -- "
            f"known sections: {sorted(KNOWN_PARAM_SECTIONS)} (a value meant for one of "
            f"those sections dropped at the top level by mistake?)"
        )

    return cfg


def get_section(
    cfg: dict, name: str, required_keys: tuple[str, ...], optional_keys: tuple[str, ...] = (),
) -> dict:
    """cfg['params'][name], asserting its keys are exactly required_keys plus
    a subset of optional_keys -- no more, no less.

    No defaults for a required key (CLAUDE.md: no inferred defaults for a
    value that changes the reported numbers) and no unknown key silently
    ignored (a typo'd or misplaced key -- e.g. a recovery_validity option
    dropped at the top level instead of under params.recovery_validity --
    fails loud here instead of quietly doing nothing).

    Raises:
        KeyError: the section is missing, isn't a mapping, is missing a
            required key, or has a key outside required_keys/optional_keys.
    """
    if name not in cfg["params"]:
        raise KeyError(f"params.{name} section is required but missing")
    section = cfg["params"][name]
    if not isinstance(section, dict):
        raise KeyError(f"params.{name} must be a mapping, got {type(section).__name__}")

    actual = set(section)
    allowed = set(required_keys) | set(optional_keys)
    missing = set(required_keys) - actual
    unexpected = actual - allowed
    if missing or unexpected:
        raise KeyError(
            f"params.{name} keys {sorted(actual)} invalid -- "
            f"missing required: {sorted(missing)}, unexpected: {sorted(unexpected)}"
        )
    return section
