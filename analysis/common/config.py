"""Paths and validating loaders for analysis/analysis-configs/<type>/<id>.yaml.

Configs use the harness's id/project/description/seed/params envelope. Each type has
its own config directory and results directory, and a config filed under the wrong
type fails to load. Ground-truth files are pinned in the config (not read from the
DatasetConfig) so numbers can't drift if the dataset config is repointed.
Script-specific options live in ``params.<section>`` and are checked by ``get_section``,
so misplaced or misspelled keys fail loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ANALYSIS_CONFIGS_ROOT = _REPO_ROOT / "analysis" / "analysis-configs"
ANALYSIS_RESULTS_ROOT = _REPO_ROOT / "analysis" / "results"

# Types with both a config dir and a results dir. recovery-validity configs also drive
# the setup steps (postprocessing, match_cache, deduplicate_cache, deduplication).
ANALYSIS_TYPES = (
    "recovery-validity", "measeval", "synthetic-probe", "calibration", "calibration-validated",
    "platt-scaling", "decision-threshold", "meta", "clustering",
)
# Types with a results dir only (keyed by experiment id, or config id + experiment id).
RESULTS_ONLY_TYPES = ("match-cache", "deduplicate-cache", "deduplication")


def analysis_config_path(analysis_type: str, config_id: str) -> Path:
    """Path of an analysis config (not checked for existence).

    Args:
        analysis_type: One of ANALYSIS_TYPES.
        config_id: Config id.

    Returns:
        ``ANALYSIS_CONFIGS_ROOT/<analysis_type>/<config_id>.yaml``.

    Raises:
        ValueError: Unknown analysis type.
    """
    if analysis_type not in ANALYSIS_TYPES:
        raise ValueError(f"unknown analysis type {analysis_type!r}; known: {ANALYSIS_TYPES}")
    return ANALYSIS_CONFIGS_ROOT / analysis_type / f"{config_id}.yaml"


def analysis_results_dir(analysis_type: str) -> Path:
    """Results directory for an analysis type.

    Args:
        analysis_type: One of ANALYSIS_TYPES or RESULTS_ONLY_TYPES.

    Returns:
        ``ANALYSIS_RESULTS_ROOT/<analysis_type>``.

    Raises:
        ValueError: Unknown analysis type.
    """
    if analysis_type not in ANALYSIS_TYPES + RESULTS_ONLY_TYPES:
        raise ValueError(f"unknown analysis type {analysis_type!r}; known: {ANALYSIS_TYPES + RESULTS_ONLY_TYPES}")
    return ANALYSIS_RESULTS_ROOT / analysis_type


def check_config_type(path: Path, analysis_type: str) -> None:
    """Fail unless a config file sits directly in its type's directory.

    Args:
        path: Config path.
        analysis_type: Expected type.

    Raises:
        ValueError: Unknown type, or the file is in another directory.
    """
    if analysis_type not in ANALYSIS_TYPES:
        raise ValueError(f"unknown analysis type {analysis_type!r}; known: {ANALYSIS_TYPES}")
    if Path(path).parent.name != analysis_type:
        raise ValueError(
            f"{path}: a {analysis_type} config must live in analysis-configs/{analysis_type}/, "
            f"found in {Path(path).parent.name}/"
        )


# Allowed params sections, so a stray top-level key is rejected rather than ignored.
KNOWN_PARAM_SECTIONS = {"recovery_validity", "measeval_evaluation", "deduplicate_cache", "deduplication"}

# Params of a synthetic-probe config. use_platt_scaling picks the Platt-wrapped or
# *_noplatt pickles.
SYNTHETIC_PROBE_STR_KEYS = ("dataset", "judge_interp_id")
SYNTHETIC_PROBE_PARAM_KEYS = SYNTHETIC_PROBE_STR_KEYS + ("use_platt_scaling",)


def _resolve_ground_truth_path(ground_truth_file: str) -> Path:
    """Resolve a ground-truth path relative to the repo root unless it is absolute.

    Args:
        ground_truth_file: Path string from a config.

    Returns:
        Absolute Path.
    """
    path = Path(ground_truth_file)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def get_ground_truth_path(cfg: dict) -> Path:
    """Absolute path of a loaded config's ``params.ground_truth_file``.

    Args:
        cfg: Config from ``load_analysis_config`` (which already checked the file exists).

    Returns:
        Absolute Path.
    """
    return _resolve_ground_truth_path(cfg["params"]["ground_truth_file"])


def _load_envelope(path: Path, analysis_type: str) -> dict:
    """Parse a config and check its envelope and directory. Shared by every loader.

    Args:
        path: Config path.
        analysis_type: Expected type.

    Returns:
        The parsed config dict.

    Raises:
        ValueError: Wrong directory, missing envelope key, id != filename stem, or
            params not a mapping.
    """
    check_config_type(path, analysis_type)
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
    return cfg


def load_analysis_config(path: Path, analysis_type: str) -> dict:
    """Load a recovery-validity or measeval config (an experiment-id list plus ground truth).

    ``seed`` must be present but is not checked against the global seed: it seeds the
    analysis bootstrap, not model generation, and may be varied on purpose.

    Args:
        path: Config path.
        analysis_type: ``"recovery-validity"`` or ``"measeval"``.

    Returns:
        The config dict.

    Raises:
        ValueError: Bad envelope, empty or duplicate ``experiment_ids``, missing
            ``ground_truth_file``, or an unknown top-level params key.
    """
    if analysis_type not in ("recovery-validity", "measeval"):
        raise ValueError(f"load_analysis_config serves recovery-validity and measeval configs, got {analysis_type!r}")
    cfg = _load_envelope(path, analysis_type)

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
    """Return ``cfg['params'][name]`` after checking its keys exactly.

    Args:
        cfg: Loaded config.
        name: Section name.
        required_keys: Keys that must be present.
        optional_keys: Keys that may be present.

    Returns:
        The section dict.

    Raises:
        KeyError: Section missing or not a mapping, a required key missing, or an
            unknown key present.
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


def load_synthetic_probe_config(path: Path) -> dict:
    """Load a synthetic-probe config for synthetic_probe_train.py.

    params must be exactly ``dataset``, ``judge_interp_id`` (the synthetic-corpus
    judge run to train on) and ``use_platt_scaling``. ``seed`` seeds the training splits.

    Args:
        path: Config path.

    Returns:
        The config dict.

    Raises:
        ValueError: Bad envelope, wrong params keys or types, or non-int seed.
    """
    cfg = _load_envelope(path, "synthetic-probe")
    keys = set(cfg["params"])
    if keys != set(SYNTHETIC_PROBE_PARAM_KEYS):
        raise ValueError(
            f"{path}: params keys {sorted(keys)} must be exactly "
            f"{sorted(SYNTHETIC_PROBE_PARAM_KEYS)}"
        )
    for k in SYNTHETIC_PROBE_STR_KEYS:
        v = cfg["params"][k]
        if not isinstance(v, str) or not v:
            raise ValueError(f"{path}: params.{k} must be a non-empty string, got {v!r}")
    if not isinstance(cfg["params"]["use_platt_scaling"], bool):
        raise ValueError(
            f"{path}: params.use_platt_scaling must be a bool, got {cfg['params']['use_platt_scaling']!r}")
    if not isinstance(cfg["seed"], int) or isinstance(cfg["seed"], bool):
        raise ValueError(f"{path}: seed must be an int, got {cfg['seed']!r}")
    return cfg


# Per-dataset block keys shared by every calibration-family config:
#   extraction_id / judge_interp_id / judge_combine_id: real extraction, interp judge
#     run over it, and the combine run holding its labels.
#   ground_truth_file: ground truth the extraction is scored against.
#   synthetic_probe_config: synthetic-probe config id whose trained probe is applied.
#   syn_test_ids: {primary, diag} synthetic test runs; params.syn_split picks one.
#   use_matching_labels: True labels a row valid if judged valid OR matched.
# Cross-run consistency is checked by calibration_ids.resolve_calibration_inputs.
CALIBRATION_DATASETS = ("pond", "nfix", "supermat")
CALIBRATION_DATASET_KEYS = (
    "extraction_id", "judge_interp_id", "judge_combine_id", "ground_truth_file",
    "synthetic_probe_config", "syn_test_ids", "use_matching_labels",
)
CALIBRATION_SYN_SPLITS = ("primary", "diag")


def _validate_calibration_body(path: Path, cfg: dict, dataset_keys: tuple, datasets_expected: tuple) -> None:
    """Checks shared by the calibration loaders: seed, probe settings and dataset blocks.

    Keys a caller adds beyond CALIBRATION_DATASET_KEYS are checked for presence only;
    the caller validates their values.

    Args:
        path: Config path (for error messages).
        cfg: Loaded config.
        dataset_keys: Exact key set of each per-dataset block.
        datasets_expected: Exact set of dataset names.

    Raises:
        ValueError: Any check fails.
    """
    params = cfg["params"]
    if not isinstance(cfg["seed"], int) or isinstance(cfg["seed"], bool):
        raise ValueError(f"{path}: seed must be an int, got {cfg['seed']!r}")
    if params["probe_type"] not in ("head", "layer"):
        raise ValueError(f"{path}: params.probe_type must be 'head' or 'layer', got {params['probe_type']!r}")
    if params["probe_variant"] not in ("platt", "noplatt"):
        raise ValueError(
            f"{path}: params.probe_variant must be 'platt' or 'noplatt', got {params['probe_variant']!r}"
        )
    if params["syn_split"] not in CALIBRATION_SYN_SPLITS:
        raise ValueError(
            f"{path}: params.syn_split must be one of {list(CALIBRATION_SYN_SPLITS)}, got {params['syn_split']!r}"
        )
    datasets = params["datasets"]
    if not isinstance(datasets, dict) or set(datasets) != set(datasets_expected):
        raise ValueError(
            f"{path}: params.datasets must have exactly the keys {list(datasets_expected)}, "
            f"got {sorted(datasets) if isinstance(datasets, dict) else datasets!r}"
        )
    for ds, block in datasets.items():
        if not isinstance(block, dict) or set(block) != set(dataset_keys):
            raise ValueError(
                f"{path}: params.datasets.{ds} keys "
                f"{sorted(block) if isinstance(block, dict) else block!r} must be exactly "
                f"{sorted(dataset_keys)}"
            )
        for k in CALIBRATION_DATASET_KEYS:  # v2's extra pi_te_estimate is validated by its own loader
            if k in ("syn_test_ids", "use_matching_labels"):
                continue
            v = block[k]
            if not isinstance(v, str) or not v:
                raise ValueError(f"{path}: params.datasets.{ds}.{k} must be a non-empty string, got {v!r}")
        gt_path = _resolve_ground_truth_path(block["ground_truth_file"])
        if not gt_path.exists():
            raise ValueError(
                f"{path}: params.datasets.{ds}.ground_truth_file "
                f"{block['ground_truth_file']!r} does not exist at {gt_path}"
            )
        if not isinstance(block["use_matching_labels"], bool):
            raise ValueError(
                f"{path}: params.datasets.{ds}.use_matching_labels must be a bool, "
                f"got {block['use_matching_labels']!r}"
            )
        syn = block["syn_test_ids"]
        if not isinstance(syn, dict) or set(syn) != set(CALIBRATION_SYN_SPLITS):
            raise ValueError(
                f"{path}: params.datasets.{ds}.syn_test_ids must have exactly the keys "
                f"{list(CALIBRATION_SYN_SPLITS)}, got {sorted(syn) if isinstance(syn, dict) else syn!r}"
            )
        for split, v in syn.items():
            if not isinstance(v, str) or not v:
                raise ValueError(
                    f"{path}: params.datasets.{ds}.syn_test_ids.{split} must be a non-empty string, got {v!r}"
                )


# Top-level keys of every calibration-family config. pi_te_estimate is an assumed
# prevalence of valid rows in a dataset's real extraction.
CALIBRATION_V2_TOP_KEYS = ("probe_type", "probe_variant", "syn_split", "datasets")
CALIBRATION_V2_DATASET_KEYS = CALIBRATION_DATASET_KEYS + ("pi_te_estimate",)


# v3 schema (used by calibration_validated.py; read by calibration_latex.py): real cells
# are recalibrated on platt_n rows from the probe-training documents, with CIs from
# nested_bootstrap.py (n_fit_samples x n_doc_boot real, n_syn_boot synthetic).
CALIBRATION_V3_BOOTSTRAP_KEYS = ("n_fit_samples", "n_doc_boot", "n_syn_boot")
CALIBRATION_V3_TOP_KEYS = CALIBRATION_V2_TOP_KEYS + ("platt_n", "recalibration") + CALIBRATION_V3_BOOTSTRAP_KEYS
RECALIBRATION_METHODS = ("prior_shift", "intercept_fit", "platt_fit")


def _validate_v3_recalibration(path: Path, params: dict) -> None:
    """Check v3's ``platt_n``, bootstrap sizes and ``recalibration`` method.

    Args:
        path: Config path (for error messages).
        params: The config's params.

    Raises:
        ValueError: A count is not a positive int, or the method is unknown.
    """
    for key in ("platt_n",) + CALIBRATION_V3_BOOTSTRAP_KEYS:
        n = params[key]
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(f"{path}: params.{key} must be a positive int, got {n!r}")
    if params["recalibration"] not in RECALIBRATION_METHODS:
        raise ValueError(
            f"{path}: params.recalibration must be one of {RECALIBRATION_METHODS}, got {params['recalibration']!r}"
        )


def load_calibration_v3_config(path: Path) -> dict:
    """Load a v3-schema calibration config (read by calibration_latex.py).

    Args:
        path: Config path.

    Returns:
        The config dict.

    Raises:
        ValueError: Bad envelope, wrong keys, or bad values.
    """
    cfg = _load_envelope(path, "calibration")
    params = cfg["params"]
    if set(params) != set(CALIBRATION_V3_TOP_KEYS):
        raise ValueError(
            f"{path}: params keys {sorted(params)} must be exactly {sorted(CALIBRATION_V3_TOP_KEYS)}"
        )
    _validate_v3_recalibration(path, params)
    _validate_calibration_body(path, cfg, CALIBRATION_DATASET_KEYS, CALIBRATION_DATASETS)
    return cfg


# calibration_validated.py: v3 scored against human validations. Each dataset pins the
# validations file's sha256, since the file is rebuilt as more rows are validated.
CALIBRATION_VALIDATED_DATASETS = ("pond", "supermat")
CALIBRATION_VALIDATED_DATASET_KEYS = CALIBRATION_DATASET_KEYS + ("validation_sha256",)
VALIDATIONS_ENV = "SCHOLARLM_VALIDATIONS_DIR"


def validations_path(dataset: str) -> Path:
    """Path of a dataset's human-validation file under $SCHOLARLM_VALIDATIONS_DIR.

    The env var may come from the repo's .env. There is no default directory.

    Args:
        dataset: Dataset name.

    Returns:
        ``$SCHOLARLM_VALIDATIONS_DIR/<dataset>.json``.

    Raises:
        KeyError: The env var is unset.
        FileNotFoundError: The directory or file does not exist.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv(_REPO_ROOT / ".env")
    if VALIDATIONS_ENV not in os.environ:
        raise KeyError(f"{VALIDATIONS_ENV} is not set -- add it to {_REPO_ROOT / '.env'}")
    root = Path(os.environ[VALIDATIONS_ENV])
    if not root.is_dir():
        raise FileNotFoundError(f"{VALIDATIONS_ENV}={root} is not a directory")
    path = root / f"{dataset}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no validations for {dataset!r} at {path}")
    return path


def load_calibration_validated_config(path: Path) -> dict:
    """Load a calibration-validated config and check each validations file's pinned hash.

    Checked at load time so _resolve_job.py rejects a bad config before qsub.

    Args:
        path: Config path.

    Returns:
        The config dict.

    Raises:
        ValueError: Bad envelope, keys or values, or a hash mismatch.
        KeyError, FileNotFoundError: Validations env var unset or file missing.
    """
    import hashlib
    cfg = _load_envelope(path, "calibration-validated")
    params = cfg["params"]
    if set(params) != set(CALIBRATION_V3_TOP_KEYS):
        raise ValueError(
            f"{path}: params keys {sorted(params)} must be exactly {sorted(CALIBRATION_V3_TOP_KEYS)}"
        )
    _validate_v3_recalibration(path, params)
    _validate_calibration_body(path, cfg, CALIBRATION_VALIDATED_DATASET_KEYS, CALIBRATION_VALIDATED_DATASETS)
    for ds, block in params["datasets"].items():
        pin = block["validation_sha256"]
        if not isinstance(pin, str) or len(pin) != 64:
            raise ValueError(f"{path}: params.datasets.{ds}.validation_sha256 must be a 64-char hex string, got {pin!r}")
        actual = hashlib.sha256(validations_path(ds).read_bytes()).hexdigest()
        if actual != pin:
            raise ValueError(
                f"{path}: {validations_path(ds)} sha256 {actual} != pinned {pin} -- the validations were "
                f"rebuilt; re-pin validation_sha256 deliberately (this changes every number)"
            )
    return cfg


# platt_scaling.py: n_train_resamples fit samples per fit size n, each scored on the
# fixed real test set (no test-document bootstrap).
PLATT_SWEEP_V2_TOP_KEYS = CALIBRATION_V2_TOP_KEYS + ("platt_ns", "recalibration", "n_train_resamples")


def load_platt_sweep_v2_config(path: Path) -> dict:
    """Load a platt-scaling sweep config for platt_scaling.py.

    ``platt_ns`` is a strictly increasing list of fit sizes (0 = no recalibration)
    with at least one positive entry.

    Args:
        path: Config path.

    Returns:
        The config dict.

    Raises:
        ValueError: Bad envelope, keys or values.
    """
    cfg = _load_envelope(path, "platt-scaling")
    params = cfg["params"]
    if set(params) != set(PLATT_SWEEP_V2_TOP_KEYS):
        raise ValueError(
            f"{path}: params keys {sorted(params)} must be exactly {sorted(PLATT_SWEEP_V2_TOP_KEYS)}"
        )
    ns = params["platt_ns"]
    if (not isinstance(ns, list) or not ns
            or any(isinstance(n, bool) or not isinstance(n, int) or n < 0 for n in ns)
            or any(a >= b for a, b in zip(ns, ns[1:]))
            or ns == [0]):
        raise ValueError(f"{path}: params.platt_ns must be a strictly increasing list of non-negative ints "
                         f"with at least one positive n (0 = no-recalibration baseline), got {ns!r}")
    n = params["n_train_resamples"]
    if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"{path}: params.n_train_resamples must be a positive int, got {n!r}")
    if params["recalibration"] not in RECALIBRATION_METHODS:
        raise ValueError(
            f"{path}: params.recalibration must be one of {RECALIBRATION_METHODS}, got {params['recalibration']!r}"
        )
    _validate_calibration_body(path, cfg, CALIBRATION_DATASET_KEYS, CALIBRATION_DATASETS)
    return cfg


# calibration.py (v4): one recalibration map per (scorer, test dataset), fit once, with
# n_boot test-document bootstrap CIs. Methods are in common/recalibration.py.
#   fit_source sample: fit_n rows drawn from the probe-training pool with fit_seed.
#   fit_source manual: prior_shift only, using each dataset's pi_te_estimate.
#   fit_source oracle: fit on the test rows themselves (diagnostic only).
# fit_n / fit_seed are set iff sample, pi_te_estimate iff manual; otherwise null.
CALIBRATION_V4_TOP_KEYS = CALIBRATION_V2_TOP_KEYS + ("n_boot", "recalibration", "fit_source", "fit_n", "fit_seed")
CALIBRATION_V4_RECALIBRATIONS = ("prior_shift", "intercept_fit", "platt_fit")
CALIBRATION_V4_FIT_SOURCES = {"prior_shift": ("sample", "manual", "oracle"), "intercept_fit": ("sample", "oracle"),
                              "platt_fit": ("sample", "oracle")}


def is_int(v) -> bool:
    """True for an int that is not a bool.

    Args:
        v: Any value.

    Returns:
        Whether ``v`` is a non-bool int.
    """
    return isinstance(v, int) and not isinstance(v, bool)


def load_calibration_v4_config(path: Path) -> dict:
    """Load a v4 calibration config for calibration.py.

    A value the chosen ``fit_source`` would ignore (e.g. ``fit_n`` with ``manual``)
    is an error, not silently unused.

    Args:
        path: Config path.

    Returns:
        The config dict.

    Raises:
        ValueError: Bad envelope, keys or values, or fit settings inconsistent with
            ``recalibration`` / ``fit_source``.
    """
    cfg = _load_envelope(path, "calibration")
    params = cfg["params"]
    if set(params) != set(CALIBRATION_V4_TOP_KEYS):
        raise ValueError(
            f"{path}: params keys {sorted(params)} must be exactly {sorted(CALIBRATION_V4_TOP_KEYS)}"
        )
    _validate_calibration_body(path, cfg, CALIBRATION_V2_DATASET_KEYS, CALIBRATION_DATASETS)
    if not is_int(params["n_boot"]) or params["n_boot"] <= 0:
        raise ValueError(f"{path}: params.n_boot must be a positive int, got {params['n_boot']!r}")
    recal = params["recalibration"]
    if recal not in CALIBRATION_V4_RECALIBRATIONS:
        raise ValueError(
            f"{path}: params.recalibration must be one of {CALIBRATION_V4_RECALIBRATIONS}, got {recal!r}"
        )
    source = params["fit_source"]
    if source not in CALIBRATION_V4_FIT_SOURCES[recal]:
        raise ValueError(
            f"{path}: params.fit_source must be one of {CALIBRATION_V4_FIT_SOURCES[recal]} "
            f"for recalibration {recal!r}, got {source!r}"
        )
    n, fit_seed = params["fit_n"], params["fit_seed"]
    if source == "sample":
        if not is_int(n) or n <= 0:
            raise ValueError(f"{path}: params.fit_n must be a positive int for fit_source sample, got {n!r}")
        if not is_int(fit_seed) or fit_seed < 0:
            raise ValueError(f"{path}: params.fit_seed must be a non-negative int for fit_source sample, got {fit_seed!r}")
    else:
        for key, v in (("fit_n", n), ("fit_seed", fit_seed)):
            if v is not None:
                raise ValueError(f"{path}: params.{key} must be null for fit_source {source!r}, got {v!r}")
    for ds, block in params["datasets"].items():
        pi = block["pi_te_estimate"]
        if source == "manual":
            if isinstance(pi, bool) or not isinstance(pi, (int, float)) or not 0 < pi < 1:
                raise ValueError(
                    f"{path}: params.datasets.{ds}.pi_te_estimate must be a float in (0, 1) "
                    f"for fit_source manual, got {pi!r}"
                )
        elif pi is not None:
            raise ValueError(
                f"{path}: params.datasets.{ds}.pi_te_estimate must be null for fit_source {source!r}, got {pi!r}"
            )
    return cfg
