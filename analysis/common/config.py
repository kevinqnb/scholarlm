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

_REPO_ROOT = Path(__file__).parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ANALYSIS_CONFIGS_ROOT = _REPO_ROOT / "analysis" / "analysis-configs"

# Every params section name a consumer script owns, so load_analysis_config
# can reject a stray top-level key (e.g. a value meant for
# params.recovery_validity dropped at the top level instead) instead of
# silently ignoring it. Add a script's section name here when it grows one.
KNOWN_PARAM_SECTIONS = {"recovery_validity", "measeval_evaluation"}

# Params keys for analysis/synthetic_probe_train.py, which trains on one
# judge_interp run and so has no experiment_ids / ground_truth_file.
# use_platt_scaling (bool, required) picks head_probe.pkl / ntp_calibrator.pkl
# (CalibratedClassifierCV, probe_variant: platt) vs the *_noplatt.pkl variants.
SYNTHETIC_PROBE_STR_KEYS = ("dataset", "judge_interp_id")
SYNTHETIC_PROBE_PARAM_KEYS = SYNTHETIC_PROBE_STR_KEYS + ("use_platt_scaling",)


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


def _load_envelope(path: Path) -> dict:
    """Parse path and check the id/project/description/seed/params envelope
    (id == filename stem, params a mapping). Shared by every loader here."""
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
    cfg = _load_envelope(path)

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


def load_synthetic_probe_config(path: Path) -> dict:
    """Load analysis/synthetic_probe_train.py's analysis-configs/<id>.yaml.

    Same envelope as load_analysis_config, but params carries exactly
    ``dataset`` (pond / nfix / supermat), ``judge_interp_id`` (the
    judge_interp run on the synthetic corpus to train on) and
    ``use_platt_scaling`` (bool: True wraps the head probe / NTP calibrator in
    CalibratedClassifierCV and saves the unsuffixed pickles; False saves the
    ``_noplatt`` pickles) -- no experiment_ids / ground_truth_file, which belong
    to the scoring consumers. Any other params key is an error.

    Raises:
        ValueError: malformed envelope, or params keys != SYNTHETIC_PROBE_PARAM_KEYS,
            dataset / judge_interp_id not a non-empty string, use_platt_scaling
            not a bool, or seed not an int.
    ``seed`` seeds every split and LogisticRegression in synthetic_probe_train.py.
    """
    cfg = _load_envelope(path)
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


# Shared by every calibration-family loader below. Per-dataset blocks carry exactly
# CALIBRATION_DATASET_KEYS (plus a loader's own additions) for each of CALIBRATION_DATASETS:
#   - extraction_id / judge_interp_id / judge_combine_id: the real extraction run, the
#     qwen interp-judge run over it, and the judge_combine run holding its labels.
#   - ground_truth_file: the ground-truth CSV/JSON this extraction is scored against
#     (must exist; repo-root-relative if not absolute).
#   - synthetic_probe_config: the id of the analysis-configs/ yaml (loadable by
#     load_synthetic_probe_config) whose cached probe is applied.
#   - syn_test_ids: {primary: <id>, diag: <id>}, this dataset's synthetic judge_interp
#     test runs; both required, ``params.syn_split`` names which one is used.
#   - use_matching_labels: bool. True labels a real extraction valid if the judge said
#     so OR it matched a ground-truth row; False uses ``judgement_combined`` alone.
# Cross-run consistency (ids really are what they claim, judge model agreement,
# corpus-version agreement) is checked by calibration_ids.resolve_calibration_inputs.
CALIBRATION_DATASETS = ("pond", "nfix", "supermat")
CALIBRATION_DATASET_KEYS = (
    "extraction_id", "judge_interp_id", "judge_combine_id", "ground_truth_file",
    "synthetic_probe_config", "syn_test_ids", "use_matching_labels",
)
CALIBRATION_SYN_SPLITS = ("primary", "diag")


def _validate_calibration_body(path: Path, cfg: dict, dataset_keys: tuple, datasets_expected: tuple) -> None:
    """Checks shared by the calibration loaders: seed, probe_type/
    probe_variant/syn_split, and every per-dataset block against ``dataset_keys``
    (the keys of CALIBRATION_DATASET_KEYS plus whatever the caller adds, which
    it validates itself). ``params.datasets`` must have exactly ``datasets_expected``."""
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


# Top-level keys every calibration-family config carries, and the per-dataset blocks with
# one pi_te_estimate per dataset (the assumed prevalence of valid rows in that dataset's
# real extraction). Named V2 after the retired calibration_updated_v2.py that introduced
# them; calibration_updated_v4.py and platt_scaling_v2.py build on both.
CALIBRATION_V2_TOP_KEYS = ("probe_type", "probe_variant", "syn_split", "datasets")
CALIBRATION_V2_DATASET_KEYS = CALIBRATION_DATASET_KEYS + ("pi_te_estimate",)


# analysis/calibration_updated_v3.py: v2 minus pi_te_estimate. Real-extraction
# cells are instead recalibrated on platt_n rows sampled from the test
# dataset's own probe-training documents (labelled by judge_combine, plus
# matching if that dataset's use_matching_labels is on), by the required
# recalibration method: platt_fit (slope + intercept), intercept_fit (slope
# fixed at 1, intercept by MLE) or prior_shift (label-shift correction from the
# scorer's synthetic training prevalence to the sample's label rate).
# CIs and curve bands come from analysis/common/nested_bootstrap.py: n_fit_samples
# recalibration fit samples x n_doc_boot test-document resamples for real cells,
# n_syn_boot test-document resamples for (un-recalibrated) synthetic cells.
CALIBRATION_V3_BOOTSTRAP_KEYS = ("n_fit_samples", "n_doc_boot", "n_syn_boot")
CALIBRATION_V3_TOP_KEYS = CALIBRATION_V2_TOP_KEYS + ("platt_n", "recalibration") + CALIBRATION_V3_BOOTSTRAP_KEYS
RECALIBRATION_METHODS = ("prior_shift", "intercept_fit", "platt_fit")


def _validate_v3_recalibration(path: Path, params: dict) -> None:
    for key in ("platt_n",) + CALIBRATION_V3_BOOTSTRAP_KEYS:
        n = params[key]
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0:
            raise ValueError(f"{path}: params.{key} must be a positive int, got {n!r}")
    if params["recalibration"] not in RECALIBRATION_METHODS:
        raise ValueError(
            f"{path}: params.recalibration must be one of {RECALIBRATION_METHODS}, got {params['recalibration']!r}"
        )


def load_calibration_v3_config(path: Path) -> dict:
    """Load a calibration_updated_v3-schema analysis-configs/<id>.yaml.

    The v3 script itself is retired; this loader stays because
    calibration_latex.py still formats v3 output and calibration_validated.py
    shares its top-level keys. Per-dataset blocks are CALIBRATION_DATASET_KEYS
    (no pi_te_estimate anywhere), plus required top-level ``params.platt_n``, a positive int, the
    number of real rows per dataset each recalibrator is fit on, and
    ``params.recalibration``, one of RECALIBRATION_METHODS.

    Raises:
        ValueError: malformed envelope, wrong/missing/extra keys, bad value types.
    """
    cfg = _load_envelope(path)
    params = cfg["params"]
    if set(params) != set(CALIBRATION_V3_TOP_KEYS):
        raise ValueError(
            f"{path}: params keys {sorted(params)} must be exactly {sorted(CALIBRATION_V3_TOP_KEYS)}"
        )
    _validate_v3_recalibration(path, params)
    _validate_calibration_body(path, cfg, CALIBRATION_DATASET_KEYS, CALIBRATION_DATASETS)
    return cfg


# analysis/calibration_validated.py: v3 evaluated against human validations instead
# of LLM+matching labels. Only the datasets with validations (pond, supermat); each
# block additionally pins validation_sha256, the sha256 of $SCHOLARLM_VALIDATIONS_DIR/
# <ds>.json -- those files are rebuilt as more measurements get judged, so an
# unpinned file would change the reported numbers silently.
CALIBRATION_VALIDATED_DATASETS = ("pond", "supermat")
CALIBRATION_VALIDATED_DATASET_KEYS = CALIBRATION_DATASET_KEYS + ("validation_sha256",)
VALIDATIONS_ENV = "SCHOLARLM_VALIDATIONS_DIR"


def validations_path(dataset: str) -> Path:
    """$SCHOLARLM_VALIDATIONS_DIR/<dataset>.json (the env var is read from the repo's
    .env if not already exported). No default directory: unset is an error."""
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
    """Load analysis/calibration_validated.py's analysis-configs/<id>.yaml.

    Same keys as load_calibration_v3_config, except ``params.datasets`` is exactly
    CALIBRATION_VALIDATED_DATASETS and each block also carries ``validation_sha256``.
    Checks here (so _resolve_job.py rejects them before qsub) that
    $SCHOLARLM_VALIDATIONS_DIR/<ds>.json exists and hashes to the pinned value.

    Raises:
        ValueError: malformed envelope, wrong/missing/extra keys, bad value types,
            or a validations file whose sha256 differs from the pin.
        KeyError / FileNotFoundError: env var unset / validations file missing.
    """
    import hashlib
    cfg = _load_envelope(path)
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


# analysis/platt_scaling_v2.py: the sweep's inputs and platt_ns / recalibration, but
# no nested bootstrap -- n_train_resamples recalibration fit samples per n, each scored
# on the fixed (un-resampled) real test set. So no n_fit_samples / n_doc_boot.
PLATT_SWEEP_V2_TOP_KEYS = CALIBRATION_V2_TOP_KEYS + ("platt_ns", "recalibration", "n_train_resamples")


def load_platt_sweep_v2_config(path: Path) -> dict:
    """Load analysis/platt_scaling_v2.py's analysis-configs/<id>.yaml.

    Per-dataset blocks, probe_type, probe_variant and syn_split are exactly as in
    load_calibration_v3_config (syn_split is still required because
    calibration_ids.resolve_calibration_inputs cross-checks both synthetic test
    runs; the sweep itself only scores real rows). ``platt_ns`` is a strictly
    increasing list of non-negative ints (0 = the no-recalibration baseline),
    ``recalibration`` one of RECALIBRATION_METHODS, and ``n_train_resamples`` a
    positive int: the number of training-pool resamples (fit samples) per n.

    Raises:
        ValueError: malformed envelope, wrong/missing/extra keys, bad value types.
    """
    cfg = _load_envelope(path)
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


# analysis/calibration_updated_v4.py: one recalibration map per (scorer, test dataset),
# fit once -- no resampling of the fit data -- with test-document bootstrap CIs (n_boot
# resamples, real and synthetic cells alike; seeded by the envelope seed).
#   recalibration: prior_shift   -- slope 1, intercept logit(pi_te) - logit(pi_tr).
#                  intercept_fit -- slope 1, intercept by MLE on the fit rows.
#                  platt_fit     -- slope and intercept by unregularized logistic MLE on
#                                   the fit rows.
#   fit_source:    sample -- fit_n rows drawn uniformly without replacement from the test
#                            dataset's probe-training pool, by fit_seed (pi_te = their
#                            label rate for prior_shift).
#                  manual -- prior_shift only: the per-dataset pi_te_estimate.
#                  oracle -- the evaluated real test rows themselves (pi_te = their label
#                            rate; intercept_fit / platt_fit fit on them). Diagnostic only.
# fit_n and fit_seed are set iff fit_source is sample, and every dataset's
# pi_te_estimate iff it is manual; otherwise they must be null.
CALIBRATION_V4_TOP_KEYS = CALIBRATION_V2_TOP_KEYS + ("n_boot", "recalibration", "fit_source", "fit_n", "fit_seed")
CALIBRATION_V4_RECALIBRATIONS = ("prior_shift", "intercept_fit", "platt_fit")
CALIBRATION_V4_FIT_SOURCES = {"prior_shift": ("sample", "manual", "oracle"), "intercept_fit": ("sample", "oracle"),
                              "platt_fit": ("sample", "oracle")}


def is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def load_calibration_v4_config(path: Path) -> dict:
    """Load analysis/calibration_updated_v4.py's analysis-configs/<id>.yaml.

    Per-dataset blocks are v2's (CALIBRATION_V2_DATASET_KEYS: v1's keys plus
    ``pi_te_estimate``). Top level: v2's keys plus ``n_boot`` (positive int),
    ``recalibration`` (one of CALIBRATION_V4_RECALIBRATIONS), ``fit_source`` (one of
    CALIBRATION_V4_FIT_SOURCES[recalibration]), ``fit_n`` and ``fit_seed``.
    ``fit_n`` (positive int) and ``fit_seed`` (non-negative int) are set when
    ``fit_source`` is 'sample' and null otherwise; each ``pi_te_estimate`` is a float in
    (0, 1) when it is 'manual' and null otherwise -- a value the chosen source would
    ignore is an error.

    Raises:
        ValueError: malformed envelope, wrong/missing/extra keys, bad value types, or a
            fit_source / fit_n / fit_seed / pi_te_estimate inconsistent with recalibration.
    """
    cfg = _load_envelope(path)
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
