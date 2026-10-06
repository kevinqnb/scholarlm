"""Config loading, input resolution and the score join for analysis/meta_updated.py.

Split out of meta_updated.py for the same reason as calibration_ids.py: these
helpers have no import-time side effects and need no heavy data, so they can be unit
tested on a hand-built fixture (tests/test_meta_inputs.py).

Confidences are NOT recomputed here. They are the Platt-scaled probe / NTP
predictions that analysis/calibration_updated_v3.py stored in
analysis/results/calibration/<calibration config id>/predictions.pkl (the 'real'
cell with train dataset == test dataset). That file stores no measurement_ids: its
rows are the judged final.json rows whose document is outside the probe's
synthetic-training documents (and the Platt pool, which is the same set when train ==
test), in final.json order. ``stored_prediction_rows`` rebuilds that row selection and
asserts it against the pickle.

The one risky step is ``attach_scores``. Scores are per judged datapoint, keyed by
``measurement_id`` (unique in final.json).
The rows the meta-analysis weights are either those same rows (``rows: final``) or
the deduplicated ``postprocessed.json`` records (``rows: deduplicated``). Those are
NOT the same row set: postprocessed.json expands list-valued rows into several rows
that all inherit their parent's ``measurement_id``, so the join is many-to-one, and a
kept row's confidence is its parent datapoint's. ``attach_scores`` asserts the join
neither drops nor duplicates a row, leaves no row unscored, and that each row's
``document_id`` / ``attribute`` agree with the scored datapoint it joined to.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "experiments", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis import calibration_ids as cids
from analysis.analysis_config import (
    ANALYSIS_CONFIGS_ROOT, _load_envelope, get_section, load_calibration_v3_config,
)
from analysis.match_cache import _parse_numeric, repo_relative, sha256_file

SECTION = "meta"
SECTION_KEYS = ("calibration_config_id", "rows", "deduplication_config_id", "n_boot", "qq_attributes")
ROWS_CHOICES = ("final", "deduplicated")
# Columns the join compares between a kept row and the scored datapoint it joined to.
JOIN_CHECK_COLS = ("document_id", "attribute")


def numeric_point_value(point_value: pd.Series) -> pd.Series:
    """The numeric value the meta analysis uses for every row: ``point_value`` parsed with
    match_cache's ``_parse_numeric`` (plain numbers / numeric strings and "m x 10^e"
    scientific notation), i.e. the same numbers the ground-truth matching sees. Anything
    that does not parse (None, "pH", "1 m^2") is NaN and is dropped downstream by
    ``convert_units``. ``value`` is raw model text (e.g. "ca. 0.26") and is not used."""
    out = point_value.map(_parse_numeric).astype(float)
    assert len(out) == len(point_value)
    return out


def load_meta_config(path: Path) -> dict:
    """Load and validate an analysis config for meta_updated.py.

    ``params`` holds exactly the ``meta`` section (SECTION_KEYS: no defaults, no
    extras). ``deduplication_config_id`` is required to be present always: a string
    iff ``rows == 'deduplicated'``, null iff ``rows == 'final'``.
    """
    cfg = _load_envelope(path)
    unexpected = set(cfg["params"]) - {SECTION}
    if unexpected:
        raise ValueError(f"{path}: unexpected params key(s) {sorted(unexpected)}")
    sec = get_section(cfg, SECTION, SECTION_KEYS)
    if not isinstance(sec["calibration_config_id"], str) or not sec["calibration_config_id"]:
        raise ValueError(f"{path}: {SECTION}.calibration_config_id must be a non-empty string")
    if sec["rows"] not in ROWS_CHOICES:
        raise ValueError(f"{path}: {SECTION}.rows must be one of {ROWS_CHOICES}, got {sec['rows']!r}")
    dd = sec["deduplication_config_id"]
    if sec["rows"] == "deduplicated":
        if not isinstance(dd, str) or not dd:
            raise ValueError(f"{path}: {SECTION}.deduplication_config_id must be a non-empty string when rows is 'deduplicated'")
    elif dd is not None:
        raise ValueError(f"{path}: {SECTION}.deduplication_config_id must be null when rows is 'final', got {dd!r}")
    nb = sec["n_boot"]
    if isinstance(nb, bool) or not isinstance(nb, int) or nb <= 0:
        raise ValueError(f"{path}: {SECTION}.n_boot must be a positive int, got {nb!r}")
    qa = sec["qq_attributes"]
    if not isinstance(qa, list) or not qa or not all(isinstance(x, str) for x in qa) or len(set(qa)) != len(qa):
        raise ValueError(f"{path}: {SECTION}.qq_attributes must be a non-empty list of unique strings")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ValueError(f"{path}: seed must be an int, got {cfg['seed']!r}")
    return cfg


def resolve_meta_inputs(cfg: dict, dataset: str) -> dict:
    """Resolve and cross-check every run a meta config names (pure id/config/path logic).

    The calibration config (a v3 calibration analysis config) is the single source of
    truth for the extraction, judge_combine run, ground truth file, synthetic-probe
    run and probe variant: calibration_ids.resolve_calibration_inputs cross-checks all
    of them against each run's own committed config. Requires ``dataset`` in its
    datasets and a built predictions.pkl. For ``rows: deduplicated`` the deduplication
    config must list the extraction id.

    Returns dict(dataset, judge_model, extraction_id, extraction_dir, judge_combine_dir,
    ground_truth_path, probe_dir, probe_variant, predictions_path, dedup_dir | None).
    """
    sec = cfg["params"][SECTION]
    cal_id = sec["calibration_config_id"]
    cal_path = ANALYSIS_CONFIGS_ROOT / f"{cal_id}.yaml"
    if not cal_path.exists():
        raise FileNotFoundError(f"calibration config {cal_path} does not exist")
    cal_cfg = load_calibration_v3_config(cal_path)
    if dataset not in cal_cfg["params"]["datasets"]:
        raise ValueError(f"{cal_id}: no {dataset!r} block (has {sorted(cal_cfg['params']['datasets'])})")
    cal_inputs = cids.resolve_calibration_inputs(cal_cfg)
    ds = cal_inputs["datasets"][dataset]
    extraction_id = cal_cfg["params"]["datasets"][dataset]["extraction_id"]

    predictions_path = _REPO_ROOT / "analysis" / "results" / "calibration" / cal_id / "predictions.pkl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} missing -- run analysis/calibration_updated_v3.py first")

    dedup_dir = None
    if sec["rows"] == "deduplicated":
        from analysis import deduplication as dd
        dd_path = ANALYSIS_CONFIGS_ROOT / f"{sec['deduplication_config_id']}.yaml"
        if not dd_path.exists():
            raise FileNotFoundError(f"deduplication config {dd_path} does not exist")
        dd_cfg = dd.load_deduplication_config(dd_path)
        if extraction_id not in dd_cfg["params"]["experiment_ids"]:
            raise ValueError(f"{dd_cfg['id']}: experiment_ids does not contain {extraction_id!r}")
        dedup_dir = dd.deduplication_dir(dd_cfg["id"], extraction_id)

    return dict(dataset=dataset, judge_model=cal_inputs["judge_model"], extraction_id=extraction_id,
                extraction_dir=ds["extraction_dir"], judge_combine_dir=ds["judge_combine_dir"],
                ground_truth_path=ds["ground_truth_path"], probe_dir=ds["probe_dir"],
                probe_variant=cal_cfg["params"]["probe_variant"], predictions_path=predictions_path,
                dedup_dir=dedup_dir)


def stored_prediction_rows(final_df: pd.DataFrame, excluded_docs: set, probs: dict) -> pd.DataFrame:
    """Scored datapoints (one row each) for the stored calibration predictions.

    ``final_df`` is the judged final.json; ``excluded_docs`` the probe's synthetic-
    training documents (``syn_document_ids``); ``probs`` the pickle's cell with
    probe_probs / ntp_probs / labels. The stored arrays are in final.json order over
    the rows whose document is not excluded, so that selection is rebuilt here and the
    lengths must agree exactly. Returns measurement_id, document_id, attribute,
    ntp_prob, probe_prob.
    """
    idx = np.where(~final_df["document_id"].isin(excluded_docs).to_numpy())[0]
    n = len(probs["probe_probs"])
    if not (len(idx) == n == len(probs["ntp_probs"]) == len(probs["labels"])):
        raise ValueError(f"stored predictions have {n} rows but final.json has {len(idx)} rows outside "
                         f"the probe's training documents -- predictions.pkl does not match this run")
    out = final_df.iloc[idx][["measurement_id", "document_id", "attribute"]].reset_index(drop=True)
    out["ntp_prob"] = np.asarray(probs["ntp_probs"], dtype=float)
    out["probe_prob"] = np.asarray(probs["probe_probs"], dtype=float)
    if not (np.isfinite(out["ntp_prob"]).all() and np.isfinite(out["probe_prob"]).all()):
        raise ValueError("stored predictions contain non-finite values")
    return out


def load_checked_dedup_rows(dedup_dir: Path, extraction_id: str, deduplication_config_id: str) -> tuple[list[dict], dict]:
    """The kept records of a built deduplication, after checking its meta.json against
    the postprocessed.json now on disk (a regenerated extraction cannot be silently
    paired with a stale dedup). Returns (records, meta)."""
    meta_path = dedup_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} missing -- run analysis/deduplication.py first")
    meta = json.loads(meta_path.read_text())
    if meta["experiment_id"] != extraction_id or meta["analysis_config_id"] != deduplication_config_id:
        raise ValueError(f"{meta_path}: built for ({meta['analysis_config_id']!r}, {meta['experiment_id']!r}), "
                         f"asked for ({deduplication_config_id!r}, {extraction_id!r})")
    src = _REPO_ROOT / meta["extraction_file"]
    if sha256_file(src) != meta["extraction_sha256"]:
        raise ValueError(f"{src} changed since the deduplication was built (sha256 mismatch) -- rebuild it")
    records = json.loads((dedup_dir / "deduplicated.json").read_text())
    if len(records) != meta["rows_out"]:
        raise ValueError(f"{dedup_dir}: deduplicated.json has {len(records)} rows, meta.json says {meta['rows_out']}")
    return records, meta


def attach_scores(rows: pd.DataFrame, scored: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    """Left-join per-datapoint ``score_cols`` from ``scored`` onto ``rows`` by measurement_id.

    ``scored`` is one row per judged datapoint (unique measurement_id); ``rows`` may
    hold several rows per measurement_id (list-expanded children share their
    parent's id), so the join is many-to-one and each child inherits its parent's
    scores. Fails loud if: ``scored`` ids are not unique; any row's id is absent from
    ``scored``; the join changes the row count or order; any score is NaN; or a row's
    ``document_id`` / ``attribute`` differs from the scored datapoint it joined to.
    """
    if not scored["measurement_id"].is_unique:
        raise ValueError("scored datapoints do not have unique measurement_id")
    missing = sorted(set(rows["measurement_id"]) - set(scored["measurement_id"]))
    if missing:
        raise ValueError(f"{len(missing)} row measurement_ids have no scored datapoint (first: {missing[:5]})")
    keep = ["measurement_id", *JOIN_CHECK_COLS, *score_cols]
    suffix = "__scored"
    out = rows.merge(scored[keep], on="measurement_id", how="left", validate="many_to_one", suffixes=("", suffix))
    if len(out) != len(rows) or not (out["measurement_id"].to_numpy() == rows["measurement_id"].to_numpy()).all():
        raise AssertionError("score join changed the row count or order")
    for c in JOIN_CHECK_COLS:
        bad = out[c] != out[c + suffix]
        if bad.any():
            raise ValueError(f"{int(bad.sum())} rows disagree with their scored datapoint on {c!r} "
                             f"(first measurement_id {out.loc[bad, 'measurement_id'].iloc[0]})")
    out = out.drop(columns=[c + suffix for c in JOIN_CHECK_COLS])
    for c in score_cols:
        if out[c].isna().any():
            raise ValueError(f"{int(out[c].isna().sum())} rows have NaN {c!r} after the join")
    return out
