"""Config loading, input resolution and the score join for the pond meta analysis
(analysis/meta_updated_v2.py, analysis/clustering.py; data loading in pond_meta.py).

Kept apart from those scripts for the same reason as calibration_ids.py: these
helpers have no import-time side effects and need no heavy data, so they can be unit
tested on a hand-built fixture (tests/test_meta_inputs.py).

Confidences are NOT recomputed here. They are the recalibrated probe / NTP
predictions that analysis/calibration_updated_v4.py stored in
analysis/results/calibration/<calibration config id>/predictions.pkl (the 'real'
cell with train dataset == test dataset). Each real cell stores its own
measurement_ids (plus document_ids, attributes, the final.json / combined.json sha256
and the Platt sample's ids; see analysis/common/prediction_store.py), so
``stored_prediction_rows`` joins scores by id and verifies the cell against the
final.json in use. A pickle without that provenance is refused.

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

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "experiments", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis.common import calibration_ids as cids, dedup
from analysis.common.config import (
    ANALYSIS_CONFIGS_ROOT, _load_envelope, get_section, load_calibration_v4_config,
)
from analysis.common.prediction_store import check_real_cell
from analysis.common.matching import parse_numeric
from analysis.common.provenance import sha256_file

# meta_updated_v2.py's config section: hard confidence thresholds.
SECTION_V2 = "meta_v2"
SECTION_V2_KEYS = ("calibration_config_id", "calibration_version", "rows", "deduplication_config_id", "confidence", "n_boot",
                   "reference", "ecosystems", "attributes", "qq_attributes", "thresholds", "min_n",
                   "n_shuffle_samples", "outlier_adjust", "threshold_mode")
# Which calibration script built calibration_config_id's predictions.pkl (and so which
# config loader validates it). Only calibration_updated_v4.py's are read now; the v3
# entry was dropped with the retired v1 meta analysis.
CALIBRATION_LOADERS = {"v4": load_calibration_v4_config}
# The distribution every extracted setting is compared against (Q-Q x-axis, W2):
# ground_truth: the curated GT rows; valid: the extracted rows whose stored calibration
# label is positive (judge OR ground-truth match) -- the same label the probe is
# calibrated against.
REFERENCE_CHOICES = ("ground_truth", "valid")
# final: judged final.json rows (list values unexpanded); postprocessed: postprocessed.json
# (list values expanded to one row per entry, not deduplicated); deduplicated: the
# deduplication of postprocessed.json.
ROWS_CHOICES = ("final", "postprocessed", "deduplicated")
# For rows == 'deduplicated': a kept row's probe/NTP confidence is its cluster center's
# own (the center's parent datapoint's) or the mean over every cluster member's.
CONFIDENCE_CHOICES = ("center", "cluster_mean")
# Columns the join compares between a kept row and the scored datapoint it joined to.
JOIN_CHECK_COLS = ("document_id", "attribute")


def numeric_point_value(point_value: pd.Series) -> pd.Series:
    """The numeric value the meta analysis uses for every row: ``point_value`` parsed with
    matching's ``parse_numeric`` (plain numbers / numeric strings and "m x 10^e"
    scientific notation), i.e. the same numbers the ground-truth matching sees. Anything
    that does not parse (None, "pH", "1 m^2") is NaN and is dropped downstream by
    ``convert_units``. ``value`` is raw model text (e.g. "ca. 0.26") and is not used."""
    out = point_value.map(parse_numeric).astype(float)
    assert len(out) == len(point_value)
    return out


def load_meta_v2_config(path: Path) -> dict:
    """Load and validate an analysis config for meta_updated_v2.py.

    ``params`` holds exactly the ``meta_v2`` section (SECTION_V2_KEYS: no defaults, no
    extras). ``reference`` is one of REFERENCE_CHOICES; ``ecosystems`` / ``attributes``
    are the cells analysed (membership in pond_meta's canonical lists is checked by the
    script); ``qq_attributes`` must be a subset of ``attributes``.
    ``deduplication_config_id`` and ``confidence`` are required to be present always: a
    string (``confidence`` one of CONFIDENCE_CHOICES) iff ``rows == 'deduplicated'``,
    null otherwise.
    ``thresholds``: non-empty, strictly increasing list of numbers in [0, 1) starting at
    0 -- a threshold-t setting keeps the extracted rows with confidence >= t
    (``threshold_mode: value``) or drops the bottom fraction t of each cell's rows by
    confidence (``threshold_mode: percentile``); either way t = 0 is the unfiltered set. ``min_n``: a
    positive int; a Q-Q line / W1 score is only computed for a sample of at least
    that many rows. ``n_shuffle_samples``: a positive int, the number of confidence
    permutations behind the W1 permutation control. ``outlier_adjust``: a bool; true
    multiplies the NTP / probe confidences by the non-outlier factor of
    analysis/common/outlier_weight.py before thresholding.
    """
    cfg = _load_envelope(path)
    unexpected = set(cfg["params"]) - {SECTION_V2}
    if unexpected:
        raise ValueError(f"{path}: unexpected params key(s) {sorted(unexpected)}")
    sec = get_section(cfg, SECTION_V2, SECTION_V2_KEYS)
    _check_common_section(path, cfg, sec, SECTION_V2)
    th = sec["thresholds"]
    if (not isinstance(th, list) or not th
            or not all(isinstance(t, (int, float)) and not isinstance(t, bool) for t in th)):
        raise ValueError(f"{path}: {SECTION_V2}.thresholds must be a non-empty list of numbers")
    if not all(0.0 <= t < 1.0 for t in th) or not all(a < b for a, b in zip(th, th[1:])):
        raise ValueError(f"{path}: {SECTION_V2}.thresholds must be strictly increasing, in [0, 1), got {th}")
    if th[0] != 0:
        raise ValueError(f"{path}: {SECTION_V2}.thresholds must start at 0 (the unfiltered set), got {th}")
    if sec["calibration_version"] not in CALIBRATION_LOADERS:
        raise ValueError(f"{path}: {SECTION_V2}.calibration_version must be one of {sorted(CALIBRATION_LOADERS)}, "
                         f"got {sec['calibration_version']!r}")
    ns = sec["n_shuffle_samples"]
    if isinstance(ns, bool) or not isinstance(ns, int) or ns <= 0:
        raise ValueError(f"{path}: {SECTION_V2}.n_shuffle_samples must be a positive int, got {ns!r}")
    if sec["threshold_mode"] not in ("value", "percentile"):
        raise ValueError(f"{path}: {SECTION_V2}.threshold_mode must be 'value' or 'percentile', "
                         f"got {sec['threshold_mode']!r}")
    if not isinstance(sec["outlier_adjust"], bool):
        raise ValueError(f"{path}: {SECTION_V2}.outlier_adjust must be a bool, got {sec['outlier_adjust']!r}")
    mn = sec["min_n"]
    if isinstance(mn, bool) or not isinstance(mn, int) or mn <= 0:
        raise ValueError(f"{path}: {SECTION_V2}.min_n must be a positive int, got {mn!r}")
    return cfg


def _check_common_section(path: Path, cfg: dict, sec: dict, section: str) -> None:
    """Checks on the input-selection and cell keys of a meta section (and the envelope seed)."""
    SECTION = section  # noqa: N806 -- error messages name the section being checked
    if not isinstance(sec["calibration_config_id"], str) or not sec["calibration_config_id"]:
        raise ValueError(f"{path}: {SECTION}.calibration_config_id must be a non-empty string")
    if sec["rows"] not in ROWS_CHOICES:
        raise ValueError(f"{path}: {SECTION}.rows must be one of {ROWS_CHOICES}, got {sec['rows']!r}")
    dd = sec["deduplication_config_id"]
    if sec["rows"] == "deduplicated":
        if not isinstance(dd, str) or not dd:
            raise ValueError(f"{path}: {SECTION}.deduplication_config_id must be a non-empty string when rows is 'deduplicated'")
        if sec["confidence"] not in CONFIDENCE_CHOICES:
            raise ValueError(f"{path}: {SECTION}.confidence must be one of {CONFIDENCE_CHOICES} when rows is 'deduplicated', got {sec['confidence']!r}")
    else:
        if dd is not None:
            raise ValueError(f"{path}: {SECTION}.deduplication_config_id must be null unless rows is 'deduplicated', got {dd!r}")
        if sec["confidence"] is not None:
            raise ValueError(f"{path}: {SECTION}.confidence must be null unless rows is 'deduplicated', got {sec['confidence']!r}")
    nb = sec["n_boot"]
    if isinstance(nb, bool) or not isinstance(nb, int) or nb <= 0:
        raise ValueError(f"{path}: {SECTION}.n_boot must be a positive int, got {nb!r}")
    if sec["reference"] not in REFERENCE_CHOICES:
        raise ValueError(f"{path}: {SECTION}.reference must be one of {REFERENCE_CHOICES}, got {sec['reference']!r}")
    for key in ("ecosystems", "attributes", "qq_attributes"):
        v = sec[key]
        if not isinstance(v, list) or not v or not all(isinstance(x, str) for x in v) or len(set(v)) != len(v):
            raise ValueError(f"{path}: {SECTION}.{key} must be a non-empty list of unique strings")
    stray_qq = sorted(set(sec["qq_attributes"]) - set(sec["attributes"]))
    if stray_qq:
        raise ValueError(f"{path}: {SECTION}.qq_attributes not in {SECTION}.attributes: {stray_qq}")
    if isinstance(cfg["seed"], bool) or not isinstance(cfg["seed"], int):
        raise ValueError(f"{path}: seed must be an int, got {cfg['seed']!r}")


def resolve_meta_inputs(sec: dict, dataset: str) -> dict:
    """Resolve and cross-check every run a meta / clustering config names (pure
    id/config/path logic). ``sec`` is the caller's config section; only its
    ``calibration_config_id``, ``calibration_version``, ``rows`` and
    ``deduplication_config_id`` are read.

    The calibration config (validated by ``calibration_version``'s loader in
    CALIBRATION_LOADERS) is the single source of
    truth for the extraction, judge_combine run, ground truth file, synthetic-probe
    run and probe variant: calibration_ids.resolve_calibration_inputs cross-checks all
    of them against each run's own committed config. Requires ``dataset`` in its
    datasets and a built predictions.pkl. For ``rows: deduplicated`` the deduplication
    config must list the extraction id.

    Returns dict(dataset, calibration_config_id, calibration_version, judge_model, extraction_id, extraction_dir, judge_combine_dir,
    ground_truth_path, probe_dir, probe_variant, predictions_path, dedup_dir | None).
    """
    cal_id, calibration_version = sec["calibration_config_id"], sec["calibration_version"]
    cal_path = ANALYSIS_CONFIGS_ROOT / f"{cal_id}.yaml"
    if not cal_path.exists():
        raise FileNotFoundError(f"calibration config {cal_path} does not exist")
    if calibration_version not in CALIBRATION_LOADERS:
        raise ValueError(f"calibration_version must be one of {sorted(CALIBRATION_LOADERS)}, got {calibration_version!r}")
    cal_cfg = CALIBRATION_LOADERS[calibration_version](cal_path)
    if dataset not in cal_cfg["params"]["datasets"]:
        raise ValueError(f"{cal_id}: no {dataset!r} block (has {sorted(cal_cfg['params']['datasets'])})")
    cal_inputs = cids.resolve_calibration_inputs(cal_cfg)
    ds = cal_inputs["datasets"][dataset]
    extraction_id = cal_cfg["params"]["datasets"][dataset]["extraction_id"]

    predictions_path = _REPO_ROOT / "analysis" / "results" / "calibration" / cal_id / "predictions.pkl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} missing -- run analysis/calibration_updated_v4.py first")

    dedup_dir = None
    if sec["rows"] == "deduplicated":
        dd_path = ANALYSIS_CONFIGS_ROOT / f"{sec['deduplication_config_id']}.yaml"
        if not dd_path.exists():
            raise FileNotFoundError(f"deduplication config {dd_path} does not exist")
        dd_cfg = dedup.load_deduplication_config(dd_path)
        if extraction_id not in dd_cfg["params"]["experiment_ids"]:
            raise ValueError(f"{dd_cfg['id']}: experiment_ids does not contain {extraction_id!r}")
        dedup_dir = dedup.deduplication_dir(dd_cfg["id"], extraction_id)

    return dict(dataset=dataset, calibration_config_id=cal_id, calibration_version=calibration_version,
                judge_model=cal_inputs["judge_model"], extraction_id=extraction_id,
                extraction_dir=ds["extraction_dir"], judge_combine_dir=ds["judge_combine_dir"],
                ground_truth_path=ds["ground_truth_path"], probe_dir=ds["probe_dir"],
                probe_variant=cal_cfg["params"]["probe_variant"], predictions_path=predictions_path,
                dedup_dir=dedup_dir)


def stored_prediction_rows(final_df: pd.DataFrame, excluded_docs: set, probs: dict, final_sha256: str,
                           calibration_config_id: str) -> pd.DataFrame:
    """Scored datapoints (one row each) for the stored calibration predictions.

    ``final_df`` is the judged final.json; ``excluded_docs`` the probe's synthetic-
    training documents (``syn_document_ids``); ``probs`` the pickle's real cell. The
    cell carries its own measurement_ids, so scores are keyed by id and the cell is
    verified against final_df (``prediction_store.check_real_cell``): no positional
    rebuild. Returns measurement_id, document_id, attribute, ntp_prob, probe_prob,
    label.
    """
    return check_real_cell(probs, final_df, excluded_docs, final_sha256, calibration_config_id)


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


def _dumps(x) -> str:
    return json.dumps(x, default=str)


def row_provenance(rows: pd.DataFrame) -> pd.DataFrame:
    """Tag postprocessed.json rows with their identity. ``measurement_id`` is the PARENT
    datapoint's id, not a row id: a list-valued datapoint is expanded into one row per
    entry and every child keeps the parent's id. Row identity here is the position.

    Adds ``row`` (position 0..n-1), ``n_siblings`` (rows sharing this measurement_id) and
    ``list_index`` (0.. among siblings, -1 for a row that is the only one with its id).
    Fails loud unless: measurement_id is an int on every row; the rows of one id are
    contiguous; and siblings agree on every field except ``point_value`` and
    ``list_values`` (postprocessing.expand_list_values only sets those).
    """
    mid = rows["measurement_id"]
    if not all(isinstance(m, (int, np.integer)) and not isinstance(m, bool) for m in mid):
        raise ValueError("measurement_id must be an int on every row")
    run = (mid != mid.shift()).cumsum()
    if run.nunique() != mid.nunique():
        raise ValueError("rows sharing a measurement_id are not contiguous")
    out = rows.copy()
    out["row"] = np.arange(len(rows))
    out["n_siblings"] = out.groupby("measurement_id")["row"].transform("size")
    out["list_index"] = np.where(out["n_siblings"] > 1, out.groupby("measurement_id").cumcount(), -1)
    free = {"point_value", "list_values"}
    for m, g in out[out["n_siblings"] > 1].groupby("measurement_id"):
        for c in rows.columns:
            if c in free:
                continue
            if g[c].map(_dumps).nunique() != 1:
                raise ValueError(f"measurement_id {m}: sibling rows disagree on {c!r}")
    return out


def dedup_rows_with_scores(kept: pd.DataFrame, post: pd.DataFrame, clusters: pd.DataFrame,
                           scored: pd.DataFrame, score_cols: list[str], mean_cols: list[str],
                           confidence: str, excluded_docs: set) -> pd.DataFrame:
    """Deduplicated rows (outside ``excluded_docs``) with scores attached.

    ``kept`` is deduplicated.json (the cluster centers, in row order, with merged
    provenance); ``post`` the postprocessed.json rows it was built from (output of
    ``row_provenance``); ``clusters`` its clusters.csv (row, measurement_id, cluster_id,
    center_row, is_center, ...); ``scored`` one row per judged final.json datapoint.

    Every postprocessed row is scored through its parent's measurement_id (children
    inherit). A kept row takes its center's scores (``confidence == 'center'``) or, for
    ``mean_cols`` only, the mean over all rows in its cluster (``'cluster_mean'``);
    the remaining ``score_cols`` (the boolean judge label) are always the center's.
    The kept rows are located through clusters.csv and verified against ``post``;
    clusters must lie within one document (so dropping excluded documents never splits one).
    """
    if confidence not in CONFIDENCE_CHOICES:
        raise ValueError(f"confidence must be one of {CONFIDENCE_CHOICES}, got {confidence!r}")
    n = len(post)
    if len(clusters) != n or clusters["row"].tolist() != list(range(n)) \
            or clusters["measurement_id"].tolist() != post["measurement_id"].tolist():
        raise ValueError("clusters.csv does not describe these postprocessed rows (row / measurement_id mismatch)")
    centers = clusters[clusters["is_center"]].sort_values("row")
    if len(centers) != len(kept) or centers["cluster_id"].nunique() != len(centers):
        raise ValueError(f"{len(centers)} cluster centers vs {len(kept)} deduplicated rows")
    if not (clusters.loc[clusters["is_center"], "center_row"].to_numpy() == clusters.loc[clusters["is_center"], "row"].to_numpy()).all():
        raise ValueError("clusters.csv: a center's center_row is not itself")
    crow = centers["row"].to_numpy()
    for c in ("measurement_id", "document_id", "attribute", "point_value"):
        a = kept[c].map(_dumps).tolist()
        b = post.iloc[crow][c].map(_dumps).tolist()
        if a != b:
            raise ValueError(f"deduplicated.json row {next(i for i in range(len(a)) if a[i] != b[i])} "
                             f"does not match its center row in postprocessed.json on {c!r}")
    doc = post["document_id"].to_numpy()
    cid = clusters["cluster_id"].to_numpy()
    if (pd.Series(doc).groupby(cid).nunique() != 1).any():
        raise ValueError("a deduplication cluster spans more than one document")

    live = post[~post["document_id"].isin(excluded_docs)]
    live_scored = attach_scores(live, scored, score_cols).set_index("row")
    in_live = ~kept["document_id"].isin(excluded_docs).to_numpy()
    out = kept[in_live].reset_index(drop=True)
    out_rows = crow[in_live]
    out["row"] = out_rows
    out["cluster_id"] = centers["cluster_id"].to_numpy()[in_live]
    out["cluster_size"] = centers["cluster_size"].to_numpy()[in_live]
    out["n_siblings"] = post["n_siblings"].to_numpy()[out_rows]
    out["list_index"] = post["list_index"].to_numpy()[out_rows]
    for c in score_cols:
        out[c] = live_scored.loc[out_rows, c].to_numpy()
    if confidence == "cluster_mean":
        members = live_scored.join(clusters.set_index("row")["cluster_id"])
        mean = members.groupby("cluster_id")[mean_cols].mean()
        size = members.groupby("cluster_id").size()
        if not (size.loc[out["cluster_id"]].to_numpy() == out["cluster_size"].to_numpy()).all():
            raise ValueError("a kept cluster lost members to the excluded-document filter")
        for c in mean_cols:
            out[c] = mean.loc[out["cluster_id"], c].to_numpy()
    for c in score_cols:
        if out[c].isna().any():
            raise ValueError(f"NaN {c!r} after scoring the deduplicated rows")
    return out
