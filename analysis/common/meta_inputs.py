"""Config loading, input resolution and score joins for the pond meta analysis.

Used by pond_meta_analysis.py and pond_clustering.py. Confidences are not recomputed:
they are read from calibration.py's predictions.pkl (the real cell with train ==
test dataset) and joined to rows by ``measurement_id``.

The risky step is the join. Scores are per judged datapoint, but postprocessed and
deduplicated rows can share a parent's ``measurement_id``, so the join is many-to-one;
``attach_scores`` asserts it drops, duplicates and mislabels nothing.
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
    _load_envelope, analysis_config_path, analysis_results_dir, get_section, load_calibration_v4_config,
)
from analysis.common.prediction_store import check_real_cell
from analysis.common.matching import parse_numeric
from analysis.common.provenance import sha256_file

# pond_meta_analysis.py's config section: hard confidence thresholds.
SECTION_V2 = "meta_v2"
SECTION_V2_KEYS = ("calibration_config_id", "calibration_version", "rows", "deduplication_config_id", "confidence", "n_boot",
                   "reference", "ecosystems", "attributes", "qq_attributes", "thresholds", "min_n",
                   "n_shuffle_samples", "outlier_adjust", "threshold_mode", "w1_curve_scale")
# Which W1 the W1-vs-threshold figure plots: raw values for every attribute, or each
# attribute's native scale (log10 values for pond_meta.LOG_SCALE_ATTRIBUTES, raw otherwise).
W1_CURVE_SCALES = ("raw", "native")
# calibration_version -> loader that validates the calibration config.
CALIBRATION_LOADERS = {"v4": load_calibration_v4_config}
# Reference distribution for Q-Q / W2: curated ground-truth rows, or extracted rows
# whose stored calibration label is positive.
REFERENCE_CHOICES = ("ground_truth", "valid")
# Rows analysed: judged final.json, postprocessed.json (lists expanded), or its deduplication.
ROWS_CHOICES = ("final", "postprocessed", "deduplicated")
# For deduplicated rows: confidence from the cluster center, or mean over the cluster.
CONFIDENCE_CHOICES = ("center", "cluster_mean")
# Columns the join compares between a kept row and the scored datapoint it joined to.
JOIN_CHECK_COLS = ("document_id", "attribute")


def numeric_point_value(point_value: pd.Series) -> pd.Series:
    """Parse ``point_value`` to float the same way ground-truth matching does.

    Args:
        point_value: Raw ``point_value`` column.

    Returns:
        Float Series, NaN where unparseable (dropped downstream by ``convert_units``).
    """
    out = point_value.map(parse_numeric).astype(float)
    assert len(out) == len(point_value)
    return out


def load_meta_v2_config(path: Path) -> dict:
    """Load and validate a ``meta_v2`` config for pond_meta_analysis.py.

    Key rules: ``thresholds`` is strictly increasing in [0, 1) and starts at 0 (the
    unfiltered set); ``threshold_mode`` is ``value`` (keep confidence >= t) or
    ``percentile`` (drop the bottom fraction t per cell); ``min_n`` is the smallest
    sample scored; ``n_shuffle_samples`` is the number of random-subset draws in the baseline;
    ``outlier_adjust`` applies outlier_weight.py before thresholding; ``w1_curve_scale``
    is one of W1_CURVE_SCALES.

    Args:
        path: Path to the config YAML.

    Returns:
        The full config dict.

    Raises:
        ValueError, KeyError: Missing, extra or malformed keys.
    """
    cfg = _load_envelope(path, "meta")
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
    if sec["w1_curve_scale"] not in W1_CURVE_SCALES:
        raise ValueError(f"{path}: {SECTION_V2}.w1_curve_scale must be one of {W1_CURVE_SCALES}, "
                         f"got {sec['w1_curve_scale']!r}")
    return cfg


def _check_common_section(path: Path, cfg: dict, sec: dict, section: str) -> None:
    """Validate a meta section's input-selection and cell keys, and the envelope seed.

    ``deduplication_config_id`` and ``confidence`` must be set iff ``rows`` is
    ``deduplicated``, and null otherwise.

    Args:
        path: Config path (for error messages).
        cfg: Full config dict.
        sec: The section being checked.
        section: Section name (for error messages).

    Raises:
        ValueError: Any key is malformed.
    """
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
    """Resolve the inputs of a meta / clustering config through its calibration config.

    The calibration config is the single source of truth for the extraction, judge,
    ground truth and probe; ``resolve_calibration_inputs`` cross-checks them.

    Args:
        sec: Config section; reads ``calibration_config_id``, ``calibration_version``,
            ``rows``, ``deduplication_config_id``.
        dataset: Dataset to resolve.

    Returns:
        Dict with ``dataset``, ``calibration_config_id``, ``calibration_version``,
        ``judge_model``, ``extraction_id``, ``extraction_dir``, ``judge_combine_dir``,
        ``ground_truth_path``, ``probe_dir``, ``probe_variant``, ``predictions_path``,
        ``dedup_dir`` (None unless rows is ``deduplicated``).

    Raises:
        FileNotFoundError: A config or predictions.pkl is missing.
        ValueError: Unknown version, dataset not in the calibration config, or the
            deduplication config does not cover the extraction.
    """
    cal_id, calibration_version = sec["calibration_config_id"], sec["calibration_version"]
    cal_path = analysis_config_path("calibration", cal_id)
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

    predictions_path = analysis_results_dir("calibration") / cal_id / "predictions.pkl"
    if not predictions_path.exists():
        raise FileNotFoundError(f"{predictions_path} missing -- run analysis/calibration.py first")

    dedup_dir = None
    if sec["rows"] == "deduplicated":
        # deduplication runs on a recovery-validity config (see common/dedup.py)
        dd_path = analysis_config_path("recovery-validity", sec["deduplication_config_id"])
        if not dd_path.exists():
            raise FileNotFoundError(f"deduplication (recovery-validity) config {dd_path} does not exist")
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
    """Verified per-datapoint scores from a stored calibration cell.

    Args:
        final_df: Judged final.json rows.
        excluded_docs: The probe's synthetic-training documents.
        probs: The predictions.pkl real cell.
        final_sha256: Hash of the final.json in use.
        calibration_config_id: Expected calibration config id.

    Returns:
        DataFrame (measurement_id, document_id, attribute, ntp_prob, probe_prob, label).
    """
    return check_real_cell(probs, final_df, excluded_docs, final_sha256, calibration_config_id)


def load_checked_dedup_rows(dedup_dir: Path, extraction_id: str, deduplication_config_id: str) -> tuple[list[dict], dict]:
    """Load a built deduplication after checking it matches the current extraction file.

    Args:
        dedup_dir: Deduplication output directory.
        extraction_id: Expected extraction id.
        deduplication_config_id: Expected config id.

    Returns:
        ``(records, meta)``: deduplicated.json records and its meta.json.

    Raises:
        FileNotFoundError: meta.json missing.
        ValueError: Ids, extraction hash or row count disagree.
    """
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
    """Many-to-one join of per-datapoint scores onto rows by ``measurement_id``.

    Args:
        rows: Rows to score; several may share a parent ``measurement_id``.
        scored: One row per datapoint, unique ``measurement_id``.
        score_cols: Columns of ``scored`` to attach.

    Returns:
        ``rows`` (same count and order) with ``score_cols`` added.

    Raises:
        ValueError: Non-unique scored ids, unscored rows, NaN scores, or
            document_id/attribute disagreement.
        AssertionError: The join changed row count or order.
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
    """JSON string of a cell value, so lists/dicts/NaN compare by content.

    Args:
        x: Any cell value.

    Returns:
        ``json.dumps(x, default=str)``.
    """
    return json.dumps(x, default=str)


def row_provenance(rows: pd.DataFrame) -> pd.DataFrame:
    """Add row-identity columns to postprocessed.json rows.

    ``measurement_id`` is the parent datapoint's id, shared by list-expanded siblings,
    so row identity is position.

    Args:
        rows: postprocessed.json rows in file order.

    Returns:
        Copy with ``row`` (position), ``n_siblings`` (rows sharing the id) and
        ``list_index`` (index among siblings, -1 if the row has no siblings).

    Raises:
        ValueError: Non-int ids, non-contiguous siblings, or siblings that differ on a
            field other than ``point_value`` / ``list_values``.
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
    """Score deduplicated rows outside ``excluded_docs``.

    A kept row takes its cluster center's scores. With ``cluster_mean``, the
    ``mean_cols`` are averaged over the cluster instead; other ``score_cols`` (e.g.
    the judge label) stay the center's.

    Args:
        kept: deduplicated.json rows (cluster centers, in row order).
        post: postprocessed.json rows, after ``row_provenance``.
        clusters: clusters.csv (row, measurement_id, cluster_id, center_row,
            is_center, cluster_size, ...).
        scored: One row per judged datapoint.
        score_cols: Score columns to attach.
        mean_cols: Subset of ``score_cols`` averaged under ``cluster_mean``.
        confidence: ``"center"`` or ``"cluster_mean"``.
        excluded_docs: Documents to drop.

    Returns:
        Kept rows outside ``excluded_docs`` with ``row``, ``cluster_id``,
        ``cluster_size``, ``n_siblings``, ``list_index`` and the score columns.

    Raises:
        ValueError: clusters.csv, deduplicated.json and postprocessed.json disagree, a
            cluster spans documents, or a score is NaN.
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
