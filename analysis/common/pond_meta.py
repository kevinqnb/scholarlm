"""Pond meta-analysis data layer, shared by analysis/pond_meta_analysis.py and
analysis/pond_clustering.py: the canonical ecosystem/attribute universe, unit conversion and
plausibility bounds, the ground-truth unit fix, and ``load_data`` (held-out GT +
extraction rows with their stored, recalibrated NTP / probe confidences).

Extracted from the retired analysis/meta_updated.py (v1). Pond-specific throughout:
the ecosystem bucketing, ATTRIBUTES and PHYSICAL_BOUNDS only make sense for pond.
"""
from __future__ import annotations

import json
import pickle

import joblib
import numpy as np
import pandas as pd

from analysis.common.config import analysis_results_dir
from analysis.common.loaders import load_ground_truth_file
from analysis.common.meta_inputs import (
    _REPO_ROOT as REPO_ROOT,
    attach_scores, dedup_rows_with_scores, load_checked_dedup_rows, numeric_point_value, row_provenance,
    stored_prediction_rows,
)
from analysis.common.provenance import repo_relative, sha256_file
from experiments.run_extraction import load_dataset_config

# Paper figure style (ACL-style Times metrics). Not applied on import: each plotting
# entry point calls mpl.rcParams.update(PAPER_RCPARAMS) itself.
PAPER_RCPARAMS = {
    "font.family": "serif",
    # Nimbus Roman / Liberation Serif are the metric-compatible Times substitutes that
    # LaTeX's `times` package resolves to on Linux -- i.e. the actual glyphs an ACL-style
    # (\usepackage{times}) PDF renders with, not just a Times New Roman lookalike.
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",  # STIX matches Times metrics; "cm" (Computer Modern) clashes visually
    "text.usetex": False,
    "font.size": 15, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 12, "legend.title_fontsize": 13,
    "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.format": "pdf", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}

# Every run the meta analysis reads is named by one analysis config
# (analysis/analysis-configs/meta/<id>.yaml, see meta_inputs.load_meta_v2_config); outputs go to
# analysis/results/meta/<config id>/. Nothing is read from the legacy data/experiments tree.
META_ROOT = analysis_results_dir("meta")

# ── Parameters ───────────────────────────────────────────────────────────────
# Label for the `dataset` column of the output CSVs. The ecosystem bucketing,
# ATTRIBUTES and PHYSICAL_BOUNDS below are pond-specific; resolve_meta_inputs is always
# called with this dataset, so a config whose calibration has no pond block fails there.
DATASET = 'pond'

# Canonical universe of cells. A config's ecosystems / attributes pick the subset actually
# analysed; pond_meta_analysis.py keys its RNG streams on positions in THESE lists, so
# subsetting never changes a retained cell's CI -- never reorder them.
ECOSYSTEMS = ['pond', 'lake', 'wetland']
ATTRIBUTES = ['surface_area', 'max_depth', 'vegetation_cover', 'ph', 'tn', 'tp', 'chla']

METHODS = ['ntp', 'probe']
METHOD_PROB_COL = {'ntp': 'ntp_prob', 'probe': 'probe_prob'}

# Short Q-Q x-axis label for each reference.
REFERENCE_AXIS_LABEL = {'ground_truth': 'GT', 'valid': 'Valid extracted'}

STANDARD_UNITS = {
    'max_depth': 'm', 'surface_area': 'm^2', 'vegetation_cover': 'percent',
    'tn': 'µg/L', 'tp': 'µg/L', 'chla': 'µg/L', 'ph': None,
}

# Multiply-to-standard factors: standard_value = raw_value * UNIT_CONVERSION[attr][unit].
# Units not listed here are treated as unconvertible for that attribute (row dropped) --
# this includes fundamentally different measurands (e.g. µg/cm^2 chla, % dry wt tn/tp,
# pounds surface_area) that must not be silently passed through.
UNIT_CONVERSION = {
    'max_depth':        {'m': 1.0, 'cm': 0.01, 'feet': 0.3048, 'ft': 0.3048, 'km': 1000.0},
    'surface_area':     {'m^2': 1.0, 'm²': 1.0, 'km^2': 1e6, 'km²': 1e6, 'ha': 1e4,
                          'acres': 4046.86, 'x10^-2 km^2': 1e4, 'x10^-6 m^2': 1e6},
    'vegetation_cover': {'percent': 1.0, '%': 1.0, 'fraction': 100.0},
    'tn':  {'µg/L': 1.0, 'μg/L': 1.0, 'µg L⁻¹': 1.0, 'mg/L': 1000.0, 'mg L⁻¹': 1000.0,
            'mg/m^3': 1.0, 'µmol/L': 14.01, 'μmol/L': 14.01},
    'tp':  {'µg/L': 1.0, 'μg/L': 1.0, 'µg L⁻¹': 1.0, 'mg/L': 1000.0, 'mg L⁻¹': 1000.0,
            'mg/m^3': 1.0, 'µmol/L': 30.97, 'μmol/L': 30.97},
    'chla': {'µg/L': 1.0, 'μg/L': 1.0, 'mg/L': 1000.0, 'mg/m^3': 1.0, 'mg L⁻¹': 1000.0},
    'ph': {},  # dimensionless: any unit string accepted, factor 1.0 (handled specially below)
}

# Physical/domain plausibility bounds, in the standard unit for each attribute.
# Values outside these bounds are dropped (-> NaN), same as an unrecognized unit.
#
# Deliberately "reasonably unlikely" rather than "physically impossible": world-record
# ceilings (Caspian Sea, Lake Baikal, ...) let through a specific recurring extraction
# bug where a real, correctly-read reference/comparison lake cited in a source paper's
# table (e.g. Lake Superior at 8.2e10 m^2, cited for context in a pond/wetland paper)
# gets extracted as if it were one of the paper's own study systems -- the number is
# faithful to the text, so no confidence signal catches it, but it has no business in
# a per-ecosystem pond/lake/wetland comparison. Same story for tn/tp/chla: a "mg/L"
# unit tag that should have been "mg/m^3" (numerically = ug/L, 1000x smaller) survives
# UNIT_CONVERSION as a legally recognized unit and inflates the tail by exactly 1000x.
#
# Each ceiling below is calibrated against the empirical max observed in the *full*
# ground-truth corpus (all documents, not just the held-out set used for the final
# comparison, to avoid tuning bounds to the eval slice) plus a several-fold safety
# margin -- generous enough to keep legitimate extremes on record (e.g. a 1704 ug/L
# chla reading from a genuinely tiny, bloom-choked shallow pond; a 9850 ug/L tp
# reading from Lake Nakuru, a documented hypereutrophic soda lake), while sitting
# far below the contaminating values found in practice (reference lakes at 1e9-1e11
# m^2; a mg/L-mislabeled tp cluster at 31,000-44,000 ug/L; mg/L-mislabeled chla at
# 5,000-10,000 ug/L). See docs/plans or commit history for the full audit.
PHYSICAL_BOUNDS = {
    'max_depth':        (0, 50),          # full-corpus GT max observed: 9 m
    'surface_area':     (0, 1e6),         # full-corpus GT max observed: 1.938e5 m^2
    'ph':                (0, 14),         # standard aqueous pH scale
    'vegetation_cover':  (0, 100),        # definition of a percentage
    'tn':                (0, 50_000),     # full-corpus GT max observed: 3.12e4 ug/L
    'tp':                (0, 15_000),     # full-corpus GT max observed: 9,850 ug/L (Lake Nakuru)
    'chla':              (0, 3_000),      # full-corpus GT max observed: 1,704 ug/L
}

# Log-scale attributes span several orders of magnitude; the rest read fine on a linear axis.
# max_depth is log-scale too: extraction noise includes implausible outliers (e.g. a
# 108,000 m "depth" for a wetland treatment cell) that otherwise flatten the whole panel.
LOG_SCALE_ATTRIBUTES = {'surface_area', 'max_depth', 'tn', 'tp', 'chla'}

# Quantile probability grid for the Q-Q lines, capped to [0.025, 0.975] so a single
# extreme outlier in either tail can't stretch the panel.
QLEVELS = np.linspace(0.025, 0.975, 100)


# ── Ecosystem bucketing ─────────────────────────────────────────────────────

def bucket_ecosystem(raw: str | None) -> str:
    """Map a raw free-text ecosystem string to pond / lake / wetland / other.

    Single-keyword strings (containing exactly one of wetland/pond/pool/lake)
    are bucketed to that class. Compounds ("wetland vs. lake") and terms with
    no keyword match ("pothole", "reservoir") fall to 'other' and are excluded
    from the analysis, per instructions to disregard the 'other' category.
    """
    if not raw:
        return 'other'
    s = str(raw).lower()
    hits = {b for b, kw in [('wetland', 'wetland'), ('pond', 'pond'), ('pond', 'pool'), ('lake', 'lake')]
            if kw in s}
    return hits.pop() if len(hits) == 1 else 'other'


# ── Unit conversion ─────────────────────────────────────────────────────────

def fix_fish_production_units(gt_df: pd.DataFrame, config) -> pd.DataFrame:
    """Fix a data bug: 56 GT surface_area rows for 'fish_production_in_lakes' have
    `units` corrupted with a near-duplicate of `value` instead of the real unit.
    The paper's actual surface_area unit ('acres') is recovered from directory.json.

    NOTE: This is not for extracted data at all. We are fixing the GROUND TRUTH ONLY. 
    """
    gt_df = gt_df.copy()
    metadata_path = REPO_ROOT / config.metadata_file
    with open(metadata_path) as f:
        directory = json.load(f)
    true_unit = directory['fish_production_in_lakes']['units']['surface_area']
    mask = (gt_df['document_id'] == 'fish_production_in_lakes') & (gt_df['attribute'] == 'surface_area')
    n_fixed = int(mask.sum())
    if n_fixed:
        gt_df.loc[mask, 'units'] = true_unit
        print(f"[meta] fixed {n_fixed} corrupted 'surface_area' units for "
              f"fish_production_in_lakes -> {true_unit!r}")
    return gt_df


def convert_units(
    df: pd.DataFrame,
    unit_conversion: dict,
    value_col: str = 'value',
    unit_col: str = 'units',
    attribute_col: str = 'attribute',
    out_col: str = 'converted_value',
) -> pd.DataFrame:
    """Convert values to the standard unit per attribute; unconvertible rows -> NaN.

    ``unit_conversion`` is the caller's multiply-to-standard table, shaped like
    UNIT_CONVERSION (exactly the same attribute keys, asserted): pond_clustering.py passes
    UNIT_CONVERSION, pond_meta_analysis.py its own UNIT_CONVERSION_V2.

    Unlike scholarlm.utils.unit_conversion.apply_unit_conversion, a unit that is not
    in unit_conversion[attribute] yields NaN (dropped), not a factor-of-1.0 passthrough
    -- we do not want to silently treat e.g. a 'pounds' surface_area as if it were m^2.
    pH is the one exception: it is dimensionless, so any unit string is accepted.

    All of these attributes (surface_area, max_depth, vegetation_cover, tn, tp, chla,
    ph) are non-negative physical quantities, so a negative converted value is never a
    real measurement -- it is dropped (-> NaN) rather than plotted as-is. In practice
    this catches cases where the source paper reported a log-transformed value (e.g.
    "value": -0.54, sometimes labeled "units": "log") that the extraction model
    mislabeled with a real physical unit on some duplicate mentions of the same entity,
    producing a nonsensical negative area/depth/concentration after conversion.

    Values are also dropped (-> NaN) if they fall outside PHYSICAL_BOUNDS for their
    attribute -- e.g. a "53,010,000 km^2" lake surface area, which is larger than
    Earth. These bounds are real-world extremes chosen independent of this dataset
    (see PHYSICAL_BOUNDS), applied uniformly to ground truth and extraction alike, so
    this is a plausibility check, not a fit to what we expect the answer to be.
    """
    assert set(unit_conversion) == set(UNIT_CONVERSION), (
        f'unit_conversion attributes {sorted(unit_conversion)} != {sorted(UNIT_CONVERSION)}')
    df = df.copy()
    numeric_values = pd.to_numeric(df[value_col], errors='coerce')

    factors = pd.Series(np.nan, index=df.index)
    for attribute, unit_map in unit_conversion.items():
        attr_mask = df[attribute_col] == attribute
        if attribute == 'ph':
            factors.loc[attr_mask] = 1.0
            continue
        for unit, factor in unit_map.items():
            factors.loc[attr_mask & (df[unit_col] == unit)] = factor

    converted = numeric_values * factors
    converted = converted.where(converted >= 0)

    for attribute, (lo, hi) in PHYSICAL_BOUNDS.items():
        attr_mask = df[attribute_col] == attribute
        out_of_bounds = attr_mask & ((converted < lo) | (converted > hi))
        converted = converted.where(~out_of_bounds)

    df[out_col] = converted
    return df


# ── Data loading ─────────────────────────────────────────────────────────────

def _load_stored_scores(final_df: pd.DataFrame, combined_df: pd.DataFrame, inputs: dict):
    """Recalibrated NTP / probe confidence for the judged datapoints, read from the
    predictions analysis/calibration.py stored -- nothing is recomputed.

    Uses the 'real' cell with train dataset == test dataset (this dataset's own probe on
    its own real extraction); see meta_inputs.stored_prediction_rows for how the pickle's
    rows are mapped back to measurement_ids. Returns (scored_df, syn_docs, input_files);
    scored_df has measurement_id, document_id, attribute, judgement_combined, ntp_prob,
    probe_prob, label, one row per datapoint outside the probe's training documents.
    ``label`` is the stored calibration label (judge OR ground-truth match): the
    'valid' reference setting filters on it.
    """
    judge = inputs['judge_model']
    for col in ('measurement_id', 'document_id', 'attribute'):
        assert len(final_df) == len(combined_df) and (final_df[col].to_numpy() == combined_df[col].to_numpy()).all(), (
            f'final.json and combined.json disagree on {col}')
    assert final_df['measurement_id'].is_unique, 'final.json measurement_id is not unique'

    suffix = '' if inputs['probe_variant'] == 'platt' else '_noplatt'
    probe_path = inputs['probe_dir'] / f'head_probe{suffix}.pkl'
    probe_art = joblib.load(probe_path)  # only for syn_document_ids; its predict_proba is never called
    assert probe_art['judge_model'] == judge and probe_art['dataset'] == DATASET, probe_path
    syn_docs = set(probe_art['syn_document_ids'])
    # Held-out must mean held out from BOTH confidence models: the NTP calibrator's
    # training documents have to be the probe's.
    ntp_path = inputs['probe_dir'] / f'ntp_calibrator{suffix}.pkl'
    assert set(joblib.load(ntp_path)['syn_document_ids']) == syn_docs, (
        f'{ntp_path} was trained on different documents than {probe_path}')

    with open(inputs['predictions_path'], 'rb') as f:
        cell = pickle.load(f)['real'][judge][DATASET][DATASET]
    # v4 real cells are recalibrated by a map fit once on rows outside the scored set.
    assert inputs['calibration_version'] == 'v4', inputs['calibration_version']
    assert cell['recal_map'] is not None and cell['fit_on_test_rows'] is False, (
        'v4 real-cell predictions should be recalibrated on rows outside the test set')
    # The recalibration fit rows must come from training documents, not merely be
    # different measurement_ids (check_real_cell checks only the latter).
    fit_docs = set(final_df.set_index('measurement_id').loc[cell['platt_measurement_ids'], 'document_id'])
    assert fit_docs <= syn_docs, f'recalibration fit rows from held-out documents: {sorted(fit_docs - syn_docs)}'
    scored = stored_prediction_rows(final_df, syn_docs, cell, sha256_file(inputs['extraction_dir'] / 'final.json'),
                                    inputs['calibration_config_id'])

    # Every row the judge accepted is a positive label in the pickle (labels = judge OR matched).
    jc = combined_df.set_index('measurement_id')['judgement_combined'].astype(bool)
    assert jc.index.is_unique and set(scored['measurement_id']) <= set(jc.index)
    scored['judgement_combined'] = jc.loc[scored['measurement_id']].to_numpy()
    assert scored.loc[scored['judgement_combined'], 'label'].all(), 'stored labels disagree with judgement_combined'
    assert scored['label'].dtype == bool
    return scored, syn_docs, [probe_path, inputs['predictions_path']]


def load_data(sec: dict, inputs: dict, unit_conversion: dict):
    """Load GT + extraction rows, restrict to held-out documents, and attach
    judgement_combined / ntp_prob / probe_prob to the extraction rows.

    Held-out means outside the probe/NTP training documents (syn_document_ids). Each
    side keeps all its held-out documents: GT and extraction need not cover the same
    documents. ``unit_conversion`` is the table convert_units applies to both sides (no
    default: each caller names its own, see convert_units); it is recorded in the manifest.

    ``sec`` is the caller's config section; only its ``rows``, ``confidence`` and
    ``deduplication_config_id`` are read. Scores are computed on the judged run's
    final.json rows and joined by measurement_id onto the rows named by ``rows``
    (final.json itself, postprocessed.json, or the deduplicated records -- see
    meta_inputs.attach_scores for the many-to-one join).
    Returns (gt_df, ext_df, manifest) where manifest records row counts and input hashes.
    """
    config = load_dataset_config(DATASET)

    gt_df = load_ground_truth_file(inputs['ground_truth_path'])
    gt_df = fix_fish_production_units(gt_df, config)

    final_path = inputs['extraction_dir'] / 'final.json'
    combined_path = inputs['judge_combine_dir'] / 'combined.json'
    final_df = pd.DataFrame(json.loads(final_path.read_text()))
    combined_df = pd.DataFrame(json.loads(combined_path.read_text()))
    scored, syn_docs, scored_inputs = _load_stored_scores(final_df, combined_df, inputs)
    # label (judge OR match) is boolean like judgement_combined: a deduplicated row
    # always takes its cluster center's (dedup_rows_with_scores), never a mean.
    score_cols = ['judgement_combined', 'label', 'ntp_prob', 'probe_prob']

    input_files = [inputs['ground_truth_path'], final_path, combined_path, *scored_inputs]
    if sec['rows'] == 'final':
        rows_df = final_df
        n_rows = len(rows_df)
        # Rows from the probe's own training documents have no held-out score (and are
        # excluded below anyway); drop them before the join so every remaining row must score.
        rows_df = rows_df[~rows_df['document_id'].isin(syn_docs)].reset_index(drop=True)
        ext_df = attach_scores(rows_df, scored, score_cols)
    else:
        post_path = inputs['extraction_dir'] / 'postprocessed.json'
        post_df = row_provenance(pd.DataFrame(json.loads(post_path.read_text())))
        input_files.append(post_path)
        n_rows = len(post_df)
        if sec['rows'] == 'postprocessed':
            rows_df = post_df[~post_df['document_id'].isin(syn_docs)].reset_index(drop=True)
            ext_df = attach_scores(rows_df, scored, score_cols)
        else:
            records, dedup_meta = load_checked_dedup_rows(
                inputs['dedup_dir'], inputs['extraction_id'], sec['deduplication_config_id'])
            assert REPO_ROOT / dedup_meta['extraction_file'] == post_path, (
                f"deduplication was built from {dedup_meta['extraction_file']}, not {post_path}")
            assert dedup_meta['rows_in'] == len(post_df)
            clusters_df = pd.read_csv(inputs['dedup_dir'] / 'clusters.csv')
            input_files += [inputs['dedup_dir'] / 'deduplicated.json', inputs['dedup_dir'] / 'clusters.csv']
            print(f"[meta] deduplicated rows: {dedup_meta['rows_in']} -> {dedup_meta['rows_out']} "
                  f"(confidence: {sec['confidence']})")
            ext_df = dedup_rows_with_scores(
                pd.DataFrame(records), post_df, clusters_df, scored, score_cols,
                ['ntp_prob', 'probe_prob'], sec['confidence'], syn_docs)
            rows_df = ext_df

    shared_docs = set(gt_df['document_id']) & set(ext_df['document_id'])
    gt_df = gt_df[~gt_df['document_id'].isin(syn_docs)].reset_index(drop=True)
    assert not set(ext_df['document_id']) & syn_docs, 'extracted rows from probe-training documents'
    heldout_docs = set(gt_df['document_id']) | set(ext_df['document_id'])
    print(f"[meta] held out (non-training) docs: gt={gt_df['document_id'].nunique()}, "
          f"ext={ext_df['document_id'].nunique()}, shared={len(shared_docs - syn_docs)}")
    print(f"[meta] rows after held-out filter: gt={len(gt_df)}, ext={len(ext_df)}")

    gt_df['ecosystem_bucket'] = gt_df['ecosystem'].map(bucket_ecosystem)
    ext_df['ecosystem_bucket'] = ext_df['ecosystem'].map(bucket_ecosystem)

    # Every row's numeric value is its parsed point_value (see numeric_point_value). In the
    # ground truth point_value == value numerically (asserted), so only the extraction side
    # changes relative to reading `value`.
    gt_df['meta_value'] = numeric_point_value(gt_df['point_value'])
    gt_value = pd.to_numeric(gt_df['value'], errors='coerce')
    assert ((gt_df['meta_value'] == gt_value) | (gt_df['meta_value'].isna() & gt_value.isna())).all(), (
        'ground truth point_value and value disagree')
    ext_df['meta_value'] = numeric_point_value(ext_df['point_value'])
    gt_df = convert_units(gt_df, unit_conversion, value_col='meta_value')
    ext_df = convert_units(ext_df, unit_conversion, value_col='meta_value')

    manifest = dict(
        rows=sec['rows'], n_final_rows=len(final_df), n_rows_before_doc_filter=n_rows, confidence=sec['confidence'], n_rows_scored=len(rows_df),
        n_shared_docs=len(shared_docs), n_heldout_docs=len(heldout_docs),
        n_gt_docs=int(gt_df['document_id'].nunique()), n_ext_docs=int(ext_df['document_id'].nunique()),
        n_gt_rows=len(gt_df), n_ext_rows=len(ext_df),
        n_ext_list_children=int((ext_df['n_siblings'] > 1).sum()) if 'n_siblings' in ext_df else None,
        n_ext_list_children_unparseable=(int(((ext_df['n_siblings'] > 1) & ext_df['meta_value'].isna()).sum())
                                         if 'n_siblings' in ext_df else None),
        n_ext_unparseable_point_value=int(ext_df['meta_value'].isna().sum()),
        n_ext_unconvertible=int(ext_df['converted_value'].isna().sum()),
        input_sha256={repo_relative(p): sha256_file(p) for p in input_files},
        unit_conversion=unit_conversion,
    )
    return gt_df, ext_df, manifest


# ── Q-Q quantile helpers ────────────────────────────────────────────────────

def _valid_range(n: int, lo_cap: float = QLEVELS.min(), hi_cap: float = QLEVELS.max()) -> tuple[float, float]:
    """Probability range for which Hazen quantiles of an n-point sample are true
    interpolations rather than clamped to the sample min/max.

    Hazen plotting positions are (i-0.5)/n for i=1..n, so the smallest and largest
    representable probabilities are 0.5/n and 1-0.5/n; requesting a level outside that
    range makes np.interp silently clamp to the extreme observed value, which reads as
    a flat, artifactual tail rather than genuine distributional agreement/disagreement.
    """
    lo = max(lo_cap, 0.5 / n)
    hi = min(hi_cap, 1 - 0.5 / n)
    return lo, hi


def _attr_title(attribute: str) -> str:
    unit = STANDARD_UNITS[attribute]
    unit_str = f' ({unit})' if unit and unit != 'percent' else ''
    return attribute.replace('_', ' ') + unit_str


def _axis_limits(values: np.ndarray, log: bool) -> tuple[float, float]:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if log:
        pad = (vmax / vmin) ** 0.05 if vmax > vmin else 1.1
        return vmin / pad, vmax * pad
    pad = (vmax - vmin) * 0.05 if vmax > vmin else max(abs(vmax), 1.0) * 0.05
    return vmin - pad, vmax + pad
