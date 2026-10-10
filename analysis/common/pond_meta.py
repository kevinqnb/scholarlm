"""Pond meta-analysis data layer, shared by pond_meta_analysis.py and pond_clustering.py.

Defines the ecosystem/attribute universe, unit conversion and plausibility bounds,
and ``load_data``, which returns held-out ground-truth and extraction rows with
their stored NTP / probe confidences. Pond-specific throughout.
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

# Paper figure style (ACL Times metrics). Applied by each plotting entry point, not on import.
PAPER_RCPARAMS = {
    "font.family": "serif",
    # Times substitutes that match what LaTeX's `times` package renders on Linux.
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

# Outputs go to analysis/results/meta/<config id>/.
META_ROOT = analysis_results_dir("meta")

# ── Parameters ───────────────────────────────────────────────────────────────
DATASET = 'pond'

# Canonical cells. Configs pick a subset. RNG streams are keyed on positions in these
# lists so subsetting never changes a cell's CI -- never reorder them.
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

# standard_value = raw_value * UNIT_CONVERSION[attr][unit]. Unlisted units are dropped,
# so different measurands (µg/cm^2 chla, % dry wt tn/tp) never pass through.
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

# UNIT_CONVERSION plus pond schema units it lacks: mi^2, and ppm / ppb as mg/L / µg/L
# (dilute water). µg/cm^2 and molar chla stay unconvertible (likely mislabelled units).
# Used by pond_meta_analysis.py and pond_clustering.py.
_V2_ADDED_UNITS = {
    'surface_area': {'mi^2': 2589988.110336},
    'tn':   {'ppm': 1000.0, 'ppb': 1.0},
    'tp':   {'ppm': 1000.0, 'ppb': 1.0},
    'chla': {'ppm': 1000.0, 'ppb': 1.0},
}
assert all(u not in UNIT_CONVERSION[a] for a, m in _V2_ADDED_UNITS.items() for u in m), 'v2 addition already in v1'
UNIT_CONVERSION_V2 = {a: {**m, **_V2_ADDED_UNITS.get(a, {})} for a, m in UNIT_CONVERSION.items()}

# Plausibility bounds in standard units; values outside are dropped like unknown units.
# Set to "reasonably unlikely", not "physically impossible", to drop two recurring
# errors: cited reference lakes extracted as study systems (1e9-1e11 m^2), and mg/m^3
# mislabelled mg/L (1000x too large). Ceilings are the full-corpus ground-truth max
# (all documents, not the eval slice) times a safety margin.
PHYSICAL_BOUNDS = {
    'max_depth':        (0, 50),          # full-corpus GT max observed: 9 m
    'surface_area':     (0, 1e6),         # full-corpus GT max observed: 1.938e5 m^2
    'ph':                (0, 14),         # standard aqueous pH scale
    'vegetation_cover':  (0, 100),        # definition of a percentage
    'tn':                (0, 50_000),     # full-corpus GT max observed: 3.12e4 ug/L
    'tp':                (0, 15_000),     # full-corpus GT max observed: 9,850 ug/L (Lake Nakuru)
    'chla':              (0, 3_000),      # full-corpus GT max observed: 1,704 ug/L
}

# Attributes plotted on a log axis (they span orders of magnitude, or have outliers).
LOG_SCALE_ATTRIBUTES = {'surface_area', 'max_depth', 'tn', 'tp', 'chla'}

# Q-Q quantile levels, capped at [0.025, 0.975] so one extreme value can't stretch a panel.
QLEVELS = np.linspace(0.025, 0.975, 100)


# ── Ecosystem bucketing ─────────────────────────────────────────────────────

def bucket_ecosystem(raw: str | None) -> str:
    """Bucket a free-text ecosystem string into pond / lake / wetland / other.

    ``pool`` counts as pond. Strings matching zero or several buckets become 'other',
    which the analysis excludes.

    Args:
        raw: Ecosystem text, or None.

    Returns:
        ``'pond'``, ``'lake'``, ``'wetland'`` or ``'other'``.
    """
    if not raw:
        return 'other'
    s = str(raw).lower()
    hits = {b for b, kw in [('wetland', 'wetland'), ('pond', 'pond'), ('pond', 'pool'), ('lake', 'lake')]
            if kw in s}
    return hits.pop() if len(hits) == 1 else 'other'


# ── Unit conversion ─────────────────────────────────────────────────────────

def fix_fish_production_units(gt_df: pd.DataFrame, config) -> pd.DataFrame:
    """Repair corrupted ground-truth units for one paper (ground truth only, never extractions).

    The surface_area rows of 'fish_production_in_lakes' have a copy of ``value`` in
    ``units``; the real unit is read from the dataset's directory.json.

    Args:
        gt_df: Ground-truth rows.
        config: Pond DatasetConfig (for ``metadata_file``).

    Returns:
        Copy of ``gt_df`` with those units fixed.
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
    """Convert values to each attribute's standard unit, NaN where not convertible.

    Unlike scholarlm's apply_unit_conversion, an unknown unit gives NaN rather than
    passing through at factor 1. Negative results (often log-scale values with a
    physical unit attached) and values outside PHYSICAL_BOUNDS are also NaN. pH
    accepts any unit string.

    Args:
        df: Rows to convert.
        unit_conversion: Multiply-to-standard table with the same attributes as
            UNIT_CONVERSION.
        value_col: Numeric value column.
        unit_col: Unit column.
        attribute_col: Attribute column.
        out_col: Output column name.

    Returns:
        Copy of ``df`` with ``out_col`` added.
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
    """Read stored recalibrated NTP / probe confidences for the judged datapoints.

    Uses the pond-on-pond real cell of calibration.py's predictions.pkl and checks
    that both confidence models and the recalibration fit rows exclude held-out documents.

    Args:
        final_df: Judged final.json rows.
        combined_df: combined.json rows, aligned to ``final_df``.
        inputs: Output of ``resolve_meta_inputs``.

    Returns:
        Tuple of:
            - scored: measurement_id, document_id, attribute, judgement_combined,
              ntp_prob, probe_prob, label (judge OR match), one row per held-out datapoint
            - syn_docs: the probe's training documents
            - input files read (for the manifest)
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
    # The probe is the only trained confidence model (NTP is the judge's raw p(true)).
    syn_docs = set(probe_art['syn_document_ids'])

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
    """Load held-out ground-truth and extraction rows with scores and converted values.

    Held-out means outside the probe's training documents. The two sides need not
    cover the same documents.

    Args:
        sec: Config section; reads ``rows``, ``confidence``, ``deduplication_config_id``.
        inputs: Output of ``resolve_meta_inputs``.
        unit_conversion: Table passed to ``convert_units`` for both sides.

    Returns:
        Tuple of:
            - gt_df: ground truth with ``ecosystem_bucket``, ``meta_value``, ``converted_value``
            - ext_df: same columns plus judgement_combined, label, ntp_prob, probe_prob
            - manifest: row counts, input hashes and the unit table
    """
    config = load_dataset_config(DATASET)

    gt_df = load_ground_truth_file(inputs['ground_truth_path'])
    gt_df = fix_fish_production_units(gt_df, config)

    final_path = inputs['extraction_dir'] / 'final.json'
    combined_path = inputs['judge_combine_dir'] / 'combined.json'
    final_df = pd.DataFrame(json.loads(final_path.read_text()))
    combined_df = pd.DataFrame(json.loads(combined_path.read_text()))
    scored, syn_docs, scored_inputs = _load_stored_scores(final_df, combined_df, inputs)
    # Boolean label columns always take the cluster center's value, never a mean.
    score_cols = ['judgement_combined', 'label', 'ntp_prob', 'probe_prob']

    input_files = [inputs['ground_truth_path'], final_path, combined_path, *scored_inputs]
    if sec['rows'] == 'final':
        rows_df = final_df
        n_rows = len(rows_df)
        # Drop training-document rows first (they have no score) so every remaining row must score.
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

    # Numeric value = parsed point_value. In ground truth it must equal `value` (asserted).
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
    """Quantile-level range that Hazen positions of an n-point sample can interpolate.

    Outside [0.5/n, 1 - 0.5/n], np.interp clamps to the sample extreme, which draws a
    flat, artificial tail.

    Args:
        n: Sample size.
        lo_cap: Lowest level allowed.
        hi_cap: Highest level allowed.

    Returns:
        ``(lo, hi)`` quantile levels.
    """
    lo = max(lo_cap, 0.5 / n)
    hi = min(hi_cap, 1 - 0.5 / n)
    return lo, hi


# Figure text for attribute keys and unit strings that don't read correctly as-is.
ATTRIBUTE_DISPLAY = {'ph': 'pH'}
UNIT_DISPLAY = {'m^2': r'$m^2$'}


def _attr_title(attribute: str) -> str:
    """Panel title for an attribute, with its standard unit.

    Args:
        attribute: Attribute key.

    Returns:
        E.g. ``"max depth (m)"``; no unit for pH and percentages.
    """
    unit = STANDARD_UNITS[attribute]
    unit_str = f' ({UNIT_DISPLAY.get(unit, unit)})' if unit and unit != 'percent' else ''
    return ATTRIBUTE_DISPLAY.get(attribute, attribute.replace('_', ' ')) + unit_str


def _axis_limits(values: np.ndarray, log: bool) -> tuple[float, float]:
    """Axis limits padding the data range by 5% (multiplicatively on a log axis).

    Args:
        values: Plotted values.
        log: Whether the axis is log-scaled.

    Returns:
        ``(low, high)`` limits.
    """
    vmin, vmax = float(np.min(values)), float(np.max(values))
    if log:
        pad = (vmax / vmin) ** 0.05 if vmax > vmin else 1.1
        return vmin / pad, vmax * pad
    pad = (vmax - vmin) * 0.05 if vmax > vmin else max(abs(vmax), 1.0) * 0.05
    return vmin - pad, vmax + pad
