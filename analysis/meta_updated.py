"""Meta-analysis experiment: does an LLM-extracted dataset reproduce the same
per-ecosystem attribute distributions as the human-curated ground truth?

Two families of Q-Q plots are produced for the pond dataset, broken down by
ecosystem class (pond / lake / wetland) and attribute:

    (a) ground truth vs. extracted data hard-filtered to NTP confidence >= t
    (b) ground truth vs. extracted data hard-filtered to probe confidence >= t

for a sequential sweep of thresholds t in THRESHOLDS = (0.0, 0.25, 0.5, 0.75).
Each Q-Q line plots ground-truth quantiles (x) against extracted quantiles (y)
at matched probability levels; a line that hugs the y=x diagonal means the
filtered extracted subset reproduces the ground-truth distribution at that
quantile. Filtering by confidence is a hard gate only -- no reliability
weighting is applied to the surviving rows, so every quantile in this file is
a plain (unweighted) Hazen-plotting-position quantile of the raw values.

A gray band around the diagonal shows the *ground truth's own* sampling
uncertainty at each quantile level, estimated by bootstrap resampling of the
ground-truth data alone. It is not a statement about the extracted lines'
uncertainty -- a threshold-0.75 line built from a handful of surviving rows
can be far noisier than the band suggests.

This produces 3 ecosystems x 2 methods (ntp/probe) = 6 Q-Q figures, each with
one subplot per attribute in a single row. The summary-statistics CSV/table
(ground truth, full extracted, judge-filtered, and each ntp_t/probe_t hard-cut
cell) is still written for the record; its median/Q1/Q3 columns are computed
on raw, non-log values -- log scaling in LOG_SCALE_ATTRIBUTES is applied only
when rendering the Q-Q axes, never before computing a statistic.

All settings are restricted to the documents shared between ground truth and
extraction, minus the documents used to train the probe/NTP calibrator
(``syn_document_ids``), so that the confidence-filtered settings are honest
out-of-sample estimates. See docs/plans or the commit history for the design
rationale.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from analysis.loaders import (
    load_extraction, load_combined_judgements, load_ground_truth,
    load_trained_probe, load_trained_ntp_calibrator, load_activations,
)
from experiments.run_extraction import load_dataset_config

mpl.rcParams.update({
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
})

FIGURES_DIR = REPO_ROOT / "figures" / "meta"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── Parameters ───────────────────────────────────────────────────────────────
DATASET = 'pond'
EXT_MODEL = 'gemma-3-27b'
EXT_DATE = '2026_05_05'
JUDGE_MODEL = 'qwen-2.5-7b'
JUDGE_DATE = '2026_05_06'
PROBE_TYPE = 'head'

ECOSYSTEMS = ['pond', 'lake', 'wetland']
ATTRIBUTES = ['surface_area', 'max_depth', 'vegetation_cover', 'ph', 'tn', 'tp', 'chla']

# Confidence-threshold sweep for the Q-Q comparisons: at each t, the extracted
# subset is hard-filtered to confidence >= t (no weighting), so as t increases
# the surviving subset shrinks and (hopefully) grows more faithful to the GT
# distribution. 0.0 keeps everything -- i.e. reduces to the full extracted set.
THRESHOLDS = [0.0, 0.25, 0.5, 0.75]
METHODS = ['ntp', 'probe']
METHOD_PROB_COL = {'ntp': 'ntp_prob', 'probe': 'probe_prob'}
METHOD_LABELS = {'ntp': 'NTP confidence', 'probe': 'Probe confidence'}

# Settings recorded in the stats CSV/table: ground truth, the two unfiltered
# reference settings, and every (method, threshold) hard-cut cell.
SETTINGS = (
    ['ground_truth', 'extracted', 'judge_filtered']
    + [f'{m}_{t:g}' for m in METHODS for t in THRESHOLDS]
)

SETTING_LABELS = {
    'ground_truth':   'Ground truth',
    'extracted':      'Extracted (unfiltered)',
    'judge_filtered': 'Extracted (judge-filtered)',
    **{f'{m}_{t:g}': f'Extracted ({METHOD_LABELS[m]} $\\geq${t:g})' for m in METHODS for t in THRESHOLDS},
}

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

MIN_RELIABLE_N = 5  # cells below this n have their stats-table row treated as unreliable
                     # (n recorded, but see plot_qq() for the Q-Q figures' own, per-curve
                     # quantile-validity handling, which is stricter and per-line rather
                     # than per-cell)

# Quantile probability grid for the Q-Q lines, capped to [0.025, 0.975] so a single
# extreme outlier in either tail can't stretch the panel.
QLEVELS = np.linspace(0.025, 0.975, 39)

# Bootstrap resamples for the ground-truth quantile uncertainty band.
N_BOOT = 1000

# Blue (threshold 0.0) -> red (threshold 0.75).
THRESHOLD_CMAP = plt.cm.coolwarm
THRESHOLD_NORM = mcolors.Normalize(vmin=min(THRESHOLDS), vmax=max(THRESHOLDS))

# Default attribute subset shown in the Q-Q figures (one subplot column each).
QQ_ATTRIBUTES = ['surface_area', 'ph', 'tn', 'tp']


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
    value_col: str = 'value',
    unit_col: str = 'units',
    attribute_col: str = 'attribute',
    out_col: str = 'converted_value',
) -> pd.DataFrame:
    """Convert values to the standard unit per attribute; unconvertible rows -> NaN.

    Unlike scholarlm.utils.unit_conversion.apply_unit_conversion, a unit that is not
    in UNIT_CONVERSION[attribute] yields NaN (dropped), not a factor-of-1.0 passthrough
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
    df = df.copy()
    numeric_values = pd.to_numeric(df[value_col], errors='coerce')

    factors = pd.Series(np.nan, index=df.index)
    for attribute, unit_map in UNIT_CONVERSION.items():
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

def load_data():
    """Load GT + extraction data, restrict to the shared held-out document set,
    and attach judgement_combined / ntp_probs / probe_probs to ext_df.
    """
    config = load_dataset_config(DATASET)

    gt_df = load_ground_truth(config)
    gt_df = fix_fish_production_units(gt_df, config)

    ext_records = load_extraction(DATASET, EXT_MODEL, EXT_DATE)
    judged_records = load_combined_judgements(DATASET, EXT_MODEL, EXT_DATE)
    ext_df = pd.DataFrame(ext_records)
    judged_df = pd.DataFrame(judged_records)
    ext_df['judgement_combined'] = judged_df['judgement_combined'].to_numpy()
    ext_df[f'judgement_p_true_{JUDGE_MODEL}'] = judged_df[f'judgement_p_true_{JUDGE_MODEL}'].to_numpy()

    pd_data = load_trained_probe(DATASET, JUDGE_MODEL, ptype=PROBE_TYPE)
    ntp_cal_data = load_trained_ntp_calibrator(DATASET, JUDGE_MODEL)
    syn_docs = set(pd_data['syn_document_ids'])

    shared_docs = set(gt_df['document_id']) & set(ext_df['document_id'])
    heldout_docs = shared_docs - syn_docs
    print(f"[meta] shared GT/extraction docs: {len(shared_docs)}, "
          f"held out (non-training): {len(heldout_docs)}")

    gt_df = gt_df[gt_df['document_id'].isin(heldout_docs)].reset_index(drop=True)
    ext_df = ext_df[ext_df['document_id'].isin(heldout_docs)].reset_index(drop=True)
    print(f"[meta] rows after held-out filter: gt={len(gt_df)}, ext={len(ext_df)}")

    raw_ntp = ext_df[f'judgement_p_true_{JUDGE_MODEL}'].to_numpy()
    ntp_probs = ntp_cal_data['calibrator'].predict_proba(raw_ntp.reshape(-1, 1))[:, 1]

    top = pd_data['top_k_heads'] if PROBE_TYPE == 'head' else [pd_data['top_layer']]
    act = load_activations(DATASET, EXT_MODEL, EXT_DATE, JUDGE_MODEL, JUDGE_DATE)
    mids = ext_df['measurement_id'].tolist()
    if PROBE_TYPE == 'head':
        X = np.stack([
            np.concatenate([np.asarray(act[str(m)], dtype=np.float32)[l, h, :] for l, h in top])
            for m in mids
        ])
    else:
        layer = pd_data['top_layer']
        X = np.stack([np.asarray(act[str(m)], dtype=np.float32)[layer] for m in mids])
    probe_probs = pd_data['probe'].predict_proba(X)[:, 1]

    ext_df['ntp_prob'] = ntp_probs
    ext_df['probe_prob'] = probe_probs

    gt_df['ecosystem_bucket'] = gt_df['ecosystem'].map(bucket_ecosystem)
    ext_df['ecosystem_bucket'] = ext_df['ecosystem'].map(bucket_ecosystem)

    gt_df = convert_units(gt_df)
    ext_df = convert_units(ext_df)

    return gt_df, ext_df


# ── Weighted statistics ──────────────────────────────────────────────────────

def weighted_stats(x, w):
    """Weighted mean/std/quartiles for one (ecosystem, attribute, setting) cell.

    Methods, and why each is the right tool here:

    - **Mean**: standard weighted mean, sum(w*x)/sum(w).

    - **Std**: "reliability weights" (a.k.a. importance/probability weights) variance
      estimator, sum(w*(x-mean)^2) / (sum(w) - sum(w^2)/sum(w)). This is the correct
      denominator when w represents a continuous confidence/reliability, as opposed to
      integer frequency weights (denominator sum(w)-1) or the naive plug-in MLE
      (denominator sum(w), which understates variance). By Cauchy-Schwarz,
      sum(w)-sum(w^2)/sum(w) >= 0, with equality iff all weight sits on one point
      (n_eff == 1, see below) -- guarded explicitly rather than left to raise or
      silently divide by zero.

    - **n_eff**: Kish's effective sample size, sum(w)^2 / sum(w^2). Reports how many
      *unweighted* observations the weighted sample is statistically equivalent to.
      All callers in this module pass w = ones (see module docstring: thresholds are a
      hard gate on the subset, never a continuous weight), so n_eff == n everywhere in
      practice; the machinery is kept general rather than special-cased.

    - **Quartiles**: weighted-ECDF via Hazen plotting positions, p_i = (cumsum(w)_i -
      0.5*w_i) / sum(w), then linear interpolation to p=0.25/0.5/0.75. This is the
      weighted generalization of the classic Hazen (1914) plotting position (i-0.5)/n
      used for unweighted quantile-quantile / ECDF estimates, and reduces to it exactly
      when every weight is 1 -- i.e. always, in this module.

    - **quantiles_clamped**: True when the requested 0.25/0.75 probability falls outside
      the observed [p_min, p_max] range, so np.interp clamped Q1 or Q3 to the extreme
      x value instead of truly interpolating. Signals a cell whose n is too small for
      the quartiles to be trustworthy -- flag/suppress these in the table.

    - **Whiskers/outliers**: standard Tukey 1.5*IQR fence built from the weighted
      quartiles above, applied only to the positive-weight observations that actually
      contribute to the distribution.

    x, w: 1-D arrays, same length. Rows with w == 0 are dropped before any statistic is
    computed -- they contribute nothing to the weighted distribution and must not be
    allowed to stretch the whiskers or register as outliers.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != w.shape or x.ndim != 1:
        raise ValueError("x and w must be 1-D and same length")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("w must be finite and non-negative")

    keep = w > 0
    x, w = x[keep], w[keep]
    n = x.size
    if n == 0:
        raise ValueError("no positive-weight observations")

    sw = w.sum()
    v2 = np.sum(w ** 2)
    n_eff = sw ** 2 / v2
    mean = np.sum(w * x) / sw
    denom = sw - v2 / sw
    std = np.sqrt(np.sum(w * (x - mean) ** 2) / denom) if denom > 0 else np.nan

    order = np.argsort(x)
    xs, ws = x[order], w[order]
    p = (np.cumsum(ws) - 0.5 * ws) / sw
    clamped = p[0] > 0.25 or p[-1] < 0.75   # tail saturation: Q1/Q3 got clamped, not interpolated
    q1, med, q3 = np.interp([0.25, 0.5, 0.75], p, xs)

    iqr = q3 - q1
    lo_f, hi_f = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    inside = (xs >= lo_f) & (xs <= hi_f)

    return dict(n=n, n_eff=n_eff, sum_w=sw, mean=mean, std=std,
                q1=q1, median=med, q3=q3,
                whisker_lo=xs[inside].min(), whisker_hi=xs[inside].max(),
                n_outliers=int((~inside).sum()), w_outliers=ws[~inside].sum(),
                quantiles_clamped=clamped, outlier_values=xs[~inside])


# ── Per-cell aggregation ─────────────────────────────────────────────────────

def _setting_data(setting: str, gt_df: pd.DataFrame, ext_df: pd.DataFrame, ecosystem: str, attribute: str):
    """Return (x, w) arrays of converted_value / weight for one setting, or None if empty.

    w is always an all-ones array -- every setting here is a hard filter on which rows
    are included, never a continuous reliability weight (see module docstring).
    """
    if setting == 'ground_truth':
        sub = gt_df[(gt_df['ecosystem_bucket'] == ecosystem) & (gt_df['attribute'] == attribute)]
        sub = sub.dropna(subset=['converted_value'])
        if len(sub) == 0:
            return None
        return sub['converted_value'].to_numpy(), np.ones(len(sub))

    base = ext_df[(ext_df['ecosystem_bucket'] == ecosystem) & (ext_df['attribute'] == attribute)]
    base = base.dropna(subset=['converted_value'])
    if len(base) == 0:
        return None

    if setting == 'extracted':
        return base['converted_value'].to_numpy(), np.ones(len(base))
    if setting == 'judge_filtered':
        sub = base[base['judgement_combined'].astype(bool)]
        if len(sub) == 0:
            return None
        return sub['converted_value'].to_numpy(), np.ones(len(sub))

    for method in METHODS:
        prefix = f'{method}_'
        if setting.startswith(prefix):
            threshold = float(setting[len(prefix):])
            prob_col = METHOD_PROB_COL[method]
            sub = base.dropna(subset=[prob_col])
            sub = sub[sub[prob_col] >= threshold]
            if len(sub) == 0:
                return None
            return sub['converted_value'].to_numpy(), np.ones(len(sub))

    raise ValueError(f"Unknown setting: {setting}")


def build_stats_table(gt_df: pd.DataFrame, ext_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ecosystem in ECOSYSTEMS:
        for attribute in ATTRIBUTES:
            for setting in SETTINGS:
                data = _setting_data(setting, gt_df, ext_df, ecosystem, attribute)
                row = dict(dataset=DATASET, ecosystem=ecosystem, attribute=attribute,
                           setting=setting, unit=STANDARD_UNITS[attribute])
                if data is None:
                    row.update(n=0, n_eff=0.0, sum_w=0.0, mean=np.nan, std=np.nan,
                               q1=np.nan, median=np.nan, q3=np.nan,
                               whisker_lo=np.nan, whisker_hi=np.nan,
                               n_outliers=0, w_outliers=0.0, quantiles_clamped=False)
                else:
                    x, w = data
                    try:
                        stats = weighted_stats(x, w)
                    except ValueError:
                        stats = dict(n=0, n_eff=0.0, sum_w=0.0, mean=np.nan, std=np.nan,
                                     q1=np.nan, median=np.nan, q3=np.nan,
                                     whisker_lo=np.nan, whisker_hi=np.nan,
                                     n_outliers=0, w_outliers=0.0, quantiles_clamped=False)
                    stats.pop('outlier_values', None)
                    row.update(stats)
                rows.append(row)
    return pd.DataFrame(rows)


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


def _hazen_quantiles(x: np.ndarray, levels: np.ndarray) -> np.ndarray:
    return np.quantile(x, levels, method='hazen')


def _bootstrap_gt_band(x: np.ndarray, levels: np.ndarray, n_boot: int, ci: float, seed: int):
    """2.5/97.5th percentile band (or `ci`-equivalent) of the bootstrap distribution
    of the ground-truth Hazen quantiles at `levels`. Represents sampling noise in the
    ground-truth estimate alone -- see module docstring.
    """
    rng = np.random.default_rng(seed)
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_samples = x[idx]
    boot_q = np.quantile(boot_samples, levels, method='hazen', axis=1).T  # (n_boot, len(levels))
    alpha = (1 - ci) / 2
    lo = np.quantile(boot_q, alpha, axis=0)
    hi = np.quantile(boot_q, 1 - alpha, axis=0)
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


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_qq(
    gt_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    ecosystem: str,
    method: str,
    attributes: list[str],
    out_path: Path,
    n_boot: int = N_BOOT,
    ci: float = 0.95,
    seed: int = 0,
):
    """One Q-Q figure for a fixed (ecosystem, method): one subplot per attribute,
    laid out in a single row. Each subplot overlays, for every threshold t in
    THRESHOLDS, a line of (GT quantile, extracted-quantile-at-confidence>=t) pairs
    at matched probability levels -- i.e. a standard Q-Q plot, with the reference
    distribution's own sampling noise shown as a shaded band around the y=x diagonal.
    """
    n_attrs = len(attributes)
    fig, axes = plt.subplots(1, n_attrs, figsize=(2.6 * n_attrs, 2.9))
    if n_attrs == 1:
        axes = [axes]

    prob_col = METHOD_PROB_COL[method]

    for i, (ax, attribute) in enumerate(zip(axes, attributes)):
        log_scale = attribute in LOG_SCALE_ATTRIBUTES

        gt_data = _setting_data('ground_truth', gt_df, ext_df, ecosystem, attribute)
        gt_x = gt_data[0] if gt_data is not None else np.array([])
        if log_scale:
            gt_x = gt_x[gt_x > 0]
        if gt_x.size < 2:
            ax.text(0.5, 0.5, f'insufficient GT data\n(n={gt_x.size})',
                     ha='center', va='center', fontsize=9, color='#888888',
                     transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel('GT')
            if i == 0:
                ax.set_ylabel('Extracted')
            ax.set_title(_attr_title(attribute), fontsize=13, style='italic')
            continue

        gt_lo, gt_hi = _valid_range(gt_x.size)
        gt_levels = QLEVELS[(QLEVELS >= gt_lo) & (QLEVELS <= gt_hi)]
        gt_q = _hazen_quantiles(gt_x, gt_levels)
        boot_lo, boot_hi = _bootstrap_gt_band(gt_x, gt_levels, n_boot=n_boot, ci=ci, seed=seed)

        all_plotted = [gt_q, boot_lo, boot_hi]

        for t in THRESHOLDS:
            data = _setting_data(f'{method}_{t:g}', gt_df, ext_df, ecosystem, attribute)
            if data is None:
                continue
            ext_x = data[0]
            if log_scale:
                ext_x = ext_x[ext_x > 0]
            if ext_x.size == 0:
                continue

            ext_lo, ext_hi = _valid_range(ext_x.size)
            lo, hi = max(gt_lo, ext_lo), min(gt_hi, ext_hi)
            levels_t = QLEVELS[(QLEVELS >= lo) & (QLEVELS <= hi)]
            if levels_t.size == 0:
                continue

            gt_q_t = _hazen_quantiles(gt_x, levels_t)
            ext_q_t = _hazen_quantiles(ext_x, levels_t)
            color = THRESHOLD_CMAP(THRESHOLD_NORM(t))
            if levels_t.size == 1:
                ax.scatter(gt_q_t, ext_q_t, color=color, s=14, zorder=4)
            else:
                ax.plot(gt_q_t, ext_q_t, color=color, linewidth=1.8, zorder=4, solid_capstyle='round')
            all_plotted.extend([gt_q_t, ext_q_t])

        lo_lim, hi_lim = _axis_limits(np.concatenate(all_plotted), log=log_scale)
        ax.fill_between(gt_q, boot_lo, boot_hi, color='#888888', alpha=0.25, linewidth=0, zorder=1)
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color='#888888', linewidth=1.0,
                 linestyle='--', zorder=2)

        if log_scale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlim(lo_lim, hi_lim)
        ax.set_ylim(lo_lim, hi_lim)
        ax.set_box_aspect(1)

        ax.set_xlabel('GT')
        if i == 0:
            ax.set_ylabel('Extracted')
        ax.set_title(_attr_title(attribute), fontsize=13, style='italic')
        ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta] wrote {out_path}")


def plot_qq_legend(out_path: Path):
    """Shared legend for all six Q-Q figures: one line per threshold, plus swatches
    for the y=x reference and the ground-truth bootstrap uncertainty band.
    """
    handles = [
        mlines.Line2D([], [], color=THRESHOLD_CMAP(THRESHOLD_NORM(t)), linewidth=1.8,
                       label=f'confidence $\\geq${t:g}')
        for t in THRESHOLDS
    ]
    handles.append(mlines.Line2D([], [], color='#888888', linewidth=1.0, linestyle='--',
                                   label='y = x (perfect match)'))
    handles.append(mpatches.Patch(facecolor='#888888', alpha=0.25, edgecolor='none',
                                    label='GT bootstrap 95% CI'))

    fig, ax = plt.subplots(figsize=(1.9 * len(handles), 0.5))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(handles), fontsize=11, frameon=False)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta] wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attributes', nargs='+', default=QQ_ATTRIBUTES,
                         choices=ATTRIBUTES, help='Attribute subset (one subplot column each) for the Q-Q figures.')
    parser.add_argument('--n-boot', type=int, default=N_BOOT,
                         help='Bootstrap resamples for the ground-truth quantile uncertainty band.')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    gt_df, ext_df = load_data()

    stats_df = build_stats_table(gt_df, ext_df)
    csv_path = RESULTS_DIR / f'meta_{DATASET}_{EXT_MODEL}_{EXT_DATE}.csv'
    stats_df.to_csv(csv_path, index=False)
    print(f"[meta] wrote {csv_path}")
    print(stats_df.to_string(index=False, float_format='{:.3g}'.format))

    for method in METHODS:
        for ecosystem in ECOSYSTEMS:
            out_path = FIGURES_DIR / f'qq_{method}_{ecosystem}.pdf'
            plot_qq(gt_df, ext_df, ecosystem, method, args.attributes, out_path,
                    n_boot=args.n_boot, seed=args.seed)

    plot_qq_legend(FIGURES_DIR / 'qq_legend.pdf')


if __name__ == "__main__":
    main()
