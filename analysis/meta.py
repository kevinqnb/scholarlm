"""Meta-analysis experiment: does an LLM-extracted dataset reproduce the same
per-ecosystem attribute distributions as the human-curated ground truth?

Seven parallel settings are compared for the pond dataset, broken down by
ecosystem class (pond / lake / wetland) and attribute:

    (a) ground truth
    (b) full extracted dataset, unweighted
    (c) extracted dataset filtered by judgement_combined
    (d) full extracted dataset, weighted by NTP (next-token-probability) confidence
    (e) full extracted dataset, weighted by trained-probe confidence
    (f) extracted dataset hard-thresholded at NTP confidence >= THRESHOLD, unweighted
    (g) extracted dataset hard-thresholded at probe confidence >= THRESHOLD, unweighted

(f)/(g) are a hard-gate variant of (d)/(e): rather than continuously weighting by
confidence, rows below THRESHOLD are dropped entirely and every surviving row gets
weight 1 -- i.e. plain statistics on the filtered subset. They still go through
weighted_stats() (with an all-ones weight vector) rather than a separate plain-stats
code path, so the quantile/whisker convention (Hazen plotting positions) is identical
across all seven settings and the comparison stays apples-to-apples.

All seven settings are restricted to the documents shared between ground truth
and extraction, minus the documents used to train the probe/NTP calibrator
(``syn_document_ids``), so that (d)/(e) are honest out-of-sample estimates.
See docs/plans or the commit history for the design rationale.
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
import matplotlib.patches as mpatches

from analysis.loaders import (
    load_extraction, load_combined_judgements, load_ground_truth,
    load_trained_probe, load_trained_ntp_calibrator, load_activations,
)
from experiments.run_extraction import load_dataset_config

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
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
SETTINGS = ['ground_truth', 'extracted', 'judge_filtered', 'ntp_weighted', 'probe_weighted',
            'ntp_threshold', 'probe_threshold']

# Hard-gate cutoff for the ntp_threshold / probe_threshold settings: rows with
# confidence below THRESHOLD are dropped, surviving rows get weight 1 (plain stats).
THRESHOLD = 0.75

SETTING_LABELS = {
    'ground_truth':    'Ground truth',
    'extracted':       'Extracted (unweighted)',
    'judge_filtered':  'Extracted (judge-filtered)',
    'ntp_weighted':    'Extracted (NTP-weighted)',
    'probe_weighted':  'Extracted (probe-weighted)',
    'ntp_threshold':   f'Extracted (NTP >= {THRESHOLD:g})',
    'probe_threshold': f'Extracted (probe >= {THRESHOLD:g})',
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
# Chosen independent of this dataset from real-world extremes, so they are citable
# and not tuned to make any particular result look better. The geometric attributes
# have hard, well-known "world record" ceilings; the concentration attributes (tn/tp/
# chla) have no comparable single citable record, so these are generous literature-
# informed ceilings (documented raw-wastewater / bloom-scum ranges, rounded well
# upward) rather than an authoritative maximum -- flagged as the softer of the two.
# Values outside these bounds are dropped (-> NaN), same as an unrecognized unit.
PHYSICAL_BOUNDS = {
    'max_depth':        (0, 1642),        # Lake Baikal, the deepest lake on Earth (m)
    'surface_area':     (0, 3.71e11),     # Caspian Sea, the largest lake on Earth (371,000 km^2 -> m^2)
    'ph':                (0, 14),         # standard aqueous pH scale
    'vegetation_cover':  (0, 100),        # definition of a percentage
    'tn':                (0, 100_000),    # generous ceiling above raw sewage TN (~20-85 mg/L)
    'tp':                (0, 50_000),     # generous ceiling above raw sewage TP (~4-30 mg/L)
    'chla':              (0, 10_000),     # generous ceiling above extreme bloom-scum chla
}

# Log-scale attributes span several orders of magnitude; the rest read fine on a linear axis.
# max_depth is log-scale too: extraction noise includes implausible outliers (e.g. a
# 108,000 m "depth" for a wetland treatment cell) that otherwise flatten the whole panel.
LOG_SCALE_ATTRIBUTES = {'surface_area', 'max_depth', 'tn', 'tp', 'chla'}

MIN_RELIABLE_N = 5  # cells below this n or n_eff, or with quantiles_clamped, are flagged in the figure


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
      denominator when w represents a continuous confidence/reliability (our NTP and
      probe probabilities), as opposed to integer frequency weights (denominator
      sum(w)-1) or the naive plug-in MLE (denominator sum(w), which understates
      variance). By Cauchy-Schwarz, sum(w)-sum(w^2)/sum(w) >= 0, with equality iff all
      weight sits on one point (n_eff == 1, see below) -- guarded explicitly rather
      than left to raise or silently divide by zero.

    - **n_eff**: Kish's effective sample size, sum(w)^2 / sum(w^2). Reports how many
      *unweighted* observations the weighted sample is statistically equivalent to --
      more informative than raw n when a cell's weight mass is concentrated on a
      handful of high-probability rows.

    - **Quartiles**: weighted-ECDF via Hazen plotting positions, p_i = (cumsum(w)_i -
      0.5*w_i) / sum(w), then linear interpolation to p=0.25/0.5/0.75. This is the
      weighted generalization of the classic Hazen (1914) plotting position (i-0.5)/n
      used for unweighted quantile-quantile / ECDF estimates, and reduces to it exactly
      when every weight is 1.

    - **quantiles_clamped**: True when the requested 0.25/0.75 probability falls outside
      the observed [p_min, p_max] range, so np.interp clamped Q1 or Q3 to the extreme
      x value instead of truly interpolating. Signals a cell whose weight mass is too
      concentrated near one tail for the quartiles to be trustworthy (typically small n
      or a few dominant weights) -- flag/suppress these in the figure.

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
    """Return (x, w) arrays of converted_value / weight for one setting, or None if empty."""
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
    if setting == 'ntp_weighted':
        return base['converted_value'].to_numpy(), base['ntp_prob'].to_numpy()
    if setting == 'probe_weighted':
        return base['converted_value'].to_numpy(), base['probe_prob'].to_numpy()
    if setting == 'ntp_threshold':
        sub = base[base['ntp_prob'] >= THRESHOLD]
        if len(sub) == 0:
            return None
        return sub['converted_value'].to_numpy(), np.ones(len(sub))
    if setting == 'probe_threshold':
        sub = base[base['probe_prob'] >= THRESHOLD]
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


# ── Visualization ─────────────────────────────────────────────────────────────

def pastel(color, mix=0.45):
    """Mix a color with white to get a pastel fill, mix in [0, 1] = fraction white."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - mix) * rgb + mix * np.array([1.0, 1.0, 1.0]))


def _cell_stats(stats_df: pd.DataFrame, ecosystem: str, attribute: str, setting: str) -> dict | None:
    row = stats_df[(stats_df.ecosystem == ecosystem) & (stats_df.attribute == attribute)
                    & (stats_df.setting == setting)]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return r.to_dict()


def plot_boxplots(
    stats_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    attributes: list[str],
    settings: list[str],
    out_path: Path,
    seed: int = 0,
):
    n_attrs = len(attributes)
    n_settings = len(settings)
    tab10 = plt.cm.tab10.colors
    setting_colors = {s: pastel(tab10[i % 10]) for i, s in enumerate(settings)}
    rng = np.random.default_rng(seed)

    offsets = np.linspace(-0.3, 0.3, n_settings) if n_settings > 1 else np.array([0.0])
    box_width = min(0.6 / n_settings, 0.18)

    fig, axes = plt.subplots(1, n_attrs, figsize=(3.4 * n_attrs, 4.0))
    if n_attrs == 1:
        axes = [axes]

    for ax, attribute in zip(axes, attributes):
        for eco_idx, ecosystem in enumerate(ECOSYSTEMS):
            for s_idx, setting in enumerate(settings):
                cell = _cell_stats(stats_df, ecosystem, attribute, setting)
                x_pos = eco_idx + offsets[s_idx]
                color = setting_colors[setting]
                if cell is None or cell['n'] == 0 or np.isnan(cell['median']):
                    continue

                unreliable = (cell['n'] < MIN_RELIABLE_N or cell['n_eff'] < MIN_RELIABLE_N
                              or cell['quantiles_clamped'])
                stats_dict = [{
                    'med': cell['median'], 'q1': cell['q1'], 'q3': cell['q3'],
                    'whislo': cell['whisker_lo'], 'whishi': cell['whisker_hi'],
                    'fliers': [],
                }]
                ax.bxp(
                    stats_dict, positions=[x_pos], widths=box_width,
                    patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=color, edgecolor='#444444', linewidth=0.9,
                                  hatch='///' if unreliable else None, alpha=0.55 if unreliable else 1.0),
                    medianprops=dict(color='#222222', linewidth=1.3),
                    whiskerprops=dict(color='#444444', linewidth=0.9),
                    capprops=dict(color='#444444', linewidth=0.9),
                    zorder=3,
                )

                # Faint scatter of every underlying point, jittered around the box's x
                # position; alpha scaled by each point's weight so unweighted settings
                # show uniform faint dots and (d)/(e) visually fade out low-confidence rows.
                data = _setting_data(setting, gt_df, ext_df, ecosystem, attribute)
                if data is not None:
                    xv, wv = data
                    jitter = rng.uniform(-box_width * 0.35, box_width * 0.35, size=len(xv))
                    w_norm = wv / wv.max() if wv.max() > 0 else np.ones_like(wv)
                    ax.scatter(x_pos + jitter, xv, s=6, color='#333333',
                               alpha=np.clip(0.12 * w_norm, 0.02, 0.12), linewidths=0, zorder=2)

        if attribute in LOG_SCALE_ATTRIBUTES:
            ax.set_yscale('log')
        ax.set_xticks(range(len(ECOSYSTEMS)))
        ax.set_xticklabels([e.capitalize() for e in ECOSYSTEMS])
        unit = STANDARD_UNITS[attribute]
        ylabel = attribute.replace('_', ' ') + (f' ({unit})' if unit and unit != 'percent' else '')
        ax.set_ylabel(ylabel)
        ax.set_title(attribute.replace('_', ' '), fontsize=13, style='italic')
        ax.grid(alpha=0.25, linestyle='-', linewidth=0.4, axis='y')
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta] wrote {out_path}")


def plot_legend(settings: list[str], out_path: Path):
    tab10 = plt.cm.tab10.colors
    setting_colors = {s: pastel(tab10[i % 10]) for i, s in enumerate(settings)}
    handles = [
        mpatches.Patch(facecolor=setting_colors[s], edgecolor='#444444', linewidth=0.9,
                        label=SETTING_LABELS[s])
        for s in settings
    ]
    fig, ax = plt.subplots(figsize=(1.6 * len(settings), 0.5))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(settings), fontsize=11, frameon=False)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta] wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attributes', nargs='+', default=['surface_area', 'max_depth', 'tn', 'tp'],
                         choices=ATTRIBUTES, help='Attribute subset for the box-plot figure.')
    parser.add_argument('--settings', nargs='+', default=SETTINGS,
                         choices=SETTINGS, help='Setting subset for the box-plot figure.')
    args = parser.parse_args()

    gt_df, ext_df = load_data()

    stats_df = build_stats_table(gt_df, ext_df)
    csv_path = RESULTS_DIR / f'meta_{DATASET}_{EXT_MODEL}_{EXT_DATE}.csv'
    stats_df.to_csv(csv_path, index=False)
    print(f"[meta] wrote {csv_path}")
    print(stats_df.to_string(index=False, float_format='{:.3g}'.format))

    plot_boxplots(stats_df, gt_df, ext_df, args.attributes, args.settings,
                  FIGURES_DIR / f"box_{'_'.join(args.attributes)}.pdf")
    plot_legend(args.settings, FIGURES_DIR / 'legend.pdf')


if __name__ == "__main__":
    main()
