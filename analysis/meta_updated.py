"""Meta-analysis experiment: does an LLM-extracted dataset reproduce the same
per-ecosystem attribute distributions as the human-curated ground truth?

Extracted-side statistics are weighted by NTP/probe confidence over ALL
held-out rows -- never a hard threshold that drops rows. Quantiles (median/
Q1/Q3, every Q-Q line) use weighted-Hazen plotting positions
(weighted_hazen_quantile): p_i = (cumsum(w)_i - 0.5*w_i)/sum(w) over x sorted
ascending, then linear interpolation. This reduces exactly to
np.quantile(x, q, method='hazen') at w=1, which is why the ground-truth side
(_hazen_quantiles/_bootstrap_gt_band) stays unweighted and unchanged -- same
estimator family, no mismatch between the two axes of a Q-Q line.

The Q-Q sweep (plot_qq_smooth) draws one line per temperature gamma in GAMMAS
(1.0 -> GAMMA_FLOOR), weighting each row by confidence(x)^(1/gamma). gamma=1.0
is the plain weighted distribution; as gamma drops toward the floor, weight
concentrates on high-confidence rows -- a "soft" filter that never drops a
row outright. gamma=0 is undefined, hence the positive floor.

Because every gamma sees every row, raw n can't signal reliability the way a
hard threshold's shrinking n did. Kish's (1965) effective sample size,
n_eff = (sum w)^2/sum(w^2), does that job via two gates: a line is dropped
once n_eff < MIN_RELIABLE_N; within a surviving line, a quantile level q is
dropped if n_eff < 1/min(q, 1-q) (kish_gate_levels) -- tails need more
support than the median. weighted_valid_range is a separate, mechanical
guard: it keeps np.interp from clamping into a probability range no
observation reaches.

The gray band around the diagonal is the ground truth's own bootstrap
sampling uncertainty, not a statement about the extracted lines.

Outputs: 6 Q-Q figures (3 ecosystems x 2 methods) plus one poster figure, a
stats CSV (ground_truth, extracted, judge_filtered, ntp_weighted,
probe_weighted) via weighted_stats() -- weighted mean/std/n_eff/Hazen
median/Q1/Q3, computed on raw non-log values -- and a 2-Wasserstein CSV
(build_wasserstein_table) giving the quantile-approximated W_2 distance from
each extracted setting's per-(ecosystem, attribute) distribution to ground
truth, with a two-sample percentile bootstrap CI. Every setting is restricted to
documents shared between GT and extraction, minus the probe/NTP calibrator's
training documents, so the weighted settings are honest out-of-sample
estimates. See notes/scholarlm/builds/2026-08-21-weighted-hazen-meta-01.md
for the design rationale.
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
JUDGE_DATE = '2026_05_06'          # default; --judge-date overrides (interp judge run
                                  # supplying the real-extraction activations)
PROBE_TYPE = 'head'
PROBE_SOURCE = None               # default (baseline trained_probe/); --probe-source
                                  # selects a parallel synthetic_probe_<source>/ tree
                                  # (e.g. 'v2') for the probe + NTP calibrator.

ECOSYSTEMS = ['pond', 'lake', 'wetland']
ATTRIBUTES = ['surface_area', 'max_depth', 'vegetation_cover', 'ph', 'tn', 'tp', 'chla']

METHODS = ['ntp', 'probe']
METHOD_PROB_COL = {'ntp': 'ntp_prob', 'probe': 'probe_prob'}
METHOD_LABELS = {'ntp': 'NTP confidence', 'probe': 'Probe confidence'}

# Settings recorded in the stats CSV/table: ground truth, the two unfiltered
# reference settings, and each method's fully-weighted setting -- ALL held-out rows,
# weighted by that method's confidence, no hard cutoff (see module docstring).
SETTINGS = (
    ['ground_truth', 'extracted', 'judge_filtered']
    + [f'{m}_weighted' for m in METHODS]
)

SETTING_LABELS = {
    'ground_truth':   'Ground truth',
    'extracted':      'Extracted (unfiltered)',
    'judge_filtered': 'Extracted (judge-filtered)',
    **{f'{m}_weighted': f'Extracted ({METHOD_LABELS[m]}-weighted)' for m in METHODS},
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

# A gamma line is dropped once its Kish n_eff (module docstring) falls below this
# value. Raw n no longer shrinks with gamma (every row stays, just reweighted), so
# reliability has to be tracked via n_eff instead. Complementary to
# weighted_valid_range (narrows the line's probability window, never drops it
# outright) and kish_gate_levels (the per-level version of this gate). Ground truth
# is exempt -- see the `gt_x.size < 2` checks in plot_qq_smooth/plot_qq_poster_smooth.
MIN_RELIABLE_N = 5

# Quantile probability grid for the Q-Q lines, capped to [0.025, 0.975] so a single
# extreme outlier in either tail can't stretch the panel.
QLEVELS = np.linspace(0.025, 0.975, 100)

# Bootstrap resamples for the ground-truth quantile uncertainty band.
N_BOOT = 2000

# ── 2-Wasserstein quantile grid ─────────────────────────────────────────────
# W_2(F_ext, F_gt) = ( int_0^1 (F_ext^{-1}(u) - F_gt^{-1}(u))^2 du )^{1/2}. The
# distributions are not assumed normal, so the integral is approximated straight
# from quantile differences (midpoint rule on W2_QGRID), never a closed-form
# Gaussian expression. Domain is the fixed trim [W2_QLO, W2_QHI] -- the same as
# QLEVELS' cap, so the score summarizes the Q-Q panel beside it, and identical for
# every cell so W_2 values are comparable across cells (an intersect-per-cell
# domain would not be). GT needs 0.5/n <= W2_QLO for its Hazen plotting positions
# to span the grid without np.interp clamping (n >= 20 at this trim); widen the
# trim here to relax that.
W2_QLO, W2_QHI = 0.025, 0.975
W2_NGRID = 999
W2_QGRID = W2_QLO + (np.arange(W2_NGRID) + 0.5) * (W2_QHI - W2_QLO) / W2_NGRID
W2_MIN_GT_N = int(np.ceil(0.5 / W2_QLO))

# Bootstrap replicates for the W_2 CI reuse N_BOOT. A replicate that trips
# wasserstein2_quantile's skip guards is a genuine wide-tail draw, not noise to
# discard -- dropping it biases the CI down (see _bootstrap_w2_ci). The CI is
# emitted only when at least this fraction of replicates survived; below it the
# interval was never meaningful and the cell gets a NaN CI instead.
W2_BOOT_MIN_OK_FRAC = 0.99

# Temperature grid for the Q-Q weight sweep: each extracted row is weighted by
# confidence(x)^(1/gamma) (see module docstring). gamma=1.0 (required -- recovers the
# plain weighted distribution) down to a small positive floor in steps of 0.05, since
# gamma=0 is undefined (1/gamma).
GAMMA_FLOOR = 0.20
GAMMAS = np.round(np.arange(1.0, GAMMA_FLOOR - 0.01, -0.05), 2)

# Diverging red-blue "cool-warm" scale: RdBu's native direction puts red at its low
# end and blue at its high end, so normalizing directly on gamma gives
# gamma=GAMMA_FLOOR (heaviest filtering) red and gamma=1.0 (no reweighting) blue.
GAMMA_CMAP = plt.cm.RdBu
QQ_GAMMA_NORM = mcolors.Normalize(vmin=GAMMA_FLOOR, vmax=1.0)


def gamma_color(gamma: float):
    """Color for one GAMMAS line -- see QQ_GAMMA_NORM's docstring comment above."""
    return GAMMA_CMAP(QQ_GAMMA_NORM(gamma))

# Default attribute subset shown in the Q-Q figures (one subplot column each).
QQ_ATTRIBUTES = ['surface_area', 'ph', 'tn', 'tp']

# Single (ecosystem, method, attribute) cell blown up poster-size, axes flipped
# (GT on y, Extracted on x) and fonts enlarged relative to the QQ_ATTRIBUTES grid --
# matches the one-off qq_probe_pond_tn_poster.pdf committed in cdd9f39, whose
# generating code was never itself committed (see build note
# 2026-08-18-smoothed-qq-sweep-01 for the reconstruction).
POSTER_ECOSYSTEM = 'pond'
POSTER_METHOD = 'probe'
POSTER_ATTRIBUTE = 'tn'

# Fixed y-axis (Ground Truth) crop for the poster figure only, still log-scale --
# tighter than the auto-computed range so the sweep isn't dominated by the sparse,
# noisy high-threshold tail. Does not affect the x-axis (Extracted), which stays
# auto-ranged, or any other Q-Q figure.
POSTER_YLIM = (100, 5000)


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

    pd_data = load_trained_probe(DATASET, JUDGE_MODEL, ptype=PROBE_TYPE, source=PROBE_SOURCE)
    ntp_cal_data = load_trained_ntp_calibrator(DATASET, JUDGE_MODEL, source=PROBE_SOURCE)
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
# See the module docstring for design rationale and tests/test_meta_weighted_stats.py
# for the correctness checks (w=1 exactness, hand-computed fixtures) these functions
# are validated against.

def weighted_hazen_quantile(x: np.ndarray, w: np.ndarray, q) -> np.ndarray:
    """Weighted Hazen plotting-position quantile: p_i = (cumsum(w)_i - 0.5*w_i) /
    sum(w) over x sorted ascending, then linear interpolation to q. Reduces exactly
    to np.quantile(x, q, method='hazen') when every weight is 1.

    Rows with w == 0 are dropped. q outside the representable range (see
    weighted_valid_range) is clamped by np.interp rather than raising.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != w.shape or x.ndim != 1:
        raise ValueError("x and w must be 1-D and same length")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("w must be finite and non-negative")
    keep = w > 0
    x, w = x[keep], w[keep]
    if x.size == 0:
        raise ValueError("no positive-weight observations")
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    sw = ws.sum()
    p = (np.cumsum(ws) - 0.5 * ws) / sw
    return np.interp(q, p, xs)


def weighted_valid_range(
    x: np.ndarray, w: np.ndarray,
    lo_cap: float = QLEVELS.min(), hi_cap: float = QLEVELS.max(),
) -> tuple[float, float]:
    """Weight-aware _valid_range: the probability window [p_1, p_n] where
    weighted_hazen_quantile truly interpolates rather than clamping.

    Keyed on the *specific* weights of the smallest-x/largest-x rows (not a global
    min), since Hazen positions are monotone in sorted-x order:
        p_min = 0.5 * w_1 / sum(w);  p_max = 1 - 0.5 * w_n / sum(w)
    A large weight on an extreme-x row widens the excluded region around it.
    """
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.shape != w.shape or x.ndim != 1:
        raise ValueError("x and w must be 1-D and same length")
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("w must be finite and non-negative")
    keep = w > 0
    x, w = x[keep], w[keep]
    if x.size == 0:
        raise ValueError("no positive-weight observations")
    order = np.argsort(x)
    ws = w[order]
    sw = ws.sum()
    lo = max(lo_cap, 0.5 * ws[0] / sw)
    hi = min(hi_cap, 1 - 0.5 * ws[-1] / sw)
    return lo, hi


def kish_n_eff(w: np.ndarray) -> float:
    """Kish's (1965) effective sample size: n_eff = (sum w)^2 / sum(w^2) -- how
    many unweighted observations the weighted sample is statistically equivalent
    to. Rows with w == 0 are dropped, so an all-zero input raises rather than
    silently returning 0/0 -> NaN.
    """
    w = np.asarray(w, dtype=float)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("w must be finite and non-negative")
    w = w[w > 0]
    if w.size == 0:
        raise ValueError("no positive-weight observations")
    sw = w.sum()
    return float(sw ** 2 / np.sum(w ** 2))


def kish_gate_levels(levels: np.ndarray, n_eff: float) -> np.ndarray:
    """Drop levels where n_eff < 1/min(q, 1-q) -- e.g. q=0.5 needs n_eff >= 2,
    q=0.025 needs n_eff >= 40. Complements weighted_valid_range: that guards
    mechanical interpolability, this guards statistical support within it.
    """
    levels = np.asarray(levels, dtype=float)
    required = 1.0 / np.minimum(levels, 1.0 - levels)
    return levels[n_eff >= required]


def weighted_stats(x, w) -> dict:
    """Weighted mean/std/median/Q1/Q3 + Kish n_eff for one (ecosystem, attribute,
    setting) cell.

    Std uses the reliability-weights estimator, sum(w*(x-mean)^2) / (sum(w) -
    sum(w^2)/sum(w)) -- correct for continuous confidence weights, unlike integer
    frequency weights (denominator sum(w)-1) or the naive sum(w) denominator (which
    understates variance). The denominator hits 0 iff n_eff == 1 (all weight on one
    point); guarded to NaN rather than raising or dividing by zero.

    Rows with w == 0 are dropped; unweighted settings just pass w=1.
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
    q1, med, q3 = weighted_hazen_quantile(x, w, [0.25, 0.5, 0.75])

    return dict(n=n, n_eff=n_eff, sum_w=sw, mean=mean, std=std, q1=q1, median=med, q3=q3)


# ── Per-cell aggregation ─────────────────────────────────────────────────────

def _setting_data(
    setting: str, gt_df: pd.DataFrame, ext_df: pd.DataFrame, ecosystem: str, attribute: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (x, w) arrays of converted_value / weight for one setting, or None if
    empty. ground_truth/extracted/judge_filtered are unit-weight (w=1 for every
    surviving row); ntp_weighted/probe_weighted are ALL held-out extracted rows
    weighted by that method's confidence, with no hard cutoff (see module docstring).
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
        if setting == f'{method}_weighted':
            prob_col = METHOD_PROB_COL[method]
            sub = base.dropna(subset=[prob_col])
            if len(sub) == 0:
                return None
            return sub['converted_value'].to_numpy(), sub[prob_col].to_numpy()

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
                               q1=np.nan, median=np.nan, q3=np.nan)
                else:
                    x, w = data
                    try:
                        stats = weighted_stats(x, w)
                    except ValueError:
                        # Every row has zero weight for this cell -- legitimate "no
                        # usable data", same bucket as data is None above.
                        stats = dict(n=0, n_eff=0.0, sum_w=0.0, mean=np.nan, std=np.nan,
                                     q1=np.nan, median=np.nan, q3=np.nan)
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


# ── 2-Wasserstein distance (quantile approximation) ─────────────────────────
# See W2_QGRID's block comment above for the estimator and the domain choice.
# Both sides use the same Hazen estimator as the Q-Q figures:
# weighted_hazen_quantile for the confidence-weighted extracted side (reduces
# exactly to np.quantile(method='hazen') at unit weight), _hazen_quantiles for
# the unweighted ground-truth side -- same estimator family on both axes, no
# mismatch. The score is in the raw value units, and additionally in log10 units
# for LOG_SCALE_ATTRIBUTES so the table reads across attributes that span orders
# of magnitude (a raw W_2 in m^2 and one in pH units are not comparable).

W2_SETTINGS = ['extracted', 'judge_filtered'] + [f'{m}_weighted' for m in METHODS]


def wasserstein2_quantile(gt_x: np.ndarray, ext_x: np.ndarray, ext_w: np.ndarray) -> dict:
    """Quantile-approximated 2-Wasserstein distance from the confidence-weighted
    extracted sample (ext_x, ext_w) to the unweighted ground-truth sample (gt_x),
    in the units of the inputs.

    W_2 ~= sqrt( mean_i (q_ext(u_i) - q_gt(u_i))^2 ) over the midpoint grid
    W2_QGRID on [W2_QLO, W2_QHI]. The mean (not the width-scaled integral) keeps
    the statistic in data units, so a pure location shift ext = gt + c gives
    W_2 = |c| exactly.

    Returns dict(w2, w2_skip, ext_qlo, ext_qhi). w2 is NaN, with a non-empty
    w2_skip, when a sample cannot support W2_QGRID without np.interp clamping into
    an unobserved probability range (a clamped quantile reads as a flat artifact,
    not a measurement):
      - 'gt_undersupported' : gt_x has fewer than W2_MIN_GT_N points
      - 'ext_no_weight'     : ext_x empty, or every weight is zero
      - 'ext_clamped'       : weighted_valid_range(ext_x, ext_w) does not cover
                              [W2_QLO, W2_QHI] -- e.g. a heavy-weight row at the
                              sample max
    """
    gt_x = np.asarray(gt_x, dtype=float)
    ext_x = np.asarray(ext_x, dtype=float)
    ext_w = np.asarray(ext_w, dtype=float)

    if gt_x.size < W2_MIN_GT_N:
        return dict(w2=np.nan, w2_skip='gt_undersupported', ext_qlo=np.nan, ext_qhi=np.nan)
    if ext_x.size == 0 or not np.any(ext_w > 0):
        return dict(w2=np.nan, w2_skip='ext_no_weight', ext_qlo=np.nan, ext_qhi=np.nan)

    ext_qlo, ext_qhi = weighted_valid_range(ext_x, ext_w, lo_cap=0.0, hi_cap=1.0)
    if ext_qlo > W2_QLO or ext_qhi < W2_QHI:
        return dict(w2=np.nan, w2_skip='ext_clamped', ext_qlo=ext_qlo, ext_qhi=ext_qhi)

    gt_q = _hazen_quantiles(gt_x, W2_QGRID)
    ext_q = weighted_hazen_quantile(ext_x, ext_w, W2_QGRID)
    w2 = float(np.sqrt(np.mean((ext_q - gt_q) ** 2)))
    return dict(w2=w2, w2_skip='', ext_qlo=float(ext_qlo), ext_qhi=float(ext_qhi))


def _boot_rng(boot_seed: int, ecosystem: str, attribute: str, stream: int) -> np.random.Generator:
    """Deterministic per-(cell, stream) RNG, so a cell's bootstrap CI does not
    depend on the order cells are iterated in (seed-determinism control,
    CLAUDE.md). `stream` separates the independent resampling streams inside one
    (ecosystem, attribute):
      0 = GT rows, raw            2 = extracted rows, raw  (all-rows settings)
      1 = GT rows, log10-positive 3 = extracted rows, raw  (judge_filtered)
                                  4 = extracted rows, log  (all-rows settings)
                                  5 = extracted rows, log  (judge_filtered)
    extracted / ntp_weighted / probe_weighted draw from the identical row set, so
    sharing a stream across them makes their replicates paired -- the right basis
    for the within-cell 'does weighting beat unweighted' comparison these numbers
    exist for (a paired-difference CI is a later, separate addition).
    """
    return np.random.default_rng(
        [int(boot_seed), ECOSYSTEMS.index(ecosystem), ATTRIBUTES.index(attribute), int(stream)]
    )


def _bootstrap_w2_ci(gt_x, ext_x, ext_w, gt_rng, ext_rng, n_boot: int, ci: float = 0.95):
    """Percentile bootstrap CI for wasserstein2_quantile(gt_x, ext_x, ext_w).

    Resamples BOTH samples with replacement. This differs on purpose from
    _bootstrap_gt_band, which resamples GT alone: there the extracted line is the
    estimand and only GT's sampling noise is in question; here W_2 is a two-sample
    statistic and both samples contribute to its sampling distribution.

    Each drawn extracted row keeps its ORIGINAL weight -- NOT a multinomial count
    applied to the weight vector, which is not exact for Hazen plotting positions
    (the position depends on the weight, not just the resulting rank; see
    tests/test_meta_weighted_stats.py's module docstring).

    Replicates that trip wasserstein2_quantile's skip guards (a resampled
    heavy-weight row landing at the sample extreme -> 'ext_clamped', etc.) are the
    widest-tailed, largest-W_2 draws in the set. Dropping them and renormalizing
    would pull the CI down and narrow it in exactly the borderline cells where the
    CI matters most, so they are NOT dropped silently: n_ok counts the survivors
    and the caller emits a NaN CI when n_ok / n_boot < W2_BOOT_MIN_OK_FRAC.

    NOTE ON INTERPRETATION: the plug-in W_2 is biased upward -- two samples drawn
    from the *same* distribution still give W_2 > 0 -- so this percentile CI will
    essentially never contain 0. It is the spread of the W_2 estimate, NOT a test
    of "is the extracted distribution different from GT".

    Returns (lo, hi, n_ok).
    """
    gt_x = np.asarray(gt_x, dtype=float)
    ext_x = np.asarray(ext_x, dtype=float)
    ext_w = np.asarray(ext_w, dtype=float)
    G, E = gt_x.size, ext_x.size

    gt_idx = gt_rng.integers(0, G, size=(n_boot, G))
    ext_idx = ext_rng.integers(0, E, size=(n_boot, E))

    vals = np.full(n_boot, np.nan)
    for b in range(n_boot):
        res = wasserstein2_quantile(gt_x[gt_idx[b]], ext_x[ext_idx[b]], ext_w[ext_idx[b]])
        vals[b] = res['w2']

    ok = np.isfinite(vals)
    n_ok = int(ok.sum())
    if n_ok == 0:
        return np.nan, np.nan, 0
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(vals[ok], [alpha, 1.0 - alpha])
    return float(lo), float(hi), n_ok


def build_wasserstein_table(gt_df: pd.DataFrame, ext_df: pd.DataFrame, shuffle_seed: int,
                            n_boot: int = N_BOOT) -> pd.DataFrame:
    """One row per (ecosystem, attribute, setting): the quantile-approximated
    2-Wasserstein distance from that extracted setting's distribution to ground
    truth, raw units and (for LOG_SCALE_ATTRIBUTES) log10 units.

    `extracted` / `judge_filtered` are the unweighted baselines the two
    confidence-weighted settings are read against -- a probe/NTP-weighted W_2 is
    uninterpretable alone; the claim it supports is "confidence weighting moves
    the extracted distribution toward GT", i.e. {ntp,probe}_weighted W_2 below
    `extracted`. `w2_shuffled` is the permutation control: ext weights shuffled
    within the cell (seed `shuffle_seed`), which should regress W_2 back toward
    the `extracted` baseline -- if it does not, the weights carry no
    distributional signal.

    `w2_lo` / `w2_hi` (and `w2_log_lo` / `w2_log_hi`) are the percentile bootstrap
    CI from _bootstrap_w2_ci -- both samples resampled, `n_boot` replicates,
    `shuffle_seed` reused as the bootstrap seed. `w2_n_boot_ok` is the surviving
    replicate count; the CI is NaN when it drops below W2_BOOT_MIN_OK_FRAC * n_boot
    (see _bootstrap_w2_ci -- the CI is a spread, never a test against 0).
    """
    rng = np.random.default_rng(shuffle_seed)
    min_ok = int(np.ceil(W2_BOOT_MIN_OK_FRAC * n_boot))
    rows = []
    for ecosystem in ECOSYSTEMS:
        for attribute in ATTRIBUTES:
            log_scale = attribute in LOG_SCALE_ATTRIBUTES
            gt_data = _setting_data('ground_truth', gt_df, ext_df, ecosystem, attribute)
            gt_x = gt_data[0] if gt_data is not None else np.array([])
            gt_pos = gt_x[gt_x > 0] if log_scale else np.array([])
            for setting in W2_SETTINGS:
                row = dict(dataset=DATASET, ecosystem=ecosystem, attribute=attribute,
                           setting=setting, unit=STANDARD_UNITS[attribute],
                           n_gt=int(gt_x.size), n_ext=0, n_eff=0.0,
                           ext_qlo=np.nan, ext_qhi=np.nan,
                           w2=np.nan, w2_skip='no_data',
                           w2_lo=np.nan, w2_hi=np.nan, w2_n_boot_ok=0,
                           w2_log=np.nan, w2_log_skip='no_data',
                           w2_log_lo=np.nan, w2_log_hi=np.nan, w2_log_n_boot_ok=0,
                           w2_shuffled=np.nan)
                ext_data = _setting_data(setting, gt_df, ext_df, ecosystem, attribute)
                all_rows = setting != 'judge_filtered'  # shared ext resample stream
                if ext_data is not None:
                    ext_x, ext_w = ext_data
                    row['n_ext'] = int(ext_x.size)
                    row['n_eff'] = kish_n_eff(ext_w) if np.any(ext_w > 0) else 0.0

                    raw = wasserstein2_quantile(gt_x, ext_x, ext_w)
                    row.update(w2=raw['w2'], w2_skip=raw['w2_skip'],
                               ext_qlo=raw['ext_qlo'], ext_qhi=raw['ext_qhi'])
                    if np.isfinite(raw['w2']):
                        lo, hi, n_ok = _bootstrap_w2_ci(
                            gt_x, ext_x, ext_w,
                            _boot_rng(shuffle_seed, ecosystem, attribute, 0),
                            _boot_rng(shuffle_seed, ecosystem, attribute, 2 if all_rows else 3),
                            n_boot=n_boot)
                        row['w2_n_boot_ok'] = n_ok
                        if n_ok >= min_ok:
                            row['w2_lo'], row['w2_hi'] = lo, hi

                    if log_scale:
                        ext_pos = ext_x > 0
                        lgx, lgw = np.log10(ext_x[ext_pos]), ext_w[ext_pos]
                        gt_logx = np.log10(gt_pos)
                        lg = wasserstein2_quantile(gt_logx, lgx, lgw)
                        row.update(w2_log=lg['w2'], w2_log_skip=lg['w2_skip'])
                        if np.isfinite(lg['w2']):
                            lo, hi, n_ok = _bootstrap_w2_ci(
                                gt_logx, lgx, lgw,
                                _boot_rng(shuffle_seed, ecosystem, attribute, 1),
                                _boot_rng(shuffle_seed, ecosystem, attribute, 4 if all_rows else 5),
                                n_boot=n_boot)
                            row['w2_log_n_boot_ok'] = n_ok
                            if n_ok >= min_ok:
                                row['w2_log_lo'], row['w2_log_hi'] = lo, hi
                    else:
                        row['w2_log'], row['w2_log_skip'] = np.nan, 'n/a'

                    if np.any(ext_w > 0):
                        shuf = wasserstein2_quantile(gt_x, ext_x, rng.permutation(ext_w))
                        row['w2_shuffled'] = shuf['w2']
                rows.append(row)
    return pd.DataFrame(rows)


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_qq_smooth(
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
    """One Q-Q figure for a fixed (ecosystem, method): one subplot per attribute.
    Each subplot overlays, for every gamma in GAMMAS, a line of (GT quantile,
    weighted-extracted quantile) pairs -- every held-out row included at every
    gamma, weighted by confidence(x)^(1/gamma), colored via gamma_color(). The gray
    band around the diagonal is GT's own bootstrap sampling uncertainty.

    A line is dropped once its Kish n_eff falls below MIN_RELIABLE_N; within a
    surviving line, individual levels below their own n_eff requirement are
    dropped by kish_gate_levels.
    """
    n_attrs = len(attributes)
    fig, axes = plt.subplots(1, n_attrs, figsize=(2.6 * n_attrs, 2.9))
    if n_attrs == 1:
        axes = [axes]

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

        ext_data = _setting_data(f'{method}_weighted', gt_df, ext_df, ecosystem, attribute)
        if ext_data is not None:
            ext_x_all, ext_p_all = ext_data
            if log_scale:
                pos = ext_x_all > 0
                ext_x_all, ext_p_all = ext_x_all[pos], ext_p_all[pos]
        else:
            ext_x_all, ext_p_all = np.array([]), np.array([])

        for t in GAMMAS:
            if ext_x_all.size == 0:
                continue
            w_t = ext_p_all ** (1.0 / t)
            if not np.any(w_t > 0):
                continue
            n_eff = kish_n_eff(w_t)
            if n_eff < MIN_RELIABLE_N:
                continue

            ext_lo, ext_hi = weighted_valid_range(ext_x_all, w_t)
            lo, hi = max(gt_lo, ext_lo), min(gt_hi, ext_hi)
            levels_t = QLEVELS[(QLEVELS >= lo) & (QLEVELS <= hi)]
            levels_t = kish_gate_levels(levels_t, n_eff)
            if levels_t.size == 0:
                continue

            gt_q_t = _hazen_quantiles(gt_x, levels_t)
            ext_q_t = weighted_hazen_quantile(ext_x_all, w_t, levels_t)
            color = gamma_color(t)
            if levels_t.size == 1:
                ax.scatter(gt_q_t, ext_q_t, color=color, s=10, alpha=0.85, zorder=4)
            else:
                ax.plot(gt_q_t, ext_q_t, color=color, linewidth=1.0, alpha=0.85,
                        zorder=4, solid_capstyle='round')
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


def plot_qq_legend_smooth(out_path: Path):
    """Horizontal colorbar spanning GAMMAS, red (heavy filtering) to blue (no
    reweighting). Colorbar only -- the y=x reference and GT band live on each
    Q-Q panel itself.
    """
    fig, ax = plt.subplots(figsize=(4.5, 0.6))
    ax.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.35, 0.7, 0.3])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=QQ_GAMMA_NORM, cmap=GAMMA_CMAP),
        cax=cbar_ax, orientation='horizontal',
    )
    cbar.set_label(r'Temperature $\gamma$', fontsize=11)
    tick_gammas = [GAMMA_FLOOR, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(tick_gammas)
    cbar.set_ticklabels([f'{g:g}' for g in tick_gammas])

    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta] wrote {out_path}")


def plot_qq_poster_smooth(
    gt_df: pd.DataFrame,
    ext_df: pd.DataFrame,
    out_path: Path,
    n_boot: int = N_BOOT,
    ci: float = 0.95,
    seed: int = 0,
):
    """Poster-size Q-Q figure for the single POSTER_ECOSYSTEM/METHOD/ATTRIBUTE
    cell, matching qq_probe_pond_tn_poster.pdf's layout: axes flipped (GT on y)
    vs. plot_qq_smooth, enlarged fonts. Same gates as plot_qq_smooth.
    """
    ecosystem, method, attribute = POSTER_ECOSYSTEM, POSTER_METHOD, POSTER_ATTRIBUTE
    log_scale = attribute in LOG_SCALE_ATTRIBUTES

    fig, ax = plt.subplots(figsize=(3.3, 3.3))

    gt_data = _setting_data('ground_truth', gt_df, ext_df, ecosystem, attribute)
    gt_x = gt_data[0] if gt_data is not None else np.array([])
    if log_scale:
        gt_x = gt_x[gt_x > 0]
    assert gt_x.size >= 2, f"insufficient GT data for poster cell ({ecosystem}, {attribute}): n={gt_x.size}"

    gt_lo, gt_hi = _valid_range(gt_x.size)
    gt_levels = QLEVELS[(QLEVELS >= gt_lo) & (QLEVELS <= gt_hi)]
    gt_q = _hazen_quantiles(gt_x, gt_levels)
    boot_lo, boot_hi = _bootstrap_gt_band(gt_x, gt_levels, n_boot=n_boot, ci=ci, seed=seed)

    all_plotted = [gt_q, boot_lo, boot_hi]

    ext_data = _setting_data(f'{method}_weighted', gt_df, ext_df, ecosystem, attribute)
    if ext_data is not None:
        ext_x_all, ext_p_all = ext_data
        if log_scale:
            pos = ext_x_all > 0
            ext_x_all, ext_p_all = ext_x_all[pos], ext_p_all[pos]
    else:
        ext_x_all, ext_p_all = np.array([]), np.array([])

    for t in GAMMAS:
        if ext_x_all.size == 0:
            continue
        w_t = ext_p_all ** (1.0 / t)
        if not np.any(w_t > 0):
            continue
        n_eff = kish_n_eff(w_t)
        if n_eff < MIN_RELIABLE_N:
            continue

        ext_lo, ext_hi = weighted_valid_range(ext_x_all, w_t)
        lo, hi = max(gt_lo, ext_lo), min(gt_hi, ext_hi)
        levels_t = QLEVELS[(QLEVELS >= lo) & (QLEVELS <= hi)]
        levels_t = kish_gate_levels(levels_t, n_eff)
        if levels_t.size == 0:
            continue

        gt_q_t = _hazen_quantiles(gt_x, levels_t)
        ext_q_t = weighted_hazen_quantile(ext_x_all, w_t, levels_t)
        color = gamma_color(t)
        if levels_t.size == 1:
            ax.scatter(ext_q_t, gt_q_t, color=color, s=14, alpha=0.85, zorder=4)
        else:
            ax.plot(ext_q_t, gt_q_t, color=color, linewidth=1.4, alpha=0.85,
                    zorder=4, solid_capstyle='round')
        all_plotted.extend([gt_q_t, ext_q_t])

    lo_lim, hi_lim = _axis_limits(np.concatenate(all_plotted), log=log_scale)
    ax.fill_between(gt_q, boot_lo, boot_hi, color='#888888', alpha=0.25, linewidth=0, zorder=1)
    ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color='#888888', linewidth=1.0,
             linestyle='--', zorder=2)

    if log_scale:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.set_xlim(lo_lim, hi_lim)
    ax.set_ylim(*POSTER_YLIM)
    ax.set_box_aspect(1)

    ax.set_xlabel('Extracted', fontsize=20)
    ax.set_ylabel('Ground Truth', fontsize=20)
    ax.set_title(_attr_title(attribute), fontsize=20, style='italic')
    ax.tick_params(labelsize=15)
    ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta] wrote {out_path}")


def plot_qq_legend_poster_smooth(out_path: Path):
    """Poster-size counterpart to plot_qq_legend_smooth (enlarged fonts), matching
    qq_legend_poster.pdf's poster-scale styling. Colorbar only -- see
    plot_qq_legend_smooth.
    """
    fig, ax = plt.subplots(figsize=(6.0, 0.9))
    ax.axis('off')

    cbar_ax = fig.add_axes([0.15, 0.35, 0.7, 0.4])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=QQ_GAMMA_NORM, cmap=GAMMA_CMAP),
        cax=cbar_ax, orientation='horizontal',
    )
    cbar.set_label(r'Temperature $\gamma$', fontsize=16)
    tick_gammas = [GAMMA_FLOOR, 0.4, 0.6, 0.8, 1.0]
    cbar.set_ticks(tick_gammas)
    cbar.set_ticklabels([f'{g:g}' for g in tick_gammas])
    cbar.ax.tick_params(labelsize=13)

    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta] wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    global PROBE_SOURCE, JUDGE_DATE, FIGURES_DIR
    _judge_date_default = JUDGE_DATE
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--attributes', nargs='+', default=QQ_ATTRIBUTES,
                         choices=ATTRIBUTES, help='Attribute subset (one subplot column each) for the Q-Q figures.')
    parser.add_argument('--n-boot', type=int, default=N_BOOT,
                         help='Bootstrap resamples for the ground-truth quantile uncertainty band.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--probe-source', default=None,
                         help="Synthetic training corpus for the probe + NTP calibrator "
                              "(default: baseline trained_probe/). E.g. 'v2' reads the "
                              "parallel synthetic_probe_<source>/ tree. Also suffixes the "
                              "output CSVs and routes figures under figures/meta/<source>/ "
                              "so baseline artifacts are never overwritten.")
    parser.add_argument('--judge-date', default=_judge_date_default,
                         help=f"Date tag of the qwen-2.5-7b interp judge run supplying the "
                              f"real-extraction activations (default: {_judge_date_default}).")
    args = parser.parse_args()

    PROBE_SOURCE = args.probe_source
    JUDGE_DATE = args.judge_date
    _suffix = f'_{PROBE_SOURCE}' if PROBE_SOURCE else ''
    if PROBE_SOURCE:
        FIGURES_DIR = FIGURES_DIR / PROBE_SOURCE
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gt_df, ext_df = load_data()

    stats_df = build_stats_table(gt_df, ext_df)
    csv_path = RESULTS_DIR / f'meta_{DATASET}_{EXT_MODEL}_{EXT_DATE}{_suffix}.csv'
    stats_df.to_csv(csv_path, index=False)
    print(f"[meta] wrote {csv_path}")
    print(stats_df.to_string(index=False, float_format='{:.3g}'.format))

    w2_df = build_wasserstein_table(gt_df, ext_df, shuffle_seed=args.seed, n_boot=args.n_boot)
    w2_path = RESULTS_DIR / f'wasserstein_{DATASET}_{EXT_MODEL}_{EXT_DATE}{_suffix}.csv'
    w2_df.to_csv(w2_path, index=False)
    print(f"[meta] wrote {w2_path}")
    print(w2_df.to_string(index=False, float_format='{:.3g}'.format))

    for method in METHODS:
        for ecosystem in ECOSYSTEMS:
            out_path = FIGURES_DIR / f'qq_{method}_{ecosystem}_smooth.pdf'
            plot_qq_smooth(gt_df, ext_df, ecosystem, method, args.attributes, out_path,
                            n_boot=args.n_boot, seed=args.seed)

    plot_qq_legend_smooth(FIGURES_DIR / 'qq_legend_smooth.pdf')

    plot_qq_poster_smooth(gt_df, ext_df, FIGURES_DIR / 'qq_probe_pond_tn_poster_smooth.pdf',
                           n_boot=args.n_boot, seed=args.seed)
    plot_qq_legend_poster_smooth(FIGURES_DIR / 'qq_legend_poster_smooth.pdf')


if __name__ == "__main__":
    main()
