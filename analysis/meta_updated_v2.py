"""Meta-analysis, v2: does an LLM-extracted dataset reproduce the per-ecosystem
attribute distributions of the human-curated ground truth -- and does keeping only
the confidently-scored extractions bring it closer?

Simplified from analysis/meta_updated.py:

1. Confidence is used as a HARD filter, not a weight: for each method (NTP, probe)
   and each t in params.meta_v2.thresholds, setting ``{method}_ge_{t:.2f}`` keeps the
   held-out extracted rows with confidence >= t (ties at t are kept). The grid must
   start at t = 0, which keeps every row: it IS the unfiltered extracted set (there is
   no separate ``extracted`` setting in the tables or figures). The confidence is the
   adjusted one when params.meta_v2.outlier_adjust.
2. Every sample is unweighted, so quantiles, Q-Q lines and distances come straight
   from library functions: np.quantile(method='hazen') for quantiles (both Q-Q axes,
   stats table), scipy.stats.wasserstein_distance (W1, over the full empirical
   distributions -- not comparable to v1's trimmed quantile-W2), and
   scipy.stats.bootstrap (percentile) for the reference band and the W1 CIs.
3. Data filtering is limited to what meta_updated.load_data does with
   restrict_to_shared_docs=False: GT and extracted rows each drop the probe/NTP
   training documents (scores exist, and are out-of-sample, only outside them) but
   are NOT restricted to documents both sides cover; a row with an
   unparseable point_value or a unit not in UNIT_CONVERSION_V2 has no standard-unit
   value; and a converted value outside PHYSICAL_BOUNDS is dropped. Nothing else --
   no Kish/n_eff gates. A Q-Q line or W1 score needs at least params.meta_v2.min_n
   rows. On log axes (LOG_SCALE_ATTRIBUTES) non-positive values cannot be drawn: they
   are left out of the log Q-Q line / log10 W1 only, and counted (n_nonpos) in the
   CSVs; the stats table and raw W1 keep them.

Outputs under analysis/results/meta/<config id>/:
  meta_stats.csv   n, n_docs, mean, std, Hazen Q1/median/Q3 per (ecosystem,
                   attribute, setting), plus method/threshold columns.
  survival.csv     per (ecosystem, attribute, method, threshold): rows and documents the
                   threshold keeps (n_ext, n_docs_ext), the reference's n_ref, and the
                   fractions of the t = 0 rows / documents that survive.
  wasserstein.csv  W1 (raw, and log10 for LOG_SCALE_ATTRIBUTES) from each compared
                   setting to the reference, two-sample percentile bootstrap CI, and
                   the permutation control w1_shuffled_{mean,lo,hi}[_log]: over
                   params.meta_v2.n_shuffle_samples permutations of the method's RAW
                   confidences within the cell, each multiplied by its row's own
                   non-outlier factor (1 unless params.meta_v2.outlier_adjust); mean and
                   2.5/97.5 percentiles of the resulting W1. Without adjustment the same
                   threshold keeps exactly as many rows as the real one; with it the count
                   differs (n_ext_shuffled_mean). A real W1 inside that range means thresholding on the confidence
                   does no better than keeping the same number of random rows. NB the
                   shuffled subsets spread over more documents than a real high-t
                   subset (n_docs_ext_shuffled_mean vs n_docs_ext).
  qq_lines.csv     per Q-Q line: n, n_docs, n_nonpos, whether it was drawn.
  figures/qq_{method}_{ecosystem}.pdf, figures/qq_legend.pdf
  figures/w1_vs_threshold_{ecosystem}.pdf, figures/w1_vs_threshold_legend.pdf
                   one panel per attribute, drawn from wasserstein.csv: raw-scale W1 to
                   the reference divided by the reference's range (ref_range), vs threshold
                   for NTP and probe (no CI drawn) and each method's shuffled
                   control (mean, 2.5-97.5 percentile band).
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats

# Importing meta_updated also applies its matplotlib rcParams (paper fonts/sizes).
from analysis.meta_updated import (
    ATTRIBUTES, DATASET, ECOSYSTEMS, LOG_SCALE_ATTRIBUTES, META_ROOT, METHOD_PROB_COL, METHODS,
    QLEVELS, REFERENCE_AXIS_LABEL, STANDARD_UNITS, UNIT_CONVERSION,
    _attr_title, _axis_limits, _valid_range, load_data,
)
from analysis.outlier_weight import add_outlier_columns
from analysis.meta_inputs import SECTION as META_SECTION, SECTION_V2, load_meta_v2_config, resolve_meta_inputs

# Non-threshold settings, per reference: the reference's own setting first. `valid` is
# the extracted rows whose stored calibration label is positive (judge OR GT match).
# v1's judge-only `judge_filtered` is dropped (it is `valid` minus the GT matches), and
# so is `extracted` (it is the t = 0 threshold setting).
BASE_SETTINGS = {
    'ground_truth': ['ground_truth', 'valid'],
    'valid':        ['ground_truth', 'valid'],
}
# Non-threshold lines drawn on every Q-Q panel, over the threshold lines.
QQ_BASE_LINES = {
    'ground_truth': ['valid'],
    'valid':        ['ground_truth'],
}
# Line colors: every threshold line (t = 0, the unfiltered set, is just the first of them)
# is solid, stepped evenly from the cool end to the warm end of THRESHOLD_CMAP; valid is
# a black dotted line.
THRESHOLD_CMAP = mpl.colormaps['coolwarm']
QQ_BASE_STYLE = {
    'ground_truth': dict(color='#2a7d3a', linestyle=(0, (6, 2)), linewidth=1.6),
    'valid':        dict(color='black', linestyle=':', linewidth=2.0),
}
QQ_BASE_LEGEND = {
    'ground_truth': 'Ground truth',
    'valid':        'Valid extracted (judge or GT match)',
}
# v1's UNIT_CONVERSION plus pond schema units it lacked (experiments/dataset-configs/
# pond.py lists them as extraction units): mi^2, and ppm / ppb for tn / tp / chla, taken
# as mg/L / µg/L (mass ratios in dilute water, density ~1 kg/L). Still deliberately
# unconvertible: µg/cm^2 (an areal density, not a water concentration) and molar chla
# (chl-a is essentially never reported in µmol/L; such a row is far more likely a
# mislabeled unit than a real measurement).
_V2_ADDED_UNITS = {
    'surface_area': {'mi^2': 2589988.110336},
    'tn':   {'ppm': 1000.0, 'ppb': 1.0},
    'tp':   {'ppm': 1000.0, 'ppb': 1.0},
    'chla': {'ppm': 1000.0, 'ppb': 1.0},
}
assert all(u not in UNIT_CONVERSION[a] for a, m in _V2_ADDED_UNITS.items() for u in m), 'v2 addition already in v1'
UNIT_CONVERSION_V2 = {a: {**m, **_V2_ADDED_UNITS.get(a, {})} for a, m in UNIT_CONVERSION.items()}

# Codes keying the per-(cell, setting) bootstrap RNG (see _rng); stable across configs.
SETTING_CODES = {'ground_truth': 0, 'extracted': 1, 'valid': 3,
                 **{m: 4 + i for i, m in enumerate(METHODS)}}
# Offset added to a method's code for its shuffled-confidence permutation stream.
SHUFFLE_STREAM = 100

# W1-vs-threshold curves: tab10 colors mixed with white into pastels (NTP = tab10 blue,
# probe = tab10 green); real arm solid and thick, its shuffled control dotted.
PASTEL_WHITE_FRACTION = 0.35


def _pastel(color) -> tuple:
    """`color` mixed with PASTEL_WHITE_FRACTION white."""
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - PASTEL_WHITE_FRACTION) * rgb + PASTEL_WHITE_FRACTION)


_TAB10 = sns.color_palette('tab10', 10)
CURVE_STYLE = {
    'ntp':            dict(color=_pastel(_TAB10[0]), ls='-', lw=2.5, label='NTP'),
    'probe':          dict(color=_pastel(_TAB10[2]), ls='-', lw=2.5, label='Probe'),
    'ntp_shuffled':   dict(color=_pastel(_TAB10[0]), ls=':', lw=2.0, label='NTP (shuffled)'),
    'probe_shuffled': dict(color=_pastel(_TAB10[2]), ls=':', lw=2.0, label='Probe (shuffled)'),
}


def threshold_setting(method: str, t: float) -> str:
    return f'{method}_ge_{t:.2f}'


def threshold_style(t: float, thresholds: list[float]) -> dict:
    """Solid line, the i-th of k thresholds (t = 0 is i = 0, like any other) at
    THRESHOLD_CMAP(i / (k - 1)): first threshold the cool end, last the warm end."""
    assert thresholds[0] == 0.0 and len(thresholds) > 1, thresholds
    return dict(color=THRESHOLD_CMAP(thresholds.index(t) / (len(thresholds) - 1)), linestyle='-', linewidth=1.5)


def threshold_label(t: float) -> str:
    return rf'Confidence $\geq {t:g}$'


def _rng(seed: int, ecosystem: str, attribute: str, code: int, t: float, log: bool) -> np.random.Generator:
    """Deterministic RNG per (cell, setting, threshold, scale): a cell's CI never
    depends on which other cells/thresholds the config selects or their order.
    Keyed on the canonical ECOSYSTEMS/ATTRIBUTES positions and t in 1e-4 units."""
    return np.random.default_rng([int(seed), ECOSYSTEMS.index(ecosystem), ATTRIBUTES.index(attribute),
                                  int(code), int(round(t * 10_000)), int(log)])


# ── Settings ────────────────────────────────────────────────────────────────

def cell_rows(gt_df: pd.DataFrame, ext_df: pd.DataFrame, ecosystem: str, attribute: str):
    """(GT rows, extracted rows) of one cell that carry a standard-unit value."""
    g = gt_df[(gt_df['ecosystem_bucket'] == ecosystem) & (gt_df['attribute'] == attribute)]
    e = ext_df[(ext_df['ecosystem_bucket'] == ecosystem) & (ext_df['attribute'] == attribute)]
    return g.dropna(subset=['converted_value']), e.dropna(subset=['converted_value'])


def setting_rows(setting: str, gt: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Rows of one cell (gt/ext from cell_rows) belonging to `setting`.

    Unfiltered: ground_truth, extracted, valid
    (stored calibration label, judge OR GT match). Thresholded: '{method}_ge_{t:.2f}'
    keeps extracted rows whose METHOD_PROB_COL[method] >= t.
    """
    if setting == 'ground_truth':
        return gt
    if setting == 'extracted':
        return ext
    if setting == 'valid':
        assert ext['label'].dtype == bool, ext['label'].dtype
        return ext[ext['label']]
    method, sep, t = setting.partition('_ge_')
    assert sep and method in METHODS, f'unknown setting {setting!r}'
    prob = ext[METHOD_PROB_COL[method]].to_numpy()
    assert prob.shape == (len(ext),) and np.isfinite(prob).all(), 'confidence missing or misshapen'
    return ext[prob >= float(t)]


def all_settings(reference: str, thresholds: list[float]) -> list[str]:
    names = BASE_SETTINGS[reference] + [threshold_setting(m, t) for m in METHODS for t in thresholds]
    assert len(set(names)) == len(names), f'threshold settings collide at 2 decimals: {thresholds}'
    return names


def _setting_meta(setting: str) -> dict:
    method, sep, t = setting.partition('_ge_')
    return dict(method=method, threshold=float(t)) if sep else dict(method='', threshold=np.nan)


# ── Stats table ─────────────────────────────────────────────────────────────

def summary_stats(x: np.ndarray) -> dict:
    """Mean, sample std (ddof=1), Hazen Q1/median/Q3 of an unweighted sample."""
    assert x.ndim == 1 and np.isfinite(x).all()
    if x.size == 0:
        return dict(mean=np.nan, std=np.nan, q1=np.nan, median=np.nan, q3=np.nan)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75], method='hazen')
    return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
                q1=float(q1), median=float(med), q3=float(q3))


def build_stats_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds) -> pd.DataFrame:
    rows = []
    for ecosystem in ecosystems:
        for attribute in attributes:
            gt, ext = cell_rows(gt_df, ext_df, ecosystem, attribute)
            for setting in all_settings(reference, thresholds):
                sub = setting_rows(setting, gt, ext)
                x = sub['converted_value'].to_numpy(dtype=float)
                rows.append(dict(dataset=DATASET, ecosystem=ecosystem, attribute=attribute, setting=setting,
                                 **_setting_meta(setting), unit=STANDARD_UNITS[attribute],
                                 n=int(x.size), n_docs=int(sub['document_id'].nunique()),
                                 n_nonpos=int((x <= 0).sum()), **summary_stats(x)))
    return pd.DataFrame(rows)


# ── Survival table ──────────────────────────────────────────────────────────

def build_survival_table(stats_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Rows / documents surviving each threshold, from build_stats_table's output: one row
    per (ecosystem, attribute, method, threshold) with n_ref (the `reference` setting's n),
    n_ext, n_docs_ext and their fractions of the t = 0 (unfiltered) values."""
    ref = stats_df[stats_df['setting'] == reference].set_index(['ecosystem', 'attribute'])
    assert ref.index.is_unique and len(ref) > 0, 'reference setting missing or duplicated in the stats table'
    thr = stats_df[stats_df['method'].isin(METHODS)]
    out = thr.rename(columns={'n': 'n_ext', 'n_docs': 'n_docs_ext'})[
        ['dataset', 'ecosystem', 'attribute', 'method', 'threshold', 'n_ext', 'n_docs_ext']].copy()
    key = ['ecosystem', 'attribute']
    out['n_ref'] = [int(ref.loc[(e, a), 'n']) for e, a in zip(out['ecosystem'], out['attribute'])]
    t0 = out[out['threshold'] == 0.0].set_index(key + ['method'])
    assert t0.index.is_unique and len(t0) == len(out.groupby(key + ['method'])), 'every cell needs one t = 0 row'
    assert (t0['n_ext'] > 0).all(), 'a cell has no extracted rows at t = 0: survival fractions undefined'
    base = out.join(t0[['n_ext', 'n_docs_ext']].rename(columns={'n_ext': 'n_ext_t0', 'n_docs_ext': 'n_docs_ext_t0'}),
                    on=key + ['method'])
    assert len(base) == len(out), 'the t = 0 join changed the row count'
    out['frac_rows_vs_t0'] = base['n_ext'] / base['n_ext_t0']
    out['frac_docs_vs_t0'] = base['n_docs_ext'] / base['n_docs_ext_t0']
    assert (out['frac_rows_vs_t0'] <= 1).all() and (out['frac_docs_vs_t0'] <= 1).all(), 'a threshold kept more than t = 0'
    return out.reset_index(drop=True)


# ── W1 table ────────────────────────────────────────────────────────────────

def _scale(x: np.ndarray, log: bool) -> np.ndarray:
    return np.log10(x[x > 0]) if log else x


def w1_with_ci(ref_x, ext_x, min_n: int, n_boot: int, rng: np.random.Generator, ci: float = 0.95) -> dict:
    """scipy W1 between two unweighted samples plus a two-sample (unpaired) percentile
    bootstrap CI. NaN, with a skip reason, when either sample has < min_n rows.

    The plug-in W1 is biased upward (two samples from one distribution still give
    W1 > 0), so the CI is the spread of the estimate, never a test against 0."""
    if ref_x.size < min_n:
        return dict(w1=np.nan, w1_lo=np.nan, w1_hi=np.nan, w1_skip='ref_n<min_n')
    if ext_x.size < min_n:
        return dict(w1=np.nan, w1_lo=np.nan, w1_hi=np.nan, w1_skip='ext_n<min_n')
    w1 = stats.wasserstein_distance(ref_x, ext_x)
    res = stats.bootstrap((ref_x, ext_x), stats.wasserstein_distance, paired=False, vectorized=False,
                          n_resamples=n_boot, confidence_level=ci, method='percentile', random_state=rng)
    lo, hi = res.confidence_interval
    assert np.isfinite(lo) and np.isfinite(hi), 'W1 bootstrap CI is not finite'
    return dict(w1=float(w1), w1_lo=float(lo), w1_hi=float(hi), w1_skip='')


def _shuffle_rng(seed: int, ecosystem: str, attribute: str, method: str, sample: int) -> np.random.Generator:
    """RNG of shuffle `sample` of `method`'s confidences in one cell. Five seed words,
    so it never coincides with a six-word _rng bootstrap stream."""
    return np.random.default_rng([int(seed), ECOSYSTEMS.index(ecosystem), ATTRIBUTES.index(attribute),
                                  SHUFFLE_STREAM + SETTING_CODES[method], int(sample)])


def _summarize_shuffles(vals: np.ndarray, ci: float) -> dict:
    """Mean and central `ci` percentile range of the per-shuffle W1s -- only when every
    shuffle produced one (a partial set is not averaged; NaN + skip reason instead)."""
    n_ok = int(np.isfinite(vals).sum())
    if n_ok < vals.size:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n_ok=n_ok, skip='none_ok' if n_ok == 0 else 'partial_ok')
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return dict(mean=float(vals.mean()), lo=float(lo), hi=float(hi), n_ok=n_ok, skip='')


def shuffled_w1(ref: np.ndarray, ext: pd.DataFrame, method: str, thresholds: list[float], log_scale: bool,
                min_n: int, n_shuffle: int, seed: int, ecosystem: str, attribute: str, ci: float = 0.95) -> dict:
    """Permutation control for one (cell, method): per shuffle, permute the method's RAW
    confidences (``{prob_col}_raw``) over the cell's extracted rows, multiply by each row's
    own ``outlier_factor`` (so the non-outlier filter is kept and only the link between
    the model's confidence and the row is broken), then apply every threshold (one
    permutation shared by all thresholds, so shuffled subsets are nested like the real
    ones) and take W1 to `ref`.

    With no outlier adjustment every factor is 1, the shuffled scores are a permutation of
    the real ones, and a threshold keeps exactly as many rows as the real one (asserted).
    With adjustment the products form a different multiset, so the shuffled row count
    differs from the real one; it is reported as n_ext_shuffled_mean, since W1 depends on
    the sample size.

    Returns {t: dict(w1_shuffled_{mean,lo,hi,n_ok,skip}, w1_shuffled_log_*,
    n_docs_ext_shuffled_mean, n_ext_shuffled_mean)}; log columns NaN / 'n/a' off
    LOG_SCALE_ATTRIBUTES."""
    col = METHOD_PROB_COL[method]
    raw = ext[f'{col}_raw'].to_numpy(dtype=float)
    factor = ext['outlier_factor'].to_numpy(dtype=float)
    assert np.isfinite(raw).all() and np.isfinite(factor).all() and ((factor > 0) & (factor <= 1)).all()
    prob = ext[col].to_numpy(dtype=float)
    assert np.array_equal(prob, raw * factor), f'{col} != {col}_raw * outlier_factor'
    exact = bool((factor == 1.0).all())
    x = ext['converted_value'].to_numpy(dtype=float)
    doc_codes, _ = pd.factorize(ext['document_id'])
    real_n = {t: int((prob >= t).sum()) for t in thresholds}
    ref_log = _scale(ref, True)
    w_raw = {t: np.full(n_shuffle, np.nan) for t in thresholds}
    lg = {t: np.full(n_shuffle, np.nan) for t in thresholds}
    n_docs = {t: np.zeros(n_shuffle) for t in thresholds}
    n_rows = {t: np.zeros(n_shuffle) for t in thresholds}
    for s in range(n_shuffle):
        p = _shuffle_rng(seed, ecosystem, attribute, method, s).permutation(raw) * factor
        for t in thresholds:
            keep = p >= t
            if exact:
                assert int(keep.sum()) == real_n[t], 'a permutation changed the number of rows kept'
            xs = x[keep]
            n_rows[t][s] = xs.size
            n_docs[t][s] = np.unique(doc_codes[keep]).size
            if ref.size >= min_n and xs.size >= min_n:
                w_raw[t][s] = stats.wasserstein_distance(ref, xs)
            if log_scale:
                xl = _scale(xs, True)
                if ref_log.size >= min_n and xl.size >= min_n:
                    lg[t][s] = stats.wasserstein_distance(ref_log, xl)
    out = {}
    for t in thresholds:
        r = _summarize_shuffles(w_raw[t], ci)
        l = (_summarize_shuffles(lg[t], ci) if log_scale
             else dict(mean=np.nan, lo=np.nan, hi=np.nan, n_ok=0, skip='n/a'))
        out[t] = {**{f'w1_shuffled_{k}': v for k, v in r.items()},
                  **{f'w1_shuffled_log_{k}': v for k, v in l.items()},
                  'n_docs_ext_shuffled_mean': float(n_docs[t].mean()),
                  'n_ext_shuffled_mean': float(n_rows[t].mean())}
    return out


def build_w1_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds,
                   min_n: int, n_boot: int, n_shuffle: int, seed: int) -> pd.DataFrame:
    """One row per (ecosystem, attribute, compared setting): W1 from that setting to the
    `reference` setting, raw and (LOG_SCALE_ATTRIBUTES) log10 units, with its bootstrap
    CI, plus the shuffled_w1 permutation control (over `n_shuffle` permutations) on the
    threshold settings -- NaN for the non-threshold settings.
    """
    rows = []
    for ecosystem in ecosystems:
        for attribute in attributes:
            log_scale = attribute in LOG_SCALE_ATTRIBUTES
            gt, ext = cell_rows(gt_df, ext_df, ecosystem, attribute)
            ref = setting_rows(reference, gt, ext)['converted_value'].to_numpy(dtype=float)
            ref_stats = summary_stats(ref)
            ref_iqr = ref_stats['q3'] - ref_stats['q1']   # NaN for an empty reference
            assert np.isnan(ref_iqr) or ref_iqr > 0, f'reference IQR is {ref_iqr} for {ecosystem}/{attribute}'
            ref_range = float(ref.max() - ref.min()) if ref.size else np.nan
            assert np.isnan(ref_range) or ref_range > 0, f'reference range is {ref_range} for {ecosystem}/{attribute}'
            shuffled = {m: shuffled_w1(ref, ext, m, thresholds, log_scale, min_n, n_shuffle, seed, ecosystem, attribute)
                        for m in METHODS}
            empty = {k: (np.nan if isinstance(v, float) else (0 if isinstance(v, int) else '')) for k, v in
                     shuffled[METHODS[0]][thresholds[0]].items()}
            for setting in all_settings(reference, thresholds):
                if setting == reference:
                    continue
                meta = _setting_meta(setting)
                sub = setting_rows(setting, gt, ext)
                x = sub['converted_value'].to_numpy(dtype=float)
                code, t = SETTING_CODES[meta['method'] or setting], (0.0 if meta['method'] == '' else meta['threshold'])
                row = dict(dataset=DATASET, ecosystem=ecosystem, attribute=attribute, setting=setting,
                           reference=reference, **meta, unit=STANDARD_UNITS[attribute],
                           n_ref=int(ref.size), ref_iqr=ref_iqr, ref_range=ref_range, n_ext=int(x.size),
                           n_docs_ext=int(sub['document_id'].nunique()))
                row.update(w1_with_ci(ref, x, min_n, n_boot, _rng(seed, ecosystem, attribute, code, t, False)))
                if log_scale:
                    lg = w1_with_ci(_scale(ref, True), _scale(x, True), min_n, n_boot,
                                    _rng(seed, ecosystem, attribute, code, t, True))
                else:
                    lg = dict(w1=np.nan, w1_lo=np.nan, w1_hi=np.nan, w1_skip='n/a')
                row.update({k.replace('w1', 'w1_log'): v for k, v in lg.items()})

                row.update(shuffled[meta['method']][meta['threshold']] if meta['method'] else empty)
                rows.append(row)
    return pd.DataFrame(rows)


# ── Q-Q figures ─────────────────────────────────────────────────────────────

def qq_levels(n_ref: int, n_ext: int) -> np.ndarray:
    """QLEVELS inside both samples' Hazen range [0.5/n, 1-0.5/n], where np.quantile
    interpolates rather than clamping to the sample min/max (a clamped tail reads as
    a flat artifact)."""
    lo_r, hi_r = _valid_range(n_ref)
    lo_e, hi_e = _valid_range(n_ext)
    lo, hi = max(lo_r, lo_e), min(hi_r, hi_e)
    return QLEVELS[(QLEVELS >= lo) & (QLEVELS <= hi)]


def qq_line(ref_x: np.ndarray, ext_x: np.ndarray, min_n: int):
    """(reference quantiles, extracted quantiles), both np.quantile(method='hazen'), or
    None if the extracted sample has < min_n rows."""
    if ext_x.size < min_n:
        return None
    levels = qq_levels(ref_x.size, ext_x.size)
    assert levels.size > 0
    return (np.quantile(ref_x, levels, method='hazen'), np.quantile(ext_x, levels, method='hazen'))


def reference_band(ref_x: np.ndarray, levels: np.ndarray, n_boot: int, rng: np.random.Generator, ci: float = 0.95):
    """Percentile bootstrap band (scipy.stats.bootstrap) of the reference sample's Hazen
    quantiles at `levels`: the reference's own sampling noise, nothing about the
    extracted lines."""
    res = stats.bootstrap((ref_x,), lambda s, axis: np.quantile(s, levels, method='hazen', axis=axis),
                          vectorized=True, n_resamples=n_boot, confidence_level=ci, method='percentile',
                          random_state=rng)
    lo, hi = res.confidence_interval
    assert lo.shape == levels.shape and hi.shape == levels.shape
    return lo, hi


def plot_qq(gt_df, ext_df, reference, ecosystem, method, attributes, thresholds, min_n, n_boot, seed,
            out_path: Path) -> list[dict]:
    """One Q-Q figure for (ecosystem, method), one panel per attribute: x = reference
    quantiles, y = extracted quantiles, one line per threshold (threshold_style) plus the
    QQ_BASE_LINES, the y=x diagonal and the reference bootstrap band.
    Returns one record per line for qq_lines.csv."""
    ref_label = REFERENCE_AXIS_LABEL[reference]
    fig, axes = plt.subplots(1, len(attributes), figsize=(2.6 * len(attributes), 2.9), squeeze=False)
    records = []
    for i, (ax, attribute) in enumerate(zip(axes[0], attributes)):
        log_scale = attribute in LOG_SCALE_ATTRIBUTES
        gt, ext = cell_rows(gt_df, ext_df, ecosystem, attribute)
        ref_all = setting_rows(reference, gt, ext)['converted_value'].to_numpy(dtype=float)
        ref_x = ref_all[ref_all > 0] if log_scale else ref_all

        ax.set_title(_attr_title(attribute), fontsize=13, style='italic')
        ax.set_xlabel(ref_label)
        if i == 0:
            ax.set_ylabel('Extracted')
        if ref_x.size < min_n:
            ax.text(0.5, 0.5, f'insufficient {ref_label} data\n(n={ref_x.size})', ha='center', va='center',
                    fontsize=9, color='#888888', transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        ref_levels = qq_levels(ref_x.size, ref_x.size)
        ref_q = np.quantile(ref_x, ref_levels, method='hazen')
        band_lo, band_hi = reference_band(ref_x, ref_levels, n_boot,
                                          _rng(seed, ecosystem, attribute, SETTING_CODES[reference], 0.0, log_scale))
        plotted = [ref_q, band_lo, band_hi]

        lines = [(threshold_setting(method, t), threshold_style(t, thresholds), 4) for t in thresholds]
        lines += [(s, QQ_BASE_STYLE[s], 5) for s in QQ_BASE_LINES[reference]]
        for setting, style, z in lines:
            sub = setting_rows(setting, gt, ext)
            x_all = sub['converted_value'].to_numpy(dtype=float)
            x = x_all[x_all > 0] if log_scale else x_all
            line = qq_line(ref_x, x, min_n)
            records.append(dict(ecosystem=ecosystem, attribute=attribute, panel_method=method, setting=setting,
                                **_setting_meta(setting), reference=reference, log_scale=log_scale,
                                n_ref=int(ref_x.size), n=int(x.size), n_nonpos_excluded=int(x_all.size - x.size),
                                n_docs=int(sub['document_id'].nunique()), drawn=line is not None))
            if line is None:
                continue
            ax.plot(*line, alpha=0.9, zorder=z, solid_capstyle='round', **style)
            plotted.extend(line)

        lo_lim, hi_lim = _axis_limits(np.concatenate(plotted), log=log_scale)
        ax.fill_between(ref_q, band_lo, band_hi, color='#888888', alpha=0.25, linewidth=0, zorder=1)
        ax.plot([lo_lim, hi_lim], [lo_lim, hi_lim], color='#888888', linewidth=1.0, linestyle='--', zorder=2)
        if log_scale:
            ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlim(lo_lim, hi_lim); ax.set_ylim(lo_lim, hi_lim)
        ax.set_box_aspect(1)
        ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta_v2] wrote {out_path}")
    return records


def plot_qq_legend(out_path: Path, reference: str, thresholds: list[float]):
    """Key for the Q-Q figures: a discrete color bar for the threshold lines (one cell per
    threshold, in the colors threshold_style draws, labelled with t), plus a legend for
    the QQ_BASE_LINES and the band."""
    colors = [threshold_style(t, thresholds)['color'] for t in thresholds]
    k = len(thresholds)
    fig = plt.figure(figsize=(7.0, 1.0))
    cax = fig.add_axes([0.04, 0.5, 0.46, 0.22])
    cmap = mcolors.ListedColormap(colors)
    sm = mpl.cm.ScalarMappable(norm=mcolors.BoundaryNorm(np.arange(k + 1) - 0.5, k), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.arange(k))
    cb.ax.set_xticklabels([f'{t:g}' for t in thresholds], fontsize=9)
    cb.ax.tick_params(length=0)
    cb.set_label(r'Confidence threshold $t$', fontsize=10)
    handles = [Line2D([], [], label=QQ_BASE_LEGEND[s], **QQ_BASE_STYLE[s]) for s in QQ_BASE_LINES[reference]]
    handles.append(Patch(color='#888888', alpha=0.25, linewidth=0, label=f'{REFERENCE_AXIS_LABEL[reference]} 95% bootstrap'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.54, 0.5), fontsize=9, handlelength=2.6, frameon=False)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta_v2] wrote {out_path}")


# ── W1 vs threshold ─────────────────────────────────────────────────────────

def _curve(w1_df: pd.DataFrame, ecosystem: str, attribute: str, method: str, thresholds: list[float]) -> pd.DataFrame:
    """The (ecosystem, attribute, method) threshold rows of wasserstein.csv, in grid order."""
    c = w1_df[(w1_df['ecosystem'] == ecosystem) & (w1_df['attribute'] == attribute) & (w1_df['method'] == method)]
    c = c.sort_values('threshold')
    assert np.allclose(c['threshold'].to_numpy(dtype=float), thresholds), (ecosystem, attribute, method)
    return c


def plot_w1_curves(w1_df: pd.DataFrame, ecosystem: str, attributes: list[str], thresholds: list[float],
                   out_path: Path):
    """One figure per ecosystem, one panel per attribute: W1 to the reference against the
    confidence threshold t. Per method, the real filter (solid) and its shuffled control
    (dotted; band = the 2.5-97.5 percentile range
    over permutations). The real curve has no CI band (the bootstrap CI stays in
    wasserstein.csv, not drawn). Raw-scale W1 (no log, even for LOG_SCALE_ATTRIBUTES) divided by the
    cell's reference range (`ref_range` = max - min of the reference sample: fixed per
    cell, so it rescales without changing a curve's shape or the real-vs-shuffled
    comparison). The y-label is on the leftmost panel only. The extracted sample can lie
    outside the reference range, so the ratio is not bounded by 1. A point the
    table skipped (n < min_n) is a gap in its line."""
    fig, axes = plt.subplots(1, len(attributes), figsize=(3.0 * len(attributes), 2.8), squeeze=False)
    for i, (ax, attribute) in enumerate(zip(axes[0], attributes)):
        for method in METHODS:
            c = _curve(w1_df, ecosystem, attribute, method, thresholds)
            rng = c['ref_range'].to_numpy(dtype=float)
            assert np.all(rng == rng[0]), 'reference range differs across the thresholds of one cell'
            real = CURVE_STYLE[method]
            ax.plot(thresholds, c['w1'] / rng, **real)
            shuf = CURVE_STYLE[f'{method}_shuffled']
            ax.plot(thresholds, c['w1_shuffled_mean'] / rng, **shuf)
            ax.fill_between(thresholds, c['w1_shuffled_lo'] / rng, c['w1_shuffled_hi'] / rng, color=shuf['color'],
                            alpha=0.12, linewidth=0)
        ax.set_title(_attr_title(attribute), fontsize=13, style='italic')
        ax.set_xlabel('Confidence threshold $t$', fontsize=11)
        if i == 0:
            ax.set_ylabel(f"$W_1$ / range of {REFERENCE_AXIS_LABEL[w1_df['reference'].iloc[0]]}", fontsize=11)
        ax.set_xlim(thresholds[0], thresholds[-1])
        ax.set_xticks(thresholds if len(thresholds) <= 6 else thresholds[::3])   # grid points only, never interpolated ticks
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta_v2] wrote {out_path}")


def plot_w1_curves_legend(out_path: Path):
    """Line key (as clustering.plot_legend) plus the meaning of the shuffled band."""
    order = ('probe', 'ntp', 'probe_shuffled', 'ntp_shuffled')
    handles = [Line2D([], [], color=CURVE_STYLE[a]['color'], lw=4 if CURVE_STYLE[a]['ls'] == '-' else 2.5,
                      linestyle=CURVE_STYLE[a]['ls'], label=CURVE_STYLE[a]['label']) for a in order]
    handles.append(Patch(color='#888888', alpha=0.18, linewidth=0, label='Shuffled: 2.5-97.5% over permutations'))
    fig, ax = plt.subplots(figsize=(10.0, 0.7))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=3, fontsize=12, frameon=False, handlelength=2.0)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta_v2] wrote {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help="analysis/analysis-configs/<id>.yaml with params.meta_v2")
    args = parser.parse_args()

    cfg = load_meta_v2_config(args.config)
    sec = cfg['params'][SECTION_V2]
    ecosystems, attributes, reference = sec['ecosystems'], sec['attributes'], sec['reference']
    assert not set(ecosystems) - set(ECOSYSTEMS), f"ecosystems not in ECOSYSTEMS: {ecosystems}"
    assert not set(attributes) - set(ATTRIBUTES), f"attributes not in ATTRIBUTES: {attributes}"
    thresholds, min_n, n_boot, seed = sec['thresholds'], sec['min_n'], sec['n_boot'], cfg['seed']
    n_shuffle = sec['n_shuffle_samples']

    # resolve_meta_inputs / load_data read their inputs from params.meta; hand them
    # exactly the input-selection keys they use, nothing that could change v1 behavior.
    input_keys = ('calibration_config_id', 'rows', 'deduplication_config_id', 'confidence')
    v1_cfg = {**cfg, 'params': {META_SECTION: {k: sec[k] for k in input_keys}}}
    inputs = resolve_meta_inputs(v1_cfg, DATASET, sec['calibration_version'])
    gt_df, ext_df, manifest = load_data(v1_cfg, inputs, restrict_to_shared_docs=False,
                                        unit_conversion=UNIT_CONVERSION_V2)
    # Thresholds, survival and the real W1 read ext_df's ntp_prob / probe_prob (adjusted when
    # outlier_adjust); the shuffled control permutes the *_raw columns and re-applies the
    # row's own outlier_factor (see shuffled_w1).
    ext_df, moments = add_outlier_columns(ext_df, sec['outlier_adjust'])
    if moments is not None:
        print(f"[meta_v2] outlier_adjust: confidences x exp(-(x-mu)^2/2sigma^2)\n{moments}")

    out_dir = META_ROOT / cfg['id']
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats_df = build_stats_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds)
    stats_df.to_csv(out_dir / 'meta_stats.csv', index=False)
    print(f"[meta_v2] wrote {out_dir / 'meta_stats.csv'}")

    survival_df = build_survival_table(stats_df, reference)
    survival_df.to_csv(out_dir / 'survival.csv', index=False)
    print(f"[meta_v2] wrote {out_dir / 'survival.csv'}")

    w1_df = build_w1_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds, min_n, n_boot, n_shuffle, seed)
    w1_df.to_csv(out_dir / 'wasserstein.csv', index=False)
    check = survival_df.merge(w1_df[w1_df['method'].isin(METHODS)], on=['ecosystem', 'attribute', 'method', 'threshold'],
                              suffixes=('', '_w1'))
    assert len(check) == len(survival_df) == len(w1_df[w1_df['method'].isin(METHODS)]), 'survival / W1 tables cover different cells'
    assert (check['n_ext'] == check['n_ext_w1']).all() and (check['n_docs_ext'] == check['n_docs_ext_w1']).all() \
        and (check['n_ref'] == check['n_ref_w1']).all(), 'survival.csv disagrees with wasserstein.csv on row/doc counts'
    print(f"[meta_v2] wrote {out_dir / 'wasserstein.csv'}")
    print(w1_df[['ecosystem', 'attribute', 'setting', 'n_ext', 'n_docs_ext', 'w1', 'w1_lo', 'w1_hi',
                 'w1_shuffled_mean', 'w1_log', 'w1_shuffled_log_mean']].to_string(index=False, float_format='{:.3g}'.format))

    records = []
    for method in METHODS:
        for ecosystem in ecosystems:
            records += plot_qq(gt_df, ext_df, reference, ecosystem, method, sec['qq_attributes'], thresholds,
                               min_n, n_boot, seed, figures_dir / f'qq_{method}_{ecosystem}.pdf')
    pd.DataFrame(records).to_csv(out_dir / 'qq_lines.csv', index=False)
    plot_qq_legend(figures_dir / 'qq_legend.pdf', reference, thresholds)
    for ecosystem in ecosystems:
        plot_w1_curves(w1_df, ecosystem, attributes, thresholds, figures_dir / f'w1_vs_threshold_{ecosystem}.pdf')
    plot_w1_curves_legend(figures_dir / 'w1_vs_threshold_legend.pdf')

    manifest.update(analysis_config_id=cfg['id'], script='analysis/meta_updated_v2.py', seed=seed, n_boot=n_boot,
                    n_shuffle_samples=n_shuffle,
                    reference=reference, ecosystems=ecosystems, attributes=attributes, thresholds=thresholds,
                    min_n=min_n, outlier_adjust=sec['outlier_adjust'],
                    outlier_moments=None if moments is None else moments.to_dict('index'),
                    calibration_version=sec['calibration_version'], extraction_id=inputs['extraction_id'], **{k: sec[k] for k in input_keys})
    (out_dir / 'meta.json').write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
