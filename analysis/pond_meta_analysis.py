"""Pond meta-analysis: do confidence-filtered extractions reproduce the ground truth's distributions?

For each (ecosystem, attribute) cell and method (NTP, probe), extracted rows are
hard-filtered at each confidence threshold t, and the distribution is compared to a
reference (ground truth or valid extractions) with Q-Q plots and Wasserstein-1 (W1).
- threshold_mode ``value`` keeps confidence >= t; ``percentile`` drops the bottom
  fraction t of the cell. t = 0 is the unfiltered set.
- Samples are unweighted: Hazen quantiles, scipy W1, and scipy percentile-bootstrap CIs.
- Permutation control: shuffle the raw confidences within a cell (keeping each row's
  outlier factor) and recompute W1. A real W1 inside the shuffled range means the
  filter does no better than random rows of the same count.
- Log-scale attributes drop non-positive values from log Q-Q / log W1 only.

Outputs in analysis/results/meta/<config id>/: meta_stats.csv, survival.csv,
wasserstein.csv, qq_lines.csv, meta.json, and Q-Q and W1-vs-threshold figures.

Usage
-----
    python analysis/pond_meta_analysis.py analysis/analysis-configs/meta/<id>.yaml
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

from analysis.common.pond_meta import (
    ATTRIBUTES, DATASET, ECOSYSTEMS, LOG_SCALE_ATTRIBUTES, META_ROOT, METHOD_PROB_COL, METHODS, PAPER_RCPARAMS,
    QLEVELS, REFERENCE_AXIS_LABEL, STANDARD_UNITS, UNIT_CONVERSION,
    _attr_title, _axis_limits, _valid_range, load_data,
)
from analysis.common.outlier_weight import add_outlier_columns
from analysis.common.meta_inputs import SECTION_V2, load_meta_v2_config, resolve_meta_inputs

mpl.rcParams.update(PAPER_RCPARAMS)

# Non-threshold settings per reference. `valid` = extracted rows labelled valid
# (judge OR GT match). The unfiltered set is the t = 0 threshold setting.
BASE_SETTINGS = {
    'ground_truth': ['ground_truth', 'valid'],
    'valid':        ['ground_truth', 'valid'],
}
# Non-threshold lines drawn on every Q-Q panel, over the threshold lines.
QQ_BASE_LINES = {
    'ground_truth': ['valid'],
    'valid':        ['ground_truth'],
}
# Threshold lines are solid, stepped cool -> warm along THRESHOLD_CMAP.
THRESHOLD_CMAP = mpl.colormaps['coolwarm']
QQ_BASE_STYLE = {
    'ground_truth': dict(color='#2a7d3a', linestyle=(0, (6, 2)), linewidth=1.6),
    'valid':        dict(color='black', linestyle=':', linewidth=2.0),
}
QQ_BASE_LEGEND = {
    'ground_truth': 'Ground truth',
    'valid':        'Valid extracted (judge or GT match)',
}
# UNIT_CONVERSION plus pond schema units it lacks: mi^2, and ppm / ppb as mg/L / µg/L
# (dilute water). µg/cm^2 and molar chla stay unconvertible (likely mislabelled units).
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

# W1-vs-threshold curves: pastel tab10 (NTP blue, probe green); real solid, shuffled dotted.
PASTEL_WHITE_FRACTION = 0.35


def _pastel(color) -> tuple:
    """Lighten a color by mixing in PASTEL_WHITE_FRACTION white.

    Args:
        color: Any matplotlib color.

    Returns:
        RGB tuple.
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - PASTEL_WHITE_FRACTION) * rgb + PASTEL_WHITE_FRACTION)


_TAB10 = sns.color_palette('tab10', 10)
CURVE_STYLE = {
    'ntp':            dict(color=_pastel(_TAB10[0]), ls='-', lw=2.5, label='NTP'),
    'probe':          dict(color=_pastel(_TAB10[2]), ls='-', lw=2.5, label='Probe'),
    'ntp_shuffled':   dict(color=_pastel(_TAB10[0]), ls=':', lw=2.0, label='NTP (shuffled)'),
    'probe_shuffled': dict(color=_pastel(_TAB10[2]), ls=':', lw=2.0, label='Probe (shuffled)'),
}


THRESHOLD_MODES = ('value', 'percentile')
SETTING_INFIX = {'value': '_ge_', 'percentile': '_pct_'}
THRESHOLD_AXIS_LABEL = {'value': r'Confidence threshold $t$',
                        'percentile': r'Bottom fraction of confidence removed $q$'}


def threshold_setting(method: str, t: float, mode: str) -> str:
    """Setting name for a threshold, e.g. ``probe_ge_0.50`` or ``ntp_pct_0.25``.

    Args:
        method: ``"ntp"`` or ``"probe"``.
        t: Threshold.
        mode: ``"value"`` or ``"percentile"``.

    Returns:
        Setting name.
    """
    return f'{method}{SETTING_INFIX[mode]}{t:.2f}'


def parse_threshold_setting(setting: str):
    """Inverse of ``threshold_setting``.

    Args:
        setting: Setting name.

    Returns:
        ``(method, mode, t)``, or None for a non-threshold setting.
    """
    for mode, infix in SETTING_INFIX.items():
        method, sep, t = setting.partition(infix)
        if sep:
            assert method in METHODS, f'unknown setting {setting!r}'
            return method, mode, float(t)
    return None


def keep_mask(prob: np.ndarray, t: float, mode: str) -> np.ndarray:
    """Which rows of one cell survive threshold t.

    Value mode keeps prob >= t. Percentile mode keeps prob >= the lower t-quantile
    of ``prob``, so ties at the cutoff are kept.

    Args:
        prob: The cell's confidences.
        t: Threshold (or fraction to drop).
        mode: ``"value"`` or ``"percentile"``.

    Returns:
        Boolean keep mask.
    """
    assert prob.ndim == 1 and np.isfinite(prob).all(), 'confidence missing or misshapen'
    if mode == 'value':
        return prob >= t
    assert mode == 'percentile' and 0.0 <= t < 1.0, (mode, t)
    if prob.size == 0:
        return np.zeros(0, dtype=bool)
    return prob >= np.quantile(prob, t, method='lower')


def threshold_style(t: float, thresholds: list[float]) -> dict:
    """Line style for a threshold, coloured by its position in the grid (cool to warm).

    Args:
        t: Threshold (must be in ``thresholds``).
        thresholds: Full grid, starting at 0.

    Returns:
        matplotlib line kwargs.
    """
    assert thresholds[0] == 0.0 and len(thresholds) > 1, thresholds
    return dict(color=THRESHOLD_CMAP(thresholds.index(t) / (len(thresholds) - 1)), linestyle='-', linewidth=1.5)


def threshold_label(t: float) -> str:
    """LaTeX legend label "Confidence >= t".

    Args:
        t: Threshold.

    Returns:
        Label string.
    """
    return rf'Confidence $\geq {t:g}$'


def _rng(seed: int, ecosystem: str, attribute: str, code: int, t: float, log: bool) -> np.random.Generator:
    """Bootstrap RNG for one (cell, setting, threshold, scale).

    Keyed on canonical positions, so a cell's CI doesn't depend on what else the config selects.

    Args:
        seed: Global seed.
        ecosystem: Ecosystem (position in ECOSYSTEMS).
        attribute: Attribute (position in ATTRIBUTES).
        code: SETTING_CODES value.
        t: Threshold (in 1e-4 units in the key).
        log: Log-scale stream.

    Returns:
        Generator.
    """
    return np.random.default_rng([int(seed), ECOSYSTEMS.index(ecosystem), ATTRIBUTES.index(attribute),
                                  int(code), int(round(t * 10_000)), int(log)])


# ── Settings ────────────────────────────────────────────────────────────────

def cell_rows(gt_df: pd.DataFrame, ext_df: pd.DataFrame, ecosystem: str, attribute: str):
    """Ground-truth and extracted rows of one cell that have a converted value.

    Args:
        gt_df: Ground-truth rows.
        ext_df: Extracted rows.
        ecosystem: Ecosystem bucket.
        attribute: Attribute.

    Returns:
        ``(gt_rows, ext_rows)``.
    """
    g = gt_df[(gt_df['ecosystem_bucket'] == ecosystem) & (gt_df['attribute'] == attribute)]
    e = ext_df[(ext_df['ecosystem_bucket'] == ecosystem) & (ext_df['attribute'] == attribute)]
    return g.dropna(subset=['converted_value']), e.dropna(subset=['converted_value'])


def setting_rows(setting: str, gt: pd.DataFrame, ext: pd.DataFrame) -> pd.DataFrame:
    """Rows of one cell belonging to a setting.

    Args:
        setting: ``ground_truth``, ``extracted``, ``valid``, or a threshold setting.
        gt: The cell's ground-truth rows.
        ext: The cell's extracted rows.

    Returns:
        The selected rows.
    """
    if setting == 'ground_truth':
        return gt
    if setting == 'extracted':
        return ext
    if setting == 'valid':
        assert ext['label'].dtype == bool, ext['label'].dtype
        return ext[ext['label']]
    parsed = parse_threshold_setting(setting)
    assert parsed is not None, f'unknown setting {setting!r}'
    method, mode, t = parsed
    prob = ext[METHOD_PROB_COL[method]].to_numpy(dtype=float)
    assert prob.shape == (len(ext),), 'confidence misshapen'
    return ext[keep_mask(prob, t, mode)]


def all_settings(reference: str, thresholds: list[float], threshold_mode: str) -> list[str]:
    """Every setting compared for a reference: base settings, then each method x threshold.

    Args:
        reference: ``"ground_truth"`` or ``"valid"``.
        thresholds: Threshold grid.
        threshold_mode: ``"value"`` or ``"percentile"``.

    Returns:
        Setting names.
    """
    names = BASE_SETTINGS[reference] + [threshold_setting(m, t, threshold_mode) for m in METHODS for t in thresholds]
    assert len(set(names)) == len(names), f'threshold settings collide at 2 decimals: {thresholds}'
    return names


def _setting_meta(setting: str) -> dict:
    """Method / mode / threshold columns for a setting (blank and NaN if not thresholded).

    Args:
        setting: Setting name.

    Returns:
        Dict with ``method``, ``threshold_mode``, ``threshold``.
    """
    parsed = parse_threshold_setting(setting)
    if parsed is None:
        return dict(method='', threshold_mode='', threshold=np.nan)
    method, mode, t = parsed
    return dict(method=method, threshold_mode=mode, threshold=t)


# ── Stats table ─────────────────────────────────────────────────────────────

def summary_stats(x: np.ndarray) -> dict:
    """Mean, sample std and Hazen quartiles of a sample (NaN if empty).

    Args:
        x: 1-D finite values.

    Returns:
        Dict with ``mean``, ``std``, ``q1``, ``median``, ``q3``.
    """
    assert x.ndim == 1 and np.isfinite(x).all()
    if x.size == 0:
        return dict(mean=np.nan, std=np.nan, q1=np.nan, median=np.nan, q3=np.nan)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75], method='hazen')
    return dict(mean=float(np.mean(x)), std=float(np.std(x, ddof=1)) if x.size > 1 else np.nan,
                q1=float(q1), median=float(med), q3=float(q3))


def build_stats_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds, threshold_mode) -> pd.DataFrame:
    """Summary statistics per (ecosystem, attribute, setting).

    Args:
        gt_df: Ground-truth rows.
        ext_df: Extracted rows.
        reference: Reference setting.
        ecosystems: Ecosystems to include.
        attributes: Attributes to include.
        thresholds: Threshold grid.
        threshold_mode: ``"value"`` or ``"percentile"``.

    Returns:
        meta_stats.csv rows: n, n_docs, n_nonpos and ``summary_stats`` per setting.
    """
    rows = []
    for ecosystem in ecosystems:
        for attribute in attributes:
            gt, ext = cell_rows(gt_df, ext_df, ecosystem, attribute)
            for setting in all_settings(reference, thresholds, threshold_mode):
                sub = setting_rows(setting, gt, ext)
                x = sub['converted_value'].to_numpy(dtype=float)
                rows.append(dict(dataset=DATASET, ecosystem=ecosystem, attribute=attribute, setting=setting,
                                 **_setting_meta(setting), unit=STANDARD_UNITS[attribute],
                                 n=int(x.size), n_docs=int(sub['document_id'].nunique()),
                                 n_nonpos=int((x <= 0).sum()), **summary_stats(x)))
    return pd.DataFrame(rows)


# ── Survival table ──────────────────────────────────────────────────────────

def build_survival_table(stats_df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """How many rows and documents survive each threshold.

    Args:
        stats_df: Output of ``build_stats_table``.
        reference: Reference setting (for ``n_ref``).

    Returns:
        One row per (ecosystem, attribute, method, threshold) with ``n_ref``, ``n_ext``,
        ``n_docs_ext`` and their fractions of the t = 0 values.
    """
    ref = stats_df[stats_df['setting'] == reference].set_index(['ecosystem', 'attribute'])
    assert ref.index.is_unique and len(ref) > 0, 'reference setting missing or duplicated in the stats table'
    thr = stats_df[stats_df['method'].isin(METHODS)]
    out = thr.rename(columns={'n': 'n_ext', 'n_docs': 'n_docs_ext'})[
        ['dataset', 'ecosystem', 'attribute', 'method', 'threshold_mode', 'threshold', 'n_ext', 'n_docs_ext']].copy()
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
    """log10 of the positive values if ``log``, else ``x`` unchanged.

    Args:
        x: Values.
        log: Use log10.

    Returns:
        Scaled values (non-positive dropped when ``log``).
    """
    return np.log10(x[x > 0]) if log else x


def w1_with_ci(ref_x, ext_x, min_n: int, n_boot: int, rng: np.random.Generator, ci: float = 0.95) -> dict:
    """W1 between two samples with an unpaired percentile-bootstrap CI.

    Plug-in W1 is biased upward, so the CI shows spread and is not a test against 0.

    Args:
        ref_x: Reference sample.
        ext_x: Compared sample.
        min_n: Minimum size of each sample.
        n_boot: Bootstrap resamples.
        rng: Random generator.
        ci: Confidence level.

    Returns:
        Dict with ``w1``, ``w1_lo``, ``w1_hi``, ``w1_skip`` (NaN and a reason if too small).
    """
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
    """RNG for one confidence shuffle in one cell (distinct from the ``_rng`` streams).

    Args:
        seed: Global seed.
        ecosystem: Ecosystem.
        attribute: Attribute.
        method: ``"ntp"`` or ``"probe"``.
        sample: Shuffle index.

    Returns:
        Generator.
    """
    return np.random.default_rng([int(seed), ECOSYSTEMS.index(ecosystem), ATTRIBUTES.index(attribute),
                                  SHUFFLE_STREAM + SETTING_CODES[method], int(sample)])


def _summarize_shuffles(vals: np.ndarray, ci: float) -> dict:
    """Mean and percentile range of per-shuffle W1s, only if every shuffle produced one.

    Args:
        vals: One W1 per shuffle (NaN where skipped).
        ci: Range coverage.

    Returns:
        Dict with ``mean``, ``lo``, ``hi``, ``n_ok``, ``skip`` (NaN and a reason if any shuffle failed).
    """
    n_ok = int(np.isfinite(vals).sum())
    if n_ok < vals.size:
        return dict(mean=np.nan, lo=np.nan, hi=np.nan, n_ok=n_ok, skip='none_ok' if n_ok == 0 else 'partial_ok')
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(vals, [alpha, 1.0 - alpha])
    return dict(mean=float(vals.mean()), lo=float(lo), hi=float(hi), n_ok=n_ok, skip='')


def shuffled_w1(ref: np.ndarray, ext: pd.DataFrame, method: str, thresholds: list[float], threshold_mode: str,
                log_scale: bool, min_n: int, n_shuffle: int, seed: int, ecosystem: str, attribute: str, ci: float = 0.95) -> dict:
    """Permutation control for one (cell, method).

    Each shuffle permutes the raw confidences, reapplies each row's own outlier factor,
    applies every threshold, and takes W1 to ``ref``. Without outlier adjustment the
    shuffled subsets keep exactly the real row counts (asserted); with it, counts can
    differ, so the realized size is reported.

    Args:
        ref: Reference sample.
        ext: The cell's extracted rows.
        method: ``"ntp"`` or ``"probe"``.
        thresholds: Threshold grid.
        threshold_mode: ``"value"`` or ``"percentile"``.
        log_scale: Also compute log10 W1.
        min_n: Minimum sample size for W1.
        n_shuffle: Number of shuffles.
        seed: Global seed.
        ecosystem: Ecosystem (RNG key).
        attribute: Attribute (RNG key).
        ci: Range coverage.

    Returns:
        ``{t: {w1_shuffled_*, w1_shuffled_log_*, n_docs_ext_shuffled_mean, n_ext_shuffled_mean}}``;
        log fields are NaN / 'n/a' for linear-scale attributes.
    """
    col = METHOD_PROB_COL[method]
    raw = ext[f'{col}_raw'].to_numpy(dtype=float)
    factor = ext['outlier_factor'].to_numpy(dtype=float)
    assert np.isfinite(raw).all() and np.isfinite(factor).all() and ((factor > 0) & (factor <= 1)).all()
    prob = ext[col].to_numpy(dtype=float)
    assert np.array_equal(prob, raw * factor), f'{col} != {col}_raw * outlier_factor'
    exact = bool((factor == 1.0).all())
    x = ext['converted_value'].to_numpy(dtype=float)
    doc_codes, _ = pd.factorize(ext['document_id'])
    real_n = {t: int(keep_mask(prob, t, threshold_mode).sum()) for t in thresholds}
    ref_log = _scale(ref, True)
    w_raw = {t: np.full(n_shuffle, np.nan) for t in thresholds}
    lg = {t: np.full(n_shuffle, np.nan) for t in thresholds}
    n_docs = {t: np.zeros(n_shuffle) for t in thresholds}
    n_rows = {t: np.zeros(n_shuffle) for t in thresholds}
    for s in range(n_shuffle):
        p = _shuffle_rng(seed, ecosystem, attribute, method, s).permutation(raw) * factor
        for t in thresholds:
            keep = keep_mask(p, t, threshold_mode)
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


def build_w1_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds, threshold_mode,
                   min_n: int, n_boot: int, n_shuffle: int, seed: int) -> pd.DataFrame:
    """W1 from every compared setting to the reference, with CIs and the permutation control.

    Args:
        gt_df: Ground-truth rows.
        ext_df: Extracted rows.
        reference: Reference setting.
        ecosystems: Ecosystems to include.
        attributes: Attributes to include.
        thresholds: Threshold grid.
        threshold_mode: ``"value"`` or ``"percentile"``.
        min_n: Minimum sample size.
        n_boot: Bootstrap resamples.
        n_shuffle: Permutation-control shuffles.
        seed: Global seed.

    Returns:
        wasserstein.csv: one row per (ecosystem, attribute, setting) with raw and log W1,
        CIs, reference IQR/range, and shuffled-control columns (NaN for non-threshold settings).
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
            shuffled = {m: shuffled_w1(ref, ext, m, thresholds, threshold_mode, log_scale, min_n, n_shuffle, seed, ecosystem, attribute)
                        for m in METHODS}
            empty = {k: (np.nan if isinstance(v, float) else (0 if isinstance(v, int) else '')) for k, v in
                     shuffled[METHODS[0]][thresholds[0]].items()}
            for setting in all_settings(reference, thresholds, threshold_mode):
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
    """QLEVELS that both samples can interpolate (see ``_valid_range``).

    Args:
        n_ref: Reference sample size.
        n_ext: Compared sample size.

    Returns:
        Quantile levels.
    """
    lo_r, hi_r = _valid_range(n_ref)
    lo_e, hi_e = _valid_range(n_ext)
    lo, hi = max(lo_r, lo_e), min(hi_r, hi_e)
    return QLEVELS[(QLEVELS >= lo) & (QLEVELS <= hi)]


def qq_line(ref_x: np.ndarray, ext_x: np.ndarray, min_n: int):
    """Q-Q line of Hazen quantiles.

    Args:
        ref_x: Reference sample.
        ext_x: Compared sample.
        min_n: Minimum compared-sample size.

    Returns:
        ``(reference quantiles, extracted quantiles)``, or None if ``ext_x`` is too small.
    """
    if ext_x.size < min_n:
        return None
    levels = qq_levels(ref_x.size, ext_x.size)
    assert levels.size > 0
    return (np.quantile(ref_x, levels, method='hazen'), np.quantile(ext_x, levels, method='hazen'))


def reference_band(ref_x: np.ndarray, levels: np.ndarray, n_boot: int, rng: np.random.Generator, ci: float = 0.95):
    """Percentile-bootstrap band of the reference's own quantiles (its sampling noise).

    Args:
        ref_x: Reference sample.
        levels: Quantile levels.
        n_boot: Bootstrap resamples.
        rng: Random generator.
        ci: Confidence level.

    Returns:
        ``(lo, hi)`` arrays at ``levels``.
    """
    res = stats.bootstrap((ref_x,), lambda s, axis: np.quantile(s, levels, method='hazen', axis=axis),
                          vectorized=True, n_resamples=n_boot, confidence_level=ci, method='percentile',
                          random_state=rng)
    lo, hi = res.confidence_interval
    assert lo.shape == levels.shape and hi.shape == levels.shape
    return lo, hi


def plot_qq(gt_df, ext_df, reference, ecosystem, method, attributes, thresholds, threshold_mode, min_n, n_boot, seed,
            out_path: Path) -> list[dict]:
    """Save a Q-Q figure for one (ecosystem, method), one panel per attribute.

    Each panel shows one line per threshold plus the base lines, the diagonal, and
    the reference bootstrap band.

    Args:
        gt_df: Ground-truth rows.
        ext_df: Extracted rows.
        reference: Reference setting (x-axis).
        ecosystem: Ecosystem.
        method: ``"ntp"`` or ``"probe"``.
        attributes: Panel attributes.
        thresholds: Threshold grid.
        threshold_mode: ``"value"`` or ``"percentile"``.
        min_n: Minimum sample size per line.
        n_boot: Bootstrap resamples for the band.
        seed: Global seed.
        out_path: Figure path.

    Returns:
        One qq_lines.csv record per line.
    """
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

        lines = [(threshold_setting(method, t, threshold_mode), threshold_style(t, thresholds), 4) for t in thresholds]
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


def plot_qq_legend(out_path: Path, reference: str, thresholds: list[float], threshold_mode: str):
    """Save the Q-Q key: a discrete threshold colour bar plus base-line and band legend.

    Args:
        out_path: Figure path.
        reference: Reference setting.
        thresholds: Threshold grid.
        threshold_mode: ``"value"`` or ``"percentile"``.
    """
    colors = [threshold_style(t, thresholds)['color'] for t in thresholds]
    k = len(thresholds)
    fig = plt.figure(figsize=(7.0, 1.0))
    cax = fig.add_axes([0.04, 0.5, 0.46, 0.22])
    cmap = mcolors.ListedColormap(colors)
    sm = mpl.cm.ScalarMappable(norm=mcolors.BoundaryNorm(np.arange(k + 1) - 0.5, k), cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation='horizontal', ticks=np.arange(k))
    cb.ax.set_xticklabels([f'{t:g}' for t in thresholds], fontsize=9)
    cb.ax.tick_params(length=0)
    cb.set_label(THRESHOLD_AXIS_LABEL[threshold_mode], fontsize=10)
    handles = [Line2D([], [], label=QQ_BASE_LEGEND[s], **QQ_BASE_STYLE[s]) for s in QQ_BASE_LINES[reference]]
    handles.append(Patch(color='#888888', alpha=0.25, linewidth=0, label=f'{REFERENCE_AXIS_LABEL[reference]} 95% bootstrap'))
    fig.legend(handles=handles, loc='center left', bbox_to_anchor=(0.54, 0.5), fontsize=9, handlelength=2.6, frameon=False)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[meta_v2] wrote {out_path}")


# ── W1 vs threshold ─────────────────────────────────────────────────────────

def _curve(w1_df: pd.DataFrame, ecosystem: str, attribute: str, method: str, thresholds: list[float]) -> pd.DataFrame:
    """One method's threshold rows for a cell from wasserstein.csv, in grid order.

    Args:
        w1_df: wasserstein.csv frame.
        ecosystem: Ecosystem.
        attribute: Attribute.
        method: ``"ntp"`` or ``"probe"``.
        thresholds: Expected grid.

    Returns:
        The rows, sorted by threshold.
    """
    c = w1_df[(w1_df['ecosystem'] == ecosystem) & (w1_df['attribute'] == attribute) & (w1_df['method'] == method)]
    c = c.sort_values('threshold')
    assert np.allclose(c['threshold'].to_numpy(dtype=float), thresholds), (ecosystem, attribute, method)
    return c


def plot_w1_curves(w1_df: pd.DataFrame, ecosystem: str, attributes: list[str], thresholds: list[float],
                   threshold_mode: str, out_path: Path):
    """Save W1 vs threshold for one ecosystem, one panel per attribute.

    Plots raw-scale W1 divided by the reference's range (a per-cell constant, so curve
    shapes are unchanged). Real curves are solid with no CI drawn; shuffled controls
    are dotted with their percentile band. Skipped points are gaps.

    Args:
        w1_df: wasserstein.csv frame.
        ecosystem: Ecosystem.
        attributes: Panel attributes.
        thresholds: Threshold grid.
        threshold_mode: ``"value"`` or ``"percentile"``.
        out_path: Figure path.
    """
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
        ax.set_xlabel(THRESHOLD_AXIS_LABEL[threshold_mode], fontsize=11)
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
    """Save the W1-curve legend (real / shuffled per method, plus the band).

    Args:
        out_path: Figure path.
    """
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
    """CLI: load inputs, build every table and figure, and write meta.json."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('config', type=Path, help="analysis/analysis-configs/meta/<id>.yaml with params.meta_v2")
    args = parser.parse_args()

    cfg = load_meta_v2_config(args.config)
    sec = cfg['params'][SECTION_V2]
    ecosystems, attributes, reference = sec['ecosystems'], sec['attributes'], sec['reference']
    assert not set(ecosystems) - set(ECOSYSTEMS), f"ecosystems not in ECOSYSTEMS: {ecosystems}"
    assert not set(attributes) - set(ATTRIBUTES), f"attributes not in ATTRIBUTES: {attributes}"
    thresholds, min_n, n_boot, seed = sec['thresholds'], sec['min_n'], sec['n_boot'], cfg['seed']
    n_shuffle, threshold_mode = sec['n_shuffle_samples'], sec['threshold_mode']

    input_keys = ('calibration_config_id', 'rows', 'deduplication_config_id', 'confidence')
    inputs = resolve_meta_inputs(sec, DATASET)
    gt_df, ext_df, manifest = load_data(sec, inputs, unit_conversion=UNIT_CONVERSION_V2)
    # Real settings use ntp_prob / probe_prob (adjusted if outlier_adjust); the shuffled
    # control permutes the *_raw columns (see shuffled_w1).
    ext_df, moments = add_outlier_columns(ext_df, sec['outlier_adjust'])
    if moments is not None:
        print(f"[meta_v2] outlier_adjust: confidences x exp(-(x-mu)^2/2sigma^2)\n{moments}")

    out_dir = META_ROOT / cfg['id']
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    stats_df = build_stats_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds, threshold_mode)
    stats_df.to_csv(out_dir / 'meta_stats.csv', index=False)
    print(f"[meta_v2] wrote {out_dir / 'meta_stats.csv'}")

    survival_df = build_survival_table(stats_df, reference)
    survival_df.to_csv(out_dir / 'survival.csv', index=False)
    print(f"[meta_v2] wrote {out_dir / 'survival.csv'}")

    w1_df = build_w1_table(gt_df, ext_df, reference, ecosystems, attributes, thresholds, threshold_mode, min_n, n_boot, n_shuffle, seed)
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
                               threshold_mode, min_n, n_boot, seed, figures_dir / f'qq_{method}_{ecosystem}.pdf')
    pd.DataFrame(records).to_csv(out_dir / 'qq_lines.csv', index=False)
    plot_qq_legend(figures_dir / 'qq_legend.pdf', reference, thresholds, threshold_mode)
    for ecosystem in ecosystems:
        plot_w1_curves(w1_df, ecosystem, attributes, thresholds, threshold_mode, figures_dir / f'w1_vs_threshold_{ecosystem}.pdf')
    plot_w1_curves_legend(figures_dir / 'w1_vs_threshold_legend.pdf')

    manifest.update(analysis_config_id=cfg['id'], script='analysis/pond_meta_analysis.py', seed=seed, n_boot=n_boot,
                    n_shuffle_samples=n_shuffle,
                    reference=reference, ecosystems=ecosystems, attributes=attributes, thresholds=thresholds, threshold_mode=threshold_mode,
                    min_n=min_n, outlier_adjust=sec['outlier_adjust'],
                    outlier_moments=None if moments is None else moments.to_dict('index'),
                    calibration_version=sec['calibration_version'], extraction_id=inputs['extraction_id'], **{k: sec[k] for k in input_keys})
    (out_dir / 'meta.json').write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
