"""Unit tests for analysis/pond_meta_analysis.py (hard confidence thresholds, unweighted
library quantiles / scipy W1) and its config loader.

Hand-built fixture, one (pond, tn) cell -- tn is a LOG_SCALE attribute:
  - 30 ground-truth rows over 3 documents;
  - 40 extracted rows over 4 documents (10 each). The first 20 rows carry a positive
    calibration label and probe confidence 0.9; rows 20-29 have probe confidence
    exactly 0.5 (a tie at the 0.5 threshold); rows 30-39 have 0.1. NTP confidence is
    a constant 0.5. Row 39's value is 0 (allowed by PHYSICAL_BOUNDS, not log-plottable).

Known answers (stated before running):
  - probe_ge_0.00 / ntp_ge_0.00 are every extracted row (the unfiltered set): same stats, same W1;
  - probe_ge_0.50 keeps rows 0-29 (ties at t are kept), 3 documents; probe_ge_0.60 keeps rows 0-19;
  - ntp_ge_0.50 keeps all 40, ntp_ge_0.60 keeps none (NaN W1, skip ext_n<min_n);
  - under reference 'valid', probe_ge_0.60 is exactly the valid rows, so W1 = 0;
  - an extracted sample equal to the reference shifted by c has W1 = |c|;
  - the zero-valued row is counted in n_nonpos and excluded from the log W1 only.
"""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats

from analysis.common.meta_inputs import load_meta_v2_config
from analysis.pond_meta_analysis import (
    _summarize_draws, all_settings, build_stats_table, build_survival_table, build_w1_table, cell_rows, qq_line, setting_rows, w1_with_ci,
)

ECO, ATTR = 'pond', 'tn'
THRESHOLDS = [0.0, 0.5, 0.6]
SRC = ['ntp', 'probe']   # threshold_sources(False): the fixture has no outlier factor
MIN_N = 5


def fixture():
    rng = np.random.default_rng(0)
    gt = pd.DataFrame({
        'ecosystem_bucket': ECO, 'attribute': ATTR, 'document_id': np.repeat(['g0', 'g1', 'g2'], 10),
        'converted_value': rng.lognormal(6.0, 0.5, 30),
    })
    vals = np.r_[rng.lognormal(6.1, 0.5, 20), rng.lognormal(8.0, 0.5, 19), 0.0]
    probe = np.r_[np.full(20, 0.9), np.full(10, 0.5), np.full(10, 0.1)]
    label = np.r_[np.ones(20, bool), np.zeros(20, bool)]
    ext = pd.DataFrame({
        'ecosystem_bucket': ECO, 'attribute': ATTR, 'document_id': np.repeat(['e0', 'e1', 'e2', 'e3'], 10),
        'converted_value': vals, 'label': label,
        'probe_prob': probe, 'ntp_prob': np.full(40, 0.5),
    })
    from analysis.common.outlier_weight import add_outlier_columns
    return gt, add_outlier_columns(ext, False)[0]


def row(df, setting):
    r = df[df['setting'] == setting]
    assert len(r) == 1
    return r.iloc[0]


def test_threshold_filter_keeps_ties_and_t0_is_everything():
    gt, ext = fixture()
    g, e = cell_rows(gt, ext, ECO, ATTR)
    pd.testing.assert_frame_equal(setting_rows('probe_ge_0.00', g, e), setting_rows('extracted', g, e))
    assert list(setting_rows('probe_ge_0.50', g, e).index) == list(range(30))
    assert list(setting_rows('probe_ge_0.60', g, e).index) == list(range(20))
    assert len(setting_rows('ntp_ge_0.50', g, e)) == 40
    assert len(setting_rows('ntp_ge_0.60', g, e)) == 0


def test_stats_table_known_answers():
    gt, ext = fixture()
    df = build_stats_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'value', SRC)
    assert list(df['setting']) == all_settings('ground_truth', THRESHOLDS, 'value', SRC)
    cols = ['n', 'n_docs', 'n_nonpos', 'mean', 'std', 'q1', 'median', 'q3']
    assert 'extracted' not in set(df['setting'])  # t = 0 is the unfiltered set
    pd.testing.assert_series_equal(row(df, 'probe_ge_0.00')[cols], row(df, 'ntp_ge_0.00')[cols], check_names=False)
    allx = ext['converted_value'].to_numpy()
    assert row(df, 'probe_ge_0.00')['n'] == 40 and row(df, 'probe_ge_0.00')['median'] == np.quantile(allx, 0.5, method='hazen')
    r = row(df, 'probe_ge_0.50')
    assert (r['n'], r['n_docs'], r['n_nonpos'], r['threshold'], r['method']) == (30, 3, 0, 0.5, 'probe')
    assert row(df, 'probe_ge_0.00')['n_nonpos'] == 1
    x = ext['converted_value'].to_numpy()[:20]
    r = row(df, 'probe_ge_0.60')
    assert r['median'] == np.quantile(x, 0.5, method='hazen') and r['std'] == np.std(x, ddof=1)
    assert row(df, 'ntp_ge_0.60')['n'] == 0 and np.isnan(row(df, 'ntp_ge_0.60')['median'])


def test_survival_table_known_answers():
    gt, ext = fixture()
    stats_df = build_stats_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'value', SRC)
    sv = build_survival_table(stats_df, 'ground_truth')
    assert len(sv) == 2 * len(THRESHOLDS) and (sv['n_ref'] == 30).all()   # 30 GT rows; 2 methods x 3 thresholds
    g = lambda m, t: sv[(sv['method'] == m) & (sv['threshold'] == t)].iloc[0]
    # probe: 40 rows / 4 docs at t=0; 30 / 3 at 0.5 (ties kept); 20 / 2 at 0.6.  ntp: all 40 at 0.5, none at 0.6.
    assert [(g('probe', t)['n_ext'], g('probe', t)['n_docs_ext']) for t in THRESHOLDS] == [(40, 4), (30, 3), (20, 2)]
    assert [g('probe', t)['frac_rows_vs_t0'] for t in THRESHOLDS] == [1.0, 0.75, 0.5]
    assert [g('probe', t)['frac_docs_vs_t0'] for t in THRESHOLDS] == [1.0, 0.75, 0.5]
    assert [g('ntp', t)['n_ext'] for t in THRESHOLDS] == [40, 40, 0] and g('ntp', 0.6)['frac_rows_vs_t0'] == 0.0
    # it agrees with the W1 table's counts
    w1 = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'value', SRC, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    for _, r in w1[w1['method'].isin(['ntp', 'probe'])].iterrows():
        s = g(r['method'], r['threshold'])
        assert (s['n_ext'], s['n_docs_ext'], s['n_ref']) == (r['n_ext'], r['n_docs_ext'], r['n_ref'])


def test_w1_table_known_answers():
    gt, ext = fixture()
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'value', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    assert 'ground_truth' not in set(df['setting'])
    ref = gt['converted_value'].to_numpy()
    assert 'extracted' not in set(df['setting'])
    e = row(df, 'probe_ge_0.00')
    assert e['w1'] == stats.wasserstein_distance(ref, ext['converted_value'].to_numpy())
    assert e['w1_lo'] <= e['w1'] <= e['w1_hi']
    # log W1 drops the zero row, raw W1 keeps it
    pos = ext['converted_value'].to_numpy()[:39]
    assert np.isclose(e['w1_log'], stats.wasserstein_distance(np.log10(ref), np.log10(pos)))
    # same rows -> same point estimates; the CIs differ (each setting has its own bootstrap stream)
    for c in ['n_ext', 'w1', 'w1_log']:
        assert row(df, 'ntp_ge_0.00')[c] == e[c], c
    # value mode has no single subset size, so no random baseline on any row
    assert df['w1_random_mean'].isna().all() and (df['w1_random_skip'] == 'n/a').all()
    assert (e['n_ext_log'], e['n_ref_log']) == (39, 30)
    n = row(df, 'ntp_ge_0.60')
    assert n['n_ext'] == 0 and np.isnan(n['w1']) and n['w1_skip'] == 'ext_n<min_n'
    # the valid set (rows 0-19) is a compared setting under the GT reference
    v = row(df, 'valid')
    assert v['n_ext'] == 20 and v['w1'] == stats.wasserstein_distance(ref, ext['converted_value'].to_numpy()[:20])
    # the high-confidence rows sit closer to GT than the unfiltered set
    assert row(df, 'probe_ge_0.60')['w1'] < e['w1']


def _plotted(tmp_path, monkeypatch, df, attributes, sources, scale, mode='percentile'):
    import matplotlib.pyplot as plt
    import analysis.pond_meta_analysis as m
    saved = {}
    monkeypatch.setattr(plt, 'close', lambda fig: saved.setdefault('fig', fig))
    m.plot_w1_curves(df, ECO, attributes, THRESHOLDS, mode, sources, scale, tmp_path / 'w1.pdf')
    return saved['fig']


def test_ref_stats_known_answer_and_w1_curve_plot_is_unnormalized(tmp_path, monkeypatch):
    gt, ext = fixture()
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    q1, q3 = np.quantile(gt['converted_value'].to_numpy(), [0.25, 0.75], method='hazen')
    assert (df['ref_iqr'] == q3 - q1).all()   # one value per cell, the reference's Hazen IQR in raw units
    assert (df['ref_range'] == gt['converted_value'].max() - gt['converted_value'].min()).all()   # ... and its max - min

    p = df[df['method'] == 'probe'].sort_values('threshold')
    for scale, col in (('raw', 'w1'), ('native', 'w1_log')):
        lines = {l.get_label(): l for l in _plotted(tmp_path, monkeypatch, df, [ATTR], SRC, scale).axes[0].lines}
        np.testing.assert_allclose(lines['Probe'].get_ydata(), p[col].to_numpy(), equal_nan=True)
        np.testing.assert_allclose(lines['Random'].get_ydata(), p[col.replace('w1', 'w1_random') + '_mean'].to_numpy(),
                                   equal_nan=True)
        assert not any('shuffled' in k for k in lines)


def test_one_grey_dotted_random_line_and_none_in_value_mode(tmp_path, monkeypatch):
    import matplotlib.colors as mcolors
    from analysis.pond_meta_analysis import CURVE_STYLE, PASTEL_WHITE_FRACTION
    gt, ext = fixture()
    pct = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    lines = _plotted(tmp_path, monkeypatch, pct, [ATTR], SRC, 'native').axes[0].lines
    rnd = [l for l in lines if l.get_label() == 'Random']
    assert len(rnd) == 1 and rnd[0].get_linestyle() == ':'
    grey = np.array(mcolors.to_rgb('#7f7f7f'))   # tab10 grey
    np.testing.assert_allclose(CURVE_STYLE['random']['color'], (1 - PASTEL_WHITE_FRACTION) * grey + PASTEL_WHITE_FRACTION, atol=1e-3)
    val = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'value', SRC, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    assert 'Random' not in [l.get_label() for l in _plotted(tmp_path, monkeypatch, val, [ATTR], SRC, 'native', 'value').axes[0].lines]


def test_valid_line_is_flat_at_the_valid_w1_and_only_against_gt(tmp_path, monkeypatch):
    gt, ext = fixture()
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    ref = gt['converted_value'].to_numpy()
    for scale, col in (('raw', 'w1'), ('native', 'w1_log')):
        lines = [l for l in _plotted(tmp_path, monkeypatch, df, [ATTR], SRC, scale).axes[0].lines
                 if l.get_label().startswith('Valid')]
        assert len(lines) == 1
        np.testing.assert_array_equal(lines[0].get_ydata(), np.full(len(THRESHOLDS), row(df, 'valid')[col]))
    # the valid rows are rows 0-19 of the fixture
    assert row(df, 'valid')['w1'] == stats.wasserstein_distance(ref, ext['converted_value'].to_numpy()[:20])
    against_valid = build_w1_table(gt, ext, 'valid', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    labels = [l.get_label() for l in _plotted(tmp_path, monkeypatch, against_valid, [ATTR], SRC, 'native').axes[0].lines]
    assert not any(lab.startswith('Valid') for lab in labels)


def test_w1_curve_ylabel_only_on_leftmost_panel(tmp_path, monkeypatch):
    gt, ext = fixture()
    gt2, ext2 = gt.assign(attribute='tp'), ext.assign(attribute='tp')
    df = build_w1_table(pd.concat([gt, gt2]), pd.concat([ext, ext2]), 'ground_truth', [ECO], [ATTR, 'tp'],
                        THRESHOLDS, 'value', SRC, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    left, right = _plotted(tmp_path, monkeypatch, df, [ATTR, 'tp'], SRC, 'native').axes
    assert left.get_ylabel() == '$W_1$' and right.get_ylabel() == ''
    assert left.get_xlabel() == right.get_xlabel() == 'Filter Level'   # x labels stay on every panel


# ── confidence-only threshold sources ──

def factor_fixture():
    """The base fixture with a hand-set outlier factor: rows 0-9 get 1.0, 10-19 get 0.8,
    20-29 get 0.5, 30-39 get 0.0 (an underflowed outlier). Confidences = raw x factor."""
    gt, ext = fixture()
    ext = ext.copy()
    f = np.repeat([1.0, 0.8, 0.5, 0.0], 10)
    ext['outlier_factor'] = f
    for c in ('ntp_prob', 'probe_prob'):
        ext[c] = ext[f'{c}_raw'] * f
    return gt, ext


def test_confidence_only_settings_and_rows():
    from analysis.pond_meta_analysis import SETTING_CODES, threshold_sources
    assert threshold_sources(False) == ['ntp', 'probe']
    assert threshold_sources(True) == ['ntp', 'probe', 'ntp_conf', 'probe_conf']
    assert not any(s.startswith('factor') for s in all_settings('ground_truth', THRESHOLDS, 'value', threshold_sources(True)))
    gt, ext = factor_fixture()
    g, e = cell_rows(gt, ext, ECO, ATTR)
    # value mode on probe x factor: 0.9*1, 0.9*0.8, 0.5*0.5, 0.1*0 -> >= 0.6 keeps rows 0-19
    assert list(setting_rows('probe_ge_0.60', g, e).index) == list(range(20))
    # confidence only ignores the factor: probe_raw 0.9 / 0.5 / 0.1 -> >= 0.5 keeps rows 0-29,
    # where probe x factor (0.9, 0.72, 0.25, 0) keeps rows 0-19
    assert list(setting_rows('probe_conf_ge_0.50', g, e).index) == list(range(30))
    assert list(setting_rows('probe_ge_0.50', g, e).index) == list(range(20))
    assert len(setting_rows('ntp_conf_ge_0.50', g, e)) == 40   # constant raw NTP 0.5
    assert len(set(SETTING_CODES.values())) == len(SETTING_CODES)   # distinct bootstrap streams
    assert (SETTING_CODES['ntp_conf'], SETTING_CODES['probe_conf']) == (7, 8)   # unchanged by retiring factor-only (6)


def test_confidence_only_curves_drawn_with_factor(tmp_path, monkeypatch):
    from analysis.pond_meta_analysis import threshold_sources
    gt, ext = factor_fixture()
    src = threshold_sources(True)
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', src, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    lines = {l.get_label(): l for l in _plotted(tmp_path, monkeypatch, df, [ATTR], src, 'native').axes[0].lines}
    for m, label in (('ntp', 'NTP (confidence only)'), ('probe', 'Probe (confidence only)')):
        c = df[df['method'] == f'{m}_conf'].sort_values('threshold')
        np.testing.assert_allclose(lines[label].get_ydata(), c['w1_log'].to_numpy(), equal_nan=True)
    plain = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    assert not any('confidence only' in l.get_label() for l in _plotted(tmp_path, monkeypatch, plain, [ATTR], SRC, 'native').axes[0].lines)


def test_threshold_rows_share_the_random_baseline():
    from analysis.pond_meta_analysis import threshold_sources
    gt, ext = factor_fixture()
    src = threshold_sources(True)
    # percentile mode: one baseline per (cell, t), copied onto every source's row; none on base settings
    pct = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', src, MIN_N, n_boot=20, n_shuffle=5, seed=0)
    cols = [c for c in pct.columns if 'random' in c]
    for t in THRESHOLDS:
        rows = pct[pct['threshold'] == t][cols]
        assert len(rows) == len(src) and (rows.nunique(dropna=False) == 1).all()
    assert row(pct, 'valid')['w1_random_skip'] == 'n/a'
    # tied raw probe scores (0.1 / 0.5 / 0.9) keep more rows than the quantile size: reported, not forced equal
    r = row(pct, 'probe_conf_pct_0.50')
    assert (r['n_ext'], r['n_ext_random']) == (30, 21)
    sv = build_survival_table(build_stats_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'value', src), 'ground_truth')
    assert [int(sv[(sv['method'] == 'probe_conf') & (sv['threshold'] == t)]['n_ext'].iloc[0]) for t in THRESHOLDS] == [40, 30, 20]
    assert set(sv['method']) == set(src)


def _ten_x_cell(attribute):
    """ext = 10 x ref plus one zero value, in one (pond, attribute) cell."""
    ref = np.array([1.0, 2.0, 3.0, 5.0, 7.0, 8.0])
    gt = pd.DataFrame({'ecosystem_bucket': ECO, 'attribute': attribute, 'document_id': 'g', 'converted_value': ref})
    ext = pd.DataFrame({'ecosystem_bucket': ECO, 'attribute': attribute, 'document_id': 'e',
                        'converted_value': np.r_[10 * ref, 0.0], 'label': True,
                        'probe_prob': 0.5, 'ntp_prob': 0.5})
    from analysis.common.outlier_weight import add_outlier_columns
    return gt, add_outlier_columns(ext, False)[0]


def test_log_w1_only_for_log_scale_attributes():
    """tn (LOG_SCALE): ext = 10 x ref gives log10 W1 = 1 exactly, the zero value dropped.
    ph (already a log scale) gets no log W1 at all, nor does its random baseline."""
    gt, ext = _ten_x_cell('tn')
    r = row(build_w1_table(gt, ext, 'ground_truth', [ECO], ['tn'], [0.0], 'percentile', SRC, MIN_N,
                           n_boot=20, n_shuffle=5, seed=0), 'probe_pct_0.00')
    assert (r['n_ext'], r['n_ext_log'], r['n_ref_log']) == (7, 6, 6)
    assert np.isclose(r['w1_log'], 1.0) and r['w1_log_skip'] == ''
    assert np.isclose(r['w1_random_log_mean'], 1.0)   # at t = 0 every draw is the whole cell
    gt, ext = _ten_x_cell('ph')
    r = row(build_w1_table(gt, ext, 'ground_truth', [ECO], ['ph'], [0.0], 'percentile', SRC, MIN_N,
                           n_boot=20, n_shuffle=5, seed=0), 'probe_pct_0.00')
    assert np.isnan(r['w1_log']) and r['w1_log_skip'] == 'n/a' and np.isnan(r['n_ext_log'])
    assert np.isnan(r['w1_random_log_mean']) and r['w1_random_log_skip'] == 'n/a'
    assert r['w1'] == stats.wasserstein_distance(gt['converted_value'], ext['converted_value'])   # raw W1 kept


def test_native_scale_plots_log_only_for_log_scale_attributes(tmp_path, monkeypatch):
    from analysis.pond_meta_analysis import LOG_TITLE_MARK
    cells = [_ten_x_cell(a) for a in ('tn', 'ph')]
    gt, ext = pd.concat([c[0] for c in cells]), pd.concat([c[1] for c in cells], ignore_index=True)
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], ['tn', 'ph'], [0.0, 0.5], 'percentile', SRC, MIN_N,
                        n_boot=20, n_shuffle=5, seed=0)
    import matplotlib.pyplot as plt
    import analysis.pond_meta_analysis as m
    for scale, tn_col in (('native', 'w1_log'), ('raw', 'w1')):
        saved = {}
        monkeypatch.setattr(plt, 'close', lambda fig: saved.setdefault('fig', fig))
        m.plot_w1_curves(df, ECO, ['tn', 'ph'], [0.0, 0.5], 'percentile', SRC, scale, tmp_path / 'w1.pdf')
        tn_ax, ph_ax = saved['fig'].axes
        for ax, attr, col in ((tn_ax, 'tn', tn_col), (ph_ax, 'ph', 'w1')):
            probe = next(l for l in ax.lines if l.get_label() == 'Probe')
            p = df[(df['attribute'] == attr) & (df['method'] == 'probe')].sort_values('threshold')
            np.testing.assert_allclose(probe.get_ydata(), p[col].to_numpy())
        assert tn_ax.get_title().endswith(LOG_TITLE_MARK) == (scale == 'native')
        assert not ph_ax.get_title().endswith(LOG_TITLE_MARK)
        assert tn_ax.get_ylabel() == '$W_1$'


def test_qq_legend_has_a_threshold_colorbar(tmp_path, monkeypatch):
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import analysis.pond_meta_analysis as m
    th = [0.0, 0.25, 0.5, 0.75]
    saved = {}
    monkeypatch.setattr(plt, 'close', lambda fig: saved.setdefault('fig', fig))
    m.plot_qq_legend(tmp_path / 'leg.pdf', 'ground_truth', th)
    fig = saved['fig']
    cax = fig.axes[0]
    assert [t.get_text() for t in cax.get_xticklabels()] == ['0', '0.25', '0.5', '0.75']
    assert cax.get_xlabel() == 'Filter Level'
    # one colorbar cell per threshold, in exactly the colors the Q-Q lines use
    mesh, = [c for c in cax.collections if type(c).__name__ == 'QuadMesh']   # (the other collection is the dividers)
    cb_colors = [mcolors.to_hex(c) for c in mesh.cmap(mesh.norm(np.arange(len(th))))]
    assert cb_colors == [mcolors.to_hex(m.threshold_style(t, th)['color']) for t in th]
    texts = [t.get_text() for t in fig.legends[0].get_texts()]
    assert not any('Confidence' in s for s in texts)   # thresholds are no longer separate legend entries
    assert any('Valid' in s for s in texts) and any('bootstrap' in s for s in texts)


def test_valid_reference_perfect_threshold_gives_zero():
    gt, ext = fixture()
    df = build_w1_table(gt, ext, 'valid', [ECO], [ATTR], THRESHOLDS, 'value', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    assert set(df['setting']) == set(all_settings('valid', THRESHOLDS, 'value', SRC)) - {'valid'}
    assert row(df, 'probe_ge_0.60')['w1'] == 0.0
    assert row(df, 'ground_truth')['n_ext'] == 30


def ranked_fixture():
    """The base fixture with distinct probe scores, highest on the 20 GT-like rows: no ties,
    so every percentile cut keeps exactly the quantile size."""
    gt, ext = fixture()
    return gt, ext.assign(probe_prob=np.linspace(1.0, 0.01, 40), probe_prob_raw=np.linspace(1.0, 0.01, 40))


def test_random_baseline_band_and_docs():
    gt, ext = ranked_fixture()
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    ref, x = gt['converted_value'].to_numpy(), ext['converted_value'].to_numpy()
    # t = 0: every draw is the whole cell, so the baseline collapses onto the real value
    e = row(df, 'probe_pct_0.00')
    for sc in ('', '_log'):
        np.testing.assert_allclose([e[f'w1_random{sc}_{k}'] for k in ('mean', 'lo', 'hi')], e[f'w1{sc}'], rtol=1e-12)
        assert e[f'w1_random{sc}_n_ok'] == 30 and e[f'w1_random{sc}_skip'] == ''
    # t = 0.5 keeps 40 - floor(0.5 * 39) = 21 rows: the 20 GT-like rows plus row 20
    r = row(df, 'probe_pct_0.50')
    assert r['n_ext'] == r['n_ext_random'] == 21
    assert r['w1'] == stats.wasserstein_distance(ref, x[:21])
    assert r['w1_random_lo'] <= r['w1_random_mean'] <= r['w1_random_hi'] and r['w1'] < r['w1_random_lo']
    # the real subset sits in 3 documents; random 21-of-40 subsets spread over more
    assert r['n_docs_ext'] == 3 and r['n_docs_ext_random_mean'] > 3


def test_quantile_size_matches_keep_mask_without_ties():
    from analysis.pond_meta_analysis import keep_mask, quantile_size
    for n in (1, 2, 7, 40, 151, 298):
        p = np.random.default_rng(n).permutation(n) / n
        for t in (0.0, 0.1, 0.25, 0.3, 0.5, 0.6, 0.9):
            assert quantile_size(n, t) == keep_mask(p, t, 'percentile').sum(), (n, t)
    assert quantile_size(0, 0.5) == 0


def test_random_baseline_ignores_scores():
    """Same rows, different scores (and outlier factors): identical baseline."""
    from analysis.pond_meta_analysis import random_w1
    gt, ext = ranked_fixture()
    ref = gt['converted_value'].to_numpy()
    a = random_w1(ref, ext, [0.0, 0.5, 0.6], True, MIN_N, 20, 0, ECO, ATTR)
    other = ext.assign(probe_prob=0.5, ntp_prob=np.linspace(0.01, 1.0, 40), outlier_factor=0.3)
    assert random_w1(ref, other, [0.0, 0.5, 0.6], True, MIN_N, 20, 0, ECO, ATTR) == a


def test_summarize_draws_refuses_partial_sets():
    assert _summarize_draws(np.array([1.0, 2.0, 3.0]), 0.95)['skip'] == ''
    p = _summarize_draws(np.array([1.0, np.nan]), 0.95)
    assert np.isnan(p['mean']) and p['skip'] == 'partial_ok' and p['n_ok'] == 1
    assert _summarize_draws(np.array([np.nan, np.nan]), 0.95)['skip'] == 'none_ok'


def test_seed_determinism_and_seed_dependence():
    gt, ext = ranked_fixture()
    a = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    b = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    c = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, 'percentile', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a['w1_lo'].dropna(), c['w1_lo'].dropna())
    assert row(a, 'probe_pct_0.50')['w1_random_mean'] != row(c, 'probe_pct_0.50')['w1_random_mean']
    # a cell's CI and baseline do not depend on the threshold grid they were computed with
    d = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], [0.6], 'percentile', SRC, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    for c in ('w1_lo', 'w1_random_mean', 'w1_random_lo', 'w1_random_log_hi'):
        assert row(d, 'probe_pct_0.60')[c] == row(a, 'probe_pct_0.60')[c], c


def test_w1_shift_known_answer_and_min_n():
    x = np.random.default_rng(3).normal(size=40)
    r = w1_with_ci(x, x + 2.5, MIN_N, n_boot=50, rng=np.random.default_rng(0))
    assert np.isclose(r['w1'], 2.5)
    assert np.isnan(w1_with_ci(x[:4], x, MIN_N, 50, np.random.default_rng(0))['w1'])


def test_qq_line_identical_samples_is_diagonal_and_min_n():
    x = np.random.default_rng(4).normal(size=30)
    rq, eq = qq_line(x, x.copy(), MIN_N)
    np.testing.assert_array_equal(rq, eq)
    assert qq_line(x, x[:4], MIN_N) is None


# ── config validation ────────────────────────────────────────────────────────

GOOD = {
    "id": "t", "project": "scholarlm", "description": "d", "seed": 0,
    "params": {"meta_v2": {
        "calibration_config_id": "cal", "calibration_version": "v4", "rows": "final",
        "deduplication_config_id": None, "confidence": None, "n_boot": 10, "reference": "valid",
        "ecosystems": ["pond"], "attributes": ["tn", "tp"], "qq_attributes": ["tn"],
        "thresholds": [0.0, 0.25, 0.5, 0.75], "min_n": 5, "n_shuffle_samples": 10, "outlier_adjust": False, "threshold_mode": "value",
        "w1_curve_scale": "native",
    }},
}


def write(tmp_path, cfg):
    p = tmp_path / "meta" / "t.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_good_config_loads(tmp_path):
    assert load_meta_v2_config(write(tmp_path, GOOD))["params"]["meta_v2"]["thresholds"] == [0.0, 0.25, 0.5, 0.75]


@pytest.mark.parametrize("mutate", [
    lambda m: m.pop("thresholds"),
    lambda m: m.pop("min_n"),
    lambda m: m.pop("calibration_version"),     # no default loader
    lambda m: m.update(calibration_version="v5"),
    lambda m: m.update(poster=False),          # v1-only key
    lambda m: m.update(thresholds=[]),
    lambda m: m.update(thresholds=[0.25, 0.5]),  # must start at 0 (the unfiltered set)
    lambda m: m.update(thresholds=[0.5, 0.2]),  # not increasing
    lambda m: m.update(thresholds=[0.5, 0.5]),
    lambda m: m.update(thresholds=[0.5, 1.0]),  # >= 1 keeps nothing
    lambda m: m.update(thresholds=[-0.1, 0.0]),
    lambda m: m.update(thresholds=[0.0, True]),
    lambda m: m.update(min_n=0),
    lambda m: m.pop("outlier_adjust"),
    lambda m: m.pop("threshold_mode"),
    lambda m: m.update(threshold_mode="quantile"),
    lambda m: m.update(outlier_adjust=1),
    lambda m: m.pop("n_shuffle_samples"),
    lambda m: m.update(n_shuffle_samples=0),
    lambda m: m.update(n_shuffle_samples=True),
    lambda m: m.update(min_n=True),
    lambda m: m.update(reference="gt"),         # shared check still applies
    lambda m: m.update(qq_attributes=["ph"]),
    lambda m: m.pop("w1_curve_scale"),          # no default scale
    lambda m: m.update(w1_curve_scale="log10"),
    lambda m: m.update(w1_curve_scale="log"),   # replaced by "native"
])
def test_bad_config_raises(tmp_path, mutate):
    cfg = copy.deepcopy(GOOD)
    mutate(cfg["params"]["meta_v2"])
    with pytest.raises((ValueError, KeyError)):
        load_meta_v2_config(write(tmp_path, cfg))


def test_v1_section_rejected(tmp_path):
    cfg = copy.deepcopy(GOOD)
    cfg["params"] = {"meta": cfg["params"]["meta_v2"]}
    with pytest.raises(ValueError):
        load_meta_v2_config(write(tmp_path, cfg))


def test_threshold_styles():
    from analysis.pond_meta_analysis import QQ_BASE_STYLE, THRESHOLD_CMAP, threshold_label, threshold_style
    import matplotlib.colors as mcolors
    assert THRESHOLD_CMAP.name == 'coolwarm'
    th = [0.0, 0.25, 0.5, 0.75]
    styles = [threshold_style(t, th) for t in th]
    # t = 0 is not special: every threshold line is solid, evenly spaced along the colormap
    assert all(s['linestyle'] == '-' for s in styles) and len({s['linewidth'] for s in styles}) == 1
    np.testing.assert_allclose([s['color'] for s in styles], [THRESHOLD_CMAP(f) for f in (0.0, 1 / 3, 2 / 3, 1.0)])
    assert len({mcolors.to_hex(s['color']) for s in styles}) == 4
    assert threshold_label(0.0) == r'Confidence $\geq 0$' and threshold_label(0.25) == r'Confidence $\geq 0.25$'
    # valid: black, dotted
    assert mcolors.to_hex(QQ_BASE_STYLE['valid']['color']) == '#000000' and QQ_BASE_STYLE['valid']['linestyle'] == ':'


def test_unit_conversion_v2_additions_only():
    """v2's table is v1's plus mi^2 and ppm / ppb for tn / tp / chla; v1 is untouched,
    and µg/cm^2 and molar chla stay unconvertible in both."""
    from analysis.common.pond_meta import UNIT_CONVERSION, convert_units
    from analysis.pond_meta_analysis import UNIT_CONVERSION_V2
    rows = pd.DataFrame({
        'attribute': ['surface_area', 'tn', 'tp', 'chla', 'chla', 'chla', 'chla', 'tn'],
        'units':     ['mi^2',         'ppm', 'ppb', 'ppm', 'µg/cm^2', 'µmol/L', 'μmol/L', 'mg/L'],
        'v':         [0.1,            0.5,  40.0,  0.01,  3.0,       1.0,      1.0,      0.5],  # 0.1 mi^2 is under PHYSICAL_BOUNDS' 1e6 m^2
    })
    v2 = convert_units(rows, UNIT_CONVERSION_V2, value_col='v')['converted_value'].to_numpy()
    v1 = convert_units(rows, UNIT_CONVERSION, value_col='v')['converted_value'].to_numpy()
    np.testing.assert_allclose(v2[:4], [0.1 * 2589988.110336, 500.0, 40.0, 10.0])
    assert np.isnan(v2[4:7]).all() and v2[7] == 500.0
    assert np.isnan(v1[:7]).all() and v1[7] == 500.0
    for a, m in UNIT_CONVERSION.items():
        assert all(UNIT_CONVERSION_V2[a][u] == f for u, f in m.items())
    with pytest.raises(AssertionError, match='unit_conversion attributes'):
        convert_units(rows, {'tn': {}}, value_col='v')


# ── percentile threshold mode ──

def test_keep_mask_percentile_hand_checked():
    from analysis.pond_meta_analysis import keep_mask
    p = np.array([0.1] * 10 + [0.5] * 10 + [0.9] * 20)   # n = 40, sorted
    # np.quantile(method='lower') index = floor(q * 39)
    assert keep_mask(p, 0.0, 'percentile').sum() == 40                 # t = 0 keeps everything
    assert keep_mask(p, 0.25, 'percentile').sum() == 40                # cutoff 0.1 is a tie: all kept
    assert keep_mask(p, 0.3, 'percentile').sum() == 30                 # cutoff 0.5
    assert keep_mask(p, 0.6, 'percentile').sum() == 20                 # cutoff 0.9
    u = np.arange(100) / 100                                           # no ties: removes floor(q * 99) rows
    for q, kept in ((0.0, 100), (0.1, 91), (0.2, 81), (0.5, 51)):
        assert keep_mask(u, q, 'percentile').sum() == kept
    assert keep_mask(np.array([]), 0.5, 'percentile').size == 0
    with pytest.raises(AssertionError):
        keep_mask(np.array([0.5, np.nan]), 0.1, 'percentile')


def test_percentile_settings_and_nesting():
    gt, ext = fixture()
    g, e = cell_rows(gt, ext, ECO, ATTR)
    assert len(setting_rows('probe_pct_0.00', g, e)) == len(e)
    assert len(setting_rows('probe_pct_0.30', g, e)) == 30
    assert len(setting_rows('probe_pct_0.60', g, e)) == 20
    assert set(all_settings('ground_truth', [0.0, 0.3], 'percentile', SRC)) == {
        'ground_truth', 'valid', 'ntp_pct_0.00', 'ntp_pct_0.30', 'probe_pct_0.00', 'probe_pct_0.30'}


def keep_mask_n(p, t):
    from analysis.pond_meta_analysis import keep_mask
    return keep_mask(p, t, 'percentile').sum()


def test_panel_titles_and_w1_legend_text(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import analysis.pond_meta_analysis as m
    from analysis.common.pond_meta import _attr_title
    assert _attr_title('surface_area') == r'surface area ($m^2$)'
    assert _attr_title('ph') == 'pH' and _attr_title('max_depth') == 'max depth (m)'
    saved = {}
    monkeypatch.setattr(plt, 'close', lambda fig: saved.setdefault('fig', fig))
    m.plot_w1_curves_legend(tmp_path / 'leg.pdf', 'ground_truth', 'percentile', m.threshold_sources(True))
    texts = [t.get_text() for t in saved['fig'].axes[0].get_legend().get_texts()]
    assert 'Random' in texts and not any('95%' in x for x in texts)
