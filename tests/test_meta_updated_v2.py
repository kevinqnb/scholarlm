"""Unit tests for analysis/meta_updated_v2.py (hard confidence thresholds, unweighted
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

from analysis.meta_inputs import load_meta_v2_config
from analysis.meta_updated_v2 import (
    _summarize_shuffles, all_settings, build_stats_table, build_w1_table, cell_rows, qq_line, setting_rows, w1_with_ci,
)

ECO, ATTR = 'pond', 'tn'
THRESHOLDS = [0.0, 0.5, 0.6]
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
    return gt, ext


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
    df = build_stats_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS)
    assert list(df['setting']) == all_settings('ground_truth', THRESHOLDS)
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


def test_w1_table_known_answers():
    gt, ext = fixture()
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, MIN_N, n_boot=50, n_shuffle=30, seed=0)
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
    # at t = 0 every shuffle keeps every row: the control collapses onto the real value
    # (to float rounding -- the mean of n identical values need not be bit-identical)
    for sc in ('', '_log'):
        np.testing.assert_allclose([e[f'w1_shuffled{sc}_{k}'] for k in ('mean', 'lo', 'hi')], e[f'w1{sc}'], rtol=1e-12)
        assert e[f'w1_shuffled{sc}_n_ok'] == 30 and e[f'w1_shuffled{sc}_skip'] == ''
    assert np.isnan(row(df, 'valid')['w1_shuffled_mean'])  # no control for a non-threshold setting
    n = row(df, 'ntp_ge_0.60')
    assert n['n_ext'] == 0 and np.isnan(n['w1']) and n['w1_skip'] == 'ext_n<min_n'
    # the valid set (rows 0-19) is a compared setting under the GT reference
    v = row(df, 'valid')
    assert v['n_ext'] == 20 and v['w1'] == stats.wasserstein_distance(ref, ext['converted_value'].to_numpy()[:20])
    # the high-confidence rows sit closer to GT than the unfiltered set
    assert row(df, 'probe_ge_0.60')['w1'] < e['w1']


def test_valid_reference_perfect_threshold_gives_zero():
    gt, ext = fixture()
    df = build_w1_table(gt, ext, 'valid', [ECO], [ATTR], THRESHOLDS, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    assert set(df['setting']) == set(all_settings('valid', THRESHOLDS)) - {'valid'}
    assert row(df, 'probe_ge_0.60')['w1'] == 0.0
    assert row(df, 'ground_truth')['n_ext'] == 30


def test_shuffled_control_band_and_docs():
    gt, ext = fixture()
    df = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    r = row(df, 'probe_ge_0.60')
    # the perfect probe keeps the 20 GT-like rows; 20 random rows sit much further from GT
    assert r['n_ext'] == 20 and r['w1_shuffled_n_ok'] == 30
    assert r['w1_shuffled_lo'] <= r['w1_shuffled_mean'] <= r['w1_shuffled_hi']
    assert r['w1'] < r['w1_shuffled_lo']
    # the real subset sits in 2 documents; random 20-of-40 subsets spread over more
    assert r['n_docs_ext'] == 2 and r['n_docs_ext_shuffled_mean'] > 2
    # a constant confidence keeps all-or-nothing, so its control equals the real value
    n = row(df, 'ntp_ge_0.50')
    np.testing.assert_allclose([n[f'w1_shuffled_{k}'] for k in ('mean', 'lo', 'hi')], n['w1'], rtol=1e-12)


def test_summarize_shuffles_refuses_partial_sets():
    assert _summarize_shuffles(np.array([1.0, 2.0, 3.0]), 0.95)['skip'] == ''
    p = _summarize_shuffles(np.array([1.0, np.nan]), 0.95)
    assert np.isnan(p['mean']) and p['skip'] == 'partial_ok' and p['n_ok'] == 1
    assert _summarize_shuffles(np.array([np.nan, np.nan]), 0.95)['skip'] == 'none_ok'


def test_seed_determinism_and_seed_dependence():
    gt, ext = fixture()
    a = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    b = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, MIN_N, n_boot=50, n_shuffle=30, seed=0)
    c = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], THRESHOLDS, MIN_N, n_boot=50, n_shuffle=30, seed=1)
    pd.testing.assert_frame_equal(a, b)
    assert not np.allclose(a['w1_lo'].dropna(), c['w1_lo'].dropna())
    # a cell's CI does not depend on the threshold grid it was computed with
    d = build_w1_table(gt, ext, 'ground_truth', [ECO], [ATTR], [0.6], MIN_N, n_boot=50, n_shuffle=30, seed=0)
    for c in ('w1_lo', 'w1_shuffled_mean', 'w1_shuffled_lo', 'w1_shuffled_log_hi'):
        assert row(d, 'probe_ge_0.60')[c] == row(a, 'probe_ge_0.60')[c], c


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
        "thresholds": [0.0, 0.25, 0.5, 0.75], "min_n": 5, "n_shuffle_samples": 10,
    }},
}


def write(tmp_path, cfg):
    p = tmp_path / "t.yaml"
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
    lambda m: m.pop("n_shuffle_samples"),
    lambda m: m.update(n_shuffle_samples=0),
    lambda m: m.update(n_shuffle_samples=True),
    lambda m: m.update(min_n=True),
    lambda m: m.update(reference="gt"),         # shared check still applies
    lambda m: m.update(qq_attributes=["ph"]),
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
    from analysis.meta_updated_v2 import DARK_BLUE, DARK_RED, QQ_BASE_STYLE, THRESHOLD_CMAP, threshold_style
    import matplotlib.colors as mcolors
    th = [0.0, 0.25, 0.5, 0.75]
    s0 = threshold_style(0.0, th)
    assert s0['color'] == DARK_BLUE and s0['linestyle'] != '-'
    assert QQ_BASE_STYLE['valid']['color'] == DARK_RED and QQ_BASE_STYLE['valid']['linestyle'] != '-'
    inner = [threshold_style(t, th) for t in th[1:]]
    assert all(s['linestyle'] == '-' for s in inner)
    np.testing.assert_allclose([s['color'] for s in inner], [THRESHOLD_CMAP(f) for f in (0.25, 0.5, 0.75)])
    assert len({mcolors.to_hex(s['color']) for s in inner} | {DARK_BLUE, DARK_RED}) == 5


def test_unit_conversion_v2_additions_only():
    """v2's table is v1's plus mi^2 and ppm / ppb for tn / tp / chla; v1 is untouched,
    and µg/cm^2 and molar chla stay unconvertible in both."""
    from analysis.meta_updated import UNIT_CONVERSION, convert_units
    from analysis.meta_updated_v2 import UNIT_CONVERSION_V2
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
