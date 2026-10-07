"""Unit tests for params.meta.reference in ``analysis.meta_updated``: which
distribution the extracted settings are compared against (Q-Q x-axis, W2).

Hand-built fixture, one (pond, ph) cell: 30 ground-truth rows and 40 extracted rows,
of which the first 25 carry a positive stored calibration label (judge OR GT match).

Known answers:
  - reference 'valid' uses exactly the labelled rows as the reference (n_ref = 25);
  - a "perfect" confidence (1 on the valid rows, 0 elsewhere) reproduces the valid
    distribution exactly, so its W2 is 0 and its Q-Q line is the diagonal; shuffling
    those weights must move it off 0;
  - under 'valid', the ground_truth row is W2(valid -> GT) computed directly;
  - reference 'ground_truth' is W2(-> GT) computed directly, with the judge-only
    judge_filtered setting and the pre-reference CSV streams.
"""
import numpy as np
import pandas as pd
import pytest

from analysis.meta_updated import (
    SETTINGS,
    W2_SETTINGS,
    _setting_data,
    _valid_range,
    build_stats_table,
    build_wasserstein_table,
    qq_line,
    wasserstein2_quantile,
)

ECO, ATTR = 'pond', 'ph'
N_VALID = 25


def fixture():
    rng = np.random.default_rng(0)
    gt = pd.DataFrame({
        'ecosystem_bucket': ECO, 'attribute': ATTR,
        'converted_value': rng.normal(7.0, 0.5, 30),
    })
    valid_vals = rng.normal(7.2, 0.6, N_VALID)
    invalid_vals = rng.normal(3.0, 1.0, 15)        # far from the valid rows
    label = np.r_[np.ones(N_VALID, bool), np.zeros(15, bool)]
    ext = pd.DataFrame({
        'ecosystem_bucket': ECO, 'attribute': ATTR,
        'converted_value': np.r_[valid_vals, invalid_vals],
        'label': label,
        # judge-only verdict differs from the label on two rows (a GT match the
        # judge rejected), so judge_filtered and valid are distinguishable.
        'judgement_combined': np.r_[np.ones(N_VALID - 2, bool), np.zeros(17, bool)],
        'probe_prob': label.astype(float),          # perfect confidence
        'ntp_prob': np.full(40, 0.5),               # uninformative constant confidence
    })
    return gt, ext


def w2_row(df, setting):
    row = df[df['setting'] == setting]
    assert len(row) == 1
    return row.iloc[0]


def test_valid_setting_is_exactly_the_labelled_rows():
    gt, ext = fixture()
    x, w = _setting_data('valid', gt, ext, ECO, ATTR)
    np.testing.assert_array_equal(x, ext['converted_value'].to_numpy()[:N_VALID])
    np.testing.assert_array_equal(w, np.ones(N_VALID))
    jx, _ = _setting_data('judge_filtered', gt, ext, ECO, ATTR)
    assert jx.size == N_VALID - 2


def test_valid_setting_refuses_non_bool_label():
    gt, ext = fixture()
    ext['label'] = ext['label'].astype(int)
    with pytest.raises(AssertionError):
        _setting_data('valid', gt, ext, ECO, ATTR)


def test_valid_reference_w2_known_answers():
    gt, ext = fixture()
    df = build_wasserstein_table(gt, ext, shuffle_seed=0, reference='valid',
                                 ecosystems=[ECO], attributes=[ATTR], n_boot=5)
    assert df['setting'].tolist() == W2_SETTINGS['valid']
    assert (df['reference'] == 'valid').all() and (df['n_ref'] == N_VALID).all()

    valid_x = ext['converted_value'].to_numpy()[:N_VALID]
    gt_x = gt['converted_value'].to_numpy()
    all_x = ext['converted_value'].to_numpy()

    # Perfect confidence reproduces the reference exactly (up to float rounding:
    # weighted plotting positions (cumsum(w)-0.5w)/sum(w) vs numpy's (i-0.5)/n).
    probe = w2_row(df, 'probe_weighted')
    assert probe['w2'] == pytest.approx(0.0, abs=1e-12) and probe['w2_skip'] == ''
    # ...and shuffling it breaks that (permutation control has something to regress to).
    assert probe['w2_shuffled'] > 0.5

    # Constant confidence is the unweighted extraction.
    assert w2_row(df, 'ntp_weighted')['w2'] == pytest.approx(w2_row(df, 'extracted')['w2'], abs=1e-12)

    # Direct computations.
    assert w2_row(df, 'extracted')['w2'] == pytest.approx(
        wasserstein2_quantile(valid_x, all_x, np.ones(40))['w2'], abs=1e-12)
    assert w2_row(df, 'ground_truth')['w2'] == pytest.approx(
        wasserstein2_quantile(valid_x, gt_x, np.ones(30))['w2'], abs=1e-12)
    # The invalid rows sit ~4 pH units low, so unweighted extraction is far from valid.
    assert w2_row(df, 'extracted')['w2'] > 1.0


def test_ground_truth_reference_unchanged_semantics():
    gt, ext = fixture()
    df = build_wasserstein_table(gt, ext, shuffle_seed=0, reference='ground_truth',
                                 ecosystems=[ECO], attributes=[ATTR], n_boot=5)
    assert df['setting'].tolist() == W2_SETTINGS['ground_truth']
    assert 'valid' not in df['setting'].tolist() and (df['n_ref'] == 30).all()
    gt_x = gt['converted_value'].to_numpy()
    jx, _ = _setting_data('judge_filtered', gt, ext, ECO, ATTR)
    assert w2_row(df, 'judge_filtered')['w2'] == pytest.approx(
        wasserstein2_quantile(gt_x, jx, np.ones(jx.size))['w2'], abs=1e-12)


def test_stats_table_settings_follow_reference():
    gt, ext = fixture()
    for reference in ('ground_truth', 'valid'):
        st = build_stats_table(gt, ext, reference, [ECO], [ATTR])
        assert st['setting'].tolist() == SETTINGS[reference]
    st = build_stats_table(gt, ext, 'valid', [ECO], [ATTR])
    assert st.set_index('setting').loc['valid', 'n'] == N_VALID


def test_qq_line_perfect_confidence_is_the_diagonal():
    gt, ext = fixture()
    ref_x, _ = _setting_data('valid', gt, ext, ECO, ATTR)
    lo, hi = _valid_range(ref_x.size)
    x, w = _setting_data('probe_weighted', gt, ext, ECO, ATTR)
    ref_q, ext_q = qq_line(ref_x, lo, hi, x, w)
    np.testing.assert_allclose(ref_q, ext_q)
    assert ref_q.size > 1


def test_qq_line_gates_low_n_eff():
    ref_x = np.linspace(0, 1, 50)
    lo, hi = _valid_range(ref_x.size)
    # Kish n_eff of 4 equal weights is 4 < MIN_RELIABLE_N (5): gated out.
    assert qq_line(ref_x, lo, hi, np.arange(4.0), np.ones(4)) is None
    assert qq_line(ref_x, lo, hi, np.array([]), np.array([])) is None
    assert qq_line(ref_x, lo, hi, np.arange(60.0), np.ones(60)) is not None
