import numpy as np
import pandas as pd
import pytest
from scipy import stats

from analysis.common.outlier_weight import MAD_TO_SIGMA, add_outlier_columns


def _df():
    # (pond, a): 0, 1, 2, 10 -> median 1.5, |dev| 1.5, 0.5, 0.5, 8.5 -> MAD 1.0.
    # (lake, a): 4, 6, 9      -> median 6,   |dev| 2, 0, 3           -> MAD 2.0.
    # (pond, b): 10, 20       -> median 15,  |dev| 5, 5              -> MAD 5.0.
    return pd.DataFrame({
        'ecosystem_bucket': ['pond'] * 4 + ['lake'] * 3 + ['pond'] * 2 + ['pond'],
        'attribute': ['a'] * 7 + ['b', 'b', 'a'],
        'converted_value': [0.0, 1.0, 2.0, 10.0, 4.0, 6.0, 9.0, 10.0, 20.0, np.nan],
        'ntp_prob': [1.0, 0.5, 1.0, 0.8, 0.8, 0.9, 0.4, 0.8, 0.8, 0.9],
        'probe_prob': [0.5, 1.0, 0.2, 1.0, 0.4, 0.9, 0.6, 1.0, 0.4, 0.9],
    })


def factor(dev, mad):
    return np.exp(-0.5 * (dev / (MAD_TO_SIGMA * mad)) ** 2)


def test_mad_constant_is_gaussian_consistent():
    assert MAD_TO_SIGMA == pytest.approx(1.4826, abs=1e-4)
    assert MAD_TO_SIGMA == 1.0 / stats.norm.ppf(0.75)


def test_hand_checked_factors_per_ecosystem():
    out, m = add_outlier_columns(_df(), True)
    pa = m[(m['ecosystem_bucket'] == 'pond') & (m['attribute'] == 'a')].iloc[0]
    la = m[(m['ecosystem_bucket'] == 'lake') & (m['attribute'] == 'a')].iloc[0]
    assert (pa['median'], pa['mad'], pa['n']) == (1.5, 1.0, 4)
    assert (la['median'], la['mad'], la['n']) == (6.0, 2.0, 3)   # same attribute, own ecosystem statistics
    f = out['outlier_factor'].to_numpy()
    np.testing.assert_allclose(f[:4], factor(np.array([1.5, 0.5, 0.5, 8.5]), 1.0))
    np.testing.assert_allclose(f[4:7], factor(np.array([2.0, 0.0, 3.0]), 2.0))
    assert f[5] == 1.0                                            # at the median: factor 1
    np.testing.assert_allclose(f[7:9], factor(np.array([5.0, 5.0]), 5.0))
    assert out['probe_prob'].iloc[3] == pytest.approx(1.0 * factor(8.5, 1.0))
    assert out[['ntp_prob', 'probe_prob', 'outlier_factor']].iloc[9].isna().all()   # no value -> NaN


def test_zero_mad_fails_loud():
    df = _df()
    df.loc[(df['ecosystem_bucket'] == 'lake'), 'converted_value'] = [4.0, 4.0, 9.0]   # median 4, |dev| 0, 0, 5 -> MAD 0
    with pytest.raises(ValueError, match='MAD'):
        add_outlier_columns(df, True)


def test_missing_ecosystem_bucket_fails_loud():
    with pytest.raises(KeyError):
        add_outlier_columns(_df().drop(columns='ecosystem_bucket'), True)


def test_far_value_underflows_to_zero_and_is_counted():
    df = _df()
    df.loc[3, 'converted_value'] = 1e3   # z = 998.5 / 1.4826 ~ 673: exp(-z^2/2) is 0.0 in float64
    out, m = add_outlier_columns(df, True)
    assert out['outlier_factor'].iloc[3] == 0.0 and out['probe_prob'].iloc[3] == 0.0
    assert m.set_index(['ecosystem_bucket', 'attribute']).loc[('pond', 'a'), 'n_zero_factor'] == 1
    assert m['n_zero_factor'].sum() == 1


def test_raw_columns_and_no_adjust_identity():
    out, m = add_outlier_columns(_df(), True)
    assert (out['ntp_prob_raw'] == _df()['ntp_prob']).all()
    assert out['ntp_prob'].iloc[:9].to_numpy() == pytest.approx((out['ntp_prob_raw'] * out['outlier_factor']).iloc[:9].to_numpy())
    off, m0 = add_outlier_columns(_df(), False)
    assert m0 is None and (off['outlier_factor'] == 1.0).all()
    assert off['ntp_prob'].equals(_df()['ntp_prob']) and off['probe_prob'].equals(_df()['probe_prob'])
