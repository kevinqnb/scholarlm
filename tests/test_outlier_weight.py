import numpy as np
import pandas as pd
import pytest

from analysis.outlier_weight import add_outlier_columns


def _df():
    # attribute a: values 0, 2, 4 -> mu 2, sigma 2 (ddof=1). attribute b: 10, 20 -> mu 15, sigma sqrt(50).
    return pd.DataFrame({
        'attribute': ['a', 'a', 'a', 'b', 'b', 'a'],
        'converted_value': [0.0, 2.0, 4.0, 10.0, 20.0, np.nan],
        'ntp_prob': [1.0, 0.5, 1.0, 0.8, 0.8, 0.9],
        'probe_prob': [0.5, 1.0, 0.2, 1.0, 0.4, 0.9],
    })


def test_hand_checked_factors():
    out, m = add_outlier_columns(_df(), True)
    assert m.loc['a', 'mu'] == 2.0 and m.loc['a', 'sigma'] == 2.0
    f_far = np.exp(-4 / 8)           # |x - mu| = 2, sigma = 2
    f_b = np.exp(-25 / 100)          # (x-mu)^2 = 25, 2 sigma^2 = 100
    assert out['ntp_prob'].iloc[0] == pytest.approx(1.0 * f_far)
    assert out['ntp_prob'].iloc[1] == pytest.approx(0.5)            # at the mean: factor 1
    assert out['probe_prob'].iloc[2] == pytest.approx(0.2 * f_far)  # symmetric
    assert out['probe_prob'].iloc[4] == pytest.approx(0.4 * f_b)
    assert out[['ntp_prob', 'probe_prob']].iloc[5].isna().all()     # no value -> NaN


def test_constant_attribute_fails_loud():
    df = _df()
    df.loc[df['attribute'] == 'b', 'converted_value'] = 5.0
    with pytest.raises(ValueError, match='no usable sigma'):
        add_outlier_columns(df, True)


def test_raw_columns_and_no_adjust_identity():
    out, m = add_outlier_columns(_df(), True)
    assert (out['ntp_prob_raw'] == _df()['ntp_prob']).all()
    assert out['ntp_prob'].iloc[:5].to_numpy() == pytest.approx((out['ntp_prob_raw'] * out['outlier_factor']).iloc[:5].to_numpy())
    off, m0 = add_outlier_columns(_df(), False)
    assert m0 is None and (off['outlier_factor'] == 1.0).all()
    assert off['ntp_prob'].equals(_df()['ntp_prob']) and off['probe_prob'].equals(_df()['probe_prob'])
