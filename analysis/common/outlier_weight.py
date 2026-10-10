"""Down-weight probe / NTP confidences for extracted values that look like outliers.

The stored confidences score validity only, not how plausible the number is.
``add_outlier_columns`` multiplies each by exp(-z^2 / 2), with the robust z-score
z = (x - median) / (MAD_TO_SIGMA * MAD) taken per (ecosystem_bucket, attribute) over
the valued rows of ``ext_df``. Median and MAD are used instead of mean and SD because
the outliers being targeted would otherwise inflate the scale they are measured by.

The factor is in [0, 1]: for |z| beyond ~38 exp(-z^2 / 2) underflows to exactly 0.0,
which is kept and counted (``n_zero_factor``). Rows without ``converted_value`` get NaN
so any downstream use fails loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

PROB_COLS = ('ntp_prob', 'probe_prob')
GROUP_COLS = ['ecosystem_bucket', 'attribute']
# Makes MAD a consistent estimator of sigma for Gaussian data (~1.4826).
MAD_TO_SIGMA = 1.0 / stats.norm.ppf(0.75)


def robust_moments(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Median, MAD and count of ``converted_value`` per (ecosystem_bucket, attribute).

    Args:
        ext_df: Extraction rows with ``ecosystem_bucket``, ``attribute`` and ``converted_value``.

    Returns:
        DataFrame indexed by (ecosystem_bucket, attribute) with ``median``, ``mad``, ``n``.

    Raises:
        ValueError: A group has n < 2 or MAD == 0.
    """
    v = ext_df.dropna(subset=['converted_value'])
    g = v.groupby(GROUP_COLS)['converted_value']
    median = g.median()
    dev = (v['converted_value'] - v.set_index(GROUP_COLS).index.map(median).to_numpy()).abs()
    mad = dev.groupby([v[c] for c in GROUP_COLS]).median()
    m = pd.DataFrame({'median': median, 'mad': mad, 'n': g.size()})
    bad = m[~(m['mad'] > 0) | (m['n'] < 2)]
    if len(bad):
        raise ValueError(f'no usable MAD (n < 2 or MAD == 0) for (ecosystem, attribute) {bad.index.tolist()}')
    return m


def non_outlier_factor(ext_df: pd.DataFrame, moments: pd.DataFrame) -> pd.Series:
    """Robust non-outlier factor exp(-z^2 / 2) for each row.

    Args:
        ext_df: Extraction rows with ``ecosystem_bucket``, ``attribute`` and ``converted_value``.
        moments: Output of ``robust_moments``.

    Returns:
        Series aligned to ``ext_df``, in [0, 1], NaN where there is no value.
    """
    key = pd.MultiIndex.from_frame(ext_df[GROUP_COLS])
    median = pd.Series(key.map(moments['median']), index=ext_df.index, dtype=float)
    mad = pd.Series(key.map(moments['mad']), index=ext_df.index, dtype=float)
    has_value = ext_df['converted_value'].notna()
    assert not (has_value & (median.isna() | mad.isna())).any(), 'a valued row has a group with no moments'
    z = (ext_df['converted_value'] - median) / (MAD_TO_SIGMA * mad)
    f = np.exp(-0.5 * z ** 2)
    assert (f[has_value] >= 0).all() and (f[has_value] <= 1).all(), 'non-outlier factor outside [0, 1]'
    assert f.isna().to_numpy().tolist() == (~has_value).to_numpy().tolist(), 'factor NaN pattern != missing-value pattern'
    return f


def add_outlier_columns(ext_df: pd.DataFrame, adjust: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add ``{c}_raw`` and ``outlier_factor`` columns and rescale each PROB_COL ``c``.

    ``{c}_raw`` keeps the unadjusted confidence: pond_clustering's shuffled control
    permutes it and reapplies each row's own factor.

    Args:
        ext_df: Extraction rows with ``ntp_prob``, ``probe_prob``, ``ecosystem_bucket``,
            ``attribute``, ``converted_value``.
        adjust: If False, the factor is 1.0 and confidences are unchanged.

    Returns:
        Tuple of:
            - copy of ``ext_df`` (same rows and order) with ``c = {c}_raw * outlier_factor``
            - for the manifest, one row per (ecosystem_bucket, attribute) with ``median``,
              ``mad``, ``n``, ``n_zero_factor``; None when ``adjust`` is False
    """
    out = ext_df.copy()
    for c in PROB_COLS:
        out[f'{c}_raw'] = ext_df[c]
    if not adjust:
        out['outlier_factor'] = 1.0
        return out, None
    moments = robust_moments(ext_df)
    f = non_outlier_factor(ext_df, moments)
    out['outlier_factor'] = f
    for c in PROB_COLS:
        assert ext_df.loc[f.notna(), c].between(0, 1).all(), f'{c} outside [0, 1] before adjustment'
        out[c] = ext_df[c] * f
    assert len(out) == len(ext_df) and out.index.equals(ext_df.index)
    zero = (f == 0.0).groupby([ext_df[c] for c in GROUP_COLS]).sum()
    moments['n_zero_factor'] = zero.reindex(moments.index).astype(int)
    return out, moments.reset_index()
