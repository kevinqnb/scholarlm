"""Down-weight probe / NTP confidences for extracted values that look like outliers.

The stored confidences score validity only, not how plausible the number is.
``add_outlier_columns`` multiplies each by exp(-(x - mu)^2 / (2 sigma^2)), with mu and
sigma (ddof=1) taken per attribute over all valued rows of ``ext_df``. The factor is in
(0, 1]. Rows without ``converted_value`` get NaN so any downstream use fails loudly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROB_COLS = ('ntp_prob', 'probe_prob')


def attribute_moments(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Per-attribute mean, sample std and count of ``converted_value``.

    Args:
        ext_df: Extraction rows with ``attribute`` and ``converted_value``.

    Returns:
        DataFrame indexed by attribute with columns ``mu``, ``sigma``, ``n``.

    Raises:
        ValueError: An attribute has n < 2 or zero variance.
    """
    v = ext_df.dropna(subset=['converted_value'])
    g = v.groupby('attribute')['converted_value']
    m = pd.DataFrame({'mu': g.mean(), 'sigma': g.std(ddof=1), 'n': g.size()})
    bad = m[~(m['sigma'] > 0) | (m['n'] < 2)]
    if len(bad):
        raise ValueError(f'no usable sigma (n < 2 or constant) for attributes {bad.index.tolist()}')
    return m


def non_outlier_factor(ext_df: pd.DataFrame, moments: pd.DataFrame) -> pd.Series:
    """Gaussian non-outlier factor exp(-(x-mu)^2 / (2 sigma^2)) for each row.

    Args:
        ext_df: Extraction rows with ``attribute`` and ``converted_value``.
        moments: Output of ``attribute_moments``.

    Returns:
        Series aligned to ``ext_df``, in (0, 1], NaN where there is no value.
    """
    mu = ext_df['attribute'].map(moments['mu'])
    sigma = ext_df['attribute'].map(moments['sigma'])
    has_value = ext_df['converted_value'].notna()
    assert not (has_value & (mu.isna() | sigma.isna())).any(), 'a valued row has an attribute with no moments'
    f = np.exp(-((ext_df['converted_value'] - mu) ** 2) / (2.0 * sigma ** 2))
    assert (f[has_value] > 0).all() and (f[has_value] <= 1).all(), 'non-outlier factor outside (0, 1]'
    assert f.isna().to_numpy().tolist() == (~has_value).to_numpy().tolist(), 'factor NaN pattern != missing-value pattern'
    return f


def add_outlier_columns(ext_df: pd.DataFrame, adjust: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Add ``{c}_raw`` and ``outlier_factor`` columns and rescale each PROB_COL ``c``.

    Shuffled controls permute ``{c}_raw`` and reapply each row's own factor, so they
    keep the outlier filter and break only the confidence-to-row link.

    Args:
        ext_df: Extraction rows with ``ntp_prob``, ``probe_prob``, ``attribute``,
            ``converted_value``.
        adjust: If False, the factor is 1.0 and confidences are unchanged.

    Returns:
        Tuple of:
            - copy of ``ext_df`` (same rows and order) with ``c = {c}_raw * outlier_factor``
            - per-attribute moments for the manifest, or None when ``adjust`` is False
    """
    out = ext_df.copy()
    for c in PROB_COLS:
        out[f'{c}_raw'] = ext_df[c]
    if not adjust:
        out['outlier_factor'] = 1.0
        return out, None
    moments = attribute_moments(ext_df)
    f = non_outlier_factor(ext_df, moments)
    out['outlier_factor'] = f
    for c in PROB_COLS:
        assert ext_df.loc[f.notna(), c].between(0, 1).all(), f'{c} outside [0, 1] before adjustment'
        out[c] = ext_df[c] * f
    assert len(out) == len(ext_df) and out.index.equals(ext_df.index)
    return out, moments
