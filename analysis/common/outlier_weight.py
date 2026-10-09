"""Non-outlier adjustment of the probe / NTP confidences.

The stored confidences score validity only; they ignore how plausible the extracted
number is. ``adjust_confidences`` multiplies each by the heuristic non-outlier
probability

    exp(-(x - mu)^2 / (2 sigma^2)),

with mu / sigma the mean / sample std (ddof=1) of the extracted ``converted_value``s of
the row's attribute, raw scale, pooled over every extracted row that has a value (all
ecosystems, all documents in ``ext_df``). The factor is in (0, 1], so adjusted
confidences stay in [0, 1]. Rows without a ``converted_value`` get NaN adjusted
confidences (they are dropped downstream; NaN keeps a stray use loud).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PROB_COLS = ('ntp_prob', 'probe_prob')


def attribute_moments(ext_df: pd.DataFrame) -> pd.DataFrame:
    """Per-attribute (mu, sigma, n) of ``converted_value`` over the valued rows."""
    v = ext_df.dropna(subset=['converted_value'])
    g = v.groupby('attribute')['converted_value']
    m = pd.DataFrame({'mu': g.mean(), 'sigma': g.std(ddof=1), 'n': g.size()})
    bad = m[~(m['sigma'] > 0) | (m['n'] < 2)]
    if len(bad):
        raise ValueError(f'no usable sigma (n < 2 or constant) for attributes {bad.index.tolist()}')
    return m


def non_outlier_factor(ext_df: pd.DataFrame, moments: pd.DataFrame) -> pd.Series:
    """exp(-(x-mu)^2 / (2 sigma^2)) per row of ext_df; NaN where there is no value."""
    mu = ext_df['attribute'].map(moments['mu'])
    sigma = ext_df['attribute'].map(moments['sigma'])
    has_value = ext_df['converted_value'].notna()
    assert not (has_value & (mu.isna() | sigma.isna())).any(), 'a valued row has an attribute with no moments'
    f = np.exp(-((ext_df['converted_value'] - mu) ** 2) / (2.0 * sigma ** 2))
    assert (f[has_value] > 0).all() and (f[has_value] <= 1).all(), 'non-outlier factor outside (0, 1]'
    assert f.isna().to_numpy().tolist() == (~has_value).to_numpy().tolist(), 'factor NaN pattern != missing-value pattern'
    return f


def add_outlier_columns(ext_df: pd.DataFrame, adjust: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Copy of ext_df with, for each PROB_COL c, ``{c}_raw`` (the stored confidence) and a
    per-row ``outlier_factor``; c itself becomes ``{c}_raw * outlier_factor``.

    adjust=False: factor is 1.0 everywhere and c is unchanged (moments is None).
    adjust=True: factor as in the module docstring (NaN, hence NaN c, for rows without a
    value). The shuffled controls permute ``{c}_raw`` and multiply by the row's own
    factor, so they keep the outlier filter and break only the confidence-to-row link.
    Row count / order unchanged. Returns (df, per-attribute moments for the manifest)."""
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
