"""Unit tests for the weighted-Hazen quantile machinery in ``analysis.meta_updated``.

These statistics feed a stats table and Q-Q figures headed for a paper, so every
claim here is checked either against an independent reference (numpy's own
unweighted Hazen quantile) or against a fixture computed by hand, not just against
the implementation's own internal consistency.

Two claims underwrite the whole design (see build note
notes/scholarlm/builds/2026-08-21-weighted-hazen-meta-01.md):
  1. weighted_hazen_quantile reduces *exactly* to np.quantile(x, q, method='hazen')
     when every weight is 1 -- this is what lets the GT-side Hazen code
     (_hazen_quantiles / _bootstrap_gt_band) stay untouched with no estimator
     mismatch against the newly-weighted extracted side.
  2. weighted_valid_range's bounds are keyed on the *specific* weights of the
     smallest-x / largest-x observations, not a global min weight.

Note: unlike np.quantile(method='inverted_cdf'), integer-weight repeat-expansion
is NOT exact for Hazen (the plotting-position formula depends on weight, not just
resulting rank, in a way repeat-expansion doesn't reproduce), so that equivalence
is deliberately not tested here -- see the hand-computed fixture instead.
"""
import numpy as np
import pytest

from analysis.meta_updated import (
    weighted_hazen_quantile,
    weighted_valid_range,
    kish_n_eff,
    kish_gate_levels,
    weighted_stats,
)


# ── weighted_hazen_quantile ─────────────────────────────────────────────────

def test_w_equals_one_matches_unweighted_hazen():
    rng = np.random.default_rng(0)
    x = rng.uniform(-5, 100, 200)
    levels = np.linspace(0.01, 0.99, 37)
    got = weighted_hazen_quantile(x, np.ones_like(x), levels)
    expected = np.quantile(x, levels, method='hazen')
    np.testing.assert_allclose(got, expected)


def test_w_equals_one_matches_unweighted_hazen_with_ties():
    x = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 2.0])
    levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    got = weighted_hazen_quantile(x, np.ones_like(x), levels)
    expected = np.quantile(x, levels, method='hazen')
    np.testing.assert_allclose(got, expected)


def test_hand_computed_unequal_weights():
    # x=[10,20,30], w=[1,2,1] -> sw=4, cumsum(w)=[1,3,4]
    # p_i = (cumsum(w)_i - 0.5*w_i)/sw = [0.125, 0.5, 0.875]
    x = np.array([10.0, 20.0, 30.0])
    w = np.array([1.0, 2.0, 1.0])
    got = weighted_hazen_quantile(x, w, [0.125, 0.5, 0.875])
    np.testing.assert_allclose(got, [10.0, 20.0, 30.0])

    # q=0.3 interpolates between p=0.125 (x=10) and p=0.5 (x=20):
    # frac = (0.3-0.125)/(0.5-0.125) = 7/15 -> x = 10 + 7/15*10 = 14.6666...7
    got_mid = weighted_hazen_quantile(x, w, 0.3)
    assert got_mid == pytest.approx(10.0 + (7.0 / 15.0) * 10.0)

    # q=0.7 interpolates between p=0.5 (x=20) and p=0.875 (x=30):
    # frac = (0.7-0.5)/(0.875-0.5) = 0.2/0.375 = 8/15 -> x = 20 + 8/15*10
    got_mid2 = weighted_hazen_quantile(x, w, 0.7)
    assert got_mid2 == pytest.approx(20.0 + (8.0 / 15.0) * 10.0)


def test_unequal_weights_diverge_from_equal_weights():
    # Sanity: heavy weight on the low point should pull the weighted median
    # below the unweighted (equal-weight) median.
    x = np.array([10.0, 20.0, 30.0])
    w_heavy_low = np.array([10.0, 1.0, 1.0])
    weighted_median = weighted_hazen_quantile(x, w_heavy_low, 0.5)
    unweighted_median = weighted_hazen_quantile(x, np.ones_like(x), 0.5)
    assert weighted_median < unweighted_median


def test_zero_weight_rows_are_dropped():
    x = np.array([1.0, 2.0, 3.0, 1_000_000.0])
    w = np.array([1.0, 1.0, 1.0, 0.0])
    got = weighted_hazen_quantile(x, w, [0.25, 0.5, 0.75])
    expected = weighted_hazen_quantile(x[:3], w[:3], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(got, expected)


def test_mismatched_shapes_raise():
    with pytest.raises(ValueError):
        weighted_hazen_quantile(np.array([1.0, 2.0]), np.array([1.0]), 0.5)


def test_negative_weight_raises():
    with pytest.raises(ValueError):
        weighted_hazen_quantile(np.array([1.0, 2.0]), np.array([1.0, -1.0]), 0.5)


def test_all_zero_weight_raises():
    with pytest.raises(ValueError):
        weighted_hazen_quantile(np.array([1.0, 2.0]), np.array([0.0, 0.0]), 0.5)


# ── weighted_valid_range ─────────────────────────────────────────────────────

def test_valid_range_matches_hand_fixture():
    # Same fixture as above: p_1=0.125, p_n=0.875.
    x = np.array([10.0, 20.0, 30.0])
    w = np.array([1.0, 2.0, 1.0])
    lo, hi = weighted_valid_range(x, w, lo_cap=0.0, hi_cap=1.0)
    assert lo == pytest.approx(0.125)
    assert hi == pytest.approx(0.875)


def test_valid_range_reduces_to_unweighted_at_w_one():
    x = np.arange(1.0, 21.0)  # n=20
    lo, hi = weighted_valid_range(x, np.ones_like(x), lo_cap=0.0, hi_cap=1.0)
    assert lo == pytest.approx(0.5 / 20)
    assert hi == pytest.approx(1 - 0.5 / 20)


def test_valid_range_keyed_on_extreme_weight_not_global_min():
    # Heavy weight on the smallest-x point should *raise* p_min a lot, even
    # though the sample also contains a much smaller weight in the middle --
    # the middle weight must have zero effect on the bounds.
    x = np.array([10.0, 20.0, 30.0])
    w_heavy_extreme = np.array([5.0, 0.01, 1.0])
    lo, hi = weighted_valid_range(x, w_heavy_extreme, lo_cap=0.0, hi_cap=1.0)
    sw = 5.0 + 0.01 + 1.0
    assert lo == pytest.approx(0.5 * 5.0 / sw)
    assert hi == pytest.approx(1 - 0.5 * 1.0 / sw)

    # Confirm the tiny middle weight alone (moved to the middle, extremes
    # unchanged) doesn't move lo/hi: compare against extremes-only weights.
    w_middle_irrelevant = np.array([5.0, 999.0, 1.0])  # blow up the middle weight
    sw2 = 5.0 + 999.0 + 1.0
    lo2, hi2 = weighted_valid_range(x, w_middle_irrelevant, lo_cap=0.0, hi_cap=1.0)
    assert lo2 == pytest.approx(0.5 * 5.0 / sw2)
    assert hi2 == pytest.approx(1 - 0.5 * 1.0 / sw2)


def test_valid_range_respects_caps():
    x = np.arange(1.0, 6.0)
    lo, hi = weighted_valid_range(x, np.ones_like(x), lo_cap=0.3, hi_cap=0.6)
    assert lo == 0.3
    assert hi == 0.6


# ── kish_n_eff ────────────────────────────────────────────────────────────────

def test_kish_n_eff_equal_weights_equals_n():
    w = np.ones(10)
    assert kish_n_eff(w) == pytest.approx(10.0)


def test_kish_n_eff_hand_fixture():
    # w=[1,2,1]: sw=4, sum(w^2)=1+4+1=6, n_eff=16/6
    w = np.array([1.0, 2.0, 1.0])
    assert kish_n_eff(w) == pytest.approx(16.0 / 6.0)


def test_kish_n_eff_all_weight_on_one_point_is_one():
    w = np.array([5.0, 0.0, 0.0])
    assert kish_n_eff(w) == pytest.approx(1.0)


def test_kish_n_eff_zero_weight_rows_dont_change_result():
    assert kish_n_eff(np.array([1.0, 2.0, 1.0])) == pytest.approx(
        kish_n_eff(np.array([1.0, 2.0, 1.0, 0.0, 0.0]))
    )


def test_kish_n_eff_all_zero_raises():
    with pytest.raises(ValueError):
        kish_n_eff(np.array([0.0, 0.0]))


# ── kish_gate_levels ─────────────────────────────────────────────────────────

def test_kish_gate_levels_matches_users_rule():
    # User's stated rule: drop q where n_eff < 1/min(q, 1-q).
    # required = 1/min(q,1-q): [40, 10, 4, 2, 4, 10, 40]. n_eff=10.5 clears the
    # q=0.1/q=0.9 requirement (10) with an unambiguous margin -- deliberately not an
    # exact tie, since 1-0.9 isn't exactly 0.1 in float64 and a boundary test would be
    # asserting a floating-point coincidence rather than the rule itself.
    levels = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
    n_eff = 10.5
    got = kish_gate_levels(levels, n_eff)
    expected = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    np.testing.assert_allclose(got, expected)


def test_kish_gate_levels_below_boundary_drops():
    # n_eff clearly under the q=0.1/q=0.9 requirement (10) should drop both tails.
    levels = np.array([0.1, 0.5, 0.9])
    got = kish_gate_levels(levels, n_eff=9.5)
    np.testing.assert_allclose(got, [0.5])


def test_kish_gate_levels_above_boundary_keeps():
    # n_eff clearly over the q=0.1/q=0.9 requirement (10) should keep both tails.
    levels = np.array([0.1, 0.5, 0.9])
    got = kish_gate_levels(levels, n_eff=10.5)
    np.testing.assert_allclose(got, levels)


def test_kish_gate_levels_n_eff_two_keeps_only_median():
    levels = np.array([0.1, 0.5, 0.9])
    got = kish_gate_levels(levels, n_eff=2.0)
    np.testing.assert_allclose(got, [0.5])


def test_kish_gate_levels_large_n_eff_keeps_everything():
    levels = np.linspace(0.025, 0.975, 20)
    got = kish_gate_levels(levels, n_eff=1000.0)
    np.testing.assert_allclose(got, levels)


# ── weighted_stats ──────────────────────────────────────────────────────────

def test_weighted_stats_hand_fixture():
    # Same x=[10,20,30], w=[1,2,1] fixture threaded through the whole codepath.
    x = np.array([10.0, 20.0, 30.0])
    w = np.array([1.0, 2.0, 1.0])
    stats = weighted_stats(x, w)

    assert stats['n'] == 3
    assert stats['n_eff'] == pytest.approx(16.0 / 6.0)
    # mean = (1*10 + 2*20 + 1*30)/4 = 80/4 = 20
    assert stats['mean'] == pytest.approx(20.0)
    # denom = sw - v2/sw = 4 - 6/4 = 2.5
    # sum(w*(x-mean)^2) = 1*100 + 2*0 + 1*100 = 200 -> std = sqrt(200/2.5) = sqrt(80)
    assert stats['std'] == pytest.approx(np.sqrt(80.0))
    assert stats['median'] == pytest.approx(20.0)
    assert stats['q1'] == pytest.approx(10.0 + (1.0 / 3.0) * 10.0)   # q=0.25 -> frac 1/3
    assert stats['q3'] == pytest.approx(20.0 + (2.0 / 3.0) * 10.0)   # q=0.75 -> frac 2/3


def test_weighted_stats_matches_unweighted_at_w_one():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 50, 30)
    stats = weighted_stats(x, np.ones_like(x))
    assert stats['mean'] == pytest.approx(np.mean(x))
    assert stats['std'] == pytest.approx(np.std(x, ddof=1))
    assert stats['n_eff'] == pytest.approx(30.0)
    q1, med, q3 = np.quantile(x, [0.25, 0.5, 0.75], method='hazen')
    assert stats['q1'] == pytest.approx(q1)
    assert stats['median'] == pytest.approx(med)
    assert stats['q3'] == pytest.approx(q3)


def test_weighted_stats_all_weight_on_one_point_std_is_nan():
    # n_eff == 1 -> Cauchy-Schwarz denominator hits exactly 0 -> NaN, not a
    # divide-by-zero crash or a silently wrong number.
    x = np.array([5.0, 100.0, 200.0])
    w = np.array([3.0, 0.0, 0.0])
    stats = weighted_stats(x, w)
    assert stats['n'] == 1
    assert stats['n_eff'] == pytest.approx(1.0)
    assert stats['mean'] == pytest.approx(5.0)
    assert np.isnan(stats['std'])
    assert stats['median'] == pytest.approx(5.0)


def test_weighted_stats_no_positive_weight_raises():
    with pytest.raises(ValueError):
        weighted_stats(np.array([1.0, 2.0]), np.array([0.0, 0.0]))
