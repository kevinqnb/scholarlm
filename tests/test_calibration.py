"""Unit tests for calibration-error estimators in ``scholarlm.utils.calibration``.

Covers three reported variants and their bootstrap CIs:
  * L1 ECE, equal-width, plug-in (the standard ECE used in the reliability plots)
  * L1 ECE, adaptive equal-mass (quantile) bins, plug-in
  * Debiased L2 RMS calibration error on equal-mass bins (Kumar, Liang & Ma,
    NeurIPS 2019)

The debiased estimator is validated two ways: an exact match to an independent
closed-form reference, and its defining statistical property — unbiasedness of
the squared calibration error under perfect calibration.
"""
import numpy as np
import pytest

from scholarlm.utils.calibration import (
    compute_ece,
    bootstrap_ece,
    reliability_diagram_data,
    _ece_bin_edges,
    _ece_binned_stats,
)


# ── Reference implementations (independent of the library internals) ──────────

def _ref_ece_l1_equal_width(probs, labels, n_bins=10):
    """Plain-loop L1 ECE with fixed [lo, hi) bins, last bin inclusive of 1.0."""
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    n = len(probs)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs <= hi) if hi == 1.0 else (probs >= lo) & (probs < hi)
        if mask.any():
            total += mask.sum() * abs(probs[mask].mean() - labels[mask].mean())
    return total / n


def _ref_ce2_debiased_raw(probs, labels, edges):
    """Kumar et al. (2019) debiased *squared* calibration error, pre-sqrt/clamp.

    Returns ``sum_b w_b [ (acc_b - conf_b)^2 - acc_b(1-acc_b)/(n_b - 1) ]``.
    Kept separate from the final ``sqrt(max(0, .))`` so the unbiasedness of the
    squared estimator can be checked directly.
    """
    probs = np.asarray(probs, float)
    labels = np.asarray(labels, float)
    n = len(probs)
    idx = np.digitize(probs, edges[1:-1], right=False)
    total = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        conf = probs[mask].mean()
        acc = labels[mask].mean()
        var = acc * (1.0 - acc) / (c - 1) if c > 1 else 0.0
        total += (c / n) * ((acc - conf) ** 2 - var)
    return total


# ── L1 ECE, equal-width plug-in ──────────────────────────────────────────────

def test_l1_equal_width_matches_reliability_diagram():
    rng = np.random.default_rng(0)
    probs = rng.uniform(0, 1, 500)
    labels = (rng.uniform(0, 1, 500) < probs ** 1.3).astype(int)
    assert compute_ece(probs, labels, binning="equal_width", p=1) == pytest.approx(
        reliability_diagram_data(probs, labels)["ece"]
    )


def test_l1_equal_width_matches_reference_loop():
    rng = np.random.default_rng(1)
    probs = rng.uniform(0, 1, 400)
    labels = (rng.uniform(0, 1, 400) < probs).astype(int)
    assert compute_ece(probs, labels, binning="equal_width", p=1) == pytest.approx(
        _ref_ece_l1_equal_width(probs, labels)
    )


def test_l1_hand_computed_two_bins():
    # Two points at 0.1 (bin 0.0-0.1... actually 0.1 -> bin [0.1,0.2)) and 0.9.
    # Use values squarely inside bins to avoid edge ambiguity.
    probs = np.array([0.05, 0.05, 0.95, 0.95])
    labels = np.array([0, 0, 1, 1])
    # bin(0.0-0.1): conf=0.05, acc=0.0 -> gap 0.05, weight 0.5
    # bin(0.9-1.0): conf=0.95, acc=1.0 -> gap 0.05, weight 0.5
    assert compute_ece(probs, labels, p=1) == pytest.approx(0.05)


def test_perfect_calibration_is_zero():
    # A single bin with conf == acc (0.5 confidence, 50% positive) -> ECE 0.
    probs = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([1, 1, 0, 0])
    assert compute_ece(probs, labels, p=1) == pytest.approx(0.0)


def test_empty_input_returns_zero():
    assert compute_ece(np.array([]), np.array([]), p=1) == 0.0
    assert compute_ece(np.array([]), np.array([]), p=2, debiased=True) == 0.0


# ── Equal-mass (adaptive) binning ────────────────────────────────────────────

def test_equal_mass_bins_have_near_equal_counts():
    rng = np.random.default_rng(2)
    probs = rng.uniform(0, 1, 1000)
    labels = (rng.uniform(0, 1, 1000) < 0.5).astype(int)
    edges = _ece_bin_edges(probs, 10, "equal_mass")
    _, _, counts = _ece_binned_stats(probs, labels, edges)
    # Counts across the 10 quantile bins should be within a few of n/10.
    assert counts.sum() == 1000
    assert counts.max() - counts.min() <= 3


def test_equal_mass_handles_ties_without_error():
    # Heavy ties collapse duplicate quantile edges; must not crash or NaN.
    probs = np.concatenate([np.full(50, 0.3), np.full(50, 0.7)])
    labels = np.concatenate([np.zeros(50), np.ones(50)]).astype(int)
    edges = _ece_bin_edges(probs, 10, "equal_mass")
    assert np.all(np.diff(edges) > 0)  # strictly increasing after de-duplication
    val = compute_ece(probs, labels, binning="equal_mass", p=1)
    assert np.isfinite(val)


def test_unknown_binning_raises():
    with pytest.raises(ValueError):
        compute_ece(np.array([0.5]), np.array([1]), binning="nonsense")


# ── Debiased L2 (Kumar et al. 2019) ──────────────────────────────────────────

def test_l2_debiased_matches_closed_form_reference():
    rng = np.random.default_rng(3)
    probs = rng.uniform(0, 1, 600)
    labels = (rng.uniform(0, 1, 600) < probs ** 1.5).astype(int)
    for binning in ("equal_width", "equal_mass"):
        edges = _ece_bin_edges(probs, 10, binning)
        expected = np.sqrt(max(0.0, _ref_ce2_debiased_raw(probs, labels, edges)))
        got = compute_ece(probs, labels, binning=binning, p=2, debiased=True)
        assert got == pytest.approx(expected), binning


def test_l2_plugin_ge_debiased():
    # The variance correction is non-negative, so debiased <= plug-in (pre-clamp).
    rng = np.random.default_rng(4)
    probs = rng.uniform(0, 1, 300)
    labels = (rng.uniform(0, 1, 300) < probs).astype(int)
    plugin = compute_ece(probs, labels, binning="equal_mass", p=2, debiased=False)
    debiased = compute_ece(probs, labels, binning="equal_mass", p=2, debiased=True)
    assert debiased <= plugin + 1e-12


def test_l2_debiased_is_unbiased_under_perfect_calibration():
    """Core Kumar et al. property: E[debiased CE^2] = 0 when conf == true accuracy.

    Uses discrete predictions so each value occupies its own bin and the true
    per-bin accuracy equals the confidence exactly. The plug-in CE^2 is biased
    upward by the sampling variance; the debiased CE^2 must average to ~0.
    """
    rng = np.random.default_rng(5)
    levels = np.array([0.15, 0.45, 0.75])
    per_level = 40
    edges = np.linspace(0.0, 1.0, 11)

    n_sim = 4000
    plugin_ce2 = np.empty(n_sim)
    debiased_ce2 = np.empty(n_sim)
    probs = np.repeat(levels, per_level)
    for s in range(n_sim):
        labels = (rng.uniform(0, 1, probs.size) < probs).astype(int)
        # plug-in squared CE (no correction)
        idx = np.digitize(probs, edges[1:-1], right=False)
        n = probs.size
        p_ce2 = 0.0
        for b in np.unique(idx):
            mask = idx == b
            p_ce2 += (mask.sum() / n) * (labels[mask].mean() - probs[mask].mean()) ** 2
        plugin_ce2[s] = p_ce2
        debiased_ce2[s] = _ref_ce2_debiased_raw(probs, labels, edges)

    # Plug-in is biased clearly positive; debiased averages to ~0.
    assert plugin_ce2.mean() > 5e-4
    se = debiased_ce2.std(ddof=1) / np.sqrt(n_sim)
    assert abs(debiased_ce2.mean()) < 4 * se


def test_debiased_requires_p2():
    with pytest.raises(ValueError):
        compute_ece(np.array([0.5]), np.array([1]), p=1, debiased=True)


def test_invalid_p_raises():
    with pytest.raises(ValueError):
        compute_ece(np.array([0.5]), np.array([1]), p=3)


# ── Bootstrap CIs ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("binning,p,debiased", [
    ("equal_width", 1, False),
    ("equal_mass", 1, False),
    ("equal_mass", 2, True),
])
def test_bootstrap_point_matches_compute_ece(binning, p, debiased):
    rng = np.random.default_rng(6)
    probs = rng.uniform(0, 1, 500)
    labels = (rng.uniform(0, 1, 500) < probs).astype(int)
    out = bootstrap_ece(probs, labels, binning=binning, p=p, debiased=debiased,
                        n_boot=300, seed=0)
    assert out["ece"] == pytest.approx(
        compute_ece(probs, labels, binning=binning, p=p, debiased=debiased)
    )
    assert out["ci_low"] <= out["ci_high"]
    assert out["n_boot"] == 300


def test_bootstrap_is_reproducible():
    rng = np.random.default_rng(7)
    probs = rng.uniform(0, 1, 400)
    labels = (rng.uniform(0, 1, 400) < probs).astype(int)
    a = bootstrap_ece(probs, labels, n_boot=200, seed=42)
    b = bootstrap_ece(probs, labels, n_boot=200, seed=42)
    assert a == b


def test_bootstrap_ci_covers_point_estimate():
    # A percentile CI at 95% should almost always bracket the point estimate.
    rng = np.random.default_rng(8)
    probs = rng.uniform(0, 1, 800)
    labels = (rng.uniform(0, 1, 800) < probs ** 1.4).astype(int)
    out = bootstrap_ece(probs, labels, n_boot=1000, ci=0.95, seed=0)
    assert out["ci_low"] <= out["ece"] <= out["ci_high"]


def test_bootstrap_small_sample_returns_nan_bounds():
    out = bootstrap_ece(np.array([0.3]), np.array([1]), n_boot=100)
    assert out["n_boot"] == 0
    assert np.isnan(out["ci_low"]) and np.isnan(out["ci_high"])


# ── Optional cross-check against the reference `uncertainty-calibration` pkg ──

def test_matches_uncertainty_calibration_package_if_installed():
    """Cross-validate every reported variant against Kumar et al.'s own code.

    Skipped unless ``uncertainty-calibration`` (module ``calibration``) is
    installed — it is intentionally NOT a project dependency (unmaintained, and
    its top-level module name collides with ours). Install it in a throwaway
    env to exercise this test. Confirmed to match to machine precision against
    ``uncertainty-calibration==0.1.4``.
    """
    cal = pytest.importorskip("calibration")
    rng = np.random.default_rng(9)
    probs = rng.uniform(0, 1, 3000)
    labels = (rng.uniform(0, 1, 3000) < probs ** 1.3).astype(int)
    k = 15

    # L1 ECE, equal-width and equal-mass (binary marginal mode).
    assert compute_ece(probs, labels, k, binning="equal_width", p=1) == pytest.approx(
        cal.get_ece(probs, labels, debias=False, num_bins=k, mode="marginal"), abs=1e-9
    )
    assert compute_ece(probs, labels, k, binning="equal_mass", p=1) == pytest.approx(
        cal.get_ece_em(probs, labels, debias=False, num_bins=k, mode="marginal"), abs=1e-9
    )

    # Debiased L2 on equal-mass bins == Kumar's unbiased_l2_ce on the same bins.
    binned = cal.bin(list(zip(probs, labels)), cal.get_equal_bins(probs, num_bins=k))
    assert compute_ece(probs, labels, k, binning="equal_mass", p=2, debiased=True) == \
        pytest.approx(cal.unbiased_l2_ce(binned), abs=1e-9)
