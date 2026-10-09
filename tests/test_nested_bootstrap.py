"""Rung-1 unit tests for analysis/common/nested_bootstrap.py: the fit-sample x test-document
nested bootstrap behind calibration v3's CIs and reliability curves.

Fixtures are small enough to check by hand, plus two large known-answer cases
(calibrated vs. shifted predictions) where only the direction/size is asserted.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import relplot
from scipy.special import expit, logit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.common import nested_bootstrap as nb


def _calibrated(n, seed, n_docs):
    rng = np.random.default_rng(seed)
    p = rng.uniform(0.05, 0.95, n)
    y = rng.random(n) < p
    docs = rng.integers(0, n_docs, n)
    return p, y, docs


# ── cluster_bootstrap_indices ────────────────────────────────────────────────
def test_cluster_bootstrap_takes_whole_documents():
    docs = np.array(['A', 'A', 'B', 'B', 'B'])
    outcomes = set()
    for idx in nb.cluster_bootstrap_indices(docs, 200, np.random.default_rng(0)):
        c = np.bincount(idx, minlength=5)
        # every row of a document appears as often as the document was drawn
        assert c[0] == c[1] and c[2] == c[3] == c[4]
        # X = 2 documents drawn per resample
        assert c[0] + c[2] == 2
        assert len(idx) == 2 * c[0] + 3 * c[2]
        outcomes.add((int(c[0]), int(c[2])))
    assert outcomes == {(2, 0), (1, 1), (0, 2)}


def test_cluster_bootstrap_single_document_is_the_full_set():
    for idx in nb.cluster_bootstrap_indices(np.zeros(7), 10, np.random.default_rng(1)):
        assert np.array_equal(np.sort(idx), np.arange(7))


def test_cluster_bootstrap_is_seeded():
    docs = np.repeat(np.arange(6), 3)
    a = nb.cluster_bootstrap_indices(docs, 20, np.random.default_rng(5))
    b = nb.cluster_bootstrap_indices(docs, 20, np.random.default_rng(5))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


# ── fit_sample_rng / pool_resampled_fit_sample ───────────────────────────────
def test_fit_sample_rngs_are_distinct_and_seeded():
    draws = [nb.fit_sample_rng(7, r).integers(1 << 30) for r in range(5)]
    assert len(set(draws)) == 5
    assert draws == [nb.fit_sample_rng(7, r).integers(1 << 30) for r in range(5)]
    # never the doc-resample stream for the same seed
    assert nb.fit_sample_rng(7, 0).integers(1 << 30) != nb.doc_boot_rng(7, 'real', 'pond').integers(1 << 30)


def test_fit_sample_single_document_pool_is_uniform_without_replacement():
    # one document is always drawn exactly once, so the sample is n distinct pool rows
    seen = set()
    for r in range(200):
        s = nb.pool_resampled_fit_sample(np.zeros(6), 3, nb.fit_sample_rng(0, r))
        assert len(s) == 3 and len(set(s.tolist())) == 3 and set(s.tolist()) <= set(range(6))
        seen |= set(s.tolist())
    assert seen == set(range(6))


def test_fit_sample_row_can_repeat_when_its_document_is_drawn_twice():
    # docs A = row 0, B = row 1; resampling the 2 docs gives AA, AB or BB, all equally n = 2 rows
    outcomes = {tuple(sorted(nb.pool_resampled_fit_sample(np.array(['A', 'B']), 2, nb.fit_sample_rng(0, r)).tolist()))
                for r in range(200)}
    assert outcomes == {(0, 0), (0, 1), (1, 1)}


def test_fit_sample_label_rate_is_centred_on_pool_row_rate_and_wider_than_fixed_pool():
    # 20 docs: one big all-valid doc (40 rows) + 19 small all-invalid docs (3 rows): row rate 40/97,
    # doc-mean rate 1/20. Uniform sampling targets the row rate; resampling the pool spreads it.
    docs = np.concatenate([np.zeros(40), np.repeat(np.arange(1, 20), 3)])
    y = docs == 0
    rates = np.array([y[nb.pool_resampled_fit_sample(docs, 30, nb.fit_sample_rng(1, r))].mean() for r in range(2000)])
    fixed = np.array([y[np.random.default_rng(r).choice(len(y), 30, replace=False)].mean() for r in range(2000)])
    # Known answer, derived without the sampler: the expected big-doc row share after
    # resampling 20 docs is E[40K / (40K + 3(20 - K))], K ~ Bin(20, 1/20) (~0.323). It sits
    # below the row rate 40/97 (ratio-estimator bias, large here because one doc holds 41%
    # of rows and is absent from ~36% of resamples) and far above the doc mean 1/20.
    k = np.arange(21)
    from scipy.stats import binom
    expected = float(np.sum(binom.pmf(k, 20, 1 / 20) * 40 * k / (40 * k + 3 * (20 - k))))
    assert abs(rates.mean() - expected) < 0.02
    assert rates.std() > 2 * fixed.std()


def test_fit_sample_rejects_pool_smaller_than_n():
    with pytest.raises(AssertionError):
        nb.pool_resampled_fit_sample(np.zeros(4), 5, nb.fit_sample_rng(0, 0))


# ── nested_bootstrap ─────────────────────────────────────────────────────────
def test_identical_fit_samples_point_is_plain_smece():
    p, y, docs = _calibrated(600, 0, 12)
    boot = nb.cluster_bootstrap_indices(docs, 5, np.random.default_rng(0))
    out = nb.nested_bootstrap([p, p, p], y, boot)
    ce, sigma = relplot.metrics.smECE(p, y.astype(float), eps=0.001, return_width=True)
    # equal up to the rounding of averaging three identical values
    assert out['point']['SmECE'] == pytest.approx(ce, rel=1e-12)
    assert out['sigma_curve'] == sigma
    np.testing.assert_allclose(out['line'], nb.smooth_curve(p, y, sigma), rtol=1e-12, atol=0)
    reps = out['replicates']['SmECE']
    assert reps.shape == (3, 5) and np.array_equal(reps[0], reps[1]) and np.array_equal(reps[0], reps[2])


def test_single_document_leaves_only_fit_sample_spread():
    p, y, _ = _calibrated(500, 1, 1)
    pred_sets = [p, expit(logit(p) + 0.5), expit(logit(p) - 0.5)]
    boot = nb.cluster_bootstrap_indices(np.zeros(len(p)), 4, np.random.default_rng(0))
    out = nb.nested_bootstrap(pred_sets, y, boot)
    for m in nb.METRICS:
        base, reps = out['base'][m], out['replicates'][m]
        assert np.array_equal(reps, np.repeat(base[:, None], 4, axis=1)), m
        assert out['point'][m] == pytest.approx(base.mean())
        lo, hi = np.quantile(reps, [0.025, 0.975])
        assert (out['lo'][m], out['hi'][m]) == (lo, hi)


def test_known_answer_calibrated_vs_shifted():
    p, y, docs = _calibrated(20000, 2, 200)
    boot = nb.cluster_bootstrap_indices(docs, 10, np.random.default_rng(0))
    good = nb.nested_bootstrap([p], y, boot)
    shifted = expit(logit(p) + 1.5)
    bad = nb.nested_bootstrap([shifted], y, boot)
    assert good['point']['SmECE'] < 0.02
    assert bad['point']['SmECE'] > 0.1
    # Mean, not max, distance to the diagonal: smECE's bandwidth is ~smECE itself, so a
    # well-calibrated curve is lightly smoothed and wiggles by about its pointwise SE (~0.02).
    def mean_gap(out, f):
        inside = out['drawn'] & (nb.MESH >= f.min()) & (nb.MESH <= f.max())
        return np.abs(out['line'] - nb.MESH)[inside].mean()
    assert mean_gap(good, p) < 0.015
    assert mean_gap(bad, shifted) > 0.1


def test_band_and_drawn_region():
    rng = np.random.default_rng(3)
    p = rng.uniform(0.2, 0.6, 3000)
    y = rng.random(3000) < p
    docs = rng.integers(0, 30, 3000)
    out = nb.nested_bootstrap([p], y, nb.cluster_bootstrap_indices(docs, 50, np.random.default_rng(0)))
    d = out['drawn']
    assert d.any() and not d.all()
    s = out['sigma_curve']
    assert nb.MESH[d].min() >= 0.2 - s - 1e-9 and nb.MESH[d].max() <= 0.6 + s + 1e-9
    assert (out['lower'][d] <= out['upper'][d]).all()
    assert out['line'].shape == out['lower'].shape == out['density'].shape == nb.MESH.shape


def test_nested_bootstrap_is_deterministic():
    p, y, docs = _calibrated(800, 4, 16)
    sets = [p, expit(logit(p) - 0.3)]
    a = nb.nested_bootstrap(sets, y, nb.cluster_bootstrap_indices(docs, 6, np.random.default_rng(9)))
    b = nb.nested_bootstrap(sets, y, nb.cluster_bootstrap_indices(docs, 6, np.random.default_rng(9)))
    for m in nb.METRICS:
        assert np.array_equal(a['replicates'][m], b['replicates'][m])
    for k in ('line', 'lower', 'upper', 'drawn', 'density'):
        assert np.array_equal(a[k], b[k])


@pytest.mark.parametrize('bad', ['length', 'empty_sets', 'empty_boot', 'index'])
def test_nested_bootstrap_rejects_bad_inputs(bad):
    p, y, docs = _calibrated(50, 5, 5)
    sets, boot = [p], nb.cluster_bootstrap_indices(docs, 3, np.random.default_rng(0))
    if bad == 'length':
        sets = [p[:-1]]
    elif bad == 'empty_sets':
        sets = []
    elif bad == 'empty_boot':
        boot = []
    else:
        boot = [np.array([0, 50])]
    with pytest.raises(AssertionError):
        nb.nested_bootstrap(sets, y, boot)


# ── threshold_metrics ────────────────────────────────────────────────────────
def test_threshold_metrics_average_over_fit_samples():
    y = np.array([1, 1, 0, 0], dtype=bool)
    a = np.array([0.9, 0.8, 0.2, 0.1])   # acc 1, AUROC 1
    b = np.array([0.9, 0.4, 0.6, 0.1])   # acc 1/2, AUROC 3/4
    out = nb.threshold_metrics([a, b], y)
    assert out['acc'] == pytest.approx(0.75)
    assert out['auroc'] == pytest.approx(0.875)
    assert out['prec'] == pytest.approx((1.0 + 0.5) / 2)
    assert out['rec'] == pytest.approx((1.0 + 0.5) / 2)
    bs_a = np.mean((a - y) ** 2)
    bs_b = np.mean((b - y) ** 2)
    assert out['bs'] == pytest.approx((bs_a + bs_b) / 2)
