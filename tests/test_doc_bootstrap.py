"""Rung-1 tests for analysis/doc_bootstrap.py on hand-built fixtures whose answers can be
checked by inspection."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import relplot

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from analysis import doc_bootstrap as db  # noqa: E402
from scholarlm.utils.calibration import compute_ece  # noqa: E402


def test_document_resamples_take_whole_documents():
    # docs a (3 rows), b (1 row), c (2 rows).
    rows = {"a": [0, 1, 4], "b": [3], "c": [2, 5]}
    doc_ids = np.array(["a", "a", "c", "b", "a", "c"])
    res = db.document_resamples(doc_ids, 50, np.random.default_rng(0))
    assert len(res) == 50
    for idx in res:
        # Every drawn document contributes all its rows, as many times as it was drawn.
        counts = {d: int(np.sum(doc_ids[idx] == d)) for d in rows}
        draws = {d: counts[d] / len(rows[d]) for d in rows}
        assert all(float(k).is_integer() for k in draws.values()), draws
        assert sum(draws.values()) == 3  # 3 documents drawn per resample
        for d, r in rows.items():
            for i in r:
                assert int(np.sum(idx == i)) == draws[d]
    # Deterministic given the RNG seed.
    again = db.document_resamples(doc_ids, 50, np.random.default_rng(0))
    assert all(np.array_equal(a, b) for a, b in zip(res, again))


def test_resample_rng_keyed_by_name():
    a = db.resample_rng(0, "real", "pond").integers(0, 1 << 30, 5)
    assert np.array_equal(a, db.resample_rng(0, "real", "pond").integers(0, 1 << 30, 5))
    assert not np.array_equal(a, db.resample_rng(0, "real", "nfix").integers(0, 1 << 30, 5))
    assert not np.array_equal(a, db.resample_rng(0, "syn", "pond").integers(0, 1 << 30, 5))


def _fixture(seed=1, n_docs=12, per_doc=40):
    rng = np.random.default_rng(seed)
    f = rng.uniform(0.05, 0.95, n_docs * per_doc)
    y = rng.uniform(size=f.size) < f
    docs = np.repeat(np.arange(n_docs), per_doc)
    return f, y, docs


def test_point_estimates_are_relplot_and_compute_ece():
    f, y, docs = _fixture()
    out = db.doc_bootstrap_calibration(f, y, db.document_resamples(docs, 5, np.random.default_rng(0)))
    assert out["point"]["SmECE"] == relplot.smECE(f, y.astype(float), eps=0.001)
    d = relplot.prepare_rel_diagram(f, y, plot_confidence_band=False, report_CE_std=False)
    assert np.array_equal(out["line"], d["mu"]) and np.array_equal(out["mesh"], d["mesh"])
    assert out["sigma"] == d["sigma"]
    assert out["point"]["ECE"] == compute_ece(f, y, binning="equal_width", p=1)
    assert out["point"]["RMSCE_db"] == compute_ece(f, y, binning="equal_mass", p=2, debiased=True)
    assert out["n_boot"] == 5 and all(len(v) == 5 for v in out["replicates"].values())


def test_identical_documents_give_zero_width():
    # Every document holds the same rows, so every resample is the same multiset of rows:
    # every replicate equals the point estimate and the band collapses onto the line.
    f1 = np.array([0.1, 0.4, 0.6, 0.9, 0.3, 0.7])
    y1 = np.array([0, 0, 1, 1, 1, 0], dtype=bool)
    f, y = np.tile(f1, 5), np.tile(y1, 5)
    docs = np.repeat(np.arange(5), len(f1))
    out = db.doc_bootstrap_calibration(f, y, db.document_resamples(docs, 20, np.random.default_rng(3)))
    for m in db.METRICS:
        assert np.allclose(out["replicates"][m], out["point"][m]), m
        assert np.isclose(out["lo"][m], out["point"][m]) and np.isclose(out["hi"][m], out["point"][m])
    assert np.allclose(out["lower"], out["line"]) and np.allclose(out["upper"], out["line"])


def test_interval_brackets_and_is_percentile_of_replicates():
    f, y, docs = _fixture()
    out = db.doc_bootstrap_calibration(f, y, db.document_resamples(docs, 40, np.random.default_rng(0)))
    for m in db.METRICS:
        alpha = (1 - db.CI_LEVEL) / 2
        lo, hi = np.quantile(out["replicates"][m], [alpha, 1 - alpha])
        assert out["lo"][m] == lo and out["hi"][m] == hi
    assert np.all(out["lower"] <= out["upper"])
    assert out["drawn"].any() and out["drawn"].dtype == bool


# ── Training-pool resamples ──────────────────────────────────────────────────
def test_resampled_fit_sample_is_n_rows_of_a_document_resample():
    doc_ids = np.repeat(np.arange(6), 4)
    s = db.resampled_fit_sample(doc_ids, 10, db.fit_resample_rng(0, "pond", 3))
    assert len(s) == 10 and np.array_equal(s, np.sort(s))
    # Same RNG state: the document resample is the one document_resamples draws first, and
    # the positions are a without-replacement subset of it (multiset containment).
    expanded = db.document_resamples(doc_ids, 1, db.fit_resample_rng(0, "pond", 3))[0]
    vals, counts = np.unique(s, return_counts=True)
    assert all(counts[i] <= np.sum(expanded == v) for i, v in enumerate(vals))
    # Deterministic, and n = whole resampled pool takes every resampled row.
    assert np.array_equal(s, db.resampled_fit_sample(doc_ids, 10, db.fit_resample_rng(0, "pond", 3)))
    full = db.resampled_fit_sample(doc_ids, len(expanded), db.fit_resample_rng(0, "pond", 3))
    assert np.array_equal(full, np.sort(expanded))


def test_resampled_fit_sample_shares_documents_across_n():
    # One row per document: positions are document ids, so the sample at small n must be
    # drawn from the same resampled documents as the sample at the full size.
    doc_ids = np.arange(30)
    docs_full = set(db.resampled_fit_sample(doc_ids, 30, db.fit_resample_rng(1, "nfix", 0)).tolist())
    docs_small = set(db.resampled_fit_sample(doc_ids, 5, db.fit_resample_rng(1, "nfix", 0)).tolist())
    assert docs_small <= docs_full


def test_fit_resample_rng_streams_distinct():
    a = db.fit_resample_rng(0, "pond", 0).integers(0, 1 << 30, 5)
    assert np.array_equal(a, db.fit_resample_rng(0, "pond", 0).integers(0, 1 << 30, 5))
    assert not np.array_equal(a, db.fit_resample_rng(0, "pond", 1).integers(0, 1 << 30, 5))
    assert not np.array_equal(a, db.fit_resample_rng(0, "nfix", 0).integers(0, 1 << 30, 5))
    # SeedSequence zero-pads entropy: r=0 must not reproduce any resample_rng stream
    # (the collision the 0xF17 tag word exists to prevent).
    for dtype, ds in [("fit", "pond"), ("real", "pond"), ("syn", "pond")]:
        assert not np.array_equal(a, db.resample_rng(0, dtype, ds).integers(0, 1 << 30, 5))


def test_two_class_fit_samples_no_skips_when_both_classes_certain():
    # Every document has one valid and one invalid row, and n is the whole resampled pool.
    doc_ids = np.repeat(np.arange(5), 2)
    y = np.tile([True, False], 5)
    kept, n_draws = db.two_class_fit_samples(doc_ids, y, 10, 7, seed=2, ds="pond")
    assert n_draws == 7 and [r for r, _ in kept] == list(range(7))
    for r, s in kept:
        assert np.array_equal(s, db.resampled_fit_sample(doc_ids, 10, db.fit_resample_rng(2, "pond", r)))


def test_two_class_fit_samples_skips_single_class():
    # docs 0..3 valid, 4..7 invalid, one row each: a size-2 sample from one half is skipped.
    doc_ids = np.arange(8)
    y = doc_ids < 4
    kept, n_draws = db.two_class_fit_samples(doc_ids, y, 2, 30, seed=0, ds="pond")
    assert len(kept) == 30 and n_draws > 30
    assert all(y[s].sum() == 1 for _, s in kept)
    rs = [r for r, _ in kept]
    skipped = sorted(set(range(n_draws)) - set(rs))
    assert rs == sorted(rs) and skipped
    assert all(y[db.resampled_fit_sample(doc_ids, 2, db.fit_resample_rng(0, "pond", r))].sum() in (0, 2)
               for r in skipped)
    assert db.two_class_fit_samples(doc_ids, y, 2, 30, seed=0, ds="pond")[1] == n_draws


def test_two_class_fit_samples_fails_loud_on_single_class_pool():
    import pytest
    with pytest.raises(RuntimeError):
        db.two_class_fit_samples(np.arange(10), np.zeros(10, dtype=bool), 3, 4, seed=0, ds="pond")
