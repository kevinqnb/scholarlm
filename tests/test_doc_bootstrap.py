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
