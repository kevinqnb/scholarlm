"""Document-level bootstrap of an evaluation set's calibration, for calibration_updated_v4.py.

Every smoothed quantity is relplot's own: ``relplot.prepare_rel_diagram`` gives the
SmECE, the reliability curve on relplot's fixed mesh, the prediction density, and the
bandwidth it chose for those. It is called once on the full evaluation set (the point
estimate) and once per document resample; this module only supplies the resamples and
takes percentiles. relplot's own bootstraps (BaggingRegressor bands, a scipy CI on the
smECE) resample rows, not documents, and are switched off.

Binned calibration errors (scholarlm's compute_ece) are computed on the same resamples.

Import-side-effect free so it can be unit tested on a hand-built fixture.
"""
from __future__ import annotations

import zlib

import numpy as np
import relplot

from analysis.calibration_plot_utils import support_mask
from scholarlm.utils.calibration import compute_ece

CI_LEVEL = 0.95
METRICS = ('SmECE', 'ECE', 'ECE_em', 'RMSCE_db')


def resample_rng(seed: int, dtype: str, test_ds: str) -> np.random.Generator:
    """RNG for one evaluation set's document resamples, keyed by its name."""
    return np.random.default_rng([seed, zlib.crc32(f'{dtype}/{test_ds}'.encode())])


def document_resamples(doc_ids, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """``n_boot`` document-level resamples of the rows, as row-index arrays.

    Each resample draws as many documents as there are, uniformly with replacement,
    and takes every row of each drawn document once per draw.
    """
    doc_ids = np.asarray(doc_ids)
    assert doc_ids.ndim == 1 and len(doc_ids) > 0 and n_boot > 0, (doc_ids.shape, n_boot)
    _, inverse = np.unique(doc_ids, return_inverse=True)
    rows_by_doc = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    n_docs = len(rows_by_doc)
    return [np.concatenate([rows_by_doc[d] for d in rng.integers(0, n_docs, n_docs)]) for _ in range(n_boot)]


def _relplot(probs, labels) -> dict:
    # Point curve + smECE only: no row-level bootstrap band, no scipy CI.
    return relplot.prepare_rel_diagram(probs, labels, plot_confidence_band=False, report_CE=True,
                                       report_CE_std=False)


def _errors(probs, labels, diagram) -> dict:
    return dict(
        SmECE=float(diagram['ce']),
        ECE=compute_ece(probs, labels, binning='equal_width', p=1),
        ECE_em=compute_ece(probs, labels, binning='equal_mass', p=1),
        RMSCE_db=compute_ece(probs, labels, binning='equal_mass', p=2, debiased=True),
    )


def doc_bootstrap_calibration(probs, labels, resamples) -> dict:
    """Calibration of one evaluation set with document-bootstrap intervals.

    Args:
        probs, labels: the evaluation rows' predictions and binary labels.
        resamples: row-index arrays from ``document_resamples``.
    Returns:
        dict with ``point`` / ``lo`` / ``hi`` per metric in METRICS (point = full set,
        interval = percentiles over resamples) and ``replicates`` (per metric, one value
        per resample); the curve ``mesh``, ``line`` (relplot's curve on the full set),
        ``lower`` / ``upper`` (pointwise percentiles of the resampled relplot curves),
        ``density`` and ``sigma`` (relplot's, full set) and ``drawn`` (support_mask at that
        sigma); and ``n_boot``.
    """
    f = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=bool)
    assert f.ndim == 1 and f.shape == y.shape and len(f) > 0 and np.isfinite(f).all(), (f.shape, y.shape)
    assert len(resamples) > 0 and all(len(i) > 0 and i.min() >= 0 and i.max() < len(f) for i in resamples)

    full = _relplot(f, y)
    point = _errors(f, y, full)
    reps = {m: np.empty(len(resamples)) for m in METRICS}
    curves = np.empty((len(resamples), len(full['mesh'])))
    for s, idx in enumerate(resamples):
        d = _relplot(f[idx], y[idx])
        assert np.array_equal(d['mesh'], full['mesh'])
        for m, v in _errors(f[idx], y[idx], d).items():
            reps[m][s] = v
        curves[s] = d['mu']

    alpha = (1 - CI_LEVEL) / 2
    lower, upper = np.quantile(curves, [alpha, 1 - alpha], axis=0)
    out = dict(point=point, lo={}, hi={}, replicates=reps, mesh=full['mesh'], line=full['mu'],
               lower=lower, upper=upper, density=full['density'], sigma=float(full['sigma']),
               drawn=support_mask(f, full['mesh'], full['sigma']), n_boot=len(resamples))
    for m in METRICS:
        out['lo'][m], out['hi'][m] = (float(q) for q in np.quantile(reps[m], [alpha, 1 - alpha]))
    return out
