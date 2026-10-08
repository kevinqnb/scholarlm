"""Document-level bootstrap of an evaluation set's calibration, for calibration_updated_v4.py.

Every smoothed quantity is relplot's own: ``relplot.prepare_rel_diagram`` gives the
SmECE, the reliability curve on relplot's fixed mesh, the prediction density, and the
bandwidth it chose for those. It is called once on the full evaluation set (the point
estimate) and once per document resample; this module only supplies the resamples and
takes percentiles. relplot's own bootstraps (BaggingRegressor bands, a scipy CI on the
smECE) resample rows, not documents, and are switched off.

Binned calibration errors (scholarlm's compute_ece) are computed on the same resamples.

Also supplies platt_scaling_v2.py's training-pool resamples: the pool's documents
resampled, then n rows drawn from it, keeping only two-class draws.

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


# ── Training-pool resamples (platt_scaling_v2.py) ────────────────────────────
# A run fails rather than draw more than this many resamples per wanted resample:
# that many skipped draws means n is too large for the pool or too small for its
# label rate.
MAX_DRAWS_PER_SAMPLE = 10


def fit_resample_rng(seed: int, ds: str, r: int) -> np.random.Generator:
    """RNG for training-pool resample ``r`` of dataset ``ds``.

    SeedSequence zero-pads entropy, so ``[seed, h]`` and ``[seed, h, 0]`` are the same
    stream: a bare extra word would collide with resample_rng at r=0. The constant
    second word (0xF17) keeps these streams apart from resample_rng's ``[seed, crc32]``.
    """
    assert r >= 0, r
    return np.random.default_rng([seed, 0xF17, zlib.crc32(ds.encode()), r])


def resampled_fit_sample(pool_doc_ids, n: int, rng: np.random.Generator) -> np.ndarray | None:
    """One fit sample: ``n`` sorted positions into the pool (repeats possible), or None
    if the document resample has fewer than ``n`` rows.

    The pool's documents are resampled once (document_resamples), then ``n`` of the
    resampled rows are drawn uniformly without replacement. A row whose document was
    drawn k times has k copies to draw from; in a fit a repeat is a weight. The document
    resample is drawn first, so a fresh RNG in the same state resamples the same
    documents whatever ``n`` is. Documents differ in size, so a resample that misses
    the large ones can hold fewer than ``n`` rows; it then has no size-``n`` sample and
    None is returned for the caller to skip (two_class_fit_samples counts these).
    """
    assert n > 0, n
    expanded = document_resamples(pool_doc_ids, 1, rng)[0]
    if len(expanded) < n:
        return None
    return np.sort(expanded[rng.choice(len(expanded), n, replace=False)])


def two_class_fit_samples(pool_doc_ids, pool_labels, n: int, n_samples: int, seed: int, ds: str):
    """``n_samples`` fit samples of size ``n`` that contain both classes.

    Draw r is ``resampled_fit_sample(pool_doc_ids, n, fit_resample_rng(seed, ds, r))``
    for r = 0, 1, ... Two kinds of draw are skipped, and drawing continues at the next r:
      - short pool: the document resample has fewer than ``n`` rows;
      - single class: the sample has one class, so no recalibration map can be fit.
    Both condition the kept samples (on resampled pools of at least n rows, and on both
    classes being present); the skip counts are returned so the output can report them.

    Returns:
        (kept, n_draws, skips): the kept ``(r, positions)`` pairs, the number of draws
        made, and ``{'short_pool': int, 'single_class': int}`` (summing to
        n_draws - n_samples).
    Raises:
        RuntimeError past MAX_DRAWS_PER_SAMPLE * n_samples draws.
    """
    pool_labels = np.asarray(pool_labels, dtype=bool)
    assert len(pool_labels) == len(pool_doc_ids) > 0 and n_samples > 0, (len(pool_labels), len(pool_doc_ids))
    kept, r = [], 0
    skips = {'short_pool': 0, 'single_class': 0}
    while len(kept) < n_samples:
        if r >= MAX_DRAWS_PER_SAMPLE * n_samples:
            raise RuntimeError(f'{ds} n={n}: only {len(kept)}/{n_samples} usable fit samples in {r} draws '
                               f'(skipped: {skips})')
        sample = resampled_fit_sample(pool_doc_ids, n, fit_resample_rng(seed, ds, r))
        if sample is None:
            skips['short_pool'] += 1
        else:
            assert len(sample) == n
            if 0 < pool_labels[sample].sum() < n:
                kept.append((r, sample))
            else:
                skips['single_class'] += 1
        r += 1
    assert sum(skips.values()) == r - n_samples, (skips, r, n_samples)
    return kept, r, skips


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
