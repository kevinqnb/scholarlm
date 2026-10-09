"""Document-level bootstrap for calibration metrics and recalibration fit samples.

Rows within a paper are correlated, so resampling is done over documents. The
smoothed curve and SmECE come from relplot, with its row-level bootstraps turned
off. Binned ECEs come from scholarlm's compute_ece on the same resamples.
"""
from __future__ import annotations

import zlib

import numpy as np
import relplot

from analysis.common.calibration_plot_utils import support_mask
from scholarlm.utils.calibration import compute_ece

CI_LEVEL = 0.95
METRICS = ('SmECE', 'ECE', 'ECE_em', 'RMSCE_db')


def resample_rng(seed: int, dtype: str, test_ds: str) -> np.random.Generator:
    """RNG for one evaluation set's document resamples.

    Args:
        seed: Global seed.
        dtype: Dataset name, part of the stream key.
        test_ds: Evaluation-set name, part of the stream key.

    Returns:
        Generator seeded by ``[seed, crc32("dtype/test_ds")]``.
    """
    return np.random.default_rng([seed, zlib.crc32(f'{dtype}/{test_ds}'.encode())])


def document_resamples(doc_ids, n_boot: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Draw document-level bootstrap resamples of the rows.

    Each resample draws n_docs documents with replacement and includes every row of
    each drawn document once per draw.

    Args:
        doc_ids: Per-row document id (1-D, non-empty).
        n_boot: Number of resamples.
        rng: Random generator.

    Returns:
        List of ``n_boot`` row-index arrays.
    """
    doc_ids = np.asarray(doc_ids)
    assert doc_ids.ndim == 1 and len(doc_ids) > 0 and n_boot > 0, (doc_ids.shape, n_boot)
    _, inverse = np.unique(doc_ids, return_inverse=True)
    rows_by_doc = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    n_docs = len(rows_by_doc)
    return [np.concatenate([rows_by_doc[d] for d in rng.integers(0, n_docs, n_docs)]) for _ in range(n_boot)]


# ── Training-pool resamples (platt_scaling.py) ────────────────────────────
# Draw budget per wanted sample. Exceeding it means n is too large for the pool or too
# small for its label rate, so the run fails instead of looping.
MAX_DRAWS_PER_SAMPLE = 10


def fit_resample_rng(seed: int, ds: str, r: int) -> np.random.Generator:
    """RNG for training-pool resample ``r`` of dataset ``ds``.

    The constant 0xF17 word keeps these streams distinct from ``resample_rng``
    (SeedSequence zero-pads, so ``[seed, h, 0]`` would equal ``[seed, h]``).

    Args:
        seed: Global seed.
        ds: Dataset name.
        r: Resample index (>= 0).

    Returns:
        Generator for this resample.
    """
    assert r >= 0, r
    return np.random.default_rng([seed, 0xF17, zlib.crc32(ds.encode()), r])


def resampled_fit_sample(pool_doc_ids, n: int, rng: np.random.Generator) -> np.ndarray | None:
    """Draw one fit sample: resample documents, then draw ``n`` of the resampled rows.

    Documents are drawn first, so the same RNG state gives the same documents for any
    ``n``. A row whose document was drawn k times can appear up to k times.

    Args:
        pool_doc_ids: Per-row document id of the training pool.
        n: Sample size.
        rng: Random generator.

    Returns:
        Sorted pool positions (may repeat), or None if the document resample has
        fewer than ``n`` rows.
    """
    assert n > 0, n
    expanded = document_resamples(pool_doc_ids, 1, rng)[0]
    if len(expanded) < n:
        return None
    return np.sort(expanded[rng.choice(len(expanded), n, replace=False)])


def two_class_fit_samples(pool_doc_ids, pool_labels, n: int, n_samples: int, seed: int, ds: str):
    """Draw fit samples until ``n_samples`` contain both classes.

    Draws that are too short (fewer than ``n`` rows) or single-class are skipped and
    counted, since they condition the kept samples and should be reported.

    Args:
        pool_doc_ids: Per-row document id of the training pool.
        pool_labels: Per-row boolean labels of the pool.
        n: Sample size.
        n_samples: Number of usable samples wanted.
        seed: Global seed.
        ds: Dataset name (RNG key).

    Returns:
        Tuple of:
            - kept: list of ``(r, positions)`` for the usable draws
            - n_draws: total draws made
            - skips: ``{'short_pool': int, 'single_class': int}``

    Raises:
        RuntimeError: More than ``MAX_DRAWS_PER_SAMPLE * n_samples`` draws needed.
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
    """relplot's smoothed reliability diagram, without its row-level bootstrap or CI.

    Args:
        probs: Predicted probabilities.
        labels: Boolean labels.

    Returns:
        relplot diagram dict (``mesh``, ``mu``, ``density``, ``sigma``, ``ce``, ...).
    """
    return relplot.prepare_rel_diagram(probs, labels, plot_confidence_band=False, report_CE=True,
                                       report_CE_std=False)


def _errors(probs, labels, diagram) -> dict:
    """Every metric in METRICS for one set of rows.

    Args:
        probs: Predicted probabilities.
        labels: Boolean labels.
        diagram: ``_relplot`` output for the same rows (supplies SmECE).

    Returns:
        Dict mapping metric name to float.
    """
    return dict(
        SmECE=float(diagram['ce']),
        ECE=compute_ece(probs, labels, binning='equal_width', p=1),
        ECE_em=compute_ece(probs, labels, binning='equal_mass', p=1),
        RMSCE_db=compute_ece(probs, labels, binning='equal_mass', p=2, debiased=True),
    )


def doc_bootstrap_calibration(probs, labels, resamples) -> dict:
    """Calibration metrics and reliability curve with document-bootstrap intervals.

    Args:
        probs: Predicted probabilities of the evaluation rows.
        labels: Boolean labels of the evaluation rows.
        resamples: Row-index arrays from ``document_resamples``.

    Returns:
        Dict with:
            - ``point``, ``lo``, ``hi``: per-metric full-set value and CI_LEVEL percentiles
            - ``replicates``: per-metric array, one value per resample
            - ``mesh``, ``line``: relplot curve on the full set
            - ``lower``, ``upper``: pointwise percentiles of the resampled curves
            - ``density``, ``sigma``: relplot's density and bandwidth on the full set
            - ``drawn``: ``support_mask`` at that bandwidth
            - ``n_boot``: number of resamples
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
