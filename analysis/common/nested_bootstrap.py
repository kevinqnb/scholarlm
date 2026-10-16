"""Nested bootstrap CIs for calibration errors and reliability curves.

A recalibrated test cell has two noise sources: which rows were labelled for the
recalibration fit, and which documents form the test set. Each replicate pairs one
fit sample with one test-document resample. Un-recalibrated (synthetic) cells pass a
single prediction set.

- Point estimates average the un-resampled values over fit samples; resamples only
  set the interval, since averaging |gap|-type statistics over resamples biases up.
- smECE picks its own bandwidth per replicate; curves share one bandwidth per cell
  so the band reflects sampling noise only.
"""
import zlib

import numpy as np
import relplot
from relplot.kernels import ReflectedGaussianKernel
from sklearn.metrics import roc_auc_score, brier_score_loss

from analysis.common.calibration_plot_utils import support_mask
from analysis.common.metrics import validity_rate_from_labels
from scholarlm.utils.calibration import apply_platt, compute_ece

# relplot's own grid and kernel: curves match what prepare_rel_diagram would draw.
assert not relplot.config.use_logit_scaling
MESH = np.linspace(0, 1, relplot.config.plot_mesh_pts)
KERNEL = ReflectedGaussianKernel
SMECE_EPS = 0.001  # prepare_rel_diagram's bandwidth-search floor

CI_LEVEL = 0.95
# A mesh point is drawn only where this share of replicates has data within a bandwidth.
SUPPORT_FRACTION = 0.95

METRICS = ('SmECE', 'ECE', 'ECE_em', 'RMSCE_db')


def fit_sample_rng(seed: int, r: int) -> np.random.Generator:
    """RNG for recalibration fit sample ``r``.

    Args:
        seed: Global seed.
        r: Fit-sample index (>= 0).

    Returns:
        Generator seeded by ``[seed, r]``.
    """
    assert r >= 0, r
    return np.random.default_rng([seed, r])


def pool_resampled_fit_sample(pool_doc_ids, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one recalibration fit sample from a document-resampled pool.

    Resampling the pool's documents first models a different set of papers forming
    the pool. Rows from a document drawn k times can appear up to k times.

    Args:
        pool_doc_ids: Per-row document id of the pool.
        n: Sample size.
        rng: Random generator.

    Returns:
        Sorted array of ``n`` pool positions (may repeat).
    """
    expanded = cluster_bootstrap_indices(pool_doc_ids, 1, rng)[0]
    assert len(expanded) >= n, f'resampled pool has {len(expanded)} rows < n={n}'
    return np.sort(rng.choice(expanded, n, replace=False))


def doc_boot_rng(seed: int, dtype: str, test_ds: str) -> np.random.Generator:
    """RNG for one test set's document resamples, distinct from ``fit_sample_rng``.

    Args:
        seed: Global seed.
        dtype: Dataset name, part of the stream key.
        test_ds: Test-set name, part of the stream key.

    Returns:
        Generator seeded by ``[seed, 0xD0C, crc32("dtype/test_ds")]``.
    """
    return np.random.default_rng([seed, 0xD0C, zlib.crc32(f'{dtype}/{test_ds}'.encode())])


def cluster_bootstrap_indices(doc_ids, n_boot: int, rng: np.random.Generator) -> list:
    """Draw document-level (cluster) bootstrap resamples of the rows.

    Args:
        doc_ids: Per-row document id (1-D, non-empty).
        n_boot: Number of resamples.
        rng: Random generator.

    Returns:
        List of ``n_boot`` row-index arrays; each includes every row of each drawn
        document once per draw.
    """
    doc_ids = np.asarray(doc_ids)
    assert doc_ids.ndim == 1 and len(doc_ids) > 0, doc_ids.shape
    assert n_boot > 0, n_boot
    _, inverse = np.unique(doc_ids, return_inverse=True)
    rows_by_doc = [np.flatnonzero(inverse == k) for k in range(inverse.max() + 1)]
    n_docs = len(rows_by_doc)
    return [np.concatenate([rows_by_doc[d] for d in rng.integers(0, n_docs, n_docs)])
            for _ in range(n_boot)]


def calibration_errors(probs, labels) -> dict:
    """smECE and the three binned calibration errors for one set of rows.

    Args:
        probs: Predicted probabilities.
        labels: Binary labels.

    Returns:
        Dict with ``SmECE``, ``sigma`` (smECE's bandwidth), ``ECE``, ``ECE_em``, ``RMSCE_db``.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(labels, dtype=float)
    smece, sigma = relplot.metrics.smECE(probs, y, eps=SMECE_EPS, return_width=True)
    return dict(
        SmECE=float(smece), sigma=float(sigma),
        ECE=compute_ece(probs, y, binning='equal_width', p=1),
        ECE_em=compute_ece(probs, y, binning='equal_mass', p=1),
        RMSCE_db=compute_ece(probs, y, binning='equal_mass', p=2, debiased=True),
    )


def smooth_curve(probs, labels, sigma: float, mesh=MESH) -> np.ndarray:
    """relplot's kernel-smoothed reliability curve at a fixed bandwidth.

    Args:
        probs: Predicted probabilities.
        labels: Binary labels.
        sigma: Kernel bandwidth.
        mesh: x-grid to evaluate on.

    Returns:
        Smoothed label rate at each mesh point.
    """
    ys, _ = KERNEL(sigma).smooth(np.asarray(probs, dtype=float), np.asarray(labels, dtype=float), mesh)
    return ys


def nested_bootstrap(pred_sets, labels, boot_idx, mesh=MESH) -> dict:
    """Calibration errors and reliability curve over every (fit sample, resample) pair.

    Args:
        pred_sets: Test-row predictions, one array per fit sample (one array if the
            cell is not recalibrated).
        labels: Test-row labels.
        boot_idx: Test-row resamples from ``cluster_bootstrap_indices``, shared by all
            fit samples.
        mesh: x-grid for the curve.

    Returns:
        Dict with:
            - ``base``: per metric, un-resampled value per fit sample
            - ``replicates``: per metric, (n_fit x n_boot) array
            - ``point``, ``lo``, ``hi``: per metric, mean of ``base`` and CI_LEVEL percentiles
            - ``sigma_curve``: shared curve bandwidth (median of base smECE bandwidths)
            - ``mesh``, ``line``: grid and mean un-resampled curve
            - ``lower``, ``upper``: pointwise percentile band
            - ``drawn``: mesh points with support in >= SUPPORT_FRACTION of replicates
            - ``density``: replicate-averaged prediction density
            - ``n_fit_samples``, ``n_doc_boot``
    """
    y = np.asarray(labels, dtype=bool)
    pred_sets = [np.asarray(p, dtype=float) for p in pred_sets]
    assert y.ndim == 1 and len(y) > 0, y.shape
    assert len(pred_sets) > 0 and len(boot_idx) > 0, (len(pred_sets), len(boot_idx))
    assert all(p.shape == y.shape and np.isfinite(p).all() for p in pred_sets)
    assert all(len(i) > 0 and i.min() >= 0 and i.max() < len(y) for i in boot_idx)
    n_fit, n_boot = len(pred_sets), len(boot_idx)

    base = [calibration_errors(p, y) for p in pred_sets]
    sigma_curve = float(np.median([b['sigma'] for b in base]))
    line = np.mean([smooth_curve(p, y, sigma_curve, mesh) for p in pred_sets], axis=0)

    reps = {m: np.empty((n_fit, n_boot)) for m in METRICS}
    curves = np.empty((n_fit * n_boot, len(mesh)))
    support = np.zeros(len(mesh))
    density = np.zeros(len(mesh))
    for r, p in enumerate(pred_sets):
        for s, idx in enumerate(boot_idx):
            f, yy = p[idx], y[idx]
            e = calibration_errors(f, yy)
            for m in METRICS:
                reps[m][r, s] = e[m]
            curves[r * n_boot + s] = smooth_curve(f, yy, sigma_curve, mesh)
            support += support_mask(f, mesh, sigma_curve)
            density += KERNEL(sigma_curve).kde(f, mesh)

    alpha = (1 - CI_LEVEL) / 2
    lower, upper = np.quantile(curves, [alpha, 1 - alpha], axis=0)
    out = dict(base={m: np.array([b[m] for b in base]) for m in METRICS}, replicates=reps,
               point={}, lo={}, hi={}, sigma_curve=sigma_curve, mesh=mesh, line=line,
               lower=lower, upper=upper, drawn=support / (n_fit * n_boot) >= SUPPORT_FRACTION,
               density=density / (n_fit * n_boot), n_fit_samples=n_fit, n_doc_boot=n_boot)
    for m in METRICS:
        out['point'][m] = float(out['base'][m].mean())
        out['lo'][m], out['hi'][m] = (float(q) for q in np.quantile(reps[m], [alpha, 1 - alpha]))
    assert out['drawn'].any(), 'no mesh point has support in enough replicates'
    return out


def cell_prediction_sets(cell: dict, method: str) -> list:
    """Test-row predictions under each of a cell's stored recalibration maps.

    Args:
        cell: Calibration cell with ``{method}_raw``, ``{method}_probs``, ``fit_maps``.
        method: ``"probe"`` or ``"ntp"``.

    Returns:
        List with one prediction array per fit sample, or ``[raw]`` if ``fit_maps``
        is None.
    """
    raw = cell[f'{method}_raw']
    if cell['fit_maps'] is None:
        return [raw]
    sets = [apply_platt(raw, coef, icpt) for coef, icpt in cell['fit_maps'][method]]
    assert np.array_equal(sets[0], cell[f'{method}_probs']), 'fit sample 0 must be the stored predictions'
    return sets


def bootstrap_cells(setting_results: dict, seed: int, n_doc_boot: int, n_syn_boot: int,
                    methods=('probe', 'ntp')) -> dict:
    """Run ``nested_bootstrap`` on every cell.

    Resamples depend only on (dtype, test_ds), so every cell scored on the same test
    set shares them and their intervals are paired.

    Args:
        setting_results: ``{dtype: {judge: {train_ds: {test_ds: cell}}}}``.
        seed: Global seed.
        n_doc_boot: Resamples for recalibrated (real) cells.
        n_syn_boot: Resamples for un-recalibrated (synthetic) cells.
        methods: Score methods to bootstrap.

    Returns:
        ``{dtype: {judge: {train_ds: {test_ds: {method: nested_bootstrap summary}}}}}``.
    """
    out = {}
    for dtype, by_judge in setting_results.items():
        out[dtype] = {}
        for judge, by_train in by_judge.items():
            out[dtype][judge] = {}
            boot = {}
            for train_ds, by_test in by_train.items():
                out[dtype][judge][train_ds] = {}
                for test_ds, cell in by_test.items():
                    n_boot = n_syn_boot if cell['fit_maps'] is None else n_doc_boot
                    key = (test_ds, n_boot)
                    docs = np.asarray(cell['document_ids'])
                    mids = np.asarray(cell['measurement_ids'])
                    assert docs.shape == mids.shape == np.shape(cell['labels']), (dtype, train_ds, test_ds)
                    if key not in boot:
                        boot[key] = (mids, cluster_bootstrap_indices(docs, n_boot, doc_boot_rng(seed, dtype, test_ds)))
                    # Shared resamples index rows, so every cell on this test set must have the same rows.
                    assert np.array_equal(boot[key][0], mids), (dtype, train_ds, test_ds)
                    print(f'  nested bootstrap {dtype} {train_ds} -> {test_ds}: '
                          f'{1 if cell["fit_maps"] is None else len(cell["fit_maps"][methods[0]])} fit sample(s) '
                          f'x {n_boot} document resamples')
                    out[dtype][judge][train_ds][test_ds] = {
                        m: nested_bootstrap(cell_prediction_sets(cell, m), cell['labels'], boot[key][1])
                        for m in methods
                    }
    return out


def threshold_metrics(pred_sets, labels, threshold: float = 0.5) -> dict:
    """Classification metrics at a threshold, plus AUROC and Brier, averaged over fit samples.

    Undefined metrics (e.g. precision with no positives predicted) are NaN, except
    ``validity``, which is 0.0 in that case.

    Args:
        pred_sets: Predictions, one array per fit sample.
        labels: Binary labels.
        threshold: Predicted positive if prob > threshold.

    Returns:
        Dict of means: ``acc``, ``prec``, ``rec``, ``f1``, ``auroc``, ``bs``, ``bss``, ``validity``.
    """
    y = np.asarray(labels, dtype=bool)
    assert len(pred_sets) > 0
    rows = []
    for p in pred_sets:
        p = np.asarray(p, dtype=float)
        assert p.shape == y.shape, (p.shape, y.shape)
        pred = p > threshold
        tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
        fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
        prec = tp / (tp + fp) if tp + fp else float('nan')
        rec = tp / (tp + fn) if tp + fn else float('nan')
        bs = float(brier_score_loss(y, p))
        rate = float(y.mean())
        rows.append(dict(
            acc=(tp + tn) / len(y), prec=prec, rec=rec,
            f1=2 * prec * rec / (prec + rec) if prec + rec > 0 else float('nan'),
            auroc=roc_auc_score(y, p) if 0 < y.sum() < len(y) else float('nan'),
            bs=bs, bss=1.0 - bs / (rate * (1 - rate)) if 0 < rate < 1 else float('nan'),
            validity=validity_rate_from_labels(y, pred),
        ))
    return {k: float(np.mean([row[k] for row in rows])) for k in rows[0]}
