"""
Calibration metrics for judge model probabilities.

Supports two probability sources:
1. **Probe probabilities** — ``predict_proba`` output from a trained probe
   (``scholarlm.utils.probe``).
2. **Next-token probabilities** — ``judgement_p_true`` values stored by
   ``run_judge_local.py`` in ``responses.json`` (local judges) or frontier judge
   response files.

Typical usage
-------------
    from scholarlm.utils.calibration import compute_ece, reliability_diagram_data
    import numpy as np

    probs = np.array([0.9, 0.7, 0.3, 0.1, ...])  # predicted P(valid)
    labels = np.array([1, 1, 0, 0, ...])           # ground truth

    ece = compute_ece(probs, labels)
    diag = reliability_diagram_data(probs, labels)
"""
from __future__ import annotations

import numpy as np

from scipy.special import logit, expit


def intercept_adjustment(
    probs: np.ndarray,
    pi_tr: float,
    pi_te: float,
    eps: float = 1e-12,
):
    """Adjust predicted probabilities for label shift via intercept adjustment.

    Args:
        probs: Predicted probabilities P_train(Y=1 | x) on the test set. Shape ``(n_test,)``.
        pi_tr: Training prevalence as a scalar in ``(0, 1)``.
        pi_te: Test prevalence as a scalar in ``(0, 1)``.
        eps: Numerical stability constant.
    Returns:
        Rescaled probabilities under test prevalence pi_te, shape ``(n_test,)``.
    """
    if not (eps < pi_tr < 1 - eps):
        raise ValueError(
            f"Training prevalence pi_tr={pi_tr:.4g} is degenerate; cannot rescale."
        )
    if not (eps < pi_te < 1 - eps):
        raise ValueError(
            f"Test prevalence pi_te={pi_te:.4g} is degenerate; cannot rescale."
        )

    probs = np.asarray(probs, dtype=float)
    probs_clipped = np.clip(probs, eps, 1 - eps)

    log_odds = logit(probs_clipped)
    log_prior_odds_tr = logit(pi_tr)
    log_prior_odds_te = logit(pi_te)
    log_odds_adjusted = log_odds + (log_prior_odds_te - log_prior_odds_tr)
    return expit(log_odds_adjusted)
    


def rescale_probabilities_em(
    probs: np.ndarray,
    train_labels: np.ndarray | None = None,
    *,
    pi_tr: float | None = None,
    max_iter: int = 1000,
    tol: float = 1e-8,
    init_pi_te: float | None = None,
    return_history: bool = False,
    eps: float = 1e-12,
) -> tuple:
    """Rescale predicted probabilities under label shift via Saerens et al. (2002) EM.

    Estimates the test-set prevalence from unlabeled test predictions and adjusts
    the probabilities accordingly.  Assumes label shift (P(X|Y) constant across
    domains) and that the source-domain probabilities are calibrated.

    Args:
        probs: Predicted probabilities P_train(Y=1 | x) on the test set. Shape ``(n_test,)``.
        train_labels: Binary training labels (0/1). Used only to estimate the
            training prevalence.  Mutually exclusive with ``pi_tr``.
        pi_tr: Training prevalence as a scalar in ``(0, 1)``.  Use this when
            the full label array is unavailable (e.g., when loading a saved
            probe).  Mutually exclusive with ``train_labels``.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on the change in estimated test prevalence.
        init_pi_te: Initial guess for test prevalence. Defaults to the training
            prevalence (neutral starting point).
        return_history: If True, also return the list of pi_te estimates per iteration.
        eps: Numerical stability constant.

    Returns:
        ``(rescaled, pi_te_hat)`` — rescaled probabilities and estimated test
        prevalence.  If ``return_history=True``, returns ``(rescaled, pi_te_hat,
        history)``.

    Reference:
        Saerens, Latinne, and Decaestecker (2002), Neural Computation 14(1):21-41.
    """
    if train_labels is None and pi_tr is None:
        raise ValueError("Provide exactly one of 'train_labels' or 'pi_tr'.")
    if train_labels is not None and pi_tr is not None:
        raise ValueError("Provide exactly one of 'train_labels' or 'pi_tr', not both.")

    probs = np.asarray(probs, dtype=float)

    if pi_tr is None:
        pi_tr = float(np.mean(np.asarray(train_labels)))
    if not (eps < pi_tr < 1 - eps):
        raise ValueError(
            f"Training prevalence pi_tr={pi_tr:.4g} is degenerate; cannot rescale."
        )

    probs_clipped = np.clip(probs, eps, 1 - eps)

    pi_te = float(init_pi_te) if init_pi_te is not None else pi_tr
    if init_pi_te is not None and not (eps < pi_te < 1 - eps):
        raise ValueError(f"init_pi_te={pi_te:.4g} must be strictly between 0 and 1.")

    history = [pi_te]

    for _ in range(max_iter):
        num = probs_clipped * (pi_te / pi_tr)
        den = num + (1 - probs_clipped) * ((1 - pi_te) / (1 - pi_tr))
        rescaled = num / den
        pi_te_new = float(np.mean(rescaled))
        history.append(pi_te_new)
        if abs(pi_te_new - pi_te) < tol:
            pi_te = pi_te_new
            break
        pi_te = pi_te_new

    num = probs_clipped * (pi_te / pi_tr)
    den = num + (1 - probs_clipped) * ((1 - pi_te) / (1 - pi_tr))
    rescaled = num / den

    if return_history:
        return rescaled, pi_te, history
    return rescaled, pi_te


def _ece_bin_edges(probs: np.ndarray, n_bins: int, binning: str) -> np.ndarray:
    """Return the ``n_bins + 1`` bin edges for the requested binning scheme.

    ``equal_width`` yields fixed edges ``linspace(0, 1, n_bins + 1)``.
    ``equal_mass`` (a.k.a. adaptive / quantile binning) places edges at the
    empirical quantiles of ``probs`` so that each bin holds roughly the same
    number of samples.  Ties can collapse adjacent quantiles, so the returned
    array may contain fewer than ``n_bins + 1`` unique edges.
    """
    if binning == "equal_width":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if binning == "equal_mass":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(probs, qs)
        # Pin the outer edges to [0, 1] and drop duplicates created by ties so
        # each interval is non-degenerate.
        edges[0], edges[-1] = 0.0, 1.0
        return np.unique(edges)
    raise ValueError(f"Unknown binning {binning!r}; use 'equal_width' or 'equal_mass'.")


def _ece_binned_stats(probs: np.ndarray, labels: np.ndarray, edges: np.ndarray):
    """Per-bin confidence, accuracy, and count for non-empty bins.

    Returns ``(conf, acc, count)`` arrays over the occupied bins only.  Bin
    assignment is left-closed/right-open with the upper boundary folded into the
    final bin, matching the original :func:`compute_ece` semantics.
    """
    # digitize against interior edges maps each point to a bin in [0, n_bins-1];
    # values equal to the top edge (e.g. 1.0) land in the last bin.
    idx = np.digitize(probs, edges[1:-1], right=False)
    n_bins = len(edges) - 1
    confs, accs, counts = [], [], []
    for b in range(n_bins):
        mask = idx == b
        c = int(mask.sum())
        if c == 0:
            continue
        confs.append(probs[mask].mean())
        accs.append(labels[mask].mean())
        counts.append(c)
    return np.array(confs), np.array(accs), np.array(counts, dtype=np.int64)


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    *,
    binning: str = "equal_width",
    p: int = 1,
    debiased: bool = False,
) -> float:
    """Compute a binned calibration error.

    Partitions predictions into ``n_bins`` confidence bins and aggregates the
    count-weighted per-bin gap between mean confidence and observed accuracy.

    With ``p=1`` (default) this is the standard Expected Calibration Error
    (ECE; Guo et al. 2017): the count-weighted mean of ``|acc - conf|``.  With
    ``p=2`` it is the root-mean-squared calibration error, the count-weighted
    mean of ``(acc - conf)**2`` under a final square root.

    Args:
        probs: Predicted probabilities for the positive class, shape ``(n,)``.
            Values should be in ``[0, 1]``.
        labels: Binary ground truth labels, shape ``(n,)``.  ``1`` / ``True``
            is positive.
        n_bins: Number of bins in ``[0, 1]``.
        binning: ``"equal_width"`` (fixed-width bins) or ``"equal_mass"``
            (adaptive quantile bins holding roughly equal counts).
        p: Norm of the calibration error.  ``1`` → ECE, ``2`` → RMS calibration
            error.  Debiasing is only defined for ``p=2``.
        debiased: If ``True`` (requires ``p=2``), apply the Kumar, Liang & Ma
            (NeurIPS 2019) bias correction.  The plug-in squared gap
            ``(acc - conf)**2`` overestimates ``(a - conf)**2`` by the sampling
            variance of ``acc``; subtracting the unbiased per-bin variance
            estimate ``acc * (1 - acc) / (n_b - 1)`` removes that bias.  The
            aggregate is clamped at 0 before the square root.

    Returns:
        Scalar calibration error.  Lower is better.  Plug-in ``p=1`` lies in
        ``[0, 1]``; the debiased ``p=2`` estimate is clamped at 0.

    Reference:
        Kumar, Liang, and Ma (2019), "Verified Uncertainty Calibration,"
        NeurIPS 2019 (debiased squared calibration error).
    """
    if p not in (1, 2):
        raise ValueError(f"p must be 1 or 2, got {p!r}.")
    if debiased and p != 2:
        raise ValueError(
            "Debiasing is only defined for the squared calibration error (p=2); "
            "there is no unbiased estimator for the L1 ECE."
        )

    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return 0.0

    edges = _ece_bin_edges(probs, n_bins, binning)
    conf, acc, count = _ece_binned_stats(probs, labels, edges)
    if len(count) == 0:
        return 0.0

    w = count / n
    if p == 1:
        return float(np.sum(w * np.abs(acc - conf)))

    # p == 2: squared calibration error (optionally debiased).
    sq = (acc - conf) ** 2
    if debiased:
        # Unbiased per-bin variance of acc: acc(1-acc)/(n_b - 1); 0 for singleton
        # bins, where acc(1-acc) is already 0.
        var = np.where(count > 1, acc * (1.0 - acc) / np.maximum(count - 1, 1), 0.0)
        sq = sq - var
    return float(np.sqrt(max(0.0, np.sum(w * sq))))


def bootstrap_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
    *,
    binning: str = "equal_width",
    p: int = 1,
    debiased: bool = False,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> dict[str, float]:
    """Point estimate and bootstrap confidence interval for a calibration error.

    Resamples ``(probs, labels)`` pairs with replacement ``n_boot`` times,
    recomputing the calibration error (with the requested ``binning`` / ``p`` /
    ``debiased`` options — quantile edges are re-derived on each resample) to
    obtain a percentile interval and standard error for the estimator.

    Args:
        probs: Predicted probabilities, shape ``(n,)``.
        labels: Binary ground truth labels, shape ``(n,)``.
        n_bins: Number of bins.
        binning: ``"equal_width"`` or ``"equal_mass"`` (see :func:`compute_ece`).
        p: Norm of the calibration error, ``1`` or ``2`` (see :func:`compute_ece`).
        debiased: Whether to bootstrap the debiased estimator (requires ``p=2``;
            see :func:`compute_ece`).
        n_boot: Number of bootstrap resamples.
        ci: Central coverage of the returned interval (e.g. ``0.95``).
        seed: Seed for the resampling RNG (reproducible).

    Returns:
        Dict with keys ``ece`` (point estimate on the full sample), ``ci_low`` /
        ``ci_high`` (percentile interval), ``se`` (bootstrap standard error),
        ``boot_mean`` (mean over resamples), and ``n_boot``.  When ``n < 2`` the
        interval fields are ``nan``.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    n = len(probs)
    point = compute_ece(probs, labels, n_bins, binning=binning, p=p, debiased=debiased)
    if n < 2:
        return dict(ece=point, ci_low=float("nan"), ci_high=float("nan"),
                    se=float("nan"), boot_mean=float("nan"), n_boot=0)

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[i] = compute_ece(
            probs[idx], labels[idx], n_bins, binning=binning, p=p, debiased=debiased
        )

    alpha = (1.0 - ci) / 2.0
    return dict(
        ece=point,
        ci_low=float(np.quantile(boots, alpha)),
        ci_high=float(np.quantile(boots, 1.0 - alpha)),
        se=float(boots.std(ddof=1)),
        boot_mean=float(boots.mean()),
        n_boot=n_boot,
    )


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute data for a reliability (calibration) diagram.

    Args:
        probs: Predicted probabilities for the positive class, shape ``(n,)``.
        labels: Binary ground truth labels, shape ``(n,)``.
        n_bins: Number of equal-width bins.

    Returns:
        Dict with keys:
        - ``"bin_centers"``  — midpoint of each bin, shape ``(n_bins,)``.
        - ``"bin_accuracy"`` — mean label in each bin (fraction positive),
          shape ``(n_bins,)``.  ``np.nan`` for empty bins.
        - ``"bin_confidence"`` — mean predicted probability in each bin,
          shape ``(n_bins,)``.  ``np.nan`` for empty bins.
        - ``"bin_counts"``   — number of samples in each bin, shape ``(n_bins,)``.
        - ``"ece"``          — scalar ECE value.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    bin_accuracy = np.full(n_bins, np.nan)
    bin_accuracy_sem = np.full(n_bins, np.nan)
    bin_confidence = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=np.int64)

    for i, (lo, hi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
        if hi == 1.0:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        count = mask.sum()
        bin_counts[i] = count
        if count > 0:
            p = labels[mask].mean()
            bin_accuracy[i] = p
            bin_confidence[i] = probs[mask].mean()
            bin_accuracy_sem[i] = np.sqrt(p * (1 - p) / count)

    return {
        "bin_centers": bin_centers,
        "bin_accuracy": bin_accuracy,
        "bin_accuracy_sem": bin_accuracy_sem,
        "bin_confidence": bin_confidence,
        "bin_counts": bin_counts,
        "ece": compute_ece(probs, labels, n_bins=n_bins),
    }
