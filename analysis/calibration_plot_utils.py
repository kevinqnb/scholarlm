import numpy as np


def support_mask(probs, mesh, sigma):
    """Boolean mask of mesh points inside the data's support, padded by one kernel width.

    relplot's smoother is mu(x) = sum K(x-p_i) y_i / (sum K(x-p_i) + 1e-4), so where
    no predictions lie near x the curve is an artifact (0/1e-4 -> 0, or an upward
    extrapolation), not an estimate. Only mesh points in [min(p) - sigma, max(p) + sigma]
    are drawn.
    """
    probs, mesh = np.asarray(probs, dtype=float), np.asarray(mesh, dtype=float)
    assert probs.ndim == 1 and len(probs) > 0 and np.isfinite(probs).all()
    assert mesh.ndim == 1 and sigma > 0, sigma
    mask = (mesh >= probs.min() - sigma) & (mesh <= probs.max() + sigma)
    assert mask.any(), 'support mask excludes the whole mesh'
    return mask
