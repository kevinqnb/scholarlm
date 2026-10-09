"""Shared drawing helpers for reliability-diagram plots (support masking, curve drawing)."""
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


def support_mask(probs, mesh, sigma):
    """Mask of mesh points within the predictions' range, padded by one kernel width.

    relplot's kernel smoother has no data far from the predictions, so the curve
    there is an artifact rather than an estimate; those points should not be drawn.

    Args:
        probs: 1-D predicted probabilities.
        mesh: 1-D x-grid the curve is evaluated on.
        sigma: Kernel bandwidth used as padding.

    Returns:
        Boolean array, True for mesh points in [min(probs) - sigma, max(probs) + sigma].
    """
    probs, mesh = np.asarray(probs, dtype=float), np.asarray(mesh, dtype=float)
    assert probs.ndim == 1 and len(probs) > 0 and np.isfinite(probs).all()
    assert mesh.ndim == 1 and sigma > 0, sigma
    mask = (mesh >= probs.min() - sigma) & (mesh <= probs.max() + sigma)
    assert mask.any(), 'support mask excludes the whole mesh'
    return mask


# Floor on the density-normalized alpha of the curve, so low-density regions fade
# toward transparent without a segment fully disappearing.
CURVE_DENSITY_ALPHA_FLOOR = 0.15

# Dash pattern as (period, on) in mesh points. matplotlib's '--' renders solid on a
# LineCollection of 2-point segments, so dashes are made by dropping "off" segments.
DASH_PERIOD, DASH_ON = 10, 7


def draw_reliability_curve(ax, summary, color, *, linestyle, lw, line_zorder, band_zorder):
    """Draw one nested_bootstrap summary: a density-faded curve plus its percentile band.

    Args:
        ax: Matplotlib axes to draw on.
        summary: Dict with arrays ``mesh``, ``line``, ``density``, ``drawn`` (bool mask),
            ``lower`` and ``upper`` (band edges), all the same shape.
        color: Matplotlib color for line and band.
        linestyle: ``'-'`` or ``'--'``.
        lw: Line width.
        line_zorder: z-order of the curve.
        band_zorder: z-order of the band.
    """
    mesh, line, density, drawn = summary['mesh'], summary['line'], summary['density'], summary['drawn']
    assert mesh.shape == line.shape == density.shape == drawn.shape
    density_norm = density / density.max() if density.max() > 0 else np.ones_like(density)
    alpha = CURVE_DENSITY_ALPHA_FLOOR + (1 - CURVE_DENSITY_ALPHA_FLOOR) * density_norm

    points = np.array([mesh, line]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = np.tile(mcolors.to_rgba(color), (len(segments), 1))
    seg_colors[:, 3] = (alpha[:-1] + alpha[1:]) / 2
    keep = drawn[:-1] & drawn[1:]
    segments, seg_colors = segments[keep], seg_colors[keep]
    assert len(segments) > 0
    if linestyle == '--':
        dash = (np.arange(len(segments)) % DASH_PERIOD) < DASH_ON
        segments, seg_colors = segments[dash], seg_colors[dash]
    else:
        assert linestyle == '-', linestyle

    ax.add_collection(LineCollection(segments, colors=seg_colors, lw=lw, capstyle='round', zorder=line_zorder))
    ax.fill_between(mesh, summary['lower'], summary['upper'], where=drawn,
                    color=color, alpha=0.20, linewidth=0, zorder=band_zorder)
