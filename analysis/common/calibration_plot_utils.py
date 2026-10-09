import numpy as np
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection


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


# Floor on the density-normalized alpha of the curve, so low-density regions fade
# toward transparent without a segment fully disappearing.
CURVE_DENSITY_ALPHA_FLOOR = 0.15

# Dashed curves as (period, on) in mesh-point units. matplotlib's own '--' restarts
# its dash offset on every segment of a LineCollection of 2-point segments (needed
# for per-segment alpha), which renders solid, so the "off" segments are dropped.
DASH_PERIOD, DASH_ON = 10, 7


def draw_reliability_curve(ax, summary, color, *, linestyle, lw, line_zorder, band_zorder):
    """Draw one nested_bootstrap summary: the line with density-weighted alpha, and
    its percentile band, both only on the summary's ``drawn`` mesh points."""
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
