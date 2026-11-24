"""Forest-style figure of recovery and validity point estimates with bootstrap intervals.

Two columns (recovery, validity); one subfigure row per group (a single row for the
main comparison, one per underlying model for ablations). Within a subfigure each
method is a y-axis label with its CI drawn as a capped horizontal line and its point
estimate as a marker; a dashed vertical line sits at the largest CI lower bound.
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.ticker import PercentFormatter

# Same serif face as the other analysis figures (analysis/common/pond_meta.PAPER_RCPARAMS);
# only the font keys are repeated here to avoid importing pond_meta's heavy dependencies.
_RC = {
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
_MONO = ["Nimbus Mono PS", "Liberation Mono", "DejaVu Sans Mono"]
_WHITE_MIX = 0.4  # fraction of white mixed into tab10
_MAX_COLORS = 10


def pastel_palette(n: int) -> list[tuple[float, float, float]]:
    """First ``n`` tab10 colours, each mixed with white.

    Args:
        n: Number of colours, at most 10.

    Returns:
        RGB tuples.

    Raises:
        ValueError: ``n`` exceeds tab10's 10 colours (they would silently repeat).
    """
    if not 1 <= n <= _MAX_COLORS:
        raise ValueError(f"need 1..{_MAX_COLORS} colours, got {n}")
    base = [to_rgb(c) for c in plt.get_cmap("tab10").colors[:n]]
    return [tuple((1 - _WHITE_MIX) * c + _WHITE_MIX for c in rgb) for rgb in base]


def _check_entry(e: dict) -> None:
    for key in ("recovery", "validity"):
        p, lo, hi = e[key]
        if not all(math.isfinite(v) for v in (p, lo, hi)) or not 0.0 <= lo <= p <= hi <= 1.0:
            raise ValueError(f"{e['label']}: {key} (point, lo, hi)={(p, lo, hi)} is not a valid interval in [0, 1]")


def plot_recovery_validity(
    entries: list[dict], color_keys: list[str], out_path: Path, *, group_font: str, label_font: str,
) -> None:
    """Draw the recovery/validity interval figure.

    Args:
        entries: Ordered dicts with ``label`` (y tick text), ``group`` (subfigure row),
            ``color_key`` (a member of ``color_keys``), and ``recovery`` / ``validity``
            as ``(point, ci_lo, ci_hi)`` in [0, 1]. Row order within a group and group
            order follow first appearance.
        color_keys: Every colour key in the experiment, in order; colour i is the i-th
            pastelised tab10 colour, so colours stay fixed across figures.
        out_path: Output figure path.
        group_font: Style of the group names: ``"mono"`` (model names) or ``"italic"``.
        label_font: Font of the y tick labels: ``"mono"`` (model/method names) or ``"serif"``.

    Raises:
        ValueError: No entries, an unknown colour key, or an invalid interval.
    """
    if group_font not in ("mono", "italic") or label_font not in ("mono", "serif"):
        raise ValueError(f"group_font must be 'mono'/'italic' and label_font 'mono'/'serif', got {group_font!r}, {label_font!r}")
    if not entries:
        raise ValueError("no entries to plot")
    if len(set(color_keys)) != len(color_keys):
        raise ValueError(f"duplicate colour keys: {color_keys}")
    colors = dict(zip(color_keys, pastel_palette(len(color_keys))))
    for e in entries:
        if e["color_key"] not in colors:
            raise ValueError(f"{e['label']}: colour key {e['color_key']!r} not in {color_keys}")
        _check_entry(e)

    groups: dict[str, list[dict]] = {}
    for e in entries:
        groups.setdefault(e["group"], []).append(e)
    n_groups = len(groups)
    row_height = 0.27
    height = sum(len(g) for g in groups.values()) * row_height + 0.6 * n_groups + 0.5

    with mpl.rc_context(_RC):
        fig, axes = plt.subplots(
            n_groups, 2, figsize=(7.4, height), sharex="col", squeeze=False,
            gridspec_kw={"height_ratios": [len(g) + 1.5 for g in groups.values()], "wspace": 0.15},
        )
        for r, (group, rows) in enumerate(groups.items()):
            for c, metric in enumerate(("recovery", "validity")):
                ax = axes[r, c]
                ys = np.arange(len(rows))
                for y, e in zip(ys, rows):
                    p, lo, hi = e[metric]
                    col = colors[e["color_key"]]
                    ax.hlines(y, lo, hi, color=col, lw=2.4, zorder=2, capstyle="round")
                    ax.vlines([lo, hi], y - 0.22, y + 0.22, color=col, lw=2.4, zorder=2, capstyle="round")
                    ax.scatter([p], [y], s=46, color=col, edgecolors="0.25", linewidths=0.9, zorder=3)
                frontier = max(e[metric][1] for e in rows)
                ax.axvline(frontier, color="0.35", ls="--", lw=0.9, zorder=1)
                ax.set_ylim(len(rows) - 0.5, -0.5)  # first row on top
                ax.set_yticks(ys)
                if c == 0:  # names only on the left column; the validity column shares the rows
                    ax.set_yticklabels([e["label"] for e in rows], family=_MONO if label_font == "mono" else None)
                else:
                    ax.set_yticklabels([])
                ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
                ax.grid(axis="x", alpha=0.25, lw=0.4)
                ax.set_axisbelow(True)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                ax.tick_params(axis="y", length=0)
                if r == 0:
                    ax.set_title(metric.capitalize())
                if n_groups > 1 and c == 0:
                    if group_font == "mono":
                        ax.set_ylabel(group, family=_MONO, fontsize=10, labelpad=8)
                    else:
                        ax.set_ylabel(group, style="italic", fontsize=11, labelpad=8)
        # Shared x-limits per column: all intervals plus a small pad, clipped to [0, 1].
        for c, metric in enumerate(("recovery", "validity")):
            lo = min(e[metric][1] for e in entries)
            hi = max(e[metric][2] for e in entries)
            pad = 0.04 * (hi - lo)
            axes[0, c].set_xlim(max(-0.02, lo - pad), min(1.02, hi + pad))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight", pad_inches=0.03)
        plt.close(fig)
