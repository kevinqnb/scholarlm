"""Quick 2-D PCA scatter of the RepresentationLM key-term representations.

Loads representations.npz (n x 4096 last-layer post-final-norm vectors for
pond/lake/wetland key-term tokens) and plots the first two principal
components, coloured by key term.

Exploratory look only — no claim about separability is being made here.

Usage:
    uv run python analysis/representation_lm_pca.py \
        --npz data/experiments/pond/representation_lm/llama-3.1-8b-base/2026_08_31/representations.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
import paths

# Categorical palette — dataviz skill reference palette, first 3 slots
# (pre-validated all-pairs, light mode). Fixed order, not cycled.
TERM_COLOR = {
    "pond": "#2a78d6",     # blue
    "lake": "#eb6834",     # orange
    "wetland": "#1baf7a",  # aqua
}
INK = "#1a1a19"
MUTED = "#6b6a63"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", required=True, type=Path, help="Path to representations.npz")
    p.add_argument("--seed", type=int, default=342, help="RNG seed for plot-order shuffle.")
    args = p.parse_args(argv)

    z = np.load(args.npz, allow_pickle=False)
    X = z["representations"].astype(np.float64)
    labels = z["labels"]
    model_name = z["model_name"].item()
    n, d = X.shape
    terms = list(z["key_terms"])
    assert set(np.unique(labels)) <= set(terms)
    print(f"Loaded {n} x {d} from {args.npz}")
    for t in terms:
        print(f"  {t:8s} {int((labels == t).sum())}")

    pca = PCA(n_components=2, svd_solver="full", random_state=args.seed)
    Y = pca.fit_transform(X)  # PCA centers internally
    evr = pca.explained_variance_ratio_
    print(f"Explained variance ratio: PC1 {evr[0]:.3f}, PC2 {evr[1]:.3f}")

    # Shuffle draw order so no class is systematically buried under another.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)

    fig, axes = plt.subplots(
        1, 4, figsize=(15, 4.2), sharex=True, sharey=True, constrained_layout=True
    )
    Ys, Ls = Y[order], labels[order]
    point_colors = np.array([TERM_COLOR[t] for t in Ls])
    panels = [("all key terms", terms)] + [(t, [t]) for t in terms]

    for ax, (title, show) in zip(axes, panels):
        if len(show) == 1:  # single-class facet
            m = Ls == show[0]
            ax.scatter(Ys[m, 0], Ys[m, 1], s=5, c=TERM_COLOR[show[0]],
                       alpha=0.22, linewidths=0, rasterized=True)
        else:  # combined panel — one interleaved scatter so no class sits on top
            ax.scatter(Ys[:, 0], Ys[:, 1], s=5, c=point_colors,
                       alpha=0.22, linewidths=0, rasterized=True)
        ax.set_title(title, fontsize=11, color=INK)
        ax.axhline(0, color=MUTED, lw=0.6, alpha=0.4, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.6, alpha=0.4, zorder=0)
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)", fontsize=9, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
    axes[0].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)", fontsize=9, color=MUTED)

    # Legend: opaque proxy handles (the scatter points are alpha 0.22).
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=7, mec="none", mfc=TERM_COLOR[t], label=t)
        for t in terms
    ]
    axes[0].legend(
        handles=handles, loc="upper left", frameon=False, fontsize=9,
        labelcolor=INK, handletextpad=0.3,
    )

    fig.suptitle(
        f"RepresentationLM key-term representations — 2-D PCA\n"
        f"{model_name}, last layer post-final-norm · n={n:,} · pond dataset",
        fontsize=12, color=INK,
    )

    out_dir = paths.figures_dir("pond")
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fp = out_dir / f"representation_lm_pca_{model_name.split('/')[-1]}.{ext}"
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
