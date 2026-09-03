"""Row of 2-D PCA scatters — one panel per collected layer.

Loads representations.npz (per-layer n x 4096 hidden-state arrays for
pond/lake/wetland key-term tokens) and plots, for every layer present in the
npz (or the ``--layers`` subset), the first two principal components of THAT
layer's representations, coloured by key term.

Each panel is an INDEPENDENT PCA fit: PC1/PC2 of one panel are unrelated to
those of any other — different basis, arbitrary per-panel sign and rotation,
and per-panel axis scales that differ by ~200x across layers (mean row L2
runs 0.68 at layer 0 to ~144 at the post-final-norm layer). Do not read a
cluster as "moving" across panels; that is noise. Exploratory look only — no
claim about separability is being made here.

Note: the layer-0 panel looks near-empty on purpose — token embeddings are a
lookup, so all occurrences of a given surface form ("pond"/"ponds"/...) map to
the same point and the whole dataset collapses to a handful of stacked dots.

Usage:
    uv run python analysis/representation_lm_pca.py \
        --npz data/experiments/pond/representation_lm/llama-3.1-8b-base/<date>/representations.npz \
        [--layers 0 8 16 24 32]
"""
from __future__ import annotations

import argparse
import json
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
    p.add_argument(
        "--layers", nargs="+", type=int, default=None, metavar="L",
        help=(
            "Subset of collected layers to plot (default: every layer in the "
            "npz). Order is forced to the npz's ascending layer order."
        ),
    )
    p.add_argument("--seed", type=int, default=342, help="RNG seed (plot-order shuffle + randomized PCA).")
    args = p.parse_args(argv)

    z = np.load(args.npz, allow_pickle=False)
    available = [int(L) for L in z["layers"]]

    # Boundary: the npz's rep_layer_* members must be exactly the declared layers.
    declared_members = {f"rep_layer_{L:02d}" for L in available}
    found_members = {f for f in z.files if f.startswith("rep_layer_")}
    assert declared_members == found_members, (
        f"npz rep_layer_* members {sorted(found_members)} != declared layers "
        f"{sorted(declared_members)} — partially written or mismatched npz"
    )

    if args.layers is None:
        sel = list(available)
    else:
        missing = [L for L in args.layers if L not in available]
        if missing:
            raise SystemExit(f"layers {missing} not in this npz; available: {available}")
        sel = [L for L in available if L in set(args.layers)]  # npz ascending order
    assert sel, "no layers selected"

    # run_metadata.json is the independent record of each layer's mean row L2 —
    # a hard cross-check that npz member rep_layer_{L} really is layer L. Its
    # absence is a hard error (CLAUDE.md: no silent fallbacks).
    meta_path = args.npz.parent / "run_metadata.json"
    if not meta_path.exists():
        raise SystemExit(
            f"missing {meta_path} — cannot cross-check per-layer row L2 norms"
        )
    meta = json.loads(meta_path.read_text())
    meta_l2 = meta["mean_row_l2_by_layer"]  # {str(L): float}

    labels = z["labels"]
    model_name = z["model_name"].item()
    hidden = int(z["hidden_size"])
    n = len(labels)
    terms = list(z["key_terms"])
    assert set(np.unique(labels)) <= set(terms)
    assert set(terms) <= set(TERM_COLOR), (
        f"key terms {terms} include one with no palette entry {sorted(TERM_COLOR)}"
    )
    print(f"Loaded n={n} from {args.npz}")
    for t in terms:
        print(f"  {t:8s} {int((labels == t).sum())}")
    print(f"Layers available : {available}")
    print(f"Layers plotted   : {sel}")

    # One shuffle, shared across panels, so no class sits systematically on top.
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(n)
    Ls = labels[order]
    point_colors = np.array([TERM_COLOR[t] for t in Ls])

    fig, axes = plt.subplots(
        1, len(sel), figsize=(3.2 * len(sel), 4.6), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, L in zip(axes, sel):
        key = f"rep_layer_{L:02d}"
        X = z[key]
        assert X.shape == (n, hidden), (L, X.shape)
        assert np.isfinite(X).all(), f"non-finite representations at layer {L}"

        obs_l2 = float(np.linalg.norm(X, axis=1).mean())
        exp_l2 = float(meta_l2[str(L)])
        assert abs(obs_l2 - exp_l2) <= 1e-3 * exp_l2, (
            f"layer {L}: mean row L2 {obs_l2:.6g} != run_metadata {exp_l2:.6g} "
            f"— npz member {key} may be mislabelled"
        )

        # svd_solver='randomized' (fixed random_state): matches 'full' EVR to
        # ~6 dp on this shape at ~2.6s vs ~45s/layer — the login node is shared.
        pca = PCA(n_components=2, svd_solver="randomized", random_state=args.seed)
        Y = pca.fit_transform(X)  # PCA centers internally
        del X
        evr = pca.explained_variance_ratio_
        Ys = Y[order]

        ax.scatter(Ys[:, 0], Ys[:, 1], s=5, c=point_colors,
                   alpha=0.22, linewidths=0, rasterized=True)

        ax.set_title(
            f"layer {L}\nPC1 {evr[0] * 100:.1f}% · PC2 {evr[1] * 100:.1f}%",
            fontsize=10, color=INK,
        )
        ax.axhline(0, color=MUTED, lw=0.6, alpha=0.4, zorder=0)
        ax.axvline(0, color=MUTED, lw=0.6, alpha=0.4, zorder=0)
        ax.set_xlabel("PC1", fontsize=9, color=MUTED)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)

    axes[0].set_ylabel("PC2", fontsize=9, color=MUTED)

    # Legend: opaque proxy handles (the scatter points are alpha 0.22).
    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=7, mec="none", mfc=TERM_COLOR[t], label=t)
        for t in terms
    ]
    axes[0].legend(
        handles=handles, loc="upper left", frameon=False, fontsize=9,
        labelcolor=INK, handletextpad=0.3,
    )

    out_dir = paths.figures_dir("pond")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "-".join(f"{L:02d}" for L in sel)
    for ext in ("png", "pdf"):
        fp = out_dir / (
            f"representation_lm_pca_layers_{model_name.split('/')[-1]}_L{tag}.{ext}"
        )
        fig.savefig(fp, dpi=150, bbox_inches="tight")
        print(f"wrote {fp}")


if __name__ == "__main__":
    main()
