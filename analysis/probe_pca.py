"""PCA visualization of real judgement activations against a trained head probe.

Loads the real (non-synthetic) judgement activations for the test subset --
extractions whose document was *not* part of the probe's synthetic training
set -- using the same construction as ``analysis/calibration.py``'s real-data
branch: documents held out of ``syn_document_ids``, labelled by
``judgement_combined OR ground-truth-match``. Plots a 2D PCA, fit over the
full test subset, of the probe's concatenated top-k-head activations,
colored by label.

Usage
-----
    python analysis/probe_pca.py
    python analysis/probe_pca.py --dataset nfix --extraction-model gemma-3-27b --judge-model mistral-7b
"""
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import argparse
import re

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

from analysis.loaders import (
    load_activations, load_combined_judgements, load_extraction,
    load_ground_truth, load_trained_probe, cached_match,
)
from analysis.ablation import get_matching_rules
from scholarlm.utils.unit_conversion import apply_unit_conversion
from experiments.run_extraction import load_dataset_config
import paths

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "font.size": 15, "axes.labelsize": 15, "axes.titlesize": 15,
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 12, "legend.title_fontsize": 13,
    "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 3, "ytick.major.size": 3,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "lines.linewidth": 1.2, "lines.markersize": 4,
    "legend.frameon": False,
    "figure.dpi": 150, "savefig.dpi": 300,
    "savefig.format": "pdf", "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

palette = sns.color_palette("husl", 10)
COLOR_VALID   = palette[7]  # blue
COLOR_INVALID = palette[0]  # red


_DATE_RE = re.compile(r'^\d{4}_\d{2}_\d{2}$')


def latest_dated_subdir(base: Path, required_file: str) -> str:
    """Return the most recent ``YYYY_mm_dd``-named subdirectory of ``base`` containing ``required_file``.

    Unlike a bare ``sorted(base.iterdir(), reverse=True)``, this ignores
    non-date directories (e.g. stray ``copy/`` or ``demo/`` scratch runs)
    that would otherwise sort ahead of real dated runs and be silently
    picked as "latest".
    """
    candidates = sorted(
        (d.name for d in base.iterdir() if d.is_dir() and _DATE_RE.match(d.name) and (d / required_file).exists()),
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No dated subdirectory of {base} contains {required_file}")
    return candidates[0]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', default='pond')
    p.add_argument('--extraction-model', default='gemma-3-27b')
    p.add_argument('--judge-model', default='qwen-2.5-7b')
    p.add_argument('--extraction-date', default=None, help='Auto-detects latest if omitted.')
    p.add_argument('--judge-date', default=None, help='Auto-detects latest if omitted.')
    return p.parse_args()


def main():
    args = parse_args()
    dataset, extraction_model, judge_model = args.dataset, args.extraction_model, args.judge_model

    # ── Load the cached head probe ──────────────────────────────────────
    probe_data = load_trained_probe(dataset, judge_model, ptype='head')
    top_k_heads = probe_data['top_k_heads']
    head_dim = probe_data['head_dim']
    syn_docs = set(probe_data['syn_document_ids'])

    # ── Resolve extraction date ─────────────────────────────────────────
    # paths.find_extraction_final's own auto-latest (bare reverse-sorted
    # iterdir) picks up stray non-date scratch directories (e.g. a "copy/"
    # that happens to sort after every real date and contains a leftover
    # final.json) -- resolve locally against a YYYY_mm_dd-only listing instead.
    if args.extraction_date is not None:
        assert _DATE_RE.match(args.extraction_date), f"--extraction-date must be YYYY_mm_dd, got {args.extraction_date!r}"
        extraction_date = args.extraction_date
    else:
        extraction_date = latest_dated_subdir(
            paths.EXPERIMENTS_ROOT / dataset / 'extraction' / extraction_model, 'final.json'
        )

    # ── Real extraction + combined judgements + ground truth (mirrors
    # analysis/calibration.py's test_data construction exactly) ─────────
    config = load_dataset_config(dataset)
    records = load_extraction(dataset, extraction_model, extraction_date)
    ext_df = pd.DataFrame(records)
    ext_df = apply_unit_conversion(ext_df, {})
    if dataset == 'nfix':
        ext_df['attribute'] = ext_df['attribute'].map({
            'nfix_rate_areal': 'nfix_rate', 'nfix_rate_volumetric': 'nfix_rate',
            'nfix_rate_mass':  'nfix_rate', 'nfix_rate': 'nfix_rate',
        })

    real_df = pd.DataFrame(load_combined_judgements(dataset, extraction_model, extraction_date))
    assert len(real_df) == len(ext_df), (
        f"combined.json ({len(real_df)} rows) and final.json ({len(ext_df)} rows) "
        f"disagree in length for {dataset}/{extraction_model}/{extraction_date}"
    )
    gt_df = load_ground_truth(config)

    strict, fuzzy, fuzzy_threshold = get_matching_rules(dataset)
    cache_path = paths.extraction(dataset, extraction_model, extraction_date) / 'match_cache.pkl'
    matching, edges, edge_weights = cached_match(
        gt_df, ext_df, strict_matching=strict, fuzzy_matching=fuzzy,
        fuzzy_threshold=0.0, cache_path=cache_path,
    )

    ex_edge_exists = np.zeros(len(ext_df), dtype=bool)
    for (gt_idx, ex_idx), w in zip(edges, edge_weights):
        if w > fuzzy_threshold:
            ex_edge_exists[int(ex_idx)] = True
    judge_labels = real_df['judgement_combined'].to_numpy(dtype=bool)
    combined_labels = judge_labels | ex_edge_exists

    # ── Test subset: extractions from documents held out of the probe's
    # synthetic training set ────────────────────────────────────────────
    mask = ~real_df['document_id'].isin(syn_docs)
    idx = np.where(mask.to_numpy())[0]
    assert len(idx) > 0, f"No test-subset rows left for {dataset} after excluding {len(syn_docs)} training documents"

    mids = real_df['measurement_id'].iloc[idx].tolist()
    labels = combined_labels[idx]
    print(f'[probe_pca] test subset: {len(idx)} rows '
          f'({labels.sum()} valid / {(~labels).sum()} invalid), '
          f'{len(syn_docs)} training documents excluded')

    # ── Load real judgement activations and assemble the probe's feature space ─
    judge_dir = paths.EXPERIMENTS_ROOT / dataset / 'judge' / extraction_model / extraction_date / judge_model
    if args.judge_date is not None:
        assert _DATE_RE.match(args.judge_date), f"--judge-date must be YYYY_mm_dd, got {args.judge_date!r}"
        judge_date = args.judge_date
    else:
        judge_date = latest_dated_subdir(judge_dir, 'attention_outputs.npz')
    act_path = judge_dir / judge_date / 'attention_outputs.npz'
    assert act_path.exists(), f"No attention_outputs.npz at {act_path}"
    real_act = np.load(act_path)

    X = np.concatenate([
        np.stack([np.array(real_act[str(mid)], dtype=np.float32)[l, h, :] for mid in mids], axis=0)
        for l, h in top_k_heads
    ], axis=1)
    assert X.shape == (len(mids), len(top_k_heads) * head_dim), (
        f"Feature matrix shape {X.shape} != expected {(len(mids), len(top_k_heads) * head_dim)}"
    )
    assert not np.isnan(X).any(), "NaN values in loaded activations"

    print(f'[probe_pca] dataset={dataset} extraction_model={extraction_model} judge_model={judge_model} '
          f'extraction_date={extraction_date} judge_date={judge_date}')

    # ── Standardize features for PCA (matches the probe's own preprocessing;
    # without this, PCA is dominated by whichever head has the largest raw
    # variance rather than by class-relevant structure) ─────────────────
    x_mean, x_std = X.mean(axis=0), X.std(axis=0)
    assert np.all(x_std > 0), (
        f"{int((x_std == 0).sum())} feature column(s) have zero variance across the "
        f"test subset -- standardization would produce NaN/inf"
    )
    Xz = (X - x_mean) / x_std

    # PCA is fit on the entire test subset in one pass -- every held-out row
    # assembled above, not a per-fold or per-document subsample.
    print(f'[probe_pca] fitting PCA on all {Xz.shape[0]} test-subset points')
    pca = PCA(n_components=2, random_state=0)
    Xz_2d = pca.fit_transform(Xz)

    # ── Plot ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.set_axisbelow(True)
    ax.grid(True, alpha=0.25, linewidth=0.5, color='0.75')

    ax.scatter(Xz_2d[~labels, 0], Xz_2d[~labels, 1], s=8, alpha=0.35,
               color=COLOR_INVALID, edgecolors='none', label='Invalid', rasterized=True)
    ax.scatter(Xz_2d[labels, 0], Xz_2d[labels, 1], s=8, alpha=0.35,
               color=COLOR_VALID, edgecolors='none', label='Valid', rasterized=True)

    var_pct = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f'PC1 ({var_pct[0]:.1f}% var.)')
    ax.set_ylabel(f'PC2 ({var_pct[1]:.1f}% var.)')
    ax.legend(loc='best', markerscale=2.5)
    fig.tight_layout()

    out_dir = REPO_ROOT / 'figures' / 'probe_pca' / judge_model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'probe_pca_{dataset}_{extraction_model}.pdf'
    fig.savefig(out_path, bbox_inches='tight')
    print(f'[probe_pca] saved -> {out_path}')


if __name__ == '__main__':
    main()
