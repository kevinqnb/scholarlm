import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from analysis.loaders import (
    load_synthetic_activations, load_synthetic_layer_outputs,
    load_synthetic_responses, load_trained_probe, load_trained_ntp_calibrator,
)
from scholarlm.utils.calibration import reliability_diagram_data

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

FIGURES_DIR = REPO_ROOT / "figures/sensitivity/"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# ── Parameters ────────────────────────────────────────────────────────────────
DATASETS         = ['pond', 'nfix']
EXTRACTION_MODEL = 'gemma-3-27b'
JUDGE_MODELS     = ['qwen-2.5-7b']#['llama-3.1-8b', 'mistral-7b', 'qwen-2.5-7b']
PROBE_TYPE       = "head"

JUDGE_DATES_SYN = {
    'pond': {
        #'llama-3.1-8b': '2026_05_04',
        #'mistral-7b':   '2026_05_04',
        'qwen-2.5-7b':  '2026_05_04',
    },
    'nfix': {
        #'llama-3.1-8b': '2026_05_04',
        #'mistral-7b':   '2026_05_04',
        'qwen-2.5-7b':  '2026_05_04',
    },
}

P_VALUES  = np.linspace(0, 0.5, 21)
N_TRIALS  = 100

palette       = sns.color_palette("husl", 10)
COLOR_WITHIN  = palette[7]   # blue (pond colour from calibration.py)
COLOR_CROSS   = palette[1]   # orange (nfix colour from calibration.py)


# ── Pre-load probes and NTP calibrators ───────────────────────────────────────
ntp_cal_cache = {}
probe_cache   = {}
for ds in DATASETS:
    ntp_cal_cache[ds] = {}
    probe_cache[ds]   = {}
    for jm in JUDGE_MODELS:
        ntp_cal_cache[ds][jm] = load_trained_ntp_calibrator(ds, jm)
        probe_cache[ds][jm]   = load_trained_probe(ds, jm, ptype=PROBE_TYPE)


# ── Data loading ──────────────────────────────────────────────────────────────
def load_syn_predictions(judge_model: str, train_ds: str, test_ds: str):
    """Return (probe_probs, ntp_probs, labels) for synthetic test split."""
    pd_data = probe_cache[train_ds][judge_model]
    ntp_cal = ntp_cal_cache[train_ds][judge_model]
    top     = pd_data['top_k_heads'] if PROBE_TYPE == "head" else pd_data['top_layer']

    jdate   = JUDGE_DATES_SYN[test_ds][judge_model]
    syn_df  = pd.DataFrame(load_synthetic_responses(test_ds, judge_model, jdate, split='test'))
    mids    = syn_df['measurement_id'].tolist()
    labels  = (syn_df['label'] == 'valid').to_numpy(dtype=bool)

    raw_ntp   = syn_df['judgement_p_true'].to_numpy()
    ntp_probs = ntp_cal['calibrator'].predict_proba(raw_ntp.reshape(-1, 1))[:, 1]

    if PROBE_TYPE == "layer":
        syn_lo = load_synthetic_layer_outputs(test_ds, judge_model, jdate, split='test')
        X = np.stack(
            [np.array(syn_lo[str(mid)], dtype=np.float32)[top] for mid in mids],
            axis=0,
        )
    else:
        syn_act = load_synthetic_activations(test_ds, judge_model, jdate, split='test')
        X = np.concatenate(
            [
                np.stack(
                    [np.array(syn_act[str(mid)], dtype=np.float32)[l, h, :] for mid in mids],
                    axis=0,
                )
                for l, h in top
            ],
            axis=1,
        )
    probe_probs = pd_data['probe'].predict_proba(X)[:, 1]
    return probe_probs, ntp_probs, labels


# ── Sensitivity core ──────────────────────────────────────────────────────────
def sensitivity_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    p_values: np.ndarray,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray]:
    """For each noise fraction p, randomly flip p·n labels and return mean/std ECE over trials."""
    probs  = np.asarray(probs)
    labels = np.asarray(labels, dtype=bool)
    n      = len(labels)
    mean_ece = np.empty(len(p_values))
    std_ece  = np.empty(len(p_values))
    for i, p in enumerate(p_values):
        eces = np.empty(n_trials)
        for t in range(n_trials):
            flip_mask  = np.random.random(n) < p
            noisy      = labels ^ flip_mask
            eces[t]    = reliability_diagram_data(probs, noisy)['ece']
        mean_ece[i] = eces.mean()
        std_ece[i]  = eces.std()
    return mean_ece, std_ece


def compute_sensitivity(judge_model: str) -> dict:
    """Compute within- and cross-domain sensitivity curves per training dataset."""
    results = {}
    for train_ds in DATASETS:
        within_probe_means, within_probe_stds = [], []
        within_ntp_means,   within_ntp_stds   = [], []
        cross_probe_means,  cross_probe_stds  = [], []
        cross_ntp_means,    cross_ntp_stds    = [], []

        for test_ds in DATASETS:
            print(f'  {train_ds} → {test_ds}', flush=True)
            probe_probs, ntp_probs, labels = load_syn_predictions(judge_model, train_ds, test_ds)
            p_mean, p_std = sensitivity_ece(probe_probs, labels, P_VALUES, N_TRIALS)
            n_mean, n_std = sensitivity_ece(ntp_probs,   labels, P_VALUES, N_TRIALS)

            if train_ds == test_ds:
                within_probe_means.append(p_mean); within_probe_stds.append(p_std)
                within_ntp_means.append(n_mean);   within_ntp_stds.append(n_std)
            else:
                cross_probe_means.append(p_mean);  cross_probe_stds.append(p_std)
                cross_ntp_means.append(n_mean);    cross_ntp_stds.append(n_std)

        results[train_ds] = {
            'within_probe_mean': np.mean(within_probe_means, axis=0),
            'within_probe_std':  np.mean(within_probe_stds,  axis=0),
            'within_ntp_mean':   np.mean(within_ntp_means,   axis=0),
            'within_ntp_std':    np.mean(within_ntp_stds,    axis=0),
            'cross_probe_mean':  np.mean(cross_probe_means,  axis=0),
            'cross_probe_std':   np.mean(cross_probe_stds,   axis=0),
            'cross_ntp_mean':    np.mean(cross_ntp_means,    axis=0),
            'cross_ntp_std':     np.mean(cross_ntp_stds,     axis=0),
        }
    return results


# ── Plotting ──────────────────────────────────────────────────────────────────
def _plot_curve(ax, x, mean, std, color, linestyle, lw, zorder):
    ax.plot(x, mean, linestyle, color=color, lw=lw, zorder=zorder)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18, linewidth=0, zorder=zorder - 1)


def plot_sensitivity(all_results: dict):
    for judge_model in JUDGE_MODELS:
        for train_ds in DATASETS:
            print(f'Plotting {judge_model} / train={train_ds}...')
            res = all_results[judge_model][train_ds]

            subfigure_dir = FIGURES_DIR / f"{judge_model}/{EXTRACTION_MODEL}/{PROBE_TYPE}/{train_ds}/"
            subfigure_dir.mkdir(parents=True, exist_ok=True)

            fig, ax = plt.subplots(figsize=(4.5, 3.8))

            _plot_curve(ax, P_VALUES, res['within_probe_mean'], res['within_probe_std'],
                        COLOR_WITHIN, '-',  lw=2.5, zorder=5)
            _plot_curve(ax, P_VALUES, res['cross_probe_mean'],  res['cross_probe_std'],
                        COLOR_CROSS,  '-',  lw=2.5, zorder=5)
            _plot_curve(ax, P_VALUES, res['within_ntp_mean'],   res['within_ntp_std'],
                        COLOR_WITHIN, '--', lw=2.0, zorder=3)
            _plot_curve(ax, P_VALUES, res['cross_ntp_mean'],    res['cross_ntp_std'],
                        COLOR_CROSS,  '--', lw=2.0, zorder=3)

            ax.set_xlabel(r'Noise level $p$')
            ax.set_ylabel('ECE')
            ax.set_xlim(-0.02, 0.52)
            ax.set_ylim(bottom=0.0)
            ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
            ax.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(subfigure_dir / 'sensitivity.pdf', bbox_inches='tight', dpi=200)
            plt.show()

    # Shared legend (saved once at the top-level figures dir)
    handles = [
        mlines.Line2D([], [], color=COLOR_WITHIN, lw=2, linestyle='-',  label='Within (Probe)'),
        mlines.Line2D([], [], color=COLOR_CROSS,  lw=2, linestyle='-',  label='Cross (Probe)'),
        mlines.Line2D([], [], color=COLOR_WITHIN, lw=2, linestyle='--', label='Within (NTP)'),
        mlines.Line2D([], [], color=COLOR_CROSS,  lw=2, linestyle='--', label='Cross (NTP)'),
    ]
    fig_leg, ax_leg = plt.subplots(figsize=(8.0, 0.45))
    ax_leg.axis('off')
    ax_leg.legend(handles=handles, loc='center', ncol=4, fontsize=13,
                  frameon=False, handlelength=2.0)
    fig_leg.savefig(FIGURES_DIR / 'sensitivity_legend.pdf', bbox_inches='tight', dpi=200)
    plt.show()


if __name__ == "__main__":
    all_results = {}
    for jm in JUDGE_MODELS:
        print(f'\nComputing sensitivity for {jm}...')
        all_results[jm] = compute_sensitivity(jm)

    plot_sensitivity(all_results)
