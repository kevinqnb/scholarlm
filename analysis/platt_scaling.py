"""Platt-sample-size sweep: smECE of Platt-scaled probe / NTP calibrators vs. the
number of real rows used to fit the scaler.

Replicates the real-extraction path of analysis/calibration_updated_v3.py (same
inputs, labels, Platt pool, test split, plot layout) but, per (train probe, test
dataset, method), fits one scaler for each n in params.platt_ns and reports smECE
(relplot) on one fixed test set. The whole thing is repeated for params.n_trials
random draws of the Platt rows; the plot shows the mean over trials with a central
params.ci interval over trials.

Design points that keep the sweep comparable across n and trials:
  - Trial t permutes the test dataset's Platt pool with rng([config seed, t]); the
    n-sample is its first n rows, so within a trial every sample is a superset of the
    previous one. The same permutation is used for every train probe.
  - The test rows do not depend on n or the trial: they exclude the WHOLE pool and the
    train probe's own documents, exactly as in v3.
  - Real setting only. v3 never Platt-scales synthetic cells and the synthetic
    test sets are smaller than max(platt_ns) for nfix/supermat.
  - Once n reaches the pool size every trial draws the same rows, so the band there is
    degenerate (zero), not evidence of low variance. 'Pool N' is in the CSV.

No flags: one positional config (analysis/analysis-configs/<id>.yaml, see
load_platt_sweep_config), normally run through
`bash analysis/submit.sh platt_scaling <id> --walltime HH:MM:SS --omp N`. Output goes to
analysis/results/calibration/<config id>/.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import json
import time
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
import relplot

from analysis.analysis_config import load_platt_sweep_config
from analysis import calibration_ids as cids
from scholarlm.utils.calibration import fit_platt, apply_platt

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

# Same colors / labels as calibration_updated_v3.py (blue: 7, orange: 1, red: 0).
palette = sns.color_palette("husl", 10)
_DS_COLORS = {'pond': palette[7], 'nfix': palette[1], 'supermat': palette[0]}
_DS_LABELS = {'pond': 'PLW', 'nfix': 'NF', 'supermat': 'SM'}

# (display name, key in the scored dict, linestyle) -- identical to v3.
_METHODS = [
    ('Probe', 'probe', '-'),
    ('NTP',   'ntp',   '--'),
]


def trial_order(pool_size, seed, trial):
    """Permutation of range(pool_size) for one trial; the n-sample is its first n entries."""
    perm = np.random.default_rng([seed, trial]).permutation(pool_size)
    assert len(set(perm.tolist())) == pool_size
    return perm


def smece(probs, labels):
    """relplot smECE (deterministic: automatic-bandwidth search, no resampling)."""
    probs, labels = np.asarray(probs, dtype=float), np.asarray(labels, dtype=bool)
    assert probs.shape == labels.shape and probs.ndim == 1 and len(probs) > 0
    ce = float(relplot.smECE(probs, labels))
    assert np.isfinite(ce), ce
    return ce


def summarize_trials(trials_df, ci):
    """Mean / std / central-ci percentiles over trials for each (train, test, method, n)."""
    keys = ['Train dataset', 'Test dataset', 'Type', 'Platt N']
    lo_q, hi_q = 100 * (1 - ci) / 2, 100 * (1 + ci) / 2
    g = trials_df.groupby(keys)['SmECE']
    out = pd.DataFrame({
        'SmECE': g.mean(), 'SmECE_std': g.std(ddof=1),
        'SmECE_lo': g.quantile(lo_q / 100), 'SmECE_hi': g.quantile(hi_q / 100),
        'n_used': g.size(),
    }).reset_index()
    return out


class SweepContext:
    """Everything loaded once from the config: trained artifacts and real test data."""

    def __init__(self, cfg):
        self.cfg = cfg
        params = cfg['params']
        self.seed = cfg['seed']
        self.probe_type = params['probe_type']
        self.platt_ns = list(params['platt_ns'])
        self.n_trials = params['n_trials']
        self.ci = params['ci']
        self.single_class_policy = params['single_class_policy']
        self.datasets = list(params['datasets'])
        self.inputs = cids.resolve_calibration_inputs(cfg)
        self.judge_model = self.inputs['judge_model']
        self.out_dir = REPO_ROOT / 'analysis' / 'results' / 'calibration' / cfg['id']
        self.figures_dir = self.out_dir / 'figures'
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        # None -> Platt-scaled baseline filenames; 'noplatt' picks the suffixed variant.
        variant_kw = None if params['probe_variant'] == 'platt' else params['probe_variant']
        ntp_name = 'ntp_calibrator.pkl' if variant_kw is None else 'ntp_calibrator_noplatt.pkl'
        probe_name = ('layer_probe.pkl' if self.probe_type == 'layer'
                      else ('head_probe.pkl' if variant_kw is None else 'head_probe_noplatt.pkl'))
        self.ntp_cal = {ds: self._load_artifact(ds, ntp_name) for ds in self.datasets}
        self.probe = {ds: self._load_artifact(ds, probe_name) for ds in self.datasets}
        self.test_data = {ds: self._load_test_data(ds) for ds in self.datasets}

    def _load_artifact(self, train_ds, filename):
        path = self.inputs['datasets'][train_ds]['probe_dir'] / filename
        if not path.exists():
            raise FileNotFoundError(
                f'{path} does not exist. Run analysis/synthetic_probe_train.py '
                f'on the {train_ds} synthetic-probe config first.')
        artifact = joblib.load(path)
        assert artifact['judge_model'] == self.judge_model, (path, artifact['judge_model'])
        assert artifact['dataset'] == train_ds, (path, artifact['dataset'])
        return artifact

    def _load_test_data(self, ds):
        # Mirrors calibration_updated_v3.py's loading block (labels, matching, pool).
        ds_inputs = self.inputs['datasets'][ds]
        block = self.cfg['params']['datasets'][ds]
        with open(ds_inputs['judge_combine_dir'] / 'combined.json') as f:
            real_df = pd.DataFrame(json.load(f))
        assert f'judgement_p_true_{self.judge_model}' in real_df.columns, (ds, self.judge_model)
        with open(ds_inputs['extraction_dir'] / 'final.json') as f:
            final_df = pd.DataFrame(json.load(f))
        for col in ('measurement_id', 'document_id', 'attribute'):
            assert final_df[col].tolist() == real_df[col].tolist(), (
                f'{ds}: final.json and combined.json disagree on {col}')

        jlabels = real_df['judgement_combined'].to_numpy(dtype=bool)
        if block['use_matching_labels']:
            _gt_df, ext_df, cached_edges = cids.load_cached_matching(
                block['extraction_id'], ds_inputs['ground_truth_path'])
            judged_edges = cids.edges_to_judged_rows(cached_edges, ext_df, real_df)
            matched = np.zeros(len(real_df), dtype=bool)
            for _gt_idx, ex_idx in judged_edges:
                matched[ex_idx] = True
            print(f'  {ds}: matching labels on -- {int(matched.sum())}/{len(real_df)} judged rows matched')
            labels = jlabels | matched
        else:
            print(f'  {ds}: matching labels off -- labels are judgement_combined alone')
            labels = jlabels.copy()

        pool_docs = set(self.probe[ds]['syn_document_ids'])
        pool_idx = np.where(real_df['document_id'].isin(pool_docs).to_numpy())[0]
        assert len(pool_idx) >= self.platt_ns[-1], (
            f'{ds}: pool has {len(pool_idx)} rows < max(platt_ns)={self.platt_ns[-1]}')
        print(f'  {ds}: pool {len(pool_idx)} rows, {int(labels[pool_idx].sum())} valid')
        return {'real_df': real_df, 'labels': labels, 'pool_docs': pool_docs, 'pool_idx': pool_idx}

    def score_rows(self, train_ds, test_ds, idx):
        """Raw (un-scaled) probe and NTP-calibrator probabilities for real_df rows ``idx``."""
        td = self.test_data[test_ds]
        real_df = td['real_df']
        act_dir = self.inputs['datasets'][test_ds]['judge_interp_dir']
        mids = real_df['measurement_id'].iloc[idx].tolist()
        raw_ntp = real_df[f'judgement_p_true_{self.judge_model}'].iloc[idx].to_numpy()
        pd_data, ntp_data = self.probe[train_ds], self.ntp_cal[train_ds]
        top = pd_data['top_layer'] if self.probe_type == 'layer' else pd_data['top_k_heads']
        ntp_probs = ntp_data['calibrator'].predict_proba(raw_ntp.reshape(-1, 1))[:, 1]
        if self.probe_type == 'layer':
            lo = np.load(act_dir / 'layer_outputs.npz')
            X = np.stack([np.array(lo[str(m)], dtype=np.float32)[top] for m in mids], axis=0)
        else:
            act = np.load(act_dir / 'attention_outputs.npz')
            X = np.concatenate([
                np.stack([np.array(act[str(m)], dtype=np.float32)[l, h, :] for m in mids], axis=0)
                for l, h in top
            ], axis=1)
        assert X.shape[0] == len(mids), (X.shape, len(mids))
        probe_probs = pd_data['probe'].predict_proba(X)[:, 1]
        assert probe_probs.shape == ntp_probs.shape == (len(mids),), (train_ds, test_ds)
        assert np.isfinite(probe_probs).all() and np.isfinite(ntp_probs).all(), (train_ds, test_ds)
        return {'probe': probe_probs, 'ntp': ntp_probs}


def run_sweep(ctx):
    """One row per (train_ds, test_ds, method, trial, n): Platt scaler + smECE on the fixed test set."""
    rows, dropped = [], {}
    for train_ds in ctx.datasets:
        for test_ds in ctx.datasets:
            td = ctx.test_data[test_ds]
            real_df, labels_all, pool_idx = td['real_df'], td['labels'], td['pool_idx']

            # Test rows: outside test_ds's whole Platt pool and train_ds's probe-training docs
            # (as in v3), independent of n and trial.
            exclude = td['pool_docs'] | set(ctx.probe[train_ds]['syn_document_ids'])
            test_idx = np.where(~real_df['document_id'].isin(exclude).to_numpy())[0]
            assert len(test_idx) > 0, f'{train_ds} probe -> {test_ds}: no real test rows'
            assert not set(test_idx.tolist()) & set(pool_idx.tolist()), (train_ds, test_ds)
            test_labels = labels_all[test_idx]
            t0 = time.time()
            test_scores = ctx.score_rows(train_ds, test_ds, test_idx)
            pool_scores = ctx.score_rows(train_ds, test_ds, pool_idx)  # scored once, indexed per trial
            pool_labels = labels_all[pool_idx]
            print(f'  {train_ds} probe -> {test_ds}: {len(test_idx)} test rows, pool {len(pool_idx)}; '
                  f'scored in {time.time() - t0:.0f}s', flush=True)
            t_fit = time.time()

            for trial in range(ctx.n_trials):
                order = trial_order(len(pool_idx), ctx.seed, trial)
                for n in ctx.platt_ns:
                    sel = order[:n]
                    assert len(sel) == n and set(order[:ctx.platt_ns[0]].tolist()) <= set(sel.tolist())
                    y = pool_labels[sel]
                    if y.all() or not y.any():
                        if ctx.single_class_policy == 'error':
                            raise ValueError(f'{test_ds}: trial {trial} n={n} Platt sample is single-class')
                        dropped[(test_ds, n)] = dropped.get((test_ds, n), 0) + 1
                        continue
                    for method, key, _ in _METHODS:
                        coef, icpt = fit_platt(pool_scores[key][sel], y)
                        assert np.isfinite(coef) and np.isfinite(icpt), (train_ds, test_ds, method, trial, n)
                        scaled = apply_platt(test_scores[key], coef, icpt)
                        rows.append({
                            'Train dataset': train_ds, 'Test dataset': test_ds, 'Type': method,
                            'Trial': trial, 'Platt N': n, 'coef': coef, 'intercept': icpt,
                            'Platt label rate': float(y.mean()), 'Pool N': len(pool_idx),
                            'Test N': len(test_idx), 'Test label rate': float(test_labels.mean()),
                            'SmECE': smece(scaled, test_labels),
                        })

            print(f'    {ctx.n_trials} trials x {len(ctx.platt_ns)} n x {len(_METHODS)} methods fit+scored in '
                  f'{time.time() - t_fit:.0f}s', flush=True)

    trials_df = pd.DataFrame(rows)
    assert not trials_df.duplicated(['Train dataset', 'Test dataset', 'Type', 'Trial', 'Platt N']).any()
    summary = summarize_trials(trials_df, ctx.ci)
    # Every cell must still have trials; a cell with fewer than all of them lost some to the drop policy.
    assert len(summary) == len(ctx.datasets) ** 2 * len(_METHODS) * len(ctx.platt_ns), len(summary)
    assert (summary['n_used'] >= 2).all(), summary[summary['n_used'] < 2]
    for (ds, n), k in sorted(dropped.items()):
        print(f'  WARNING dropped {k}/{ctx.n_trials} trials (single-class Platt sample): test={ds} n={n}')
    for ds in ctx.datasets:
        pool_n = len(ctx.test_data[ds]['pool_idx'])
        for n in ctx.platt_ns:
            if n >= pool_n:
                print(f'  WARNING {ds}: n={n} >= pool {pool_n}; all trials use the same rows, band is degenerate')
    return trials_df, summary


def plot_sweep(ctx, df):
    # One figure per (method, train_ds): one curve per test_ds, colored by test_ds; same
    # titles / colors / line styles / figure size as v3's real-setting calibration plots.
    for method, _, linestyle in _METHODS:
        for train_ds in ctx.datasets:
            fig, ax = plt.subplots(figsize=(4.0, 3.8))
            for test_ds in ctx.datasets:
                sub = df[(df['Type'] == method) & (df['Train dataset'] == train_ds)
                         & (df['Test dataset'] == test_ds)].sort_values('Platt N')
                assert sub['Platt N'].tolist() == ctx.platt_ns, (method, train_ds, test_ds)
                color = _DS_COLORS[test_ds]
                ax.plot(sub['Platt N'], sub['SmECE'], linestyle=linestyle, color=color,
                        lw=2.5, marker='o', ms=3.5, zorder=3)
                ax.fill_between(sub['Platt N'], sub['SmECE_lo'], sub['SmECE_hi'],
                                color=color, alpha=0.20, linewidth=0, zorder=1)
            ax.set_xscale('log')
            ax.set_xticks(ctx.platt_ns)
            ax.set_xticklabels([str(n) for n in ctx.platt_ns])
            ax.minorticks_off()
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Platt Training Samples')
            if method == 'NTP':
                ax.set_ylabel('SmECE')
            ax.set_title(method, fontsize=15, style='italic')
            ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
            ax.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(ctx.figures_dir / f'smece_vs_n_real_{method.lower()}_train-{train_ds}.pdf',
                        bbox_inches='tight', dpi=200)
            plt.show()
            plt.close(fig)

    handles = [mlines.Line2D([], [], color=_DS_COLORS[ds], lw=2, marker='o', ms=3.5, label=_DS_LABELS[ds])
               for ds in ctx.datasets]
    fig_leg, ax_leg = plt.subplots(figsize=(6.0, 0.45))
    ax_leg.axis('off')
    ax_leg.legend(handles=handles, loc='center', ncol=len(ctx.datasets), fontsize=13,
                  frameon=False, handlelength=2.0, title='Test set', title_fontsize=13)
    fig_leg.savefig(ctx.figures_dir / 'legend_smece_vs_n.pdf', bbox_inches='tight', dpi=200)
    plt.show()


def main(config_path):
    cfg = load_platt_sweep_config(Path(config_path))
    print(f"[platt sweep] config: {cfg['id']} | ns: {cfg['params']['platt_ns']} | "
          f"n_trials: {cfg['params']['n_trials']} | ci: {cfg['params']['ci']}")
    ctx = SweepContext(cfg)
    trials_df, summary = run_sweep(ctx)
    trials_df.to_csv(ctx.out_dir / 'smece_vs_platt_n_trials.csv', index=False)
    summary.to_csv(ctx.out_dir / 'smece_vs_platt_n.csv', index=False)
    print(summary.to_string(index=False, float_format='{:.3f}'.format))
    plot_sweep(ctx, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmECE vs. number of Platt training samples (real setting).")
    parser.add_argument('config', type=Path,
                        help="analysis-configs/<id>.yaml (see load_platt_sweep_config)")
    main(parser.parse_args().config)
