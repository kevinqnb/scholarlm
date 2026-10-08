"""Recalibration-sample-size sweep: calibration error of recalibrated probe / NTP
scores vs. the number of real rows the recalibration is fit on.

Replicates the real-extraction path of analysis/calibration_updated_v3.py (same
inputs, labels, Platt pool, test split, recalibration methods, fit-sample draws,
nested bootstrap and plot layout), once for each n in params.platt_ns instead of
v3's single platt_n. Per (train probe, test dataset, method, n) the calibration
errors are analysis/nested_bootstrap.py's: n_fit_samples recalibration fit
samples, each crossed with n_doc_boot test-document resamples; point = mean over
fit samples of the un-resampled value, interval = percentiles over all replicates.

Design points that keep the sweep comparable with v3 and across n:
  - Fit sample r of size n is nb.pool_resampled_fit_sample(pool, n, fit_sample_rng(seed, r)),
    exactly v3's draw, so at n = v3's platt_n the sweep reproduces v3's real cells.
    Samples of different n are not nested, but the same r resamples the same pool
    documents at every n. Samples are shared by every train probe.
  - A single-class sample cannot be fit by any recalibration method. It is skipped
    and drawing continues at the next r until there are n_fit_samples two-class
    samples ('Fit draws' in the CSV counts every draw), so small n conditions on
    both classes being present. v3 never skips: at its platt_n no draw is single-class.
  - The test rows and their document resamples do not depend on n: they exclude the
    WHOLE pool and the train probe's own documents, as in v3, and the resamples are
    v3's (nb.doc_boot_rng(seed, 'real', test_ds)), so comparisons across n are paired.
  - Real setting only. v3 never recalibrates synthetic cells.

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
import pickle
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns

from analysis.analysis_config import load_platt_sweep_config, RECALIBRATION_METHODS as _CFG_RECALIBRATION_METHODS
from analysis import calibration_ids as cids
from analysis import nested_bootstrap as nb
from analysis.head_activations import HeadActivationCache
from scholarlm.utils.calibration import apply_platt, fit_recalibration, RECALIBRATION_METHODS

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


# A run fails rather than draw more than this many fit samples per wanted sample:
# that many single-class draws means n is too small for the pool's label rate.
MAX_DRAWS_PER_SAMPLE = 10


def draw_fit_samples(pool_doc_ids, pool_labels, n, n_fit_samples, seed):
    """``n_fit_samples`` two-class recalibration fit samples of ``n`` pool positions.

    Draw r is v3's fit sample r (nb.pool_resampled_fit_sample with
    nb.fit_sample_rng(seed, r)); single-class draws are skipped. Returns the kept
    ``(r, positions)`` pairs and the number of draws made.
    """
    pool_labels = np.asarray(pool_labels, dtype=bool)
    assert len(pool_labels) == len(pool_doc_ids) > 0, (len(pool_labels), len(pool_doc_ids))
    kept, r = [], 0
    while len(kept) < n_fit_samples:
        if r >= MAX_DRAWS_PER_SAMPLE * n_fit_samples:
            raise RuntimeError(f'n={n}: only {len(kept)}/{n_fit_samples} two-class fit samples in {r} draws')
        sample = nb.pool_resampled_fit_sample(pool_doc_ids, n, nb.fit_sample_rng(seed, r))
        assert len(sample) == n
        if 0 < pool_labels[sample].sum() < n:
            kept.append((r, sample))
        r += 1
    return kept, r


def recalibrated_sets(pool_raw, pool_labels, test_raw, samples, recalibration, pi_tr):
    """One recalibrated test-prediction array per fit sample, and the fitted maps.

    ``samples`` are ``draw_fit_samples``'s ``(r, positions)`` pairs into the pool.
    Same fit and checks as v3's compute_predictions.
    """
    pred_sets, maps = [], []
    for r, sample in samples:
        p_raw, y = pool_raw[sample], np.asarray(pool_labels, dtype=bool)[sample]
        coef, icpt = fit_recalibration(recalibration, p_raw, y, pi_tr)
        assert np.isfinite(coef) and np.isfinite(icpt), (recalibration, r, coef, icpt)
        if recalibration != 'platt_fit':
            assert coef == 1.0, (recalibration, coef)
        if recalibration == 'intercept_fit':
            # Score equation of the intercept MLE (known answer, solved to ~1e-12).
            assert abs(apply_platt(p_raw, coef, icpt).mean() - y.mean()) < 1e-8, (r, icpt)
        pred_sets.append(apply_platt(test_raw, coef, icpt))
        maps.append({'Fit sample': r, 'coef': coef, 'intercept': icpt, 'Label rate': float(y.mean())})
    return pred_sets, maps


class SweepContext:
    """Everything loaded once from the config: trained artifacts and real test data."""

    def __init__(self, cfg):
        self.cfg = cfg
        params = cfg['params']
        self.seed = cfg['seed']
        self.probe_type = params['probe_type']
        self.platt_ns = list(params['platt_ns'])
        self.recalibration = params['recalibration']
        assert tuple(RECALIBRATION_METHODS) == tuple(_CFG_RECALIBRATION_METHODS)
        assert self.recalibration in RECALIBRATION_METHODS, self.recalibration
        self.n_fit_samples = params['n_fit_samples']
        self.n_doc_boot = params['n_doc_boot']
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
        # As in v3: each activation row decompressed once, keeping every train probe's top heads.
        self.head_acts = (HeadActivationCache([lh for ds in self.datasets for lh in self.probe[ds]['top_k_heads']])
                          if self.probe_type == 'head' else None)
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
        pool_doc_ids = real_df['document_id'].to_numpy()[pool_idx]
        fit_samples, n_draws = {}, {}
        for n in self.platt_ns:
            fit_samples[n], n_draws[n] = draw_fit_samples(pool_doc_ids, labels[pool_idx], n, self.n_fit_samples, self.seed)
            rates = np.array([labels[pool_idx][s].mean() for _, s in fit_samples[n]])
            print(f'  {ds}: n={n}: {self.n_fit_samples} two-class fit samples in {n_draws[n]} draws; label rate '
                  f'mean {rates.mean():.3f}, range {rates.min():.2f}-{rates.max():.2f}')
        print(f'  {ds}: pool {len(pool_idx)} rows / {len(np.unique(pool_doc_ids))} docs, '
              f'row rate {labels[pool_idx].mean():.3f}')
        return {'real_df': real_df, 'labels': labels, 'pool_docs': pool_docs, 'pool_idx': pool_idx,
                'fit_samples': fit_samples, 'n_draws': n_draws}

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
            X = self.head_acts.features(act_dir, mids, top)
        assert X.shape[0] == len(mids), (X.shape, len(mids))
        probe_probs = pd_data['probe'].predict_proba(X)[:, 1]
        assert probe_probs.shape == ntp_probs.shape == (len(mids),), (train_ds, test_ds)
        assert np.isfinite(probe_probs).all() and np.isfinite(ntp_probs).all(), (train_ds, test_ds)
        return {'probe': probe_probs, 'ntp': ntp_probs}


def run_sweep(ctx):
    """Nested-bootstrap calibration errors per (train_ds, test_ds, method, n).

    Returns the summary table (one row per cell and n), the fitted maps (one row per
    fit sample) and the nested-bootstrap summaries keyed (train_ds, test_ds, method, n).
    """
    rows, fit_rows, boots = [], [], {}
    test_boot = {}  # test_ds -> (measurement ids, document resamples), shared by every train probe and n
    for train_ds in ctx.datasets:
        pi_tr = {'probe': ctx.probe[train_ds]['train_prevalence'],
                 'ntp': ctx.ntp_cal[train_ds]['train_prevalence']}
        for test_ds in ctx.datasets:
            td = ctx.test_data[test_ds]
            real_df, labels_all, pool_idx = td['real_df'], td['labels'], td['pool_idx']

            # Test rows: outside test_ds's whole Platt pool and train_ds's probe-training docs
            # (as in v3), independent of n.
            exclude = td['pool_docs'] | set(ctx.probe[train_ds]['syn_document_ids'])
            test_idx = np.where(~real_df['document_id'].isin(exclude).to_numpy())[0]
            assert len(test_idx) > 0, f'{train_ds} probe -> {test_ds}: no real test rows'
            assert not set(test_idx.tolist()) & set(pool_idx.tolist()), (train_ds, test_ds)
            test_labels = labels_all[test_idx]
            test_mids = real_df['measurement_id'].to_numpy()[test_idx]
            test_docs = real_df['document_id'].to_numpy()[test_idx]
            if test_ds not in test_boot:
                test_boot[test_ds] = (test_mids, nb.cluster_bootstrap_indices(
                    test_docs, ctx.n_doc_boot, nb.doc_boot_rng(ctx.seed, 'real', test_ds)))
            # Shared resamples index rows, so every train probe must leave the same test rows.
            assert np.array_equal(test_boot[test_ds][0], test_mids), (train_ds, test_ds)
            boot_idx = test_boot[test_ds][1]

            t0 = time.time()
            test_scores = ctx.score_rows(train_ds, test_ds, test_idx)
            pool_scores = ctx.score_rows(train_ds, test_ds, pool_idx)  # scored once, indexed per fit sample
            pool_labels = labels_all[pool_idx]
            print(f'  {train_ds} probe -> {test_ds}: {len(test_idx)} test rows / {len(np.unique(test_docs))} docs, '
                  f'pool {len(pool_idx)}; scored in {time.time() - t0:.0f}s', flush=True)

            for n in ctx.platt_ns:
                t_n = time.time()
                for method, key, _ in _METHODS:
                    pred_sets, maps = recalibrated_sets(
                        pool_scores[key], pool_labels, test_scores[key], td['fit_samples'][n],
                        ctx.recalibration, pi_tr[key])
                    for m in maps:
                        fit_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds, 'Type': method,
                                         'Platt N': n, 'Recalibration': ctx.recalibration, **m,
                                         'Train prevalence': float(pi_tr[key])})
                    b = nb.nested_bootstrap(pred_sets, test_labels, boot_idx)
                    boots[(train_ds, test_ds, key, n)] = b
                    row = {
                        'Train dataset': train_ds, 'Test dataset': test_ds, 'Type': method, 'Platt N': n,
                        'Recalibration': ctx.recalibration,
                        'Fit samples': b['n_fit_samples'], 'Fit draws': td['n_draws'][n],
                        'Doc resamples': b['n_doc_boot'],
                        'Pool N': len(pool_idx), 'Test N': len(test_idx), 'Test docs': len(np.unique(test_docs)),
                        'Test label rate': float(test_labels.mean()),
                        'Fit label rate': float(np.mean([m['Label rate'] for m in maps])),
                    }
                    for m in nb.METRICS:
                        row.update({m: b['point'][m], f'{m}_lo': b['lo'][m], f'{m}_hi': b['hi'][m]})
                    rows.append(row)
                print(f'    n={n}: {ctx.n_fit_samples} fit samples x {ctx.n_doc_boot} doc resamples x '
                      f'{len(_METHODS)} methods in {time.time() - t_n:.0f}s', flush=True)

    summary = pd.DataFrame(rows)
    fits_df = pd.DataFrame(fit_rows)
    keys = ['Train dataset', 'Test dataset', 'Type', 'Platt N']
    assert len(summary) == len(ctx.datasets) ** 2 * len(_METHODS) * len(ctx.platt_ns), len(summary)
    assert not summary.duplicated(keys).any()
    assert (summary['Fit samples'] == ctx.n_fit_samples).all() and (summary['Doc resamples'] == ctx.n_doc_boot).all()
    assert len(fits_df) == len(summary) * ctx.n_fit_samples, len(fits_df)
    assert not fits_df.duplicated(keys + ['Fit sample']).any()
    return summary, fits_df, boots


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
    p = cfg['params']
    print(f"[platt sweep] config: {cfg['id']} | ns: {p['platt_ns']} | recalibration: {p['recalibration']} | "
          f"fit samples: {p['n_fit_samples']} x doc resamples: {p['n_doc_boot']}")
    ctx = SweepContext(cfg)
    summary, fits_df, boots = run_sweep(ctx)
    summary.to_csv(ctx.out_dir / 'smece_vs_platt_n.csv', index=False)
    fits_df.to_csv(ctx.out_dir / 'platt_fits.csv', index=False)
    with open(ctx.out_dir / 'nested_bootstrap.pkl', 'wb') as f:
        pickle.dump(boots, f)
    print(summary.to_string(index=False, float_format='{:.3f}'.format))
    plot_sweep(ctx, summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibration error vs. number of recalibration training samples (real setting).")
    parser.add_argument('config', type=Path,
                        help="analysis-configs/<id>.yaml (see load_platt_sweep_config)")
    main(parser.parse_args().config)
