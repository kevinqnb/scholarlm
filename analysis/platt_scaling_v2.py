"""Recalibration-sample-size sweep, training-resample variability only.

Calibration error of recalibrated probe / NTP scores vs. the number of real rows n the
recalibration is fit on, for each n in params.platt_ns. Same inputs, labels, Platt pool
and test rows as calibration_updated_v4.py; replaces analysis/platt_scaling.py's nested
bootstrap. Per (train probe, test dataset, method, n):

  - n_train_resamples fit samples from doc_bootstrap.two_class_fit_samples: the test
    dataset's pool documents resampled with replacement, then n rows drawn uniformly
    without replacement from the resampled pool. Draws whose resampled pool has fewer
    than n rows, and single-class draws, are skipped ('Fit draws' counts every draw;
    'Short-pool skips' / 'Single-class skips' count each kind).
  - each sample's recalibration map (prior_shift / intercept_fit from
    analysis/recalibration.py, platt_fit from scholarlm's fit_platt) is applied to the
    fixed test rows and scored by relplot's smECE, exactly as doc_bootstrap computes it.
  - point = mean of the n_train_resamples smECE values, interval = their 2.5 / 97.5
    percentiles.

The band is the spread over training resamples ONLY: the test rows are not resampled,
so test-document sampling noise is not in it. It is narrower than, and not comparable
with, platt_scaling.py's nested-bootstrap intervals.

Fit sample r of test dataset ds resamples the same pool documents at every n, and is
shared by every train probe and method.

No flags: one positional config (analysis/analysis-configs/<id>.yaml, see
load_platt_sweep_v2_config), normally run through
`bash analysis/submit.sh platt_scaling_v2 <id> --walltime HH:MM:SS --omp N`. Output goes
to analysis/results/calibration/<config id>/.
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

from analysis.analysis_config import load_platt_sweep_v2_config
from analysis import calibration_ids as cids
from analysis import doc_bootstrap as db
from analysis.head_activations import HeadActivationCache
from analysis.recalibration import prior_shift_map, intercept_fit_map
from scholarlm.utils.calibration import apply_platt, fit_platt

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

# Same colors / labels as calibration_updated_v4.py (blue: 7, orange: 1, red: 0).
palette = sns.color_palette("husl", 10)
_DS_COLORS = {'pond': palette[7], 'nfix': palette[1], 'supermat': palette[0]}
_DS_LABELS = {'pond': 'PLW', 'nfix': 'NF', 'supermat': 'SM'}

# (display name, key in the scored dict, linestyle).
_METHODS = [
    ('Probe', 'probe', '-'),
    ('NTP',   'ntp',   '--'),
]


def fit_map(recalibration: str, probs, labels, pi_tr: float) -> tuple[float, float]:
    """``(coef, intercept)`` for apply_platt, fit on these rows."""
    labels = np.asarray(labels, dtype=bool)
    if recalibration == 'prior_shift':
        return prior_shift_map(float(labels.mean()), pi_tr)
    if recalibration == 'intercept_fit':
        return intercept_fit_map(probs, labels)
    if recalibration == 'platt_fit':
        coef, icpt = fit_platt(probs, labels)
        assert np.isfinite(coef) and np.isfinite(icpt), (coef, icpt)
        return coef, icpt
    raise ValueError(f'unknown recalibration {recalibration!r}')


def smece(probs, labels) -> float:
    """relplot's smECE, exactly as doc_bootstrap (and so calibration v4) computes it."""
    return float(db._relplot(np.asarray(probs, dtype=float), np.asarray(labels, dtype=bool))['ce'])


def summarize(values, ci_level=db.CI_LEVEL) -> dict:
    """Mean and central ``ci_level`` percentile interval of per-resample values."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), values.shape
    alpha = (1 - ci_level) / 2
    lo, hi = np.quantile(values, [alpha, 1 - alpha])
    return {'SmECE': float(values.mean()), 'SmECE_lo': float(lo), 'SmECE_hi': float(hi)}


class SweepInputs:
    """Trained probes / NTP calibrators and real labels, pool and test rows per dataset,
    loaded as in calibration_updated_v4.py."""

    def __init__(self, cfg):
        params = cfg['params']
        self.cfg = cfg
        self.seed = cfg['seed']
        self.probe_type = params['probe_type']
        self.datasets = list(params['datasets'])
        self.inputs = cids.resolve_calibration_inputs(cfg)
        self.judge_model = self.inputs['judge_model']

        # None -> Platt-scaled baseline filenames; 'noplatt' picks the suffixed variant.
        variant_kw = None if params['probe_variant'] == 'platt' else params['probe_variant']
        ntp_name = 'ntp_calibrator.pkl' if variant_kw is None else 'ntp_calibrator_noplatt.pkl'
        probe_name = ('layer_probe.pkl' if self.probe_type == 'layer'
                      else ('head_probe.pkl' if variant_kw is None else 'head_probe_noplatt.pkl'))
        self.ntp_cal = {ds: self._load_artifact(ds, ntp_name) for ds in self.datasets}
        self.probe = {ds: self._load_artifact(ds, probe_name) for ds in self.datasets}
        # Each activation row decompressed once, keeping every train probe's top heads.
        self.head_acts = (HeadActivationCache([lh for ds in self.datasets for lh in self.probe[ds]['top_k_heads']])
                          if self.probe_type == 'head' else None)
        self.data = {ds: self._load_real(ds) for ds in self.datasets}

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

    def _load_real(self, ds):
        ds_inputs = self.inputs['datasets'][ds]
        block = self.cfg['params']['datasets'][ds]
        with open(ds_inputs['judge_combine_dir'] / 'combined.json') as f:
            real_df = pd.DataFrame(json.load(f))
        assert f'judgement_p_true_{self.judge_model}' in real_df.columns, (ds, self.judge_model)
        # Scored on the final.json datapoints; combined.json must be row-for-row the same run.
        with open(ds_inputs['extraction_dir'] / 'final.json') as f:
            final_df = pd.DataFrame(json.load(f))
        for col in ('measurement_id', 'document_id', 'attribute'):
            assert final_df[col].tolist() == real_df[col].tolist(), (
                f'{ds}: final.json and combined.json disagree on {col}')

        # Labels: judgement_combined, OR'd with a cached ground-truth match when the
        # dataset's use_matching_labels is on.
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

        # Pool: this dataset's real rows whose document was in its own synthetic-probe
        # training set. Test rows: every other real row.
        pool_docs = set(self.probe[ds]['syn_document_ids'])
        in_pool = real_df['document_id'].isin(pool_docs).to_numpy()
        pool_idx, test_idx = np.where(in_pool)[0], np.where(~in_pool)[0]
        assert len(pool_idx) > 0 and len(test_idx) > 0, (ds, len(pool_idx), len(test_idx))
        doc_ids = real_df['document_id'].to_numpy()
        print(f'  {ds}: pool {len(pool_idx)} rows / {len(np.unique(doc_ids[pool_idx]))} docs '
              f'(label rate {labels[pool_idx].mean():.3f}) | test {len(test_idx)} rows / '
              f'{len(np.unique(doc_ids[test_idx]))} docs (label rate {labels[test_idx].mean():.3f})')
        return {'real_df': real_df, 'labels': labels, 'pool_docs': pool_docs,
                'pool_idx': pool_idx, 'test_idx': test_idx}

    def score_rows(self, train_ds, test_ds, idx):
        """Raw (un-mapped) probe and NTP-calibrator probabilities for real_df rows ``idx``."""
        real_df = self.data[test_ds]['real_df']
        act_dir = self.inputs['datasets'][test_ds]['judge_interp_dir']
        mids = real_df['measurement_id'].iloc[idx].tolist()
        raw_ntp = real_df[f'judgement_p_true_{self.judge_model}'].iloc[idx].to_numpy()
        pd_data = self.probe[train_ds]
        top = pd_data['top_layer'] if self.probe_type == 'layer' else pd_data['top_k_heads']
        ntp_probs = self.ntp_cal[train_ds]['calibrator'].predict_proba(raw_ntp.reshape(-1, 1))[:, 1]
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


def run_sweep(inp, platt_ns, recalibration, n_resamples):
    """smECE per (train_ds, test_ds, method, n, fit sample).

    Returns the summary table (one row per cell and n) and the per-resample table.
    """
    # Fit samples depend only on (test_ds, n): shared by every train probe and method.
    samples = {}
    for ds in inp.datasets:
        d = inp.data[ds]
        pool_doc_ids = d['real_df']['document_id'].to_numpy()[d['pool_idx']]
        for n in platt_ns:
            samples[ds, n] = db.two_class_fit_samples(pool_doc_ids, d['labels'][d['pool_idx']], n,
                                                      n_resamples, inp.seed, ds)
            print(f'  {ds}: n={n}: {n_resamples} fit samples in {samples[ds, n][1]} draws '
                  f'(skipped: {samples[ds, n][2]})')

    rows, sample_rows = [], []
    for train_ds in inp.datasets:
        pi_tr = {'probe': inp.probe[train_ds]['train_prevalence'],
                 'ntp': inp.ntp_cal[train_ds]['train_prevalence']}
        for test_ds in inp.datasets:
            d = inp.data[test_ds]
            real_df, pool_idx, test_idx = d['real_df'], d['pool_idx'], d['test_idx']
            # Excluding train_ds's own probe-training documents must remove nothing more.
            exclude = d['pool_docs'] | set(inp.probe[train_ds]['syn_document_ids'])
            assert np.array_equal(np.where(~real_df['document_id'].isin(exclude).to_numpy())[0], test_idx), (
                f'{train_ds} probe -> {test_ds}: its training documents overlap the test rows')
            test_labels, pool_labels = d['labels'][test_idx], d['labels'][pool_idx]
            n_test_docs = real_df['document_id'].iloc[test_idx].nunique()

            t0 = time.time()
            test_scores = inp.score_rows(train_ds, test_ds, test_idx)
            pool_scores = inp.score_rows(train_ds, test_ds, pool_idx)  # scored once, indexed per sample
            print(f'  {train_ds} probe -> {test_ds}: scored in {time.time() - t0:.0f}s', flush=True)

            for n in platt_ns:
                t_n = time.time()
                kept, n_draws, skips = samples[test_ds, n]
                for method, key, _ in _METHODS:
                    values, rates = [], []
                    for r, s in kept:
                        y = pool_labels[s]
                        coef, icpt = fit_map(recalibration, pool_scores[key][s], y, pi_tr[key])
                        v = smece(apply_platt(test_scores[key], coef, icpt), test_labels)
                        values.append(v)
                        rates.append(float(y.mean()))
                        sample_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds, 'Type': method,
                                            'Platt N': n, 'Recalibration': recalibration, 'Fit sample': r,
                                            'coef': coef, 'intercept': icpt, 'Label rate': rates[-1],
                                            'Train prevalence': float(pi_tr[key]), 'SmECE': v})
                    rows.append({
                        'Train dataset': train_ds, 'Test dataset': test_ds, 'Type': method, 'Platt N': n,
                        'Recalibration': recalibration, 'Train resamples': len(values), 'Fit draws': n_draws,
                        'Short-pool skips': skips['short_pool'], 'Single-class skips': skips['single_class'],
                        'Pool N': len(pool_idx), 'Test N': len(test_idx), 'Test docs': n_test_docs,
                        'Test label rate': float(test_labels.mean()), 'Fit label rate': float(np.mean(rates)),
                        **summarize(values),
                    })
                print(f'    n={n}: {n_resamples} resamples x {len(_METHODS)} methods in {time.time() - t_n:.0f}s',
                      flush=True)

    summary = pd.DataFrame(rows)
    samples_df = pd.DataFrame(sample_rows)
    keys = ['Train dataset', 'Test dataset', 'Type', 'Platt N']
    assert len(summary) == len(inp.datasets) ** 2 * len(_METHODS) * len(platt_ns), len(summary)
    assert not summary.duplicated(keys).any()
    assert (summary['Train resamples'] == n_resamples).all()
    assert len(samples_df) == len(summary) * n_resamples, len(samples_df)
    assert not samples_df.duplicated(keys + ['Fit sample']).any()
    assert (summary['SmECE_lo'] <= summary['SmECE']).all() and (summary['SmECE'] <= summary['SmECE_hi']).all()
    return summary, samples_df


def plot_sweep(df, datasets, platt_ns, figures_dir):
    # One figure per (method, train_ds): one curve per test_ds, colored by test_ds.
    # Line = mean over training resamples; band = their 2.5-97.5 percentiles (no test noise).
    for method, _, linestyle in _METHODS:
        for train_ds in datasets:
            fig, ax = plt.subplots(figsize=(4.0, 3.8))
            for test_ds in datasets:
                sub = df[(df['Type'] == method) & (df['Train dataset'] == train_ds)
                         & (df['Test dataset'] == test_ds)].sort_values('Platt N')
                assert sub['Platt N'].tolist() == platt_ns, (method, train_ds, test_ds)
                color = _DS_COLORS[test_ds]
                ax.plot(sub['Platt N'], sub['SmECE'], linestyle=linestyle, color=color,
                        lw=2.5, marker='o', ms=3.5, zorder=3)
                ax.fill_between(sub['Platt N'], sub['SmECE_lo'], sub['SmECE_hi'],
                                color=color, alpha=0.20, linewidth=0, zorder=1)
            ax.set_xscale('log')
            ax.set_xticks(platt_ns)
            ax.set_xticklabels([str(n) for n in platt_ns])
            ax.minorticks_off()
            ax.set_ylim(bottom=0)
            ax.set_xlabel('Platt Training Samples')
            if method == 'NTP':
                ax.set_ylabel('SmECE')
            ax.set_title(method, fontsize=15, style='italic')
            ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
            ax.set_axisbelow(True)
            fig.tight_layout()
            fig.savefig(figures_dir / f'smece_vs_n_train_resample_{method.lower()}_train-{train_ds}.pdf',
                        bbox_inches='tight', dpi=200)
            plt.close(fig)

    handles = [mlines.Line2D([], [], color=_DS_COLORS[ds], lw=2, marker='o', ms=3.5, label=_DS_LABELS[ds])
               for ds in datasets]
    fig_leg, ax_leg = plt.subplots(figsize=(6.0, 0.45))
    ax_leg.axis('off')
    ax_leg.legend(handles=handles, loc='center', ncol=len(datasets), fontsize=13,
                  frameon=False, handlelength=2.0, title='Test set', title_fontsize=13)
    fig_leg.savefig(figures_dir / 'legend_smece_vs_n.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig_leg)


def main(config_path):
    cfg = load_platt_sweep_v2_config(Path(config_path))
    p = cfg['params']
    platt_ns, recalibration, n_resamples = list(p['platt_ns']), p['recalibration'], p['n_train_resamples']
    print(f"[platt sweep v2] config: {cfg['id']} | ns: {platt_ns} | recalibration: {recalibration} | "
          f"train resamples: {n_resamples}")
    out_dir = REPO_ROOT / 'analysis' / 'results' / 'calibration' / cfg['id']
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    inp = SweepInputs(cfg)
    summary, samples_df = run_sweep(inp, platt_ns, recalibration, n_resamples)
    summary.to_csv(out_dir / 'smece_vs_platt_n.csv', index=False)
    samples_df.to_csv(out_dir / 'smece_train_resamples.csv', index=False)
    print(summary.to_string(index=False, float_format='{:.3f}'.format))
    plot_sweep(summary, inp.datasets, platt_ns, figures_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibration error vs. number of recalibration training samples, over training resamples (real setting).")
    parser.add_argument('config', type=Path,
                        help="analysis-configs/<id>.yaml (see load_platt_sweep_v2_config)")
    main(parser.parse_args().config)
