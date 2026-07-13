"""
Calibration of probe / NTP confidence scores against HUMAN labels.

``analysis/calibration.py`` scores confidence against machine labels
(``judgement_combined`` OR'd with fuzzy ground-truth matching).  Those labels are
themselves noisy, so an ECE computed against them mixes probe miscalibration with
label error.  This script recomputes the same calibration statistics on the subset
of data points that carry a human judgement from ``experiments/validation.py``.

Predictions are joined to human labels on ``measurement_id``.  That id is a
positional index into a run's ``final.json``, so it is only meaningful within one
(dataset, extraction_model, extraction_date) triple — the human validation run must
therefore have been collected against the same extraction date this script scores
(``EXTRACTION_DATES`` in analysis/calibration.py).  We pin that date rather than
letting it resolve to the latest, so a mismatched run fails loudly instead of
silently joining misaligned rows.

Two splits are reported per setting:
    test_docs — human points outside the probe's synthetic training documents
                (mirrors calibration.py's real-data filter; leakage-free)
    all_docs  — every human-labelled point (larger n, mildly optimistic in-domain)

Usage
-----
    # Pre-flight: check the measurement_id join resolves before computing anything
    python analysis/calibration_human.py --extraction-model gemma-3-27b --check

    # Full run: metrics CSV + reliability diagrams
    python analysis/calibration_human.py --extraction-model gemma-3-27b

    python analysis/calibration_human.py --extraction-model llama-3.1-8b --probe-type layer
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))
# analysis/calibration.py does REPO_ROOT = Path.cwd() at import time.
os.chdir(_REPO_ROOT)

import argparse
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.metrics import roc_auc_score, brier_score_loss

# Importing calibration reuses the exact probes, NTP calibrators, pinned judge dates
# and prevalence adjustment behind the headline results, so the human numbers are
# directly comparable to them.  It also runs that module's loading at import time.
from analysis.calibration import (
    EXTRACTION_MODEL, PROBE_TYPE, DATASETS, JUDGE_MODELS, JUDGE_DATASETS,
    EXTRACTION_DATES, JUDGE_DATES_REAL, PI_TE_ESTIMATE, DATASET_PAIRS,
    probe_cache, ntp_cal_cache, test_data,
    FIGURES_DIR, RESULTS_DIR, _DS_COLORS, _DS_LABELS,
    ECE_N_BOOT, ECE_CI, ECE_SEED,
)
from analysis.loaders import load_human_judgements, load_activations, load_layer_outputs
from analysis.metrics import validity_rate_from_labels
from scholarlm.utils.calibration import (
    reliability_diagram_data, bootstrap_ece, intercept_adjustment,
)

SPLITS = ['test_docs', 'all_docs']
DEFAULT_N_BINS = 10


# ── Human label loading ──────────────────────────────────────────────────────

def load_human_base(dataset, judge_model, human_date=None):
    """Human labels for one (dataset, judge), joined to combined.json and activations.

    Returns a dict with the human points that resolve in both combined.json (for the
    NTP probability) and the activation file (for the probe), plus the counts needed
    to report what was dropped.  Returns None when the dataset has no human run.
    """
    ext_date = EXTRACTION_DATES[dataset]
    try:
        records, resolved = load_human_judgements(
            dataset, EXTRACTION_MODEL, extraction_date=ext_date, judge_date=human_date,
        )
    except FileNotFoundError as e:
        print(f'  [SKIP] {dataset}/{judge_model}: {e}')
        return None
    if resolved != ext_date:
        raise RuntimeError(
            f'Human run for {dataset} resolved to extraction date {resolved!r}, but '
            f'calibration scores {ext_date!r}. measurement_id is a per-run positional '
            f'index, so joining across runs would silently misalign rows.'
        )
    if not records:
        print(f'  [SKIP] {dataset}/{judge_model}: no non-skipped human judgements')
        return None

    real_df = test_data[dataset]['real_df']
    row_of_mid = {int(m): i for i, m in enumerate(real_df['measurement_id'].tolist())}

    jdate = JUDGE_DATES_REAL[dataset][judge_model]
    if PROBE_TYPE == 'layer':
        acts = load_layer_outputs(dataset, EXTRACTION_MODEL, ext_date, judge_model, jdate)
    else:
        acts = load_activations(dataset, EXTRACTION_MODEL, ext_date, judge_model, jdate)
    act_keys = set(acts.files)

    ntp_col = real_df[f'judgement_p_true_{judge_model}']

    keep, missing_combined, missing_acts, missing_ntp = [], 0, 0, 0
    for r in records:
        mid = int(r['measurement_id'])
        if mid not in row_of_mid:
            missing_combined += 1
            continue
        if str(mid) not in act_keys:
            missing_acts += 1
            continue
        # Probe and NTP must be scored on identical points for the comparison to be
        # fair, so a missing NTP probability drops the point from both.
        if pd.isna(ntp_col.iloc[row_of_mid[mid]]):
            missing_ntp += 1
            continue
        keep.append(r)

    if missing_combined or missing_acts or missing_ntp:
        print(f'  [WARN] {dataset}/{judge_model}: dropped {missing_combined} human points '
              f'absent from combined.json, {missing_acts} without activations, '
              f'{missing_ntp} without an NTP probability')
    if not keep:
        print(f'  [SKIP] {dataset}/{judge_model}: no human points resolved')
        return None

    idx = np.array([row_of_mid[int(r['measurement_id'])] for r in keep])
    return {
        'mids':        [int(r['measurement_id']) for r in keep],
        'labels':      np.array([bool(r['judgement']) for r in keep]),
        'doc_ids':     np.array([r.get('document_id') for r in keep], dtype=object),
        'error_types': [r.get('error_type') for r in keep],
        'idx':         idx,
        'raw_ntp':     ntp_col.iloc[idx].to_numpy(dtype=float),
        'acts':        acts,
        'n_records':   len(records),
        'n_dropped':   missing_combined + missing_acts + missing_ntp,
    }


def _probe_features(base, top):
    """Stack activations for the probe's selected heads / layer, in mid order."""
    acts, mids = base['acts'], base['mids']
    if PROBE_TYPE == 'layer':
        return np.stack(
            [np.array(acts[str(mid)], dtype=np.float32)[top] for mid in mids], axis=0
        )
    return np.concatenate([
        np.stack([np.array(acts[str(mid)], dtype=np.float32)[l, h, :] for mid in mids], axis=0)
        for l, h in top
    ], axis=1)


# ── Predictions ──────────────────────────────────────────────────────────────

def compute_human_predictions(judge_models, datasets, splits, human_date=None,
                              load_from_precomputed=False):
    """{split: {judge: {train_ds: {test_ds: {probe_probs, ntp_probs, labels, ...}}}}}"""
    cache_file = Path(RESULTS_DIR) / f'predictions_human_{EXTRACTION_MODEL}_{PROBE_TYPE}.pkl'
    if load_from_precomputed and cache_file.exists():
        print(f'Loading precomputed human predictions from {cache_file}...')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    results = {s: {jm: {} for jm in judge_models} for s in splits}

    for judge_model in judge_models:
        bases = {}
        for test_ds in datasets:
            if test_ds not in JUDGE_DATASETS[judge_model]:
                continue
            bases[test_ds] = load_human_base(test_ds, judge_model, human_date)

        for train_ds in datasets:
            if train_ds not in JUDGE_DATASETS[judge_model]:
                continue
            pd_data = probe_cache[train_ds][judge_model]
            ntp_cal = ntp_cal_cache[train_ds][judge_model]
            top = pd_data['top_layer'] if PROBE_TYPE == 'layer' else pd_data['top_k_heads']
            syn_docs = set(pd_data['syn_document_ids'])

            for split in splits:
                results[split][judge_model][train_ds] = {}

            for test_ds in datasets:
                if test_ds not in JUDGE_DATASETS[judge_model]:
                    continue
                base = bases.get(test_ds)
                if base is None:
                    continue

                # Probe/NTP scores for every resolved human point; splits then subset.
                probe_all = pd_data['probe'].predict_proba(_probe_features(base, top))[:, 1]
                ntp_all = ntp_cal['calibrator'].predict_proba(
                    base['raw_ntp'].reshape(-1, 1)
                )[:, 1]

                if PI_TE_ESTIMATE is not None:
                    probe_all = intercept_adjustment(
                        probe_all, pi_tr=pd_data['train_prevalence'], pi_te=PI_TE_ESTIMATE)
                    ntp_all = intercept_adjustment(
                        ntp_all, pi_tr=ntp_cal['train_prevalence'], pi_te=PI_TE_ESTIMATE)

                in_test_docs = ~np.isin(base['doc_ids'], list(syn_docs))
                for split in splits:
                    m = in_test_docs if split == 'test_docs' else np.ones(len(probe_all), bool)
                    if m.sum() == 0:
                        print(f'  [SKIP] {split} {judge_model} {train_ds}->{test_ds}: '
                              f'no points survive the filter')
                        continue
                    results[split][judge_model][train_ds][test_ds] = {
                        'probe_probs': probe_all[m],
                        'ntp_probs':   ntp_all[m],
                        'labels':      base['labels'][m],
                        'mids':        [mid for mid, k in zip(base['mids'], m) if k],
                        'error_types': [e for e, k in zip(base['error_types'], m) if k],
                    }

    print(f'Saving human predictions to {cache_file}...')
    with open(cache_file, 'wb') as f:
        pickle.dump(results, f)
    return results


# ── Metrics ──────────────────────────────────────────────────────────────────

def _human_metrics(probs, y_true, threshold=0.5, n_bins=DEFAULT_N_BINS):
    """Threshold metrics plus the three calibration-error variants, each bootstrapped.

    Mirrors calibration._probe_metrics minus `recovery`, which needs the full
    ground-truth denominator and is not interpretable on a ~100-point human subset.
    """
    probs = np.asarray(probs)
    y_true = np.asarray(y_true, dtype=bool)
    preds = probs > threshold
    tp = int((preds & y_true).sum())
    tn = int((~preds & ~y_true).sum())
    fp = int((preds & ~y_true).sum())
    fn = int((~preds & y_true).sum())
    n = len(y_true)
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    rec = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float('nan')
    auroc = (roc_auc_score(y_true, probs)
             if y_true.sum() > 0 and (~y_true).sum() > 0 else float('nan'))

    kw = dict(n_bins=n_bins, n_boot=ECE_N_BOOT, ci=ECE_CI, seed=ECE_SEED)
    ece_ew = bootstrap_ece(probs, y_true, binning='equal_width', p=1, **kw)
    ece_em = bootstrap_ece(probs, y_true, binning='equal_mass', p=1, **kw)
    rmsce = bootstrap_ece(probs, y_true, binning='equal_mass', p=2, debiased=True, **kw)

    bs = float(brier_score_loss(y_true, probs))
    p_pos = float(y_true.mean())
    bss = 1.0 - bs / (p_pos * (1 - p_pos)) if p_pos not in (0.0, 1.0) else float('nan')
    return dict(
        acc=acc, prec=prec, rec=rec, f1=f1, auroc=auroc,
        ece=ece_ew['ece'], ece_lo=ece_ew['ci_low'], ece_hi=ece_ew['ci_high'], ece_se=ece_ew['se'],
        ece_em=ece_em['ece'], ece_em_lo=ece_em['ci_low'], ece_em_hi=ece_em['ci_high'],
        ece_em_se=ece_em['se'],
        rmsce_db=rmsce['ece'], rmsce_db_lo=rmsce['ci_low'], rmsce_db_hi=rmsce['ci_high'],
        rmsce_db_se=rmsce['se'],
        bs=bs, bss=bss, n=n, n_pos=int(y_true.sum()),
        validity=validity_rate_from_labels(y_true, preds),
    )


def compute_metrics(setting_results, n_bins=DEFAULT_N_BINS):
    rows = []
    for split in setting_results:
        for judge_model in setting_results[split]:
            for train_ds in setting_results[split][judge_model]:
                for test_ds in setting_results[split][judge_model][train_ds]:
                    rdict = setting_results[split][judge_model][train_ds][test_ds]
                    for probs, kind in [(rdict['ntp_probs'], 'NTP'),
                                        (rdict['probe_probs'], 'Probe')]:
                        m = _human_metrics(probs, rdict['labels'], n_bins=n_bins)
                        rows.append({
                            'Split':         split,
                            'Judge model':   judge_model,
                            'Train dataset': train_ds,
                            'Test dataset':  test_ds,
                            'Type':          kind,
                            'Accuracy':      m['acc'],
                            'Precision':     m['prec'],
                            'Recall':        m['rec'],
                            'F1':            m['f1'],
                            'AUROC':         m['auroc'],
                            # L1 ECE, equal-width plug-in + bootstrap CI
                            'ECE':           m['ece'],
                            'ECE_lo':        m['ece_lo'],
                            'ECE_hi':        m['ece_hi'],
                            'ECE_se':        m['ece_se'],
                            # L1 ECE, adaptive equal-mass plug-in + bootstrap CI
                            'ECE_em':        m['ece_em'],
                            'ECE_em_lo':     m['ece_em_lo'],
                            'ECE_em_hi':     m['ece_em_hi'],
                            'ECE_em_se':     m['ece_em_se'],
                            # Debiased L2 RMS calibration error (Kumar 2019), equal-mass + CI
                            'RMSCE_db':      m['rmsce_db'],
                            'RMSCE_db_lo':   m['rmsce_db_lo'],
                            'RMSCE_db_hi':   m['rmsce_db_hi'],
                            'RMSCE_db_se':   m['rmsce_db_se'],
                            'Brier':         m['bs'],
                            'BSS':           m['bss'],
                            'Validity':      m['validity'],
                            'N':             m['n'],
                            'N_pos':         m['n_pos'],
                        })
    return pd.DataFrame(rows)


# ── Plots ────────────────────────────────────────────────────────────────────

def plot_calibration_curves(setting_results, split, n_bins=DEFAULT_N_BINS):
    """Reliability diagrams against human labels — same layout as calibration.py."""
    # With a single dataset selected there are no pairs, so fall back to (ds, ds) —
    # otherwise the in-domain diagram would be silently skipped.
    pairs = DATASET_PAIRS or [(ds, ds) for ds in DATASETS]

    for judge_model in setting_results[split]:
        for ds_a, ds_b in pairs:
            if ds_a not in JUDGE_DATASETS[judge_model] or ds_b not in JUDGE_DATASETS[judge_model]:
                continue
            pair_datasets = [ds_a] if ds_a == ds_b else [ds_a, ds_b]
            subfigure_dir = (FIGURES_DIR /
                             f'{judge_model}/{EXTRACTION_MODEL}/{PROBE_TYPE}/{ds_a}_{ds_b}/')
            Path(subfigure_dir).mkdir(parents=True, exist_ok=True)

            for ctype in ['in-domain', 'cross-domain']:
                fig_cal, ax_cal = plt.subplots(figsize=(4.0, 3.8))
                ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, zorder=1)
                drew = False

                for train_ds in pair_datasets:
                    for test_ds in pair_datasets:
                        if (train_ds == test_ds) != (ctype == 'in-domain'):
                            continue
                        rdict = (setting_results[split][judge_model]
                                 .get(train_ds, {}).get(test_ds))
                        if rdict is None:
                            continue
                        drew = True
                        color = _DS_COLORS[train_ds]

                        for probs, style, lw, zbase in [
                            (rdict['probe_probs'], '-', 2.5, 3),
                            (rdict['ntp_probs'], '--', 2.0, 1),
                        ]:
                            d = reliability_diagram_data(probs, rdict['labels'], n_bins=n_bins)
                            v = d['bin_counts'] > 0
                            if not v.any():
                                continue
                            counts = d['bin_counts'][v].astype(float)
                            sizes = 12 + 68 * (counts / counts.max())
                            conf, acc = d['bin_confidence'][v], d['bin_accuracy'][v]
                            sem = d['bin_accuracy_sem'][v]
                            ax_cal.plot(conf, acc, style, color=color, lw=lw, zorder=zbase + 2)
                            ax_cal.scatter(conf, acc, s=sizes, color=color, zorder=zbase + 3)
                            ax_cal.fill_between(conf, acc - sem, acc + sem, color=color,
                                                alpha=0.20, linewidth=0, zorder=2)

                if not drew:
                    plt.close(fig_cal)
                    continue

                ax_cal.set_xlim(-0.02, 1.02)
                ax_cal.set_ylim(-0.02, 1.02)
                ax_cal.set_xlabel('Predicted Probability')
                if ctype == 'in-domain':
                    ax_cal.set_ylabel('Observed Frequency')
                    ax_cal.set_title('Within', fontsize=15, style='italic')
                else:
                    ax_cal.set_ylabel('')
                    ax_cal.set_title('Cross', fontsize=15, style='italic')
                ax_cal.grid(alpha=0.25, linestyle='-', linewidth=0.4)
                ax_cal.set_axisbelow(True)
                fig_cal.tight_layout()
                fig_cal.savefig(subfigure_dir / f'cal_human_{split}_{ctype}.pdf',
                                bbox_inches='tight', dpi=200)
                plt.close(fig_cal)


def save_legend():
    handles = [
        mlines.Line2D([], [], color=_DS_COLORS[ds], lw=2, marker='o', ms=3.5, label=_DS_LABELS[ds])
        for ds in DATASETS
    ] + [
        mlines.Line2D([], [], color='#444444', lw=2, linestyle='-', label='Probe'),
        mlines.Line2D([], [], color='#444444', lw=2, linestyle='--', label='NTP'),
    ]
    fig, ax = plt.subplots(figsize=(10.0, 0.45))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=6, fontsize=13, frameon=False, handlelength=2.0)
    fig.savefig(FIGURES_DIR / 'legend_calibration_human.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)


# ── Pre-flight check ─────────────────────────────────────────────────────────

def run_check(judge_models, datasets, human_date=None):
    """Report how the measurement_id join resolves, without computing any metrics."""
    for judge_model in judge_models:
        for ds in datasets:
            if ds not in JUDGE_DATASETS[judge_model]:
                continue
            print(f'\n--- {ds} / judge={judge_model} (extraction {EXTRACTION_DATES[ds]}) ---')
            base = load_human_base(ds, judge_model, human_date)
            if base is None:
                continue
            labels = base['labels']
            print(f'  human records (non-skipped): {base["n_records"]}')
            print(f'  resolved (combined + activations): {len(labels)}  '
                  f'(dropped {base["n_dropped"]})')
            print(f'  labels: {int(labels.sum())} valid / {int((~labels).sum())} invalid')

            errs = [e for e in base['error_types'] if e]
            if errs:
                counts = {e: errs.count(e) for e in sorted(set(errs))}
                print(f'  error types: {counts}')

            for train_ds in datasets:
                if train_ds not in JUDGE_DATASETS[judge_model]:
                    continue
                syn_docs = set(probe_cache[train_ds][judge_model]['syn_document_ids'])
                dropped = int(np.isin(base['doc_ids'], list(syn_docs)).sum())
                note = ''
                if train_ds != ds and dropped:
                    note = '  <-- unexpected: cross-domain filter should be a no-op'
                print(f'  probe trained on {train_ds}: test_docs keeps '
                      f'{len(labels) - dropped}/{len(labels)} (syn-doc filter drops {dropped})'
                      f'{note}')


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Consumed by calibration._select_settings at import; declared here for --help.
    p.add_argument('--extraction-model', default=None)
    p.add_argument('--probe-type', default=None, choices=['head', 'layer'])
    p.add_argument('--datasets', nargs='+', default=None)
    p.add_argument('--judge-models', nargs='+', default=None)
    p.add_argument('--splits', nargs='+', default=SPLITS, choices=SPLITS)
    p.add_argument('--n-bins', type=int, default=DEFAULT_N_BINS)
    p.add_argument('--human-date', default=None, help='Validation session date tag.')
    p.add_argument('--check', action='store_true',
                   help='Report how the measurement_id join resolves, then exit.')
    # Recompute by default: the cache is keyed only on (extraction_model, probe_type), so it
    # goes stale whenever new human labels land. Opt in only when iterating on plots/metrics.
    p.add_argument('--use-cache', action='store_true',
                   help='Reuse the cached predictions pickle instead of recomputing.')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--output', default=None, metavar='CSV')
    args = p.parse_args()

    datasets = args.datasets or DATASETS
    judge_models = args.judge_models or JUDGE_MODELS

    if args.check:
        run_check(judge_models, datasets, args.human_date)
        return

    results = compute_human_predictions(
        judge_models, datasets, args.splits, human_date=args.human_date,
        load_from_precomputed=args.use_cache,
    )

    metrics_df = compute_metrics(results, n_bins=args.n_bins)
    if metrics_df.empty:
        print('No human-labelled settings resolved — nothing to report. '
              'Run experiments/validation.py first, or use --check to diagnose.')
        return

    print(metrics_df.to_string(index=False, float_format='{:.3f}'.format))
    out = Path(args.output) if args.output else (
        RESULTS_DIR / f'metrics_human_{EXTRACTION_MODEL}_{PROBE_TYPE}.csv')
    out.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out, index=False)
    print(f'\nSaved metrics to {out}')

    if not args.no_plots:
        for split in args.splits:
            plot_calibration_curves(results, split, n_bins=args.n_bins)
        save_legend()
        print(f'Saved figures under {FIGURES_DIR}')


if __name__ == '__main__':
    main()
