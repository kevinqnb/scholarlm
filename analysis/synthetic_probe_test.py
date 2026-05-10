import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import re
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, brier_score_loss

from analysis.loaders import (
    load_activations, load_layer_outputs, load_combined_judgements,
    load_extraction, load_ground_truth, load_trained_probe, load_trained_ntp_calibrator,
    cached_match, load_synthetic_activations, load_synthetic_layer_outputs,
    load_synthetic_responses,
)
from scholarlm.utils.calibration import reliability_diagram_data, rescale_probabilities_em
from scholarlm.utils.unit_conversion import apply_unit_conversion
from experiments.run_extraction import load_dataset_config
import paths

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.title_fontsize": 9,
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

# blue: 7, orange: 1, red: 0, green: 4
palette = sns.color_palette("husl", 10)

FIGURES_DIR = "figures/synthetic_probe/"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# ── Standalone calibration legend ─────────────────────────────────────────────
_legend_handles = [
    mlines.Line2D([], [], color=palette[7], lw=2, marker='o', ms=3.5, label='Synthetic PLW'),
    mlines.Line2D([], [], color=palette[1], lw=2, marker='o', ms=3.5, label='Synthetic NF'),
    mlines.Line2D([], [], color=palette[0], lw=2, marker='o', ms=3.5, label='Extracted PLW'),
    mlines.Line2D([], [], color=palette[4], lw=2, marker='o', ms=3.5, label='Extracted NF'),
    mlines.Line2D([], [], color='#444444', lw=2, linestyle='-',  label='Probe'),
    mlines.Line2D([], [], color='#444444', lw=2, linestyle='--', label='NTP'),
]
_fig_leg, _ax_leg = plt.subplots(figsize=(10.0, 0.45))
_ax_leg.axis('off')
_ax_leg.legend(handles=_legend_handles, loc='center', ncol=6, fontsize=9,
               frameon=False, handlelength=2.0)
_fig_leg.savefig(FIGURES_DIR + 'legend_calibration.pdf', bbox_inches='tight', dpi=200)
plt.show()


# ── Parameters ───────────────────────────────────────────────────────────────
DATASETS = ['pond', 'nfix']
EXTRACTION_MODEL = 'gemma-3-27b'
JUDGE_MODELS = ['llama-3.1-8b', 'mistral-7b', 'qwen-2.5-7b']


# Extraction date per test dataset
EXTRACTION_DATES = {
    'pond': '2026_05_05',
    'nfix': '2026_05_06',
}

# Judge date for synthetic test activations: {dataset: {judge_model: date_str | None}}
# None → auto-detect latest
JUDGE_DATES_SYN = {
    'pond': {
        'llama-3.1-8b': '2026_05_04',
        'mistral-7b': '2026_05_04',
        'qwen-2.5-7b': '2026_05_04',
    },
    'nfix': {
        'llama-3.1-8b': '2026_05_04',
        'mistral-7b': '2026_05_04',
        'qwen-2.5-7b': '2026_05_04',
    },
}

# Judge date for real activations: {test_dataset: {judge_model: date_str | None}}
# None → auto-detect latest
JUDGE_DATES_REAL = {
    'pond': {
        'llama-3.1-8b': '2026_05_06',
        'mistral-7b': '2026_05_06',
        'qwen-2.5-7b': '2026_05_06',
    },
    'nfix': {
        'llama-3.1-8b': '2026_05_05',
        'mistral-7b': '2026_05_05',
        'qwen-2.5-7b': '2026_05_05',
    },
}

THRESHOLD_SWEEP = np.linspace(0.0, 0.95, 20)  # thresholds for operating-curve plot
EDGE_THRESHOLD  = 1 / 3  # minimum fuzzy weight to count as a match


# NTP calibrators are probe-type-independent — load once outside the probe loop.
ntp_cal_cache = {}
for ds in DATASETS:
    ntp_cal_cache[ds] = {}
    for jm in JUDGE_MODELS:
        ntp_cal_cache[ds][jm] = load_trained_ntp_calibrator(ds, jm)

for PROBE_TYPE in ['head', 'layer']:
    print(f'\n{"#"*80}\nPROBE TYPE: {PROBE_TYPE.upper()}\n{"#"*80}')

    # ─────────────────────────────────────────────────────────────────
    probe_cache = {}
    for ds in DATASETS:
        probe_cache[ds] = {}
        for jm in JUDGE_MODELS:
            probe_cache[ds][jm] = load_trained_probe(ds, jm, ptype = PROBE_TYPE)
    # ─────────────────────────────────────────────────────────────────

    def _probe_metrics(probs, y_true, threshold=0.5):
        """Compute metrics at a fixed threshold. Returns dict."""
        probs   = np.asarray(probs)
        y_true  = np.asarray(y_true, dtype=bool)
        preds   = probs > threshold
        tp  = int(( preds &  y_true).sum())
        tn  = int((~preds & ~y_true).sum())
        fp  = int(( preds & ~y_true).sum())
        fn  = int((~preds &  y_true).sum())
        n   = len(y_true)
        acc   = (tp + tn) / n
        prec  = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        rec   = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        f1    = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float('nan')
        auroc = roc_auc_score(y_true, probs) if y_true.sum() > 0 and (~y_true).sum() > 0 else float('nan')
        ece   = reliability_diagram_data(probs, y_true)['ece']
        bs    = float(brier_score_loss(y_true, probs))
        p_pos = float(y_true.mean())
        bss   = 1.0 - bs / (p_pos * (1 - p_pos)) if p_pos not in (0.0, 1.0) else float('nan')
        return dict(acc=acc, prec=prec, rec=rec, f1=f1, auroc=auroc,
                    ece=ece, bs=bs, bss=bss, n=n)


    # ─────────────────────────────────────────────────────────────────
    _SUPER_MAP  = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺', '0123456789-+')
    _SUB_MAP    = str.maketrans('₀₁₂₃₄₅₆₇₈₉₋₊', '0123456789-+')
    _SCRIPT_MAP = {**_SUPER_MAP, **_SUB_MAP}

    _LATEX_RE    = re.compile(r'[\^_]\{([^}]*)\}|[\^_]([+-]?\d+)')
    _COMPOUND_RE = re.compile(r'(\w+)[\s\-]([A-Z][a-zA-Z0-9]*)')


    def nfix_clean_unit(s: str) -> str:
        if not isinstance(s, str):
            return s
        s = s.translate(_SCRIPT_MAP)
        s = _LATEX_RE.sub(lambda m: m.group(1) if m.group(1) is not None else m.group(2), s)
        s = s.replace('µ', 'u').replace('μ', 'u')
        s = _COMPOUND_RE.sub(r'\1-\2', s)
        s = re.sub(r'\byr\b', 'y', s)
        s = s.lower()
        s = re.sub(r'\bday\b', 'd', s)
        s = re.sub(r'\bhr\b',  'h', s)
        return s


    def get_matching_config(dataset):
        if dataset == 'pond':
            strict = {'document_id': 'document_id', 'attribute': 'attribute',
                    'value': 'converted_value', 'units': 'units'}
            fuzzy  = {'name': 'name', 'location': 'location', 'ecosystem': 'ecosystem'}
        elif dataset == 'nfix':
            strict = {'document_id': 'document_id', 'attribute': 'attribute',
                    'value': 'converted_value', 'units': 'units'}
            fuzzy  = {'name': 'name', 'site_type': 'site_type'}
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        return strict, fuzzy
    # ─────────────────────────────────────────────────────────────────

    test_data = {}  # test_data[dataset] = dict with pre-loaded data

    for ds in DATASETS:
        print(f'Loading test data for {ds}...')
        config  = load_dataset_config(ds)
        records = load_extraction(ds, EXTRACTION_MODEL, EXTRACTION_DATES[ds])
        ext_df  = pd.DataFrame(records)
        ext_df  = apply_unit_conversion(ext_df, {})

        if ds == 'nfix':
            ext_df['attribute'] = ext_df['attribute'].map({
                'nfix_rate_areal': 'nfix_rate', 'nfix_rate_volumetric': 'nfix_rate',
                'nfix_rate_mass':  'nfix_rate', 'nfix_rate': 'nfix_rate',
            })
            ext_df['units'] = ext_df['units'].apply(nfix_clean_unit)

        real_df = pd.DataFrame(load_combined_judgements(ds, EXTRACTION_MODEL, EXTRACTION_DATES[ds]))
        gt_df   = load_ground_truth(config)

        strict, fuzzy = get_matching_config(ds)
        cache_path = paths.extraction(ds, EXTRACTION_MODEL, EXTRACTION_DATES[ds]) / 'match_cache.pkl'
        matching, edges, edge_weights = cached_match(
            gt_df, ext_df,
            strict_matching=strict,
            fuzzy_matching=fuzzy,
            fuzzy_threshold=0.0,
            cache_path=cache_path,
        )

        ex_edge_exists = np.zeros(len(ext_df), dtype=bool)
        for (gt_idx, ex_idx), w in zip(edges, edge_weights):
            if w > EDGE_THRESHOLD:
                ex_edge_exists[int(ex_idx)] = True
        jlabels     = real_df['judgement_combined'].to_numpy(dtype=bool)
        real_labels = jlabels | ex_edge_exists

        test_data[ds] = {
            'extraction_df':   ext_df,
            'real_df':         real_df,
            'ground_truth_df': gt_df,
            'real_labels':     real_labels,
            'ex_edge_exists':  ex_edge_exists,
            'edges':           edges,
            'edge_weights':    edge_weights,
        }

        pos = real_labels.sum()
        print(f'  {ds}: {len(ext_df)} extractions, {len(gt_df)} GT rows, '
            f'pos={pos} ({pos / len(real_labels):.1%})')
        print()

    # ─────────────────────────────────────────────────────────────────
    _DS_COLORS = {
        ('pond', 'syn'):  palette[7],
        ('pond', 'real'): palette[0],
        ('nfix', 'syn'):  palette[1],
        ('nfix', 'real'): palette[4],
    }
    _DTYPE_LS = {'real': '-', 'syn': '-'}


    def analyze_probe(judge_model, train_dataset):
        """Calibration curves, threshold sweep, and metrics table for one
        (judge_model, train_dataset) pair evaluated across four test settings."""
        pd_data     = probe_cache[train_dataset][judge_model]
        ntp_cal_data = ntp_cal_cache[train_dataset][judge_model]
        if PROBE_TYPE == "layer":
            top = pd_data['top_layer']
        else:
            top    = pd_data['top_k_heads']
            
        header   = f'{judge_model}  |  trained on {train_dataset}'
        SEP      = '=' * 72
        print(f'\n{SEP}\n  {header}\n{SEP}')
        subfigure_dir = FIGURES_DIR + f"{judge_model}/{EXTRACTION_MODEL}/"
        Path(subfigure_dir).mkdir(parents=True, exist_ok=True)

        # ── Collect data for each test setting ────────────────────────────────
        setting_results = []

        for dtype in ('syn', 'real'):
            for test_ds in DATASETS:
                dtype_str = 'Syn.' if dtype == 'syn' else 'Real'
                s_label  = f'{dtype_str} {test_ds}'
                color    = _DS_COLORS[(test_ds,dtype)]
                ls       = _DTYPE_LS[dtype]

                if dtype == 'syn':
                    jdate    = JUDGE_DATES_SYN[test_ds][judge_model]
                    syn_resp = load_synthetic_responses(test_ds, judge_model, jdate, split='test')
                    syn_df_s = pd.DataFrame(syn_resp)
                    mids     = syn_df_s['measurement_id'].tolist()
                    labels   = (syn_df_s['label'] == 'valid').to_numpy(dtype=bool)
                    raw_ntp_probs = syn_df_s['judgement_p_true'].to_numpy()
                    ntp_probs = ntp_cal_data['calibrator'].predict_proba(
                        raw_ntp_probs.reshape(-1, 1)
                    )[:, 1]

                    if PROBE_TYPE == "layer":
                        syn_lo  = load_synthetic_layer_outputs(test_ds, judge_model, jdate, split='test')
                        X = np.stack([
                            np.array(syn_lo[str(mid)], dtype=np.float32)[top]
                            for mid in mids
                        ], axis=0)
                        raw_probs = pd_data['probe'].predict_proba(X)[:, 1]

                    else:
                        syn_act  = load_synthetic_activations(test_ds, judge_model, jdate, split='test')
                        X = np.concatenate([
                            np.stack([
                                np.array(syn_act[str(mid)], dtype=np.float32)[l, h, :]
                                for mid in mids
                            ], axis=0)
                            for l, h in top
                        ], axis=1)
                        raw_probs = pd_data['probe'].predict_proba(X)[:, 1]

                    print(f'  {s_label}: n={len(mids)}, pos={labels.sum()} '
                        f'({labels.mean():.1%})')
                    setting_results.append({
                        'label': s_label, 'color': color, 'ls': ls,
                        'probe_probs': raw_probs, 'ntp_probs': ntp_probs,
                        'labels': labels, 'is_syn': True,
                    })

                else:  # real
                    td       = test_data[test_ds]
                    real_df  = td['real_df']
                    syn_docs = set(pd_data['syn_document_ids'])
                    mask     = ~real_df['document_id'].isin(syn_docs)
                    idx      = np.where(mask.to_numpy())[0]
                    mids     = real_df['measurement_id'].iloc[idx].tolist()
                    labels   = td['real_labels'][idx]
                    jdate    = JUDGE_DATES_REAL[test_ds][judge_model]

                    raw_ntp_probs = real_df[f'judgement_p_true_{judge_model}'].iloc[idx].to_numpy()
                    ntp_probs = ntp_cal_data['calibrator'].predict_proba(
                        raw_ntp_probs.reshape(-1, 1)
                    )[:, 1]

                    if PROBE_TYPE == "layer":
                        real_lo  = load_layer_outputs(test_ds, EXTRACTION_MODEL, EXTRACTION_DATES[test_ds], judge_model, jdate)
                        X = np.stack([
                            np.array(real_lo[str(mid)], dtype=np.float32)[top]
                            for mid in mids
                        ], axis=0)
                        raw_probs = pd_data['probe'].predict_proba(X)[:, 1]
                    else:
                        real_act = load_activations(
                            test_ds, EXTRACTION_MODEL, EXTRACTION_DATES[test_ds], judge_model, jdate
                        )
                        X = np.concatenate([
                            np.stack([
                                np.array(real_act[str(mid)], dtype=np.float32)[l, h, :]
                                for mid in mids
                            ], axis=0)
                            for l, h in top
                        ], axis=1)
                        raw_probs = pd_data['probe'].predict_proba(X)[:, 1]

                    n_gt = len(td['ground_truth_df'])
                    ex_to_test_pos = {int(idx[i]): i for i in range(len(idx))}
                    test_edges = [
                        (int(gt_i), ex_to_test_pos[int(ex_i)])
                        for (gt_i, ex_i), w in zip(td['edges'], td['edge_weights'])
                        if int(ex_i) in ex_to_test_pos and w > EDGE_THRESHOLD
                    ]
                    n_excl = int((~mask).sum())
                    n_pap  = real_df.loc[~mask, 'document_id'].nunique()
                    print(f'  {s_label}: n={len(idx)}, pos={labels.sum()} '
                        f'({labels.mean():.1%}), pi_te_act={np.mean(labels):.3f}  '
                        f'[excl {n_excl} rows / {n_pap} syn papers]')
                    setting_results.append({
                        'label': s_label, 'color': color, 'ls': ls,
                        'probe_probs': raw_probs, 'ntp_probs': ntp_probs,
                        'labels': labels, 'is_syn': False,
                        'test_edges': test_edges, 'n_gt': n_gt,
                    })

        # ── Figure 1: Calibration curves ──────────────────────────────────────
        for dtype in ['syn', 'real']:
            fig_cal, ax_cal = plt.subplots(figsize=(4.0, 3.8))
            ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, zorder=1)
            
            for r in setting_results:
                rdtype = 'syn' if r['is_syn'] else 'real'
                if rdtype != dtype:
                    continue
                    
                # Determine color based on dataset and dtype
                test_ds = 'pond' if 'pond' in r['label'].lower() else 'nfix'
                color = _DS_COLORS[(test_ds, rdtype)]
            
                # Probe — solid line with markers
                d_prb = reliability_diagram_data(r['probe_probs'], r['labels'])
                v_prb = d_prb['bin_counts'] > 0
                ax_cal.plot(
                    d_prb['bin_confidence'][v_prb], d_prb['bin_accuracy'][v_prb],
                    r['ls'], color=color, lw=2.0, marker='o', ms=3.5,
                    zorder=3,
                )

                # NTP baseline — very faint, dashed
                d_ntp = reliability_diagram_data(r['ntp_probs'], r['labels'])
                v_ntp = d_ntp['bin_counts'] > 0
                ax_cal.plot(
                    d_ntp['bin_confidence'][v_ntp], d_ntp['bin_accuracy'][v_ntp],
                    '--', color=color, lw=1.5, alpha=1.0, zorder=1, marker='o', ms=3.5,
                )
                
                # Add error bands: SEM of accuracy within each bin (very subtle)
                bin_sems = d_prb['bin_accuracy_sem'][v_prb]
                conf_valid = d_prb['bin_confidence'][v_prb]
                acc_valid = d_prb['bin_accuracy'][v_prb]
                
                ax_cal.fill_between(
                    conf_valid, 
                    acc_valid - bin_sems, 
                    acc_valid + bin_sems,
                    color=color, alpha=0.08, linewidth=0, zorder=2
                )
            
            ax_cal.set_xlim(-0.02, 1.02)
            ax_cal.set_ylim(-0.02, 1.02)
            ax_cal.set_xlabel('Predicted Probability')
            ax_cal.set_ylabel('Observed Frequency')
            ax_cal.grid(alpha=0.25, linestyle='-', linewidth=0.4)
            ax_cal.set_axisbelow(True)
            fig_cal.tight_layout()
            fig_cal.savefig(
                subfigure_dir + f'{PROBE_TYPE}/cal_{dtype}_{train_dataset}.pdf', bbox_inches='tight', dpi = 200
            )
            plt.show()

        
        # ── Figure 2: Precision-Recall curves (one per setting) ────────────────
        for r in setting_results:
            fig_pr, ax_pr = plt.subplots(figsize=(4.0, 3.8))
            
            probe_prec, probe_rec = [], []
            ntp_prec,   ntp_rec   = [], []
            
            for t in THRESHOLD_SWEEP:
                # Compute precision and recall for probe
                preds_prb = r['probe_probs'] > t
                y_true = r['labels'].astype(bool)
                tp_prb = int((preds_prb & y_true).sum())
                fp_prb = int((preds_prb & ~y_true).sum())
                fn_prb = int((~preds_prb & y_true).sum())
                
                prec_prb = tp_prb / (tp_prb + fp_prb) if (tp_prb + fp_prb) > 0 else float('nan')
                rec_prb = tp_prb / (tp_prb + fn_prb) if (tp_prb + fn_prb) > 0 else float('nan')
                probe_prec.append(prec_prb)
                probe_rec.append(rec_prb)
                
                # Compute precision and recall for NTP
                preds_ntp = r['ntp_probs'] > t
                tp_ntp = int((preds_ntp & y_true).sum())
                fp_ntp = int((preds_ntp & ~y_true).sum())
                fn_ntp = int((~preds_ntp & y_true).sum())
                
                prec_ntp = tp_ntp / (tp_ntp + fp_ntp) if (tp_ntp + fp_ntp) > 0 else float('nan')
                rec_ntp = tp_ntp / (tp_ntp + fn_ntp) if (tp_ntp + fn_ntp) > 0 else float('nan')
                ntp_prec.append(prec_ntp)
                ntp_rec.append(rec_ntp)

            pr_prec = np.array(probe_prec);  pr_rec = np.array(probe_rec)
            nt_prec = np.array(ntp_prec);    nt_rec = np.array(ntp_rec)
            v_pr = ~(np.isnan(pr_prec) | np.isnan(pr_rec))
            v_nt = ~(np.isnan(nt_prec) | np.isnan(nt_rec))

            cmap = cm.coolwarm
            norm = mcolors.Normalize(vmin=THRESHOLD_SWEEP.min(), vmax=THRESHOLD_SWEEP.max())
            sm   = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])

            # NTP — faint dotted line
            if v_nt.any():
                ax_pr.plot(nt_rec[v_nt], nt_prec[v_nt], '--', color='#888888',
                        lw=1.5, alpha=0.75, zorder=2, label='NTP')

            # Probe — solid line with colored scatter points by threshold
            if v_pr.any():
                ax_pr.plot(pr_rec[v_pr], pr_prec[v_pr], '-', color='grey',
                        lw=2.0, zorder=3, label='Probe')
                ax_pr.scatter(pr_rec[v_pr], pr_prec[v_pr], c=THRESHOLD_SWEEP[v_pr],
                            cmap=cmap, norm=norm, s=35, zorder=3)

            # Mark threshold = 0.5
            idx0 = np.argmin(np.abs(THRESHOLD_SWEEP - 0.5))
            if v_pr[idx0]:
                ax_pr.scatter([pr_rec[idx0]], [pr_prec[idx0]], s=60, c='none',
                            edgecolors='k', linewidths=1.1, zorder=4, marker='o')

            ax_pr.set_xlim(-0.02, 1.02)
            ax_pr.set_ylim(-0.02, 1.02)
            ax_pr.set_xlabel('Recall')
            ax_pr.set_ylabel('Precision')
            ax_pr.legend(fontsize=7, loc='lower left')
            ax_pr.grid(alpha=0.25, linestyle='-', linewidth=0.4)
            ax_pr.set_axisbelow(True)

            fig_pr.colorbar(sm, ax=ax_pr, label='Threshold',
                            fraction=0.046, pad=0.04)
            fig_pr.tight_layout()
            fig_pr.savefig(
                subfigure_dir + f'{PROBE_TYPE}/pr_{'syn' if r['is_syn'] else 'real'}.pdf',
                bbox_inches='tight', dpi = 200
            )
            plt.show()
            
        # ── Summary table ─────────────────────────────────────────────────────
        rows = []
        for r in setting_results:
            for probs, kind in [(r['ntp_probs'], 'NTP'), (r['probe_probs'], 'Probe')]:
                m = _probe_metrics(probs, r['labels'])
                rows.append({
                    'Test setting':  r['label'],
                    'Type':          kind,
                    'Accuracy':      m['acc'],
                    'Precision':     m['prec'],
                    'Recall':        m['rec'],
                    'F1':            m['f1'],
                    'AUROC':         m['auroc'],
                    'ECE':           m['ece'],
                })

        df = pd.DataFrame(rows)
        print(f'\nSummary — {header}  (threshold = 0.5):')
        print(df.to_string(index=False, float_format='{:.3f}'.format))
        print()
        df.to_csv(subfigure_dir + f'{PROBE_TYPE}/metrics_{train_dataset}.csv')
        return df

    # ───────────────────────────────────────────────────────
    all_results = {}
    for jm in JUDGE_MODELS:
        for train_ds in DATASETS:
            all_results[(jm, train_ds)] = analyze_probe(jm, train_ds)