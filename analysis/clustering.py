import sys
import os
from pathlib import Path

# Limit processor usage
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'
os.environ['NUMEXPR_NUM_THREADS'] = '12'

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from analysis.loaders import (
    load_combined_judgements, load_extraction, load_ground_truth,
    load_trained_probe, load_activations, load_synthetic_layer_outputs,
    load_synthetic_responses, load_synthetic_activations, load_layer_outputs
)
from scholarlm.utils.unit_conversion import apply_unit_conversion
from experiments.run_extraction import load_dataset_config
import paths

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "text.usetex": False,
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.title_fontsize": 11,
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

FIGURES_DIR = Path("figures/clustering/")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ── Standalone legend ─────────────────────────────────────────────
_legend_handles = [
    mlines.Line2D([], [], color=palette[2], lw=4, linestyle = "-", ms=3.5, label='NTP'),
    mlines.Line2D([], [], color=palette[7], lw=4, linestyle = "-", ms=3.5, label='Probe'),
    mlines.Line2D([], [], color=palette[9], lw=4, linestyle = "-", ms=3.5, label='Random'),
    #mlines.Line2D([], [], color='#444444', lw=2, linestyle='-',  label='Probe'),
    #mlines.Line2D([], [], color='#444444', lw=2, linestyle='--', label='NTP'),
    #mlines.Line2D([], [], color='#444444', lw=2, linestyle=':', label='Random'),
]
_fig_leg, _ax_leg = plt.subplots(figsize=(10.0, 0.45))
_ax_leg.axis('off')
_ax_leg.legend(handles=_legend_handles, loc='center', ncol=6, fontsize=11,
               frameon=False, handlelength=2.0)
_fig_leg.savefig(FIGURES_DIR / 'legend.pdf', bbox_inches='tight', dpi=200)
plt.show()


# ── Parameters ───────────────────────────────────────────────────────────────
DATASET          = 'pond'
EXTRACTION_MODEL = 'gemma-3-27b'
EXTRACTION_DATE  = '2026_05_05'
JUDGE_MODELS      = ['llama-3.1-8b', 'mistral-7b', 'qwen-2.5-7b']
JUDGE_DATE       = '2026_05_06'
SYN_JUDGE_DATE   = '2026_05_04'  # date for synthetic probe activations

N_CLUSTERS = 5  # chosen from elbow curve below
# ─────────────────────────────────────────────────────────────────

for PROBE_TYPE in ['layer', 'head']:
    for JUDGE_MODEL in JUDGE_MODELS:
        # ─────────────────────────────────────────────────────────────────
        config  = load_dataset_config(DATASET)
        gt_df   = load_ground_truth(config)
        records = load_extraction(DATASET, EXTRACTION_MODEL, EXTRACTION_DATE)
        ext_df  = pd.DataFrame(records)
        syn_responses = load_synthetic_responses(DATASET, JUDGE_MODEL, SYN_JUDGE_DATE, split='test')
        syn_df  = pd.DataFrame(syn_responses)

        # Unit conversion
        gt_df  = apply_unit_conversion(gt_df,  config.unit_conversion_table)
        ext_df = apply_unit_conversion(ext_df, config.unit_conversion_table)
        syn_df = apply_unit_conversion(syn_df, config.unit_conversion_table)

        # Ground truth and synthetic need a synthetic entity_id (ext_df already carries one)
        _GT_ENTITY_COLS = ['document_id', 'name', 'location', 'ecosystem']
        gt_df['entity_id'] = gt_df.groupby(_GT_ENTITY_COLS, dropna=False).ngroup()

        _SYN_ENTITY_COLS = ['document_id', 'name', 'location', 'ecosystem']
        syn_df['entity_id'] = syn_df.groupby(_SYN_ENTITY_COLS, dropna=False).ngroup()
        # ─────────────────────────────────────────────────────────────────

        pd_data     = load_trained_probe(DATASET, JUDGE_MODEL, PROBE_TYPE)
        syn_doc_ids = set(pd_data['syn_document_ids'])
        if PROBE_TYPE == 'layer':
            top = pd_data['top_layer']
        else:
            top  = pd_data['top_k_heads']

        n_gt_before  = len(gt_df)
        n_ext_before = len(ext_df)
        n_syn_before = len(syn_df)

        gt_df  = gt_df[ ~gt_df['document_id'].isin(syn_doc_ids)].reset_index(drop=True)
        ext_df = ext_df[~ext_df['document_id'].isin(syn_doc_ids)].reset_index(drop=True)
        syn_df = syn_df[~syn_df['document_id'].isin(syn_doc_ids)].reset_index(drop=True)

        # ─────────────────────────────────────────────────────────────────
        # 1. Setup
        THRESHOLD = 0.20
        # Pivot data (ensure no dropna yet)
        gt_wide = gt_df.pivot_table(
            index='entity_id', columns='attribute', values='converted_value', aggfunc='first'
        )

        results = []

        # 2. Iterate through column combinations (d=2 to 5)
        for d in range(2, 6):
            for cols in combinations(gt_wide.columns, d):
                # Start with all rows for these specific columns
                submatrix = gt_wide[list(cols)].copy()
                
                # Iteratively drop the row with the most NaNs until total missingness < 10%
                while (submatrix.isna().sum().sum() / submatrix.size) > THRESHOLD:
                    if submatrix.shape[0] <= 1: break # Safety break
                    
                    # Find row index with highest missing count
                    worst_row = submatrix.isna().sum(axis=1).idxmax()
                    submatrix = submatrix.drop(index=worst_row)
                
                # Record the largest valid version of this column set
                n_rows = submatrix.shape[0]
                actual_missing = submatrix.isna().sum().sum() / submatrix.size
                
                results.append({
                    'attributes': list(cols),
                    'd': d,
                    'n_rows': n_rows,
                    'missing_pct': actual_missing,
                    'score': n_rows * d  # Area of the submatrix
                })

        # 3. Process and Print
        res_df = pd.DataFrame(results)
        best = res_df.sort_values('score', ascending=False).iloc[0]
        # ─────────────────────────────────────────────────────────────────

        # ── Load judge data; compute probe probabilities for extraction ───────────────
        judge_df = pd.DataFrame(load_combined_judgements(DATASET, EXTRACTION_MODEL, EXTRACTION_DATE))
        judge_df = judge_df[~judge_df['document_id'].isin(syn_doc_ids)].reset_index(drop=True)
        mids_ext = judge_df['measurement_id'].tolist()


        if PROBE_TYPE == 'layer':
            lo  = load_layer_outputs(DATASET, EXTRACTION_MODEL, EXTRACTION_DATE, JUDGE_MODEL, JUDGE_DATE)    
            X_feat_ext = np.stack([
                np.array(lo[str(mid)], dtype=np.float32)[top]
                for mid in mids_ext
            ], axis=0)
        else:
            act = load_activations(DATASET, EXTRACTION_MODEL, EXTRACTION_DATE, JUDGE_MODEL, JUDGE_DATE)
            X_feat_ext = np.concatenate([
                np.stack([np.array(act[str(mid)], dtype=np.float32)[l, h, :] for mid in mids_ext], axis=0)
                for l, h in top
            ], axis=1)
        
        judge_df['p_probe'] = pd_data['probe'].predict_proba(X_feat_ext)[:, 1]

        # Merge judge probs into ext_df so pivoting stays aligned
        ext_df = ext_df.merge(
            judge_df[['measurement_id', f'judgement_p_true_{JUDGE_MODEL}', 'p_probe']],
            on='measurement_id', how='left',
        )

        # ── Compute probe probabilities for synthetic data ─────────────────────────────
        mids_syn = syn_df['measurement_id'].tolist()
        if PROBE_TYPE == "layer":
            syn_lo  = load_synthetic_layer_outputs(DATASET, JUDGE_MODEL, SYN_JUDGE_DATE, split='test')
            X_feat_syn = np.stack([
                np.array(syn_lo[str(mid)], dtype=np.float32)[top]
                for mid in mids_syn
            ], axis=0)
        else:
            syn_act  = load_synthetic_activations(DATASET, JUDGE_MODEL, SYN_JUDGE_DATE, split='test')
            X_feat_syn = np.concatenate([
                np.stack([np.array(syn_act[str(mid)], dtype=np.float32)[l, h, :] for mid in mids_syn], axis=0)
                for l, h in top
            ], axis=1)

        syn_df['p_probe'] = pd_data['probe'].predict_proba(X_feat_syn)[:, 1]

        # ── Pivot all datasets onto keep_attrs (set by enumeration cell above) ─────────

        def _pivot(df, value_col):
            piv = df.pivot_table(index='entity_id', columns='attribute',
                                values=value_col, aggfunc='first')
            piv.columns.name = None
            return piv.reindex(columns=keep_attrs)

        def get_dense_submatrix(df_piv, threshold):
            """Drops the row with the most NaNs until the total missingness is below threshold."""
            dense_df = df_piv.copy()
            while (dense_df.isna().sum().sum() / dense_df.size) > threshold:
                if dense_df.shape[0] <= 1: break
                worst_row = dense_df.isna().sum(axis=1).idxmax()
                dense_df = dense_df.drop(index=worst_row)
            return dense_df

        # 2. Pivot all datasets onto keep_attrs (from your Cell 8/11 winner)
        keep_attrs = best['attributes']

        gt_val_piv    = _pivot(gt_df,  'converted_value')
        ext_val_piv   = _pivot(ext_df, 'converted_value')
        syn_val_piv   = _pivot(syn_df, 'converted_value')

        # Pivot the probability matrices
        ext_ntp_piv   = _pivot(ext_df, f'judgement_p_true_{JUDGE_MODEL}')
        ext_probe_piv = _pivot(ext_df, 'p_probe')
        syn_ntp_piv   = _pivot(syn_df, 'judgement_p_true')
        syn_probe_piv = _pivot(syn_df, 'p_probe')

        # 3. Apply Independent Greedy Filtering
        # We use the same THRESHOLD (e.g., 0.10) for all
        gt_val_piv  = get_dense_submatrix(gt_val_piv, THRESHOLD)
        ext_val_piv = get_dense_submatrix(ext_val_piv, THRESHOLD)
        syn_val_piv = get_dense_submatrix(syn_val_piv, THRESHOLD)

        # 4. Reindex probabilities to match the rows that survived the greedy drop
        ext_ntp_piv   = ext_ntp_piv.reindex(ext_val_piv.index)
        ext_probe_piv = ext_probe_piv.reindex(ext_val_piv.index)
        syn_ntp_piv   = syn_ntp_piv.reindex(syn_val_piv.index)
        syn_probe_piv = syn_probe_piv.reindex(syn_val_piv.index)
        # ─────────────────────────────────────────────────────────────────

        def process_matrix(df_piv):
            # Independent imputer and scaler for each dataset
            imp = KNNImputer(n_neighbors=5, weights='distance')
            scl = StandardScaler()
            
            # Impute missing values (<10%) based on internal neighbors
            imputed = imp.fit_transform(df_piv)
            # Scale based on the internal distribution of this specific dataset
            return scl.fit_transform(imputed)

        # Apply independently
        X_gt  = process_matrix(gt_val_piv)
        X_ext = process_matrix(ext_val_piv)
        X_syn = process_matrix(syn_val_piv)

        # ── Entity-level probabilities ──
        def _entity_probs(prob_piv):
            # Fill missing enries with prob=1.0, so imputation does not affect later results
            return prob_piv.prod(axis=1, min_count=1).fillna(1.0).to_numpy()

        ext_ntp_probs = _entity_probs(ext_ntp_piv)
        ext_probe_probs = _entity_probs(ext_probe_piv)
        syn_ntp_probs = _entity_probs(syn_ntp_piv)
        syn_probe_probs = _entity_probs(syn_probe_piv)
        # ─────────────────────────────────────────────────────────────────

        def centroid_matching_distance(A, B, metric='euclidean'):
            """Mean optimal-assignment distance between two sets of centroids."""
            D = cdist(A, B, metric=metric)
            row_ind, col_ind = linear_sum_assignment(D, maximize=False)
            return D[row_ind, col_ind].mean()

        # ─────────────────────────────────────────────────────────────────
        kmeans_gt  = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init='auto')
        gt_labels  = kmeans_gt.fit_predict(X_gt)
        gt_centers = kmeans_gt.cluster_centers_

        N_RUNS = 1000  # Number of different initializations
        gamma_vals = np.linspace(0.0, 5.0, 101)

        _sweep_configs = {
            'ext_ntp':   (X_ext, ext_ntp_probs),
            'ext_probe': (X_ext, ext_probe_probs),
            'syn_ntp' : (X_syn , syn_ntp_probs),
            'syn_probe' : (X_syn , syn_probe_probs),
        }

        # Initialize a 2D array to store distances for each run: [run, gamma_index]
        all_runs_dist = {k: np.zeros((N_RUNS, len(gamma_vals))) for k in _sweep_configs}

        for run_idx in range(N_RUNS):
            current_seed = 42 + run_idx # Ensure a different seed per run

            for i, gamma in enumerate(gamma_vals):
                for key, (X, probs) in _sweep_configs.items():
                    km = KMeans(n_clusters=N_CLUSTERS, random_state=current_seed, n_init='auto')
                    km.fit(X, sample_weight=probs ** gamma)

                    dist = centroid_matching_distance(gt_centers, km.cluster_centers_)
                    all_runs_dist[key][run_idx, i] = dist

        # Random baselines: N_RANDOM_SAMPLES draws of uniform random probs; each draw uses
        # its own KMeans seed, collapsing both sources of variance into one loop.
        N_RANDOM_SAMPLES = 1000
        random_runs_dist = {
            'ext_random': np.zeros((N_RANDOM_SAMPLES, len(gamma_vals))),
            'syn_random': np.zeros((N_RANDOM_SAMPLES, len(gamma_vals))),
        }
        for sample_idx in range(N_RANDOM_SAMPLES):
            rng = np.random.default_rng(sample_idx)
            ext_random_probs = rng.uniform(0, 1, size=len(X_ext))
            syn_random_probs = rng.uniform(0, 1, size=len(X_syn))
            for i, gamma in enumerate(gamma_vals):
                for key, X, rprobs in [('ext_random', X_ext, ext_random_probs),
                                       ('syn_random', X_syn, syn_random_probs)]:
                    km = KMeans(n_clusters=N_CLUSTERS, random_state=sample_idx, n_init='auto')
                    km.fit(X, sample_weight=rprobs ** gamma)
                    dist = centroid_matching_distance(gt_centers, km.cluster_centers_)
                    random_runs_dist[key][sample_idx, i] = dist

        # Calculate mean and standard deviation
        avg_distances = {k: np.mean(v, axis=0) for k, v in all_runs_dist.items()}
        std_distances = {k: np.std(v, axis=0) / np.sqrt(N_RUNS) for k, v in all_runs_dist.items()}
        for key, v in random_runs_dist.items():
            avg_distances[key] = v.mean(axis=0)
            std_distances[key] = v.std(axis=0) / np.sqrt(N_RANDOM_SAMPLES)

        _STYLE = {
            'ext_ntp':     dict(color=palette[2], ls='-', lw=3.0, alpha=0.85,
                               label='Ext. NTP'),
            'ext_probe':   dict(color=palette[7], ls='-',  lw=3.0,
                               label='Ext. Probe'),
            'ext_random':  dict(color=palette[9], ls='-',  lw=3.0, alpha=0.85,
                               label='Ext. Random'),
            #'syn_ntp':     dict(color=palette[7], ls='--', lw=2.0, alpha=0.85,
            #                   label='Syn. NTP'),
            #'syn_probe':   dict(color=palette[7], ls='-',  lw=2.0,
            #                   label='Syn. Probe'),
            #'syn_random':  dict(color=palette[7], ls=':',  lw=2.0, alpha=0.85,
            #                   label='Syn. Random'),
        }

        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        for key, style in _STYLE.items():
            # Plot the mean line
            ax.plot(gamma_vals, avg_distances[key], **style)
            # Optional: Plot the shaded error region (standard error)

            ax.fill_between(
                gamma_vals, 
                avg_distances[key] - std_distances[key], 
                avg_distances[key] + std_distances[key], 
                color=style['color'], 
                alpha=0.2
            )

        ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
        ax.set_xlabel('$\\gamma$', fontsize = 13)
        ax.set_ylabel('Mean Centroid Distance')
        ax.set_xlim(gamma_vals[0], gamma_vals[-1])
        #ax.legend(fontsize=6.5, loc='upper right', bbox_to_anchor=(1.0, 0.95))
        ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
        fig.tight_layout()

        fig.savefig(
            FIGURES_DIR / f'center_dist_ext_{EXTRACTION_MODEL}_judge_{JUDGE_MODEL}_{PROBE_TYPE}.pdf',
            bbox_inches='tight', dpi = 200
        )
