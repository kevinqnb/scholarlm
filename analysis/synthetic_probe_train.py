import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

from analysis.loaders import (
    load_synthetic_activations, load_synthetic_layer_outputs, load_synthetic_responses,
)
from scholarlm.utils.probe import grouped_kfold_split, grouped_holdout_split
from scholarlm.utils.calibration import compute_ece
import paths

# blue: 7, orange: 1, red: 0, green: 4
palette = sns.color_palette("husl", 10)

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

FIGURES_DIR = "figures/synthetic_probe/"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────
DATASETS        = ['pond', 'nfix']
JUDGE_MODELS    = ['llama-3.1-8b', 'mistral-7b', 'qwen-2.5-7b']   # must match the judge used for the synthetic run
JUDGE_DATE_SYN = '2026_05_04'           # auto-detect latest synthetic probe run

TOP_K   = 10    # number of attention heads for the final probe
N_FOLDS = 5
# ─────────────────────────────────────────────────────────────────

# Dictionary to collect layer F1 scores for all judge models
collected_layer_f1_scores = {}


# Score candidate probes by F1 and ECE
def cv_score(probe, X, y, kfold_cv):
    fold_f1s, fold_eces = [], []
    for train_idx, test_idx in kfold_cv:
        probe.fit(X[train_idx], y[train_idx])
        y_pred = probe.predict(X[test_idx])
        y_true = y[test_idx]
        probs  = probe.predict_proba(X[test_idx])[:, 1]
        fold_f1s.append(f1_score(y_true, y_pred))
        fold_eces.append(compute_ece(probs, y_true, n_bins=10))
    return (
        float(np.mean(fold_f1s)),list(fold_f1s),
        float(np.mean(fold_eces)),list(fold_eces),
    )

# ─────────────────────────────────────────────────────────────────
for DATASET in DATASETS:
    for JUDGE_MODEL in JUDGE_MODELS:
        print(f'\n{"="*60}\nDataset: {DATASET}   Judge: {JUDGE_MODEL}\n{"="*60}')

        # ─────────────────────────────────────────────────────────────────
        syn_activations = load_synthetic_activations(DATASET, JUDGE_MODEL, JUDGE_DATE_SYN, split='train')
        syn_layer_outputs = load_synthetic_layer_outputs(DATASET, JUDGE_MODEL, JUDGE_DATE_SYN, split='train')
        syn_responses   = load_synthetic_responses(DATASET, JUDGE_MODEL, JUDGE_DATE_SYN, split='train')
        syn_df          = pd.DataFrame(syn_responses)

        syn_measurement_ids = syn_df['measurement_id'].tolist()
        syn_labels          = (syn_df['label'] == 'valid').to_numpy(dtype=bool)
        syn_groups          = syn_df['document_id'].to_numpy()
        # ─────────────────────────────────────────────────────────────────

        # Use all synthetic data for training (no calibration holdout).
        # Group-aware split ensures no paper appears in both train and CV test folds.
        syn_train_idx, syn_cal_idx, syn_test_idx = grouped_holdout_split(
            syn_groups, train_frac=1.0, cal_frac=0.0, random_state=42
        )

        # Papers with really really large tables that seem to throw off the activations...
        keep_mask = (syn_df.iloc[syn_train_idx]['document_id'] != 'habitat_characteristics') & (syn_df.iloc[syn_train_idx]['document_id'] != 'R164') 
        syn_train_idx = syn_train_idx[keep_mask.values]

        syn_cv_idx = syn_train_idx
        syn_labels_cv = syn_labels[syn_cv_idx]
        syn_groups_cv = syn_groups[syn_cv_idx]
        kfold_cv = list(grouped_kfold_split(syn_groups_cv, n_splits=N_FOLDS, random_state=42))

        # ─────────────────────────────────────────────────────────────────
        _arr0 = np.array(syn_activations[str(syn_measurement_ids[0])], dtype=np.float32)
        n_layers, n_heads, head_dim = _arr0.shape
        _all_syn = {
            str(mid): np.array(syn_activations[str(mid)], dtype=np.float32)
            for mid in syn_measurement_ids
        }
        head_datasets_syn: dict[tuple[int, int], np.ndarray] = {}
        for l in range(n_layers):
            for h in range(n_heads):
                head_datasets_syn[(l, h)] = np.stack(
                    [_all_syn[str(mid)][l, h, :] for mid in syn_measurement_ids], axis=0
                )
        del _all_syn

        # ─────────────────────────────────────────────────────────────────
        probe_template = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0, class_weight='balanced', solver='lbfgs',
                max_iter=1000, random_state=42,
            ))
        ])

        head_scores_f1    = np.zeros((n_layers, n_heads))
        head_scores_ece    = np.zeros((n_layers, n_heads))

        for l in range(n_layers):
            for h in range(n_heads):
                X_cv = head_datasets_syn[(l, h)][syn_cv_idx]
                (
                mean_f1, _,
                mean_ece, _,
                ) = cv_score(probe_template, X_cv, syn_labels_cv, kfold_cv)
                head_scores_f1[l, h] = mean_f1
                head_scores_ece[l, h] = mean_ece
        # ─────────────────────────────────────────────────────────────────

        # F1 heatmap:
        score_mat = head_scores_f1
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        im = ax.imshow(np.sort(score_mat, axis=1), cmap='magma', aspect='auto', origin='lower')
        ax.set_xlabel('Head (sorted)')
        ax.set_ylabel('Layer')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR + f'{JUDGE_MODEL}/synprobe_heatmap_F1_{DATASET}.pdf', bbox_inches='tight', dpi=100)


        # ECE heatmap:
        score_mat = 1 - head_scores_ece
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        im = ax.imshow(np.sort(score_mat, axis=1), cmap='magma', aspect='auto', origin='lower')
        ax.set_xlabel('Head (sorted)')
        ax.set_ylabel('Layer')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR + f'{JUDGE_MODEL}/synprobe_heatmap_ECE_{DATASET}.pdf', bbox_inches='tight', dpi=100)


        # ─────────────────────────────────────────────────────────────────
        # Sort to top-k heads (by F1):
        flat_indices = np.argsort(head_scores_f1.flatten())[-TOP_K:][::-1]
        top_k_heads  = [np.unravel_index(idx, head_scores_f1.shape) for idx in flat_indices]

        # Train:
        X_train = np.concatenate(
            [head_datasets_syn[(l, h)][syn_train_idx] for l, h in top_k_heads], axis=1
        )
        y_train = syn_labels[syn_train_idx]

        base_probe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0, class_weight=None, solver='lbfgs',
                max_iter=1000, random_state=42,
            ))
        ])

        calibrated_probe = CalibratedClassifierCV(
            estimator=base_probe,
            method='sigmoid',  # 'sigmoid' is Platt scaling
            cv=kfold_cv
        ).fit(X_train, y_train)

        # Evaluate training performance:
        y_probs = calibrated_probe.predict_proba(X_train)[:, 1]
        y_pred = (y_probs > 0.5).astype(int)
        print(f"  Accuracy  : {accuracy_score(y_train, y_pred):.4f}")
        print(f"  Precision : {precision_score(y_train, y_pred):.4f}")
        print(f"  Recall    : {recall_score(y_train, y_pred):.4f}")
        print(f"  F1-Score  : {f1_score(y_train, y_pred):.4f}")
        print(f"  AUROC     : {roc_auc_score(y_train, y_probs):.4f}")
        print(f"  ECE       : {compute_ece(y_train, y_probs):.4f}")

        # Save probe + metadata for use in synthetic_probe_test.ipynb
        probe_dir  = paths.trained_probe_dir(DATASET, JUDGE_MODEL)
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_path = probe_dir / 'head_probe.pkl'

        probe_data = {
            'probe':            calibrated_probe,
            'top_k_heads':      top_k_heads,
            'train_prevalence': float(y_train.mean()),
            'syn_document_ids': sorted(syn_df['document_id'].unique().tolist()),
            'judge_model':      JUDGE_MODEL,
            'dataset':          DATASET,
            'n_layers':         n_layers,
            'n_heads':          n_heads,
            'head_dim':         head_dim,
        }
        joblib.dump(probe_data, probe_path)

        print(f'Probe saved  → {probe_path}')
        print(f'  Top-{TOP_K} heads        : {top_k_heads}')
        # ─────────────────────────────────────────────────────────────────

        # Fit NTP calibrator — Platt scaling via 1-D logistic regression on
        # judgement_p_true, using the same CV splits as the probe so that
        # calibration is not overfit on the training labels.
        ntp_probs_train = syn_df['judgement_p_true'].to_numpy()[syn_train_idx].reshape(-1, 1)

        ntp_base = Pipeline([
            ('clf', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000, random_state=42))
        ])
        ntp_calibrated = CalibratedClassifierCV(
            estimator=ntp_base,
            method='sigmoid',
            cv=kfold_cv,
        ).fit(ntp_probs_train, y_train)

        ntp_cal_probs_tr = ntp_calibrated.predict_proba(ntp_probs_train)[:, 1]
        print(f"  NTP calibrator train ECE: {compute_ece(ntp_cal_probs_tr, y_train):.4f}")

        ntp_cal_path = probe_dir / 'ntp_calibrator.pkl'
        joblib.dump({
            'calibrator':       ntp_calibrated,
            'train_prevalence': float(y_train.mean()),
            'syn_document_ids': sorted(syn_df['document_id'].unique().tolist()),
            'judge_model':      JUDGE_MODEL,
            'dataset':          DATASET,
        }, ntp_cal_path)
        print(f'NTP calibrator saved → {ntp_cal_path}')
        # ─────────────────────────────────────────────────────────────────

        _arr0_lo = np.array(syn_layer_outputs[str(syn_measurement_ids[0])], dtype=np.float32)
        n_layers_lo, hidden_size = _arr0_lo.shape
        _all_syn_lo = {str(mid): np.array(syn_layer_outputs[str(mid)], dtype=np.float32)
                    for mid in syn_measurement_ids}
        layer_datasets_syn: dict[int, np.ndarray] = {
            l: np.stack([_all_syn_lo[str(mid)][l] for mid in syn_measurement_ids], axis=0)
            for l in range(n_layers_lo)
        }
        del _all_syn_lo

        # ─────────────────────────────────────────────────────────────────
        probe_template_lo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=0.2, class_weight='balanced', solver='lbfgs',
                max_iter=1000, random_state=42,
            ))
        ])

        layer_scores_f1    = np.zeros(n_layers_lo)
        layer_scores_ece = np.zeros(n_layers_lo)

        for l in range(n_layers_lo):
            X_cv = layer_datasets_syn[l][syn_cv_idx]
            (
                mean_f1, _,
                mean_ece, _,
                ) = cv_score(probe_template_lo, X_cv, syn_labels_cv, kfold_cv)
            layer_scores_f1[l]    = mean_f1
            layer_scores_ece[l] = mean_ece
        
        # Collect F1 scores for this judge model and dataset
        if JUDGE_MODEL not in collected_layer_f1_scores:
            collected_layer_f1_scores[JUDGE_MODEL] = {}
        collected_layer_f1_scores[JUDGE_MODEL][DATASET] = layer_scores_f1.copy()
        # ─────────────────────────────────────────────────────────────────

        # F1 by layer line plot — layer output probe
        best_layer_lo = int(layer_scores_f1.argmax())
        best_layer_f1_scores = layer_scores_f1[best_layer_lo]
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(range(n_layers_lo), layer_scores_f1, 'o-', color=palette[0], ms=4, lw = 2.0)
        ax.axvline(best_layer_lo, color="grey", lw=1.0, ls='--', label=f'Best: L{best_layer_lo} (F1={best_layer_f1_scores:.3f})')
        ax.set_xlabel('Layer')
        ax.set_ylabel('F1')
        ax.legend(fontsize=7)
        ax.set_xlim(-0.5, n_layers_lo - 0.5)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR + f'{JUDGE_MODEL}/synprobe_layer_F1_{DATASET}.pdf', bbox_inches='tight')


        # ECE by layer line plot — layer output probe
        best_layer_lo = int(layer_scores_ece.argmin())
        best_layer_ece_scores = layer_scores_ece[best_layer_lo]
        fig, ax = plt.subplots(figsize=(3.5, 2.5))
        ax.plot(range(n_layers_lo), layer_scores_ece, 'o-', color=palette[0], ms=4, lw = 2.0)
        ax.axvline(best_layer_lo, color='grey', lw=1.0, ls='--', label=f'Best: L{best_layer_lo} (ECE={best_layer_ece_scores:.3f})')
        ax.set_xlabel('Layer')
        ax.set_ylabel('ECE')
        ax.legend(fontsize=7)
        ax.set_xlim(-0.5, n_layers_lo - 0.5)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR + f'{JUDGE_MODEL}/synprobe_layer_ECE_{DATASET}.pdf', bbox_inches='tight')

        # ─────────────────────────────────────────────────────────────────
        best_layer_lo = int(layer_scores_f1.argmax())
        X_train_lo = layer_datasets_syn[best_layer_lo][syn_train_idx]
        y_train_lo = syn_labels[syn_train_idx]

        base_probe_lo = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.2, class_weight=None, solver='lbfgs',
                                    max_iter=1000, random_state=42))
        ])


        calibrated_probe_lo = CalibratedClassifierCV(
            estimator=base_probe_lo,
            method='sigmoid',  # 'sigmoid' is Platt scaling
            cv=kfold_cv
        ).fit(X_train_lo, y_train_lo)

        # Evaluate training performance:
        y_probs_lo = calibrated_probe_lo.predict_proba(X_train_lo)[:, 1]
        y_pred_lo = (y_probs_lo > 0.5).astype(int)
        print(f"  Accuracy  : {accuracy_score(y_train_lo, y_pred_lo):.4f}")
        print(f"  Precision : {precision_score(y_train_lo, y_pred_lo):.4f}")
        print(f"  Recall    : {recall_score(y_train_lo, y_pred_lo):.4f}")
        print(f"  F1-Score  : {f1_score(y_train_lo, y_pred_lo):.4f}")
        print(f"  AUROC     : {roc_auc_score(y_train_lo, y_probs_lo):.4f}")
        print(f"  ECE       : {compute_ece(y_train_lo, y_probs_lo):.4f}")

        # Save probe + metadata for use in synthetic_probe_test.ipynb
        probe_dir  = paths.trained_probe_dir(DATASET, JUDGE_MODEL)
        probe_dir.mkdir(parents=True, exist_ok=True)
        probe_path = probe_dir / 'layer_probe.pkl'

        probe_data = {
            'probe':            calibrated_probe_lo,
            'top_layer':      best_layer_lo,
            'train_prevalence': float(y_train_lo.mean()),
            'syn_document_ids': sorted(syn_df['document_id'].unique().tolist()),
            'judge_model':      JUDGE_MODEL,
            'dataset':          DATASET,
            'n_layers':         n_layers_lo,
        }
        joblib.dump(probe_data, probe_path)

        print(f'Probe saved  → {probe_path}')
        print(f'Top layer: {best_layer_lo}')
        # ─────────────────────────────────────────────────────────────────


# ─────────────────────────────────────────────────────────────────
# Create combined plot of F1 scores by layer for all judge models
# ─────────────────────────────────────────────────────────────────
palette_idx=[7, 1, 0, 4]

for DATASET in DATASETS:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    
    # Plot each judge model's F1 scores
    for model_idx, JUDGE_MODEL in enumerate(JUDGE_MODELS):
        layer_f1_scores = collected_layer_f1_scores[JUDGE_MODEL][DATASET]
        n_layers_plot = len(layer_f1_scores)
        
        # Plot line
        ax.plot(range(n_layers_plot), layer_f1_scores, 'o-', 
                color=palette[palette_idx[model_idx]], label=JUDGE_MODEL, ms=4, lw=2.0)

        # Mark best layer with a larger dark grey marker
        best_layer_idx = int(np.argmax(layer_f1_scores))
        best_f1 = layer_f1_scores[best_layer_idx]
        ax.plot(best_layer_idx, best_f1, 'o', color='darkgrey', ms=6, 
                markeredgecolor='darkgrey', markeredgewidth=1.5)
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('F1')
    ax.legend(fontsize=8, loc='best')
    ax.set_xlim(-0.5, n_layers_plot - 0.5)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR + f'synprobe_layer_F1_all_models_{DATASET}.pdf', bbox_inches='tight')
    print(f'Combined F1 plot saved → {FIGURES_DIR}synprobe_layer_F1_all_models_{DATASET}.pdf')