import sys
from pathlib import Path
import argparse
import json
import os
import re

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
import utils as paths

# blue: 7, orange: 1, red: 0, green: 4
palette = sns.color_palette("husl", 10)

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

FIGURES_DIR = "figures/synthetic_probe/"


# ─────────────────────────────────────────────────────────────────
# Run config: CLI flags → env vars, NO defaults for the three that pick which
# activations are read (a wrong date/judge silently trains on the wrong data).
# Mirrors analysis/calibration_updated.py's _select_settings pattern.


def _env_list(name):
    """Comma/space-separated env var → list; None when unset/empty."""
    raw = os.environ.get(name, '').replace(',', ' ').split()
    return raw or None


def _select_run_config():
    """Return (datasets, judges, judge_date, syn_name, probe_source).

    ``--source`` selects the synthetic training corpus:
      * ``baseline`` (default) — the ``synthetic_probe/`` judge tree + the
        ``trained_probe/`` output dir. syn_name and probe_source are both
        ``None`` → byte-for-byte the script's prior behavior.
      * any ``[a-z0-9_]`` name (e.g. ``v2``, ``rung3``) — the augmented
        ``synthetic_probe_<name>/`` judge tree (written by
        ``run_judge_interp.py --synthetic-name <name>``) + a parallel
        ``synthetic_probe_<name>/.../trained_probe/`` output dir, so the baseline
        probe / NTP-calibrator pickles are never overwritten.

    This is the legacy, date-addressed selector -- still the only path for
    any run still sitting on the old ``data/experiments/`` tree. See
    ``_select_judge_run_ids`` for the id-addressed replacement.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--judges', nargs='+', default=None)
    parser.add_argument('--judge-date', default=None)
    parser.add_argument('--source', default=None)
    args, _ = parser.parse_known_args()

    datasets   = args.datasets or _env_list('SYNTHETIC_PROBE_DATASETS')
    judges     = args.judges or _env_list('SYNTHETIC_PROBE_JUDGES')
    judge_date = args.judge_date or os.environ.get('SYNTHETIC_PROBE_JUDGE_DATE') or None
    source     = args.source or os.environ.get('SYNTHETIC_PROBE_SOURCE') or 'baseline'

    missing = [n for n, v in [('--datasets', datasets), ('--judges', judges),
                              ('--judge-date', judge_date)] if not v]
    if missing:
        raise ValueError(
            f"synthetic_probe_train.py: missing required config {missing}. Pass "
            "--datasets / --judges / --judge-date (or SYNTHETIC_PROBE_DATASETS / "
            "SYNTHETIC_PROBE_JUDGES / SYNTHETIC_PROBE_JUDGE_DATE). No defaults."
        )
    if source == 'baseline':
        syn_name = probe_source = None
    elif re.fullmatch(r'[a-z0-9][a-z0-9_]*', source):
        syn_name = probe_source = source
    else:
        raise ValueError(
            f"Unknown --source {source!r}; expected 'baseline' or a [a-z0-9_] name"
        )
    return datasets, judges, judge_date, syn_name, probe_source


def _select_judge_run_ids() -> list[str] | None:
    """``--judge-run-ids`` / ``SYNTHETIC_PROBE_JUDGE_RUN_IDS`` -- the id-addressed
    training mode (2026-09-16). Each id is a ``judge_interp`` experiment id
    whose ``synthetic_file`` run this script trains on directly off
    ``experiments/results/{dataset}/judge_interp/{id}/`` -- no ``--datasets``
    / ``--judges`` / ``--judge-date`` guessing, since the id's own committed
    config already names the (dataset, judge_model) pair and the id itself
    (not a date) picks the exact run.

    Mutually exclusive with the legacy selectors (see ``_select_run_config``)
    -- checked in ``main()``, not here, since detecting "the legacy selectors
    were left at their defaults" vs "the user actually passed them" requires
    inspecting the same argv/env this function already parsed.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--judge-run-ids', nargs='+', default=None)
    args, _ = parser.parse_known_args()
    return args.judge_run_ids or _env_list('SYNTHETIC_PROBE_JUDGE_RUN_IDS')


def _load_synthetic_run(run_id: str):
    """Resolve a ``judge_interp`` synthetic run by experiment id and load its
    responses/activations/layer-outputs straight off the id-addressed tree.

    Unlike the legacy (dataset, judge_model, judge_date) lookup, an id names
    exactly one run -- dataset and judge_model are read from the run's own
    committed config rather than repeated by hand, and there's no "most
    recent date" ambiguity to resolve.

    Returns:
        (dataset, judge_model, syn_responses, syn_activations, syn_layer_outputs, probe_dir)

    Raises:
        FileNotFoundError: If the config or any of its three output files is missing.
        ValueError: If params.judge is missing, or run_metadata.json's
            judge_model disagrees with params.judge (two names for the same
            run that have drifted apart), or the responses/activations
            measurement_id sets don't match.
    """
    config_path = paths.find_experiment_config(run_id)
    cfg = paths.load_experiment_config(config_path)
    dataset = config_path.parts[-4]
    paths.require_params(cfg['params'], 'judge', config_path=config_path)
    judge_model = cfg['params']['judge']

    run_dir = paths.result_dir(dataset, 'judge_interp', run_id)
    responses_path = run_dir / 'responses.json'
    activations_path = run_dir / 'attention_outputs.npz'
    layer_outputs_path = run_dir / 'layer_outputs.npz'
    for p in (responses_path, activations_path, layer_outputs_path):
        if not p.exists():
            raise FileNotFoundError(f"{run_id}: missing {p}")

    metadata = paths.load_run_metadata(run_dir)
    if metadata is not None and metadata.get('judge_model') not in (None, judge_model):
        raise ValueError(
            f"{run_id}: run_metadata.json judge_model={metadata.get('judge_model')!r} "
            f"disagrees with {config_path}'s params.judge={judge_model!r}"
        )

    with open(responses_path) as f:
        syn_responses = json.load(f)
    syn_activations = np.load(activations_path)
    syn_layer_outputs = np.load(layer_outputs_path)

    response_ids = {str(r['measurement_id']) for r in syn_responses}
    activation_ids = set(syn_activations.files)
    if response_ids != activation_ids:
        raise ValueError(
            f"{run_id}: measurement_id set mismatch between responses.json "
            f"({len(response_ids)} ids) and attention_outputs.npz ({len(activation_ids)} ids)"
        )

    probe_dir = run_dir / 'trained_probe'
    return dataset, judge_model, syn_responses, syn_activations, syn_layer_outputs, probe_dir


TOP_K   = 10    # number of attention heads for the final probe
N_FOLDS = 5

# Platt scaling (CalibratedClassifierCV) wraps the head probe / NTP calibrator
# by default. Set False to fit the base Pipeline's own .fit()/.predict_proba()
# directly instead -- saved under a '_noplatt' filename suffix so the
# Platt-scaled baseline artifacts are never overwritten. Flip back to True to
# restore the original behavior exactly.
# 2026-08-10-no-platt-scaling-01 found no-Platt substantially worse (see note)
# -- reverted to True, the validated default.
USE_PLATT_SCALING = True

# Layer-output probe training + its combined cross-model plot (bottom of this
# file). Set False to skip both and only train/save the head probe + NTP
# calibrator. Flip back to True to restore the original behavior exactly.
# 2026-08-10-qwen-base-answer-cue-01: head probe only, matching both
# baselines it's compared against -- no layer-probe training needed.
TRAIN_LAYER_PROBE = False
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


def _train_and_save(DATASET, JUDGE_MODEL, syn_responses, syn_activations, syn_layer_outputs, probe_dir):
    """Train the head probe + NTP calibrator (and, if TRAIN_LAYER_PROBE, the
    layer probe) for one (dataset, judge_model) synthetic run and save them
    under ``probe_dir``. Shared by both the legacy (dataset/judge/date/source
    flags, old tree) and id-addressed (--judge-run-ids, new tree) modes in
    ``main()`` -- everything below this point is byte-for-byte the script's
    prior behavior, just parameterized over where the inputs came from and
    where the outputs go.
    """
    print(f'\n{"="*60}\nDataset: {DATASET}   Judge: {JUDGE_MODEL}\n{"="*60}')

    Path(FIGURES_DIR, JUDGE_MODEL).mkdir(parents=True, exist_ok=True)

    # ─────────────────────────────────────────────────────────────────
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

    if USE_PLATT_SCALING:
        head_probe = CalibratedClassifierCV(
            estimator=base_probe,
            method='sigmoid',  # 'sigmoid' is Platt scaling
            cv=kfold_cv
        ).fit(X_train, y_train)
    else:
        head_probe = base_probe.fit(X_train, y_train)

    # Evaluate training performance:
    y_probs = head_probe.predict_proba(X_train)[:, 1]
    y_pred = (y_probs > 0.5).astype(int)
    print(f"  Accuracy  : {accuracy_score(y_train, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_train, y_pred):.4f}")
    print(f"  Recall    : {recall_score(y_train, y_pred):.4f}")
    print(f"  F1-Score  : {f1_score(y_train, y_pred):.4f}")
    print(f"  AUROC     : {roc_auc_score(y_train, y_probs):.4f}")
    print(f"  ECE       : {compute_ece(y_train, y_probs):.4f}")

    # Save probe + metadata for use in synthetic_probe_test.ipynb
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_filename = 'head_probe.pkl' if USE_PLATT_SCALING else 'head_probe_noplatt.pkl'
    probe_path = probe_dir / probe_filename

    probe_data = {
        'probe':            head_probe,
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
    if USE_PLATT_SCALING:
        ntp_calibrated = CalibratedClassifierCV(
            estimator=ntp_base,
            method='sigmoid',
            cv=kfold_cv,
        ).fit(ntp_probs_train, y_train)
    else:
        ntp_calibrated = ntp_base.fit(ntp_probs_train, y_train)

    ntp_cal_probs_tr = ntp_calibrated.predict_proba(ntp_probs_train)[:, 1]
    print(f"  NTP calibrator train ECE: {compute_ece(ntp_cal_probs_tr, y_train):.4f}")

    ntp_cal_filename = 'ntp_calibrator.pkl' if USE_PLATT_SCALING else 'ntp_calibrator_noplatt.pkl'
    ntp_cal_path = probe_dir / ntp_cal_filename
    joblib.dump({
        'calibrator':       ntp_calibrated,
        'train_prevalence': float(y_train.mean()),
        'syn_document_ids': sorted(syn_df['document_id'].unique().tolist()),
        'judge_model':      JUDGE_MODEL,
        'dataset':          DATASET,
    }, ntp_cal_path)
    print(f'NTP calibrator saved → {ntp_cal_path}')
    # ─────────────────────────────────────────────────────────────────

    if TRAIN_LAYER_PROBE:
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
        ax.legend(fontsize=9)
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
        ax.legend(fontsize=9)
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


def main():
    judge_run_ids = _select_judge_run_ids()

    legacy_argv = {'--datasets', '--judges', '--judge-date', '--source'}
    legacy_env = ('SYNTHETIC_PROBE_DATASETS', 'SYNTHETIC_PROBE_JUDGES',
                  'SYNTHETIC_PROBE_JUDGE_DATE', 'SYNTHETIC_PROBE_SOURCE')
    legacy_given = legacy_argv & set(sys.argv) or any(os.environ.get(k) for k in legacy_env)
    if judge_run_ids and legacy_given:
        raise ValueError(
            "synthetic_probe_train.py: --judge-run-ids is mutually exclusive with "
            "--datasets/--judges/--judge-date/--source (and their SYNTHETIC_PROBE_* "
            "env equivalents) -- each judge-run-id already names its own "
            "(dataset, judge_model) pair via its experiment-config, so mixing in "
            "the legacy selectors would let two sources of truth disagree about "
            "which run gets trained on."
        )

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

    DATASETS_SEEN: list[str] = []
    JUDGE_MODELS_SEEN: list[str] = []

    if judge_run_ids:
        print(f'[synthetic_probe_train] judge_run_ids={judge_run_ids}')
        for run_id in judge_run_ids:
            DATASET, JUDGE_MODEL, syn_responses, syn_activations, syn_layer_outputs, probe_dir = (
                _load_synthetic_run(run_id)
            )
            if DATASET not in DATASETS_SEEN:
                DATASETS_SEEN.append(DATASET)
            if JUDGE_MODEL not in JUDGE_MODELS_SEEN:
                JUDGE_MODELS_SEEN.append(JUDGE_MODEL)
            _train_and_save(DATASET, JUDGE_MODEL, syn_responses, syn_activations,
                             syn_layer_outputs, probe_dir)
    else:
        DATASETS, JUDGE_MODELS, JUDGE_DATE_SYN, syn_name, probe_source = _select_run_config()
        print(f'[synthetic_probe_train] datasets={DATASETS} judges={JUDGE_MODELS} '
              f'judge_date={JUDGE_DATE_SYN} syn_name={syn_name!r} probe_source={probe_source!r}')
        DATASETS_SEEN, JUDGE_MODELS_SEEN = DATASETS, JUDGE_MODELS

        for DATASET in DATASETS:
            for JUDGE_MODEL in JUDGE_MODELS:
                syn_activations = load_synthetic_activations(DATASET, JUDGE_MODEL, JUDGE_DATE_SYN, split='train', name=syn_name)
                syn_layer_outputs = load_synthetic_layer_outputs(DATASET, JUDGE_MODEL, JUDGE_DATE_SYN, split='train', name=syn_name)
                syn_responses   = load_synthetic_responses(DATASET, JUDGE_MODEL, JUDGE_DATE_SYN, split='train', name=syn_name)
                probe_dir = paths.trained_probe_dir(DATASET, JUDGE_MODEL, source=probe_source)
                _train_and_save(DATASET, JUDGE_MODEL, syn_responses, syn_activations,
                                 syn_layer_outputs, probe_dir)

    # ─────────────────────────────────────────────────────────────────
    # Create combined plot of F1 scores by layer for all judge models
    # ─────────────────────────────────────────────────────────────────
    if TRAIN_LAYER_PROBE:
        palette_idx=[7, 1, 0, 4]

        for DATASET in DATASETS_SEEN:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))

            # Plot each judge model's F1 scores
            for model_idx, JUDGE_MODEL in enumerate(JUDGE_MODELS_SEEN):
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
            ax.legend(fontsize=10, loc='best')
            ax.set_xlim(-0.5, n_layers_plot - 0.5)
            fig.tight_layout()
            fig.savefig(FIGURES_DIR + f'synprobe_layer_F1_all_models_{DATASET}.pdf', bbox_inches='tight')
            print(f'Combined F1 plot saved → {FIGURES_DIR}synprobe_layer_F1_all_models_{DATASET}.pdf')


if __name__ == "__main__":
    main()
