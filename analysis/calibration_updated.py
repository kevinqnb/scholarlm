import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import os
import re
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
import seaborn as sns
import relplot
from sklearn.metrics import precision_recall_curve, roc_auc_score, brier_score_loss

from analysis.loaders import (
    load_activations, load_layer_outputs, load_combined_judgements,
    load_extraction, load_ground_truth, load_trained_probe, load_trained_ntp_calibrator,
    cached_match, load_synthetic_activations, load_synthetic_layer_outputs,
    load_synthetic_responses,
)
from analysis.metrics import recovery_rate_from_labels, validity_rate_from_labels
from scholarlm.utils.calibration import (
    rescale_probabilities_em, bootstrap_ece, intercept_adjustment,
)
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

FIGURES_DIR = REPO_ROOT / "figures/calibration/"
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

RESULTS_DIR = REPO_ROOT / "results/"
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

# blue: 7, orange: 1, red: 0, green: 4
palette = sns.color_palette("husl", 10)

# One consistent color per dataset, reused across every plot (synthetic/real,
# within/cross) so a dataset is always the same color regardless of role.
_DS_COLORS = {
    'pond':     palette[7],
    'nfix':     palette[1],
    'supermat': palette[0],
}

_DS_LABELS = {'pond': 'PLW', 'nfix': 'NF', 'supermat': 'SM'}


# ── Parameters ───────────────────────────────────────────────────────────────
# Everything that varies with the extraction model lives in EXTRACTION_SETTINGS,
# one entry per model. Select one with `--extraction-model`, the
# CALIBRATION_EXTRACTION_MODEL env var, or by editing DEFAULT_EXTRACTION_MODEL;
# the module-level globals below are then derived from the selected entry, so the
# rest of this file (and the notebooks that import it) is unchanged.
#
# Per-entry keys:
#   datasets        — datasets this extraction model was run on
#   judge_models    — judges with interpretable results for this extraction run
#   judge_datasets  — datasets each judge has activations for. A cross-domain
#                     (train_ds → test_ds) pair is only valid for a judge when
#                     BOTH datasets appear here, since the probe lives in that
#                     judge's activation space.
#   extraction_dates / judge_dates_syn / judge_dates_real — pinned run dates
#                     (a None judge date means "auto-detect latest")
#   pi_te_estimate  — assumed test prevalence for the label-shift intercept
#                     adjustment on real data; None disables the adjustment.
EXTRACTION_SETTINGS = {
    'gemma-3-27b': {
        'datasets': ['pond', 'nfix', 'supermat'],
        'judge_models': ['llama-3.1-8b', 'mistral-7b', 'qwen-2.5-7b', 'qwen-2.5-7b-base', 'qwen-2.5-7b-base-cued', 'llama-3.1-8b-base-cued'],
        # qwen covers all three datasets → full 3×3; llama/mistral cover pond+nfix → 2×2.
        # llama-3.1-8b-base-cued is trained/tested on all three, but its comparison
        # baseline (llama-3.1-8b instruct) only covers pond+nfix -- see
        # 2026-08-11-llama-base-answer-cue-01.
        'judge_datasets': {
            'llama-3.1-8b': ['pond', 'nfix'],
            'mistral-7b':   ['pond', 'nfix'],
            'qwen-2.5-7b':  ['pond', 'nfix', 'supermat'],
            'qwen-2.5-7b-base': ['pond', 'nfix', 'supermat'],
            'qwen-2.5-7b-base-cued': ['pond', 'nfix', 'supermat'],
            'llama-3.1-8b-base-cued': ['pond', 'nfix', 'supermat'],
        },
        'extraction_dates': {
            'pond': '2026_05_05',
            'nfix': '2026_05_06',
            'supermat': '2026_07_09',
        },
        'judge_dates_syn': {
            'pond': {
                'llama-3.1-8b': '2026_05_04',
                'mistral-7b': '2026_05_04',
                'qwen-2.5-7b': '2026_05_04',
                'qwen-2.5-7b-base': '2026_08_05',
                'qwen-2.5-7b-base-cued': '2026_08_10',
                'llama-3.1-8b-base-cued': '2026_08_11',
            },
            'nfix': {
                'llama-3.1-8b': '2026_05_04',
                'mistral-7b': '2026_05_04',
                'qwen-2.5-7b': '2026_05_04',
                'qwen-2.5-7b-base': '2026_08_05',
                'qwen-2.5-7b-base-cued': '2026_08_10',
                'llama-3.1-8b-base-cued': '2026_08_11',
            },
            'supermat': {
                'qwen-2.5-7b': '2026_07_10',   # TODO: pin the supermat synthetic-probe date if not the latest
                'qwen-2.5-7b-base': '2026_08_05',
                'qwen-2.5-7b-base-cued': '2026_08_10',
                'llama-3.1-8b-base-cued': '2026_08_11',
            },
        },
        'judge_dates_real': {
            'pond': {
                'llama-3.1-8b': '2026_05_06',
                'mistral-7b': '2026_05_06',
                'qwen-2.5-7b': '2026_05_06',
                'qwen-2.5-7b-base': '2026_08_05',
                'qwen-2.5-7b-base-cued': '2026_08_10',
                'llama-3.1-8b-base-cued': '2026_08_11',
            },
            'nfix': {
                'llama-3.1-8b': '2026_05_05',
                'mistral-7b': '2026_05_05',
                'qwen-2.5-7b': '2026_05_05',
                'qwen-2.5-7b-base': '2026_08_05',
                'qwen-2.5-7b-base-cued': '2026_08_10',
                'llama-3.1-8b-base-cued': '2026_08_11',
            },
            'supermat': {
                'qwen-2.5-7b': '2026_07_09',   # TODO: pin the supermat real (extracted) judge date if not the latest
                'qwen-2.5-7b-base': '2026_08_05',
                'qwen-2.5-7b-base-cued': '2026_08_10',
                'llama-3.1-8b-base-cued': '2026_08_11',
            },
        },
        'pi_te_estimate': None,
    },

    # Previously analysis/calibration_llama.py
    'llama-3.1-8b': {
        'datasets': ['pond', 'nfix'],
        'judge_models': ['llama-3.1-8b', 'qwen-2.5-7b'],
        'judge_datasets': {
            'llama-3.1-8b': ['pond', 'nfix'],
            'qwen-2.5-7b':  ['pond', 'nfix'],
        },
        'extraction_dates': {
            'pond': '2026_05_04',
            'nfix': '2026_05_05',
        },
        'judge_dates_syn': {
            'pond': {
                'llama-3.1-8b': '2026_05_04',
                'qwen-2.5-7b': '2026_05_04',
            },
            'nfix': {
                'llama-3.1-8b': '2026_05_04',
                'qwen-2.5-7b': '2026_05_04',
            },
        },
        'judge_dates_real': {
            'pond': {
                'llama-3.1-8b': '2026_05_13',
                'qwen-2.5-7b': '2026_05_05',
            },
            'nfix': {
                'llama-3.1-8b': '2026_05_13',
                'qwen-2.5-7b': '2026_05_05',
            },
        },
        'pi_te_estimate': None,
    },

    # Previously analysis/calibration_gpt.py
    'gpt-oss-120b': {
        'datasets': ['pond', 'nfix'],
        'judge_models': ['llama-3.1-8b', 'qwen-2.5-7b'],
        'judge_datasets': {
            'llama-3.1-8b': ['pond', 'nfix'],
            'qwen-2.5-7b':  ['pond', 'nfix'],
        },
        'extraction_dates': {
            'pond': '2026_05_02',
            'nfix': '2026_05_03',
        },
        'judge_dates_syn': {
            'pond': {
                'llama-3.1-8b': '2026_05_04',
                'qwen-2.5-7b': '2026_05_04',
            },
            'nfix': {
                'llama-3.1-8b': '2026_05_04',
                'qwen-2.5-7b': '2026_05_04',
            },
        },
        'judge_dates_real': {
            'pond': {
                'llama-3.1-8b': '2026_05_13',
                'qwen-2.5-7b': '2026_05_05',
            },
            'nfix': {
                'llama-3.1-8b': '2026_05_13',
                'qwen-2.5-7b': '2026_05_05',
            },
        },
        'pi_te_estimate': 0.85,
    },
}

DEFAULT_EXTRACTION_MODEL = 'gemma-3-27b'
DEFAULT_PROBE_TYPE = 'head'
DEFAULT_PROBE_VARIANT = 'platt'


def _env_list(name):
    """Parse a comma- or space-separated env var into a list; None when unset/empty."""
    raw = os.environ.get(name, '').replace(',', ' ').split()
    return raw or None


def _select_settings():
    """Resolve the active extraction model from CLI flag → env var → default.

    parse_known_args keeps this safe under import from a notebook or another
    script, where sys.argv holds flags meant for something else.

    ``--datasets`` / ``--judge-models`` (or CALIBRATION_DATASETS /
    CALIBRATION_JUDGE_MODELS) narrow the selected entry to a subset.  Both the
    probe cache and test_data are built eagerly at import over every dataset in
    the entry, so narrowing is what lets this module import when only part of the
    run data is present locally.  Omitting them leaves the entry unchanged.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--extraction-model', default=None, choices=list(EXTRACTION_SETTINGS))
    parser.add_argument('--probe-type', default=None, choices=['head', 'layer'])
    parser.add_argument('--probe-variant', default=None, choices=['platt', 'noplatt'])
    parser.add_argument('--datasets', nargs='+', default=None)
    parser.add_argument('--judge-models', nargs='+', default=None)
    args, _ = parser.parse_known_args()

    model = (args.extraction_model
             or os.environ.get('CALIBRATION_EXTRACTION_MODEL')
             or DEFAULT_EXTRACTION_MODEL)
    if model not in EXTRACTION_SETTINGS:
        raise ValueError(
            f'Unknown extraction model {model!r}; '
            f'known: {sorted(EXTRACTION_SETTINGS)}'
        )
    probe_type = (args.probe_type
                  or os.environ.get('CALIBRATION_PROBE_TYPE')
                  or DEFAULT_PROBE_TYPE)
    # 'platt' (default) reproduces the original Platt-scaled behavior exactly,
    # including every output path below -- 'noplatt' is additive, never
    # overwrites the baseline. See 2026-08-10-no-platt-scaling-01.
    probe_variant = (args.probe_variant
                      or os.environ.get('CALIBRATION_PROBE_VARIANT')
                      or DEFAULT_PROBE_VARIANT)
    if probe_variant not in ('platt', 'noplatt'):
        raise ValueError(f"Unknown probe variant {probe_variant!r}; expected 'platt' or 'noplatt'")

    def _subset(selected, available, what):
        unknown = [x for x in selected if x not in available]
        if unknown:
            raise ValueError(
                f'Unknown {what} {unknown} for extraction model {model!r}; '
                f'available: {available}'
            )
        return [x for x in available if x in selected]

    settings = dict(EXTRACTION_SETTINGS[model])  # copy: never mutate the registry

    datasets = args.datasets or _env_list('CALIBRATION_DATASETS')
    if datasets:
        settings['datasets'] = _subset(datasets, settings['datasets'], 'dataset')

    judges = args.judge_models or _env_list('CALIBRATION_JUDGE_MODELS')
    if judges:
        settings['judge_models'] = _subset(judges, settings['judge_models'], 'judge model')

    # judge_datasets drives the probe cache, so it has to be narrowed to match or the
    # eager loading below still reaches for probes we just excluded.
    settings['judge_datasets'] = {
        jm: [ds for ds in dss if ds in settings['datasets']]
        for jm, dss in settings['judge_datasets'].items()
        if jm in settings['judge_models']
    }
    return model, probe_type, probe_variant, settings


EXTRACTION_MODEL, PROBE_TYPE, PROBE_VARIANT, _SETTINGS = _select_settings()

# None reproduces load_trained_probe/load_trained_ntp_calibrator's original
# default filenames exactly; only 'noplatt' picks the suffixed variant.
_PROBE_VARIANT_KW = None if PROBE_VARIANT == 'platt' else PROBE_VARIANT
# '' for the default 'platt' variant keeps every output path below byte-for-byte
# identical to the pre-variant behavior; only 'noplatt' gets a distinct suffix.
_PROBE_VARIANT_SUFFIX = '' if PROBE_VARIANT == 'platt' else f'_{PROBE_VARIANT}'

DATASETS         = _SETTINGS['datasets']
JUDGE_MODELS     = _SETTINGS['judge_models']
JUDGE_DATASETS   = _SETTINGS['judge_datasets']
EXTRACTION_DATES = _SETTINGS['extraction_dates']
JUDGE_DATES_SYN  = _SETTINGS['judge_dates_syn']
JUDGE_DATES_REAL = _SETTINGS['judge_dates_real']
PI_TE_ESTIMATE   = _SETTINGS['pi_te_estimate']  # test prevalence for label-shift rescaling; None → off

print(f'[calibration] extraction model: {EXTRACTION_MODEL} | probe type: {PROBE_TYPE} '
      f'| probe variant: {PROBE_VARIANT} | datasets: {DATASETS} | judges: {JUDGE_MODELS}')

THRESHOLD_SWEEP = np.linspace(0.0, 0.95, 20)  # thresholds for operating-curve plot
EDGE_THRESHOLDS  = {'pond': 1/3, 'nfix': 1/6, 'supermat': 1/3}  # minimum fuzzy weight to count as a match


def get_matching_config(dataset):
    if dataset == 'pond':
        strict = {'document_id': 'document_id', 'attribute': 'attribute',
                'value': 'converted_value', 'units': 'units'}
        fuzzy  = {'name': 'name', 'location': 'location', 'ecosystem': 'ecosystem'}
    elif dataset == 'nfix':
        strict = {'document_id': 'document_id', 'attribute': 'attribute',
                'value': 'converted_value', 'units': 'units'}
        fuzzy  = {'name': 'name', 'site_type': 'site_type'}
    elif dataset == 'supermat':
        # tc is the only attribute; entity is the material name/formula.
        # Many ground-truth `name` values are null → those rows fall back to
        # strict-only matching on document_id + attribute + value + units.
        strict = {'document_id': 'document_id', 'attribute': 'attribute',
                'value': 'converted_value', 'units': 'units'}
        fuzzy  = {'name': 'name'}
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    return strict, fuzzy


# Pre-load all probes and NTP calibrators to avoid redundant loading within the loop.
# Only load (dataset, judge) pairs that actually have a trained probe.
ntp_cal_cache = {}
for ds in DATASETS:
    ntp_cal_cache[ds] = {}
    for jm in JUDGE_MODELS:
        if ds not in JUDGE_DATASETS[jm]:
            continue
        ntp_cal_cache[ds][jm] = load_trained_ntp_calibrator(ds, jm, variant=_PROBE_VARIANT_KW)


probe_cache = {}
for ds in DATASETS:
    probe_cache[ds] = {}
    for jm in JUDGE_MODELS:
        if ds not in JUDGE_DATASETS[jm]:
            continue
        probe_cache[ds][jm] = load_trained_probe(ds, jm, ptype=PROBE_TYPE, variant=_PROBE_VARIANT_KW)


def get_matching_config(dataset):
    if dataset == 'pond':
        strict = {'document_id': 'document_id', 'attribute': 'attribute',
                'value': 'converted_value', 'units': 'units'}
        fuzzy  = {'name': 'name', 'location': 'location', 'ecosystem': 'ecosystem'}
    elif dataset == 'nfix':
        strict = {'document_id': 'document_id', 'attribute': 'attribute',
                'value': 'converted_value', 'units': 'units'}
        fuzzy  = {'name': 'name', 'site_type': 'site_type'}
    elif dataset == 'supermat':
        # tc is the only attribute; entity is the material name/formula.
        # Many ground-truth `name` values are null → those rows fall back to
        # strict-only matching on document_id + attribute + value + units.
        strict = {'document_id': 'document_id', 'attribute': 'attribute',
                'value': 'converted_value', 'units': 'units'}
        fuzzy  = {'name': 'name'}
    else:
        raise ValueError(f'Unknown dataset: {dataset}')
    return strict, fuzzy

# Pre-load all test data, including matching results, to avoid redundant loading and matching within the loop
test_data = {}
for ds in DATASETS:
    EDGE_THRESHOLD = EDGE_THRESHOLDS[ds]
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
    filtered_edges = []
    for (gt_idx, ex_idx), w in zip(edges, edge_weights):
        if w > EDGE_THRESHOLD:
            ex_edge_exists[int(ex_idx)] = True
            filtered_edges.append((int(gt_idx), int(ex_idx)))
    jlabels     = real_df['judgement_combined'].to_numpy(dtype=bool)
    combined_labels = jlabels | ex_edge_exists

    test_data[ds] = {
        'real_df': real_df,
        'gt_df': gt_df,
        'labels': combined_labels,
        'matching_labels': ex_edge_exists,
        'judge_labels': jlabels,
        'filtered_edges': filtered_edges,
    }



def compute_predictions(judge_models, datasets, probe_type, load_from_precomputed=False):
    # ── Collect data for each test setting ────────────────────────────────
    # Result format: {dataset_type: {judge_model: {train_ds: {test_ds: {probe_probs: x, ntp_probs: y, labels: z}}}}}
    
    # Define cache file path
    cache_file = Path(RESULTS_DIR) / f'predictions_{EXTRACTION_MODEL}_{probe_type}{_PROBE_VARIANT_SUFFIX}.pkl'
    
    # Try to load from precomputed cache if requested
    if load_from_precomputed and cache_file.exists():
        print(f'Loading precomputed predictions from {cache_file}...')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    setting_results = {}

    for dataset_type in ['syn', 'real']:
        setting_results[dataset_type] = {}
        for judge_model in judge_models:
            setting_results[dataset_type][judge_model] = {}
            for train_ds in datasets:
                if train_ds not in JUDGE_DATASETS[judge_model]:
                    continue
                setting_results[dataset_type][judge_model][train_ds] = {}
                pd_data = probe_cache[train_ds][judge_model]
                ntp_cal_data = ntp_cal_cache[train_ds][judge_model]
                if probe_type == "layer":
                    top = pd_data['top_layer']
                else:
                    top = pd_data['top_k_heads']

                for test_ds in datasets:
                    if test_ds not in JUDGE_DATASETS[judge_model]:
                        continue
                    if dataset_type == 'syn':
                        jdate    = JUDGE_DATES_SYN[test_ds][judge_model]
                        syn_resp = load_synthetic_responses(test_ds, judge_model, jdate, split='test')
                        syn_df_s = pd.DataFrame(syn_resp)
                        mids     = syn_df_s['measurement_id'].tolist()
                        labels   = (syn_df_s['label'] == 'valid').to_numpy(dtype=bool)
                        raw_ntp_probs = syn_df_s['judgement_p_true'].to_numpy()
                        ntp_probs = ntp_cal_data['calibrator'].predict_proba(
                            raw_ntp_probs.reshape(-1, 1)
                        )[:, 1]

                        if probe_type == "layer":
                            syn_lo  = load_synthetic_layer_outputs(test_ds, judge_model, jdate, split='test')
                            X = np.stack([
                                np.array(syn_lo[str(mid)], dtype=np.float32)[top]
                                for mid in mids
                            ], axis=0)
                            probe_probs = pd_data['probe'].predict_proba(X)[:, 1]

                        else:
                            syn_act  = load_synthetic_activations(test_ds, judge_model, jdate, split='test')
                            X = np.concatenate([
                                np.stack([
                                    np.array(syn_act[str(mid)], dtype=np.float32)[l, h, :]
                                    for mid in mids
                                ], axis=0)
                                for l, h in top
                            ], axis=1)
                            probe_probs = pd_data['probe'].predict_proba(X)[:, 1]

                        # Each GT-positive item maps to itself: GT slot k → full-array position pos_idx[k].
                        # gt_idx is the sequential slot index (0..n_gt-1); ex_idx is the original position
                        # in predicted_labels (length = len(labels)), so pos_idx[k] is always a valid index.
                        pos_idx = np.where(labels)[0]
                        test_edges = list(enumerate(pos_idx.tolist()))
                        n_ground_truth = len(pos_idx)

                    else:  # real
                        td       = test_data[test_ds]
                        real_df  = td['real_df']
                        gt_df    = td['gt_df']
                        syn_docs = set(pd_data['syn_document_ids'])

                        # Filter extractions to test documents (those not used in probe training).
                        # idx: positional indices into real_df/ext_df for the test split.
                        mask     = ~real_df['document_id'].isin(syn_docs)
                        idx      = np.where(mask.to_numpy())[0]
                        idx_set  = set(idx.tolist())

                        # Filter GT to test documents and build reindex maps so that
                        # both gt_idx and ex_idx in test_edges live in [0, their respective test-set sizes).
                        gt_mask    = ~gt_df['document_id'].isin(syn_docs)
                        gt_idx_arr = np.where(gt_mask.to_numpy())[0]
                        gt_idx_set = set(gt_idx_arr.tolist())
                        old_to_new_ex = {int(v): k for k, v in enumerate(idx)}
                        old_to_new_gt = {int(v): k for k, v in enumerate(gt_idx_arr)}
                        test_edges = [
                            (old_to_new_gt[gt_i], old_to_new_ex[ex_i])
                            for gt_i, ex_i in td['filtered_edges']
                            if ex_i in idx_set and gt_i in gt_idx_set
                        ]
                        n_ground_truth = len(gt_idx_arr)

                        mids     = real_df['measurement_id'].iloc[idx].tolist()
                        labels   = td['labels'][idx]
                        jdate    = JUDGE_DATES_REAL[test_ds][judge_model]

                        raw_ntp_probs = real_df[f'judgement_p_true_{judge_model}'].iloc[idx].to_numpy()
                        ntp_probs = ntp_cal_data['calibrator'].predict_proba(
                            raw_ntp_probs.reshape(-1, 1)
                        )[:, 1]

                        if probe_type == "layer":
                            real_lo  = load_layer_outputs(test_ds, EXTRACTION_MODEL, EXTRACTION_DATES[test_ds], judge_model, jdate)
                            X = np.stack([
                                np.array(real_lo[str(mid)], dtype=np.float32)[top]
                                for mid in mids
                            ], axis=0)
                            probe_probs = pd_data['probe'].predict_proba(X)[:, 1]
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
                            probe_probs = pd_data['probe'].predict_proba(X)[:, 1]

                    # Real extractions have a different positive rate than the synthetic
                    # training set; rescale to the assumed test prevalence when one is set.
                    if dataset_type == 'real' and PI_TE_ESTIMATE is not None:
                        probe_probs = intercept_adjustment(
                            probe_probs, pi_tr=pd_data['train_prevalence'], pi_te=PI_TE_ESTIMATE
                        )
                        ntp_probs = intercept_adjustment(
                            ntp_probs, pi_tr=ntp_cal_data['train_prevalence'], pi_te=PI_TE_ESTIMATE
                        )

                    setting_results[dataset_type][judge_model][train_ds][test_ds] = {
                        'probe_probs': probe_probs, 'ntp_probs': ntp_probs, 'labels': labels,
                        'edges': test_edges, 'n_ground_truth': n_ground_truth,
                    }

    # Save to cache for future use
    cache_file = Path(RESULTS_DIR) / f'predictions_{EXTRACTION_MODEL}_{probe_type}{_PROBE_VARIANT_SUFFIX}.pkl'
    print(f'Saving predictions to {cache_file}...')
    with open(cache_file, 'wb') as f:
        pickle.dump(setting_results, f)
    
    return setting_results


def _pool_cross_domain(train_dict, train_ds):
    """Pool every test_ds != train_ds in train_dict into one cross-domain cell.

    ``train_dict`` is ``setting_results[dtype][judge_model][train_ds]``, keyed by
    test_ds (already restricted to datasets the judge has activations for — see
    ``compute_predictions``). Concatenates ``probe_probs``/``ntp_probs``/``labels``
    across the other test_ds's, and merges their ``edges`` by offsetting each
    subsequent test_ds's ``gt_idx``/``ex_idx`` by the running totals of prior
    ``n_ground_truth``/array length, so the pooled edges index correctly into the
    pooled arrays. Returns None when train_ds has no other dataset to pool against
    (single-dataset judge coverage).
    """
    other_test_ds = [ds for ds in DATASETS if ds != train_ds and ds in train_dict]
    if not other_test_ds:
        return None

    probe_probs, ntp_probs, labels, edges = [], [], [], []
    ex_offset, gt_offset, n_ground_truth = 0, 0, 0
    for test_ds in other_test_ds:
        cell = train_dict[test_ds]
        probe_probs.append(cell['probe_probs'])
        ntp_probs.append(cell['ntp_probs'])
        labels.append(cell['labels'])
        edges.extend((gt_i + gt_offset, ex_i + ex_offset) for gt_i, ex_i in cell['edges'])
        ex_offset += len(cell['labels'])
        gt_offset += cell['n_ground_truth']
        n_ground_truth += cell['n_ground_truth']

    return {
        'probe_probs': np.concatenate(probe_probs),
        'ntp_probs': np.concatenate(ntp_probs),
        'labels': np.concatenate(labels),
        'edges': edges,
        'n_ground_truth': n_ground_truth,
        'test_ds': other_test_ds,
    }


# Bootstrap settings for ECE confidence intervals (also reused to seed
# relplot's own internal bootstraps, which take no seed argument of their
# own — see the seeding note in _plot_relplot_curve/_probe_metrics below).
ECE_N_BOOT = 2000
ECE_CI     = 0.95
ECE_SEED   = 0


# Floor on the density-normalized alpha used for the smoothed calibration
# curves below, so low-density mesh regions fade toward-transparent (the
# continuous analog of the old discrete plot dropping zero-count bins
# entirely) without a segment fully disappearing.
_CURVE_DENSITY_ALPHA_FLOOR = 0.15


# Dash pattern for the NTP curve, expressed as (period, on) in mesh-point
# units. matplotlib's own dashed linestyle can't be passed to `linestyle=`
# below: a LineCollection built from many short independent 2-point segments
# (needed for per-segment density alpha) restarts the dash offset at the
# start of every segment, which visually collapses '--' into a solid line
# (confirmed empirically) — so the dash pattern is instead emulated by
# omitting the "off" segments outright.
_NTP_DASH_PERIOD, _NTP_DASH_ON = 10, 7


def _plot_relplot_curve(ax, probs, labels, color, *, linestyle, lw, line_zorder, band_zorder):
    """Density-weighted smoothed reliability curve + bootstrap CI band.

    Replaces the discrete per-bin scatter (marker size ~ bin count) with a
    continuous LineCollection whose per-segment alpha tracks local prediction
    density — relplot's own internal convention (see its diagrams.py: both its
    bootstrapped bag lines and its main-curve scatter scale alpha, never
    linewidth, by density) — and the per-bin SEM band with relplot's bootstrap
    confidence band.
    """
    # relplot's BaggingRegressor/scipy.stats.bootstrap calls take no seed of
    # their own and draw from the global numpy RNG — reseed immediately
    # before the call so the rendered curve/band is reproducible run-to-run
    # (confirmed: identical `mu`/`lower`/`upper` across repeated seeded calls
    # on the same inputs; without this, two full-pipeline runs disagreed).
    np.random.seed(ECE_SEED)
    d = relplot.prepare_rel_diagram(np.asarray(probs), np.asarray(labels))
    mesh, mu, density = d['mesh'], d['mu'], d['density']

    density_norm = density / density.max() if density.max() > 0 else np.ones_like(density)
    alpha = _CURVE_DENSITY_ALPHA_FLOOR + (1 - _CURVE_DENSITY_ALPHA_FLOOR) * density_norm

    points = np.array([mesh, mu]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = np.tile(mcolors.to_rgba(color), (len(segments), 1))
    seg_colors[:, 3] = (alpha[:-1] + alpha[1:]) / 2

    if linestyle == '--':
        seg_idx = np.arange(len(segments))
        dash_mask = (seg_idx % _NTP_DASH_PERIOD) < _NTP_DASH_ON
        segments = segments[dash_mask]
        seg_colors = seg_colors[dash_mask]

    lc = LineCollection(
        segments, colors=seg_colors, lw=lw,
        joinstyle='round', zorder=line_zorder,
    )
    ax.add_collection(lc)

    ax.fill_between(
        mesh, d['lower'], d['upper'],
        color=color, alpha=0.20, linewidth=0, zorder=band_zorder,
    )


def plot_calibration_curves(
    setting_results, dtype
):
    # One (within, cross) pair of plots per judge model — one line per train_ds
    # in DATASETS. Within-domain plots train_ds against itself; cross-domain
    # plots train_ds against the pooled union of every other dataset that judge
    # has activations for (see _pool_cross_domain). Judges covering fewer than 3
    # datasets degrade gracefully: fewer within-domain lines, and a cross-domain
    # "union" that pools only 1 dataset instead of 2.
    for judge_model in JUDGE_MODELS:
        # Derived from setting_results itself, not JUDGE_DATASETS[judge_model]:
        # compute_predictions may have been called with a narrower `datasets`
        # list than the module-global DATASETS (the --datasets / narrowing path
        # documented in _select_settings), in which case JUDGE_DATASETS still
        # lists datasets that have no cell here.
        train_datasets = [ds for ds in DATASETS if ds in setting_results[dtype][judge_model]]
        if not train_datasets:
            continue

        subfigure_dir = FIGURES_DIR / f"{judge_model}/{EXTRACTION_MODEL}/{PROBE_TYPE}{_PROBE_VARIANT_SUFFIX}/"
        Path(subfigure_dir).mkdir(parents=True, exist_ok=True)

        for ctype in ['in-domain', 'cross-domain']:
            # Base figure
            fig_cal, ax_cal = plt.subplots(figsize=(4.0, 3.8))
            ax_cal.plot([0, 1], [0, 1], 'k--', lw=1.0, alpha=0.5, zorder=1)

            for train_ds in train_datasets:
                train_dict = setting_results[dtype][judge_model][train_ds]

                if ctype == 'in-domain':
                    rdict = train_dict[train_ds]
                else:
                    rdict = _pool_cross_domain(train_dict, train_ds)
                    if rdict is None:
                        continue

                color = _DS_COLORS[train_ds]

                # Probe — solid, density-weighted smoothed curve + bootstrap CI band
                _plot_relplot_curve(
                    ax_cal, rdict['probe_probs'], rdict['labels'], color,
                    linestyle='-', lw=2.5, line_zorder=3, band_zorder=2,
                )

                # NTP baseline — dashed, density-weighted smoothed curve + bootstrap CI band
                _plot_relplot_curve(
                    ax_cal, rdict['ntp_probs'], rdict['labels'], color,
                    linestyle='--', lw=2.0, line_zorder=1, band_zorder=2,
                )

            ax_cal.set_xlim(-0.02, 1.02)
            ax_cal.set_ylim(-0.02, 1.02)
            ax_cal.set_xlabel('Predicted Probability')

            ax_cal.set_ylabel('Observed Frequency')
            if ctype == 'in-domain':
                ax_cal.set_title(f'Within', fontsize=15, style='italic')

            ax_cal.grid(alpha=0.25, linestyle='-', linewidth=0.4)
            ax_cal.set_axisbelow(True)
            fig_cal.tight_layout()
            fig_cal.savefig(
                subfigure_dir / f'cal_{dtype}_{ctype}.pdf', bbox_inches='tight', dpi = 200
            )
            plt.show()


def _probe_metrics(probs, y_true, threshold=0.5, *, edges=None, n_ground_truth=None):
    """Compute metrics at a fixed threshold. Returns dict.

    Calibration error is reported in four variants, each with a bootstrap
    confidence interval:
      - ``ece``      — L1 ECE, equal-width bins, plug-in (matches the
                       reliability-diagram ECE used in the calibration plots).
      - ``ece_em``   — L1 ECE, adaptive equal-mass (quantile) bins, plug-in.
      - ``rmsce_db`` — debiased L2 RMS calibration error on equal-mass bins
                       (Kumar, Liang & Ma, NeurIPS 2019).  Distinct metric /
                       scale from the L1 columns; the only provably-unbiased one.
      - ``smece``    — smooth ECE (relplot), a kernel-smoothed calibration
                       distance with its own bootstrap CI, independent of any
                       binning choice.
    Each variant ``X`` carries ``X_lo`` / ``X_hi`` interval bounds.
    """
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

    # ── Calibration-error variants with bootstrap CIs ────────────────────
    ece_ew = bootstrap_ece(probs, y_true, binning='equal_width', p=1,
                           n_boot=ECE_N_BOOT, ci=ECE_CI, seed=ECE_SEED)
    ece_em = bootstrap_ece(probs, y_true, binning='equal_mass', p=1,
                           n_boot=ECE_N_BOOT, ci=ECE_CI, seed=ECE_SEED)
    rmsce  = bootstrap_ece(probs, y_true, binning='equal_mass', p=2, debiased=True,
                           n_boot=ECE_N_BOOT, ci=ECE_CI, seed=ECE_SEED)

    # relplot's own bootstrap CI on smECE — no diagram is already computed in
    # this call path (compute_metrics and plot_calibration_curves each derive
    # their own rdicts independently from setting_results), so this is the
    # cheapest correct call: report_CE/report_CE_std default True regardless
    # of plot_confidence_band/plot_bag_lines, so skipping those (both False)
    # avoids the 200-estimator bootstrap regression fit for the main curve,
    # which isn't needed here. Reseed first — see the seeding note in
    # _plot_relplot_curve; relplot's internal `scipy.stats.bootstrap` call
    # (which produces ce_ci_width) draws from the global numpy RNG and is
    # otherwise non-reproducible run-to-run.
    np.random.seed(ECE_SEED)
    smece_d = relplot.prepare_rel_diagram(
        probs, y_true, plot_confidence_band=False, plot_bag_lines=False,
    )

    bs    = float(brier_score_loss(y_true, probs))
    p_pos = float(y_true.mean())
    bss   = 1.0 - bs / (p_pos * (1 - p_pos)) if p_pos not in (0.0, 1.0) else float('nan')
    recovery = recovery_rate_from_labels(n_ground_truth, edges, preds) if edges is not None else float('nan')
    validity = validity_rate_from_labels(y_true, preds)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auroc=auroc,
                ece=ece_ew['ece'],         ece_lo=ece_ew['ci_low'],    ece_hi=ece_ew['ci_high'],
                ece_em=ece_em['ece'],      ece_em_lo=ece_em['ci_low'], ece_em_hi=ece_em['ci_high'],
                rmsce_db=rmsce['ece'],     rmsce_db_lo=rmsce['ci_low'], rmsce_db_hi=rmsce['ci_high'],
                smece=smece_d['ce'],
                smece_lo=smece_d['ce'] - smece_d['ce_ci_width'],
                smece_hi=smece_d['ce'] + smece_d['ce_ci_width'],
                bs=bs, bss=bss, n=n, recovery=recovery, validity=validity)


def compute_metrics(setting_results):
    # One in-domain row (train_ds vs itself) plus one pooled cross-domain row
    # (train_ds vs the union of every other test_ds — see _pool_cross_domain)
    # per train_ds, instead of one row per (train_ds, test_ds) pair.
    rows = []
    for dtype in setting_results:
        for judge_model in setting_results[dtype]:
            for train_ds in setting_results[dtype][judge_model]:
                train_dict = setting_results[dtype][judge_model][train_ds]

                cells = [(train_ds, train_dict[train_ds])]
                pooled = _pool_cross_domain(train_dict, train_ds)
                if pooled is not None:
                    cells.append(('+'.join(pooled['test_ds']), pooled))

                for test_ds, rdict in cells:
                    for probs, kind in [(rdict['ntp_probs'], 'NTP'), (rdict['probe_probs'], 'Probe')]:
                        m = _probe_metrics(
                            probs, rdict['labels'],
                            edges=rdict['edges'], n_ground_truth=rdict['n_ground_truth'],
                        )
                        rows.append({
                            'Dataset type':   dtype,
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
                            # L1 ECE, adaptive equal-mass plug-in + bootstrap CI
                            'ECE_em':        m['ece_em'],
                            'ECE_em_lo':     m['ece_em_lo'],
                            'ECE_em_hi':     m['ece_em_hi'],
                            # Debiased L2 RMS calibration error (Kumar 2019), equal-mass + CI
                            'RMSCE_db':      m['rmsce_db'],
                            'RMSCE_db_lo':   m['rmsce_db_lo'],
                            'RMSCE_db_hi':   m['rmsce_db_hi'],
                            # Smooth ECE (relplot), kernel-smoothed + bootstrap CI
                            'SmECE':         m['smece'],
                            'SmECE_lo':      m['smece_lo'],
                            'SmECE_hi':      m['smece_hi'],
                            'Recovery':      m['recovery'],
                            'Validity':      m['validity'],
                        })
    df = pd.DataFrame(rows)
    return df
    


def plot_pr_curves(setting_results, dtype):
    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for judge_model in JUDGE_MODELS:
        subfigure_dir = FIGURES_DIR / f"{judge_model}/{EXTRACTION_MODEL}/{PROBE_TYPE}{_PROBE_VARIANT_SUFFIX}/"
        Path(subfigure_dir).mkdir(parents=True, exist_ok=True)

        for train_ds in DATASETS:
            for ctype in ['in-domain', 'cross-domain']:
                for test_ds in DATASETS:
                    if (train_ds == test_ds) != (ctype == 'in-domain'):
                        continue
                    if train_ds not in JUDGE_DATASETS[judge_model] or test_ds not in JUDGE_DATASETS[judge_model]:
                        continue

                    fig_pr, ax_pr = plt.subplots(figsize=(4.0, 3.8))

                    rdict = setting_results[dtype][judge_model][train_ds][test_ds]

                    # NTP — faint dashed gray line
                    prec_ntp, rec_ntp, _ = precision_recall_curve(rdict['labels'], rdict['ntp_probs'])
                    #ax_pr.plot(rec_ntp, prec_ntp, '--', color='#888888',
                    #           lw=1.5, alpha=0.75, zorder=2, label='NTP')

                    # Probe — solid gray line with scatter points colored by threshold
                    prec_prb, rec_prb, thresh_prb = precision_recall_curve(rdict['labels'], rdict['probe_probs'])
                    # precision_recall_curve returns one extra point (recall=0, prec=1) with no threshold
                    ax_pr.plot(rec_prb, prec_prb, '-', color='grey', lw=2.0, zorder=3, label='Probe')
                    # Sample every other point to reduce clutter (or adjust stride as needed)
                    stride = max(1, len(thresh_prb) // 10)  # Show ~10 scatter points
                    ax_pr.scatter(rec_prb[:-1:stride], prec_prb[:-1:stride],
                                  c=thresh_prb[::stride], cmap=cmap, norm=norm, s=35, zorder=4)

                    # Mark threshold ≈ 0.5
                    idx0 = np.argmin(np.abs(thresh_prb - 0.5))
                    ax_pr.scatter([rec_prb[idx0]], [prec_prb[idx0]], s=60, c='none',
                                  edgecolors='k', linewidths=1.1, zorder=5, marker='o')

                    ax_pr.set_xlim(-0.02, 1.02)
                    ax_pr.set_ylim(-0.02, 1.02)
                    ax_pr.set_xlabel('Recovery')
                    ax_pr.set_ylabel('Validity' if ctype == 'in-domain' else '')
                    ax_pr.grid(alpha=0.25, linestyle='-', linewidth=0.4)
                    ax_pr.set_axisbelow(True)
                    fig_pr.tight_layout()
                    fig_pr.savefig(
                        subfigure_dir / f'pr_{dtype}_{train_ds}_{test_ds}.pdf',
                        bbox_inches='tight', dpi=200,
                    )
                    plt.show()

    # Save colorbar once, shared across all PR plots
    fig_cb, ax_cb = plt.subplots(figsize=(0.35, 3.2))
    plt.colorbar(sm, cax=ax_cb, label='Threshold')
    fig_cb.savefig(FIGURES_DIR / f'pr_colorbar_{dtype}.pdf', bbox_inches='tight', dpi=200)
    plt.show()


def plot_validity_recovery(setting_results, dtype):
    N_RANDOM = 50
    cmap = plt.cm.coolwarm
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    for judge_model in JUDGE_MODELS:
        subfigure_dir = FIGURES_DIR / f"{judge_model}/{EXTRACTION_MODEL}/{PROBE_TYPE}{_PROBE_VARIANT_SUFFIX}/"
        Path(subfigure_dir).mkdir(parents=True, exist_ok=True)

        for train_ds in DATASETS:
            for ctype in ['in-domain', 'cross-domain']:
                for test_ds in DATASETS:
                    if (train_ds == test_ds) != (ctype == 'in-domain'):
                        continue
                    if train_ds not in JUDGE_DATASETS[judge_model] or test_ds not in JUDGE_DATASETS[judge_model]:
                        continue

                    fig, ax = plt.subplots(figsize=(4.0, 3.8))
                    rdict = setting_results[dtype][judge_model][train_ds][test_ds]
                    labels = rdict['labels']
                    n_gt   = rdict['n_ground_truth']
                    edges  = rdict['edges']

                    def compute_vr_curve(probs):
                        """Return (validity, recovery, thresholds) skipping thresholds with no predicted positives."""
                        v, r, ts = [], [], []
                        for t in THRESHOLD_SWEEP:
                            preds = probs > t
                            if preds.sum() == 0:
                                continue
                            v.append(validity_rate_from_labels(labels, preds))
                            r.append(recovery_rate_from_labels(n_gt, edges, preds))
                            ts.append(t)
                        return np.array(v), np.array(r), np.array(ts)

                    def plot_vr_curve(probs, linestyle, zorder_base):
                        v, r, ts = compute_vr_curve(probs)
                        if len(ts) == 0:
                            return
                        ax.plot(r, v, linestyle, color='grey', lw=3.0, zorder=zorder_base)
                        n = len(ts)
                        stride = max(1, n // 10)
                        idx = sorted({0, n - 1} | set(range(0, n, stride)))
                        ax.scatter(r[idx], v[idx], c=ts[idx], cmap=cmap, norm=norm, s=45, zorder=zorder_base + 1)
                        idx0 = int(np.argmin(np.abs(ts - 0.5)))
                        ax.scatter([r[idx0]], [v[idx0]], s=60, c='none',
                                   edgecolors='k', linewidths=1.1, zorder=zorder_base + 2, marker='o')

                    # NTP — dashed (drawn first so probe sits on top)
                    plot_vr_curve(np.asarray(rdict['ntp_probs']), '--', zorder_base=3)
                    # Probe — solid
                    plot_vr_curve(np.asarray(rdict['probe_probs']), '-', zorder_base=6)

                    # Random baseline — average validity/recovery over repeated uniform draws
                    n_items = len(labels)
                    rand_v = np.full((N_RANDOM, len(THRESHOLD_SWEEP)), np.nan)
                    rand_r = np.full((N_RANDOM, len(THRESHOLD_SWEEP)), np.nan)
                    for i in range(N_RANDOM):
                        rand_probs_i = np.random.uniform(0, 1, n_items)
                        for j, t in enumerate(THRESHOLD_SWEEP):
                            preds = rand_probs_i > t
                            if preds.sum() > 0:
                                rand_v[i, j] = validity_rate_from_labels(labels, preds)
                                rand_r[i, j] = recovery_rate_from_labels(n_gt, edges, preds)
                    avg_v = np.nanmean(rand_v, axis=0)
                    avg_r = np.nanmean(rand_r, axis=0)
                    valid_rand = ~(np.isnan(avg_v) | np.isnan(avg_r))
                    if valid_rand.any():
                        ax.plot(avg_r[valid_rand], avg_v[valid_rand], ':', color='grey', lw=2.0, zorder=2)

                    #ax.set_xlim(-0.02, 1.02)
                    #ax.set_ylim(-0.02, 1.02)
                    ax.set_xlim(left=-0.02)
                    ax.set_ylim(top=1.02)
                    ax.set_xlabel('Recovery')
                    ax.set_ylabel('Validity') # if ctype == 'in-domain' else '')
                    ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
                    ax.set_axisbelow(True)
                    fig.tight_layout()
                    fig.savefig(
                        subfigure_dir / f'vr_{dtype}_{train_ds}_{test_ds}.pdf',
                        bbox_inches='tight', dpi=200,
                    )
                    plt.show()

    fig_cb, ax_cb = plt.subplots(figsize=(0.35, 3.2))
    plt.colorbar(sm, cax=ax_cb, label='Threshold')
    fig_cb.savefig(FIGURES_DIR / f'vr_colorbar_{dtype}.pdf', bbox_inches='tight', dpi=200)
    plt.show()

    _vr_legend_handles = [
        mlines.Line2D([], [], color='grey', lw=2, linestyle='-',  label='Probe'),
        mlines.Line2D([], [], color='grey', lw=2, linestyle='--', label='NTP'),
        mlines.Line2D([], [], color='grey', lw=2, linestyle=':',  label='Random'),
    ]
    fig_vr_leg, ax_vr_leg = plt.subplots(figsize=(4.0, 0.35))
    ax_vr_leg.axis('off')
    ax_vr_leg.legend(handles=_vr_legend_handles, loc='center', ncol=3, fontsize=13,
                     frameon=False, handlelength=2.0)
    fig_vr_leg.savefig(FIGURES_DIR / f'vr_legend_{dtype}.pdf', bbox_inches='tight', dpi=200)
    plt.show()


if __name__ == "__main__":
    # Set to True to load precomputed results if available, False to recompute from scratch.
    # NOTE: the cache is keyed only by (extraction_model, probe_type), so a cache built
    # before supermat was added is stale — run once with False to regenerate, then flip back.
    load_from_precomputed = False
    
    setting_results = compute_predictions(judge_models=JUDGE_MODELS, datasets=DATASETS, probe_type=PROBE_TYPE, load_from_precomputed=load_from_precomputed)
    plot_calibration_curves(setting_results, dtype='syn')
    plot_calibration_curves(setting_results, dtype='real')
    metrics_df = compute_metrics(setting_results)
    print(metrics_df.to_string(index=False, float_format='{:.3f}'.format))
    metrics_df.to_csv(RESULTS_DIR / f'metrics_{EXTRACTION_MODEL}_{PROBE_TYPE}{_PROBE_VARIANT_SUFFIX}_pooled.csv', index=False)

    #plot_pr_curves(setting_results, dtype='syn')
    #plot_pr_curves(setting_results, dtype='real')

    plot_validity_recovery(setting_results, dtype='syn')
    plot_validity_recovery(setting_results, dtype='real')

    # ── Standalone calibration legend ─────────────────────────────────────────────
    _legend_handles = [
        mlines.Line2D([], [], color=_DS_COLORS[ds], lw=2, marker='o', ms=3.5, label=_DS_LABELS[ds])
        for ds in DATASETS
    ] + [
        mlines.Line2D([], [], color='#444444', lw=2, linestyle='-',  label='Probe'),
        mlines.Line2D([], [], color='#444444', lw=2, linestyle='--', label='NTP'),
    ]
    _fig_leg, _ax_leg = plt.subplots(figsize=(10.0, 0.45))
    _ax_leg.axis('off')
    _ax_leg.legend(handles=_legend_handles, loc='center', ncol=6, fontsize=13,
                frameon=False, handlelength=2.0)
    _fig_leg.savefig(FIGURES_DIR / 'legend_calibration.pdf', bbox_inches='tight', dpi=200)
    plt.show()

