"""Probe and NTP calibration with real test cells scored against human-validated labels (v4).

Same cells as calibration.py, with the same recalibration: one map per method
(prior_shift, intercept_fit or platt_fit), fit once on ``fit_n`` rows drawn from the
probe-training pool, labelled by LLM + matching (the pool has no human labels). Only
the real evaluation changes: real cells are scored on the human-validated rows, which
must all lie outside the probe-training documents (asserted, not filtered). Synthetic
cells are identical to calibration.py. Calibration errors get document-bootstrap CIs.
Runs at import time from one config. ``fit_source: oracle`` is rejected by the loader.

Outputs in analysis/results/calibration-validated/<config id>/: predictions.pkl (with
row provenance), bootstrap.pkl, recalibration_maps.csv, human_vs_llm_matching.csv,
metrics_{probe,ntp}.csv, figures/. The real cells cover only the validated rows, so
prediction_store.check_real_cell (which expects every non-pool row) rejects them.

Usage
-----
    python analysis/calibration_validated.py analysis/analysis-configs/calibration-validated/<id>.yaml
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import hashlib
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import roc_auc_score

from analysis.common.config import analysis_results_dir, load_calibration_validated_config, validations_path
from analysis.common import calibration_ids as cids
from analysis.common import matching
from analysis.common import doc_bootstrap as db
from analysis.common.metrics import validity_rate_from_labels
from analysis.common.prediction_store import real_cell_provenance
from analysis.common.calibration_plot_utils import draw_reliability_curve
from analysis.common.head_activations import HeadActivationCache
from analysis.common.loaders import load_probe_artifact
from analysis.common.recalibration import prior_shift_map, intercept_fit_map, platt_fit_map, uniform_fit_sample
from scholarlm.utils.calibration import apply_platt, fit_prior_shift

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

# blue: 7, orange: 1, red: 0, green: 4
palette = sns.color_palette("husl", 10)

# One fixed color per dataset across every plot.
_DS_COLORS = {
    'pond':     palette[7],
    'nfix':     palette[1],
    'supermat': palette[0],
}

_DS_LABELS = {'pond': 'PLW', 'nfix': 'NF', 'supermat': 'SM'}


# ── Parameters ───────────────────────────────────────────────────────────────
# Fit rows are drawn once (no fit resampling); the only resampling is the
# document bootstrap of each evaluation set (see common/doc_bootstrap.py).
def _parse_args():
    """Parse the single positional calibration config path."""
    parser = argparse.ArgumentParser(
        description="Probe/NTP calibration analysis (v4 recalibration and document-bootstrap CIs; real test cells scored against human-validated labels)."
    )
    parser.add_argument('config', type=Path,
                        help="analysis/analysis-configs/calibration-validated/<id>.yaml (see load_calibration_validated_config)")
    return parser.parse_args()


_CFG = load_calibration_validated_config(_parse_args().config)
CONFIG_ID = _CFG['id']
SEED = _CFG['seed']
_PARAMS = _CFG['params']

PROBE_TYPE    = _PARAMS['probe_type']
PROBE_VARIANT = _PARAMS['probe_variant']
SYN_SPLIT     = _PARAMS['syn_split']
# Test-document resamples per cell (real and synthetic).
N_BOOT = _PARAMS['n_boot']
# Real-cell map expit(slope * logit(p) + intercept); methods in common/recalibration.py.
# fit_source: sample (fit_n pool rows by fit_seed) or manual (pi_te_estimate, prior_shift
# only). No oracle: the evaluated rows carry human labels, the fit rows LLM+matching.
RECALIBRATION = _PARAMS['recalibration']
FIT_SOURCE    = _PARAMS['fit_source']
FIT_N         = _PARAMS['fit_n']
FIT_SEED      = _PARAMS['fit_seed']
assert FIT_SOURCE in ('sample', 'manual'), FIT_SOURCE
DATASETS = list(_PARAMS['datasets'])
TRAIN_DATASETS = list(_PARAMS['datasets'])  # every validated dataset has its own synthetic-probe config

_INPUTS = cids.resolve_calibration_inputs(_CFG)
JUDGE_MODEL = _INPUTS['judge_model']

OUT_DIR = analysis_results_dir("calibration-validated") / CONFIG_ID
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# None selects the Platt-scaled filenames; 'noplatt' the suffixed ones.
_PROBE_VARIANT_KW = None if PROBE_VARIANT == 'platt' else PROBE_VARIANT

_DTYPES = ['syn', 'real']

print(f'[calibration validated v4] config: {CONFIG_ID} | probe type: {PROBE_TYPE} | probe variant: {PROBE_VARIANT} '
      f'| syn split: {SYN_SPLIT} | datasets: {DATASETS} | recalibration: {RECALIBRATION} '
      f'| fit source: {FIT_SOURCE} | fit n: {FIT_N} | fit seed: {FIT_SEED} | doc resamples: {N_BOOT} '
      f'| judge: {JUDGE_MODEL} '
      f'| out dir: {OUT_DIR}')


_ntp_cal_filename = 'ntp_calibrator.pkl' if _PROBE_VARIANT_KW is None else 'ntp_calibrator_noplatt.pkl'
_probe_filename = (
    'layer_probe.pkl' if PROBE_TYPE == 'layer'
    else ('head_probe.pkl' if _PROBE_VARIANT_KW is None else 'head_probe_noplatt.pkl')
)
ntp_cal_cache, probe_cache = {}, {}
for _train_ds in TRAIN_DATASETS:
    print(f'Loading trained probe/NTP calibrator ({_train_ds}, {JUDGE_MODEL}) '
          f'from {_INPUTS["datasets"][_train_ds]["syn_train_id"]}...')
    ntp_cal_cache[_train_ds] = load_probe_artifact(_INPUTS['datasets'][_train_ds]['probe_dir'], _ntp_cal_filename, _train_ds, JUDGE_MODEL)
    probe_cache[_train_ds]   = load_probe_artifact(_INPUTS['datasets'][_train_ds]['probe_dir'], _probe_filename, _train_ds, JUDGE_MODEL)

# Decompress each activation row once, keeping the union of all probes' top heads.
_HEAD_ACTS = (HeadActivationCache([lh for _tr in TRAIN_DATASETS for lh in probe_cache[_tr]['top_k_heads']])
              if PROBE_TYPE == 'head' else None)


# ── Real data, validated test rows and fit rows, per test dataset ────────────
test_data = {}
for ds in DATASETS:
    print(f'Loading test data for {ds}...')
    ds_inputs = _INPUTS['datasets'][ds]

    with open(ds_inputs['judge_combine_dir'] / 'combined.json') as f:
        real_df = pd.DataFrame(json.load(f))
    assert f'judgement_p_true_{JUDGE_MODEL}' in real_df.columns, (ds, JUDGE_MODEL)

    # Scored on the final.json datapoints; combined.json must be row-for-row the same run.
    with open(ds_inputs['extraction_dir'] / 'final.json') as f:
        final_records = json.load(f)
    final_df = pd.DataFrame(final_records)
    for col in ('measurement_id', 'document_id', 'attribute'):
        assert final_df[col].tolist() == real_df[col].tolist(), (
            f'{ds}: final.json and combined.json disagree on {col}')

    # Fit-row labels = judgement_combined, OR ground-truth match if use_matching_labels.
    jlabels = real_df['judgement_combined'].to_numpy(dtype=bool)
    if _PARAMS['datasets'][ds]['use_matching_labels']:
        _gt_df, ext_df, cached_edges = matching.load_cached_matching(
            _PARAMS['datasets'][ds]['extraction_id'], ds_inputs['ground_truth_path'])
        judged_edges = matching.edges_to_judged_rows(cached_edges, ext_df, real_df)
        ex_edge_exists = np.zeros(len(real_df), dtype=bool)
        for _gt_idx, ex_idx in judged_edges:
            ex_edge_exists[ex_idx] = True
        print(f'  {ds}: matching labels on -- {int(ex_edge_exists.sum())}/{len(real_df)} judged rows matched')
        labels = jlabels | ex_edge_exists
    else:
        print(f'  {ds}: matching labels off -- labels are judgement_combined alone')
        labels = jlabels.copy()

    # Human validations: test labels only. Each must match its final.json row on
    # document_id and every validated field (except `sampled`).
    _vpath = validations_path(ds)
    _vbytes = _vpath.read_bytes()
    _vart = json.loads(_vbytes)
    assert _vart['schema_version'] == 1 and _vart['dataset'] == ds, (ds, _vart['schema_version'], _vart['dataset'])
    _vrows = _vart['measurements']
    assert len(_vrows) > 0, ds
    assert {r['label'] for r in _vrows} <= {'valid', 'invalid'}, (ds, {r['label'] for r in _vrows})
    assert len({r['measurement_id'] for r in _vrows}) == len(_vrows), f'{ds}: validated measurement_id not unique'
    assert real_df['measurement_id'].is_unique, ds
    _pos_of = {int(m): i for i, m in enumerate(real_df['measurement_id'])}
    _final_by_mid = {int(r['measurement_id']): r for r in final_records}
    assert len(_final_by_mid) == len(final_records), ds
    for r in _vrows:
        assert r['measurement_id'] in _pos_of, (ds, 'validated measurement_id not in the judged rows', r['measurement_id'])
        assert real_df['document_id'].iloc[_pos_of[r['measurement_id']]] == r['document_id'], (ds, r['measurement_id'])
        rec = _final_by_mid[r['measurement_id']]
        for k, v in r['fields'].items():
            if k == 'sampled':
                continue
            assert k in rec and rec[k] == v, (ds, r['measurement_id'], k, v, rec.get(k, '<missing>'))
    _vorder = np.argsort([_pos_of[r['measurement_id']] for r in _vrows])
    val_pos = np.array([_pos_of[_vrows[i]['measurement_id']] for i in _vorder], dtype=np.int64)
    val_labels = np.array([_vrows[i]['label'] == 'valid' for i in _vorder], dtype=bool)
    val_flagged = np.array([_vrows[i]['flagged'] for i in _vorder], dtype=bool)
    val_example = np.array([_vrows[i]['example'] for i in _vorder], dtype=bool)
    assert (np.diff(val_pos) > 0).all()
    print(f'  {ds}: {len(val_pos)} human-validated rows ({int(val_labels.sum())} valid, '
          f'{int(val_flagged.sum())} flagged, {int(val_example.sum())} example; all kept)')

    # Fit pool = real rows from the probe's training documents. Always excluded from
    # the test rows, so every config scores the same test rows.
    pool_docs = set(probe_cache[ds]['syn_document_ids'])
    in_pool = real_df['document_id'].isin(pool_docs).to_numpy()
    pool_idx = np.where(in_pool)[0]
    test_idx = np.where(~in_pool)[0]
    assert len(test_idx) > 0, f'{ds}: no real test rows outside the fit pool'
    # Every validated row must already be a test row: none is dropped, so none of the
    # validated sample is lost to the probe's training documents.
    n_in_pool = int(in_pool[val_pos].sum())
    assert n_in_pool == 0, (
        f'{ds}: {n_in_pool}/{len(val_pos)} validated rows are in the probe-training documents '
        f'(the validation split does not match the probe split)')
    assert np.isin(val_pos, test_idx).all(), ds

    # Rows the maps are fit on (and pi_te read from); empty for manual.
    if FIT_SOURCE == 'sample':
        assert FIT_N <= len(pool_idx), f'{ds}: fit_n={FIT_N} > {len(pool_idx)} pool rows'
        fit_idx = pool_idx[uniform_fit_sample(len(pool_idx), FIT_N, FIT_SEED)]
        assert not set(fit_idx.tolist()) & set(test_idx.tolist()), ds
        assert not set(fit_idx.tolist()) & set(val_pos.tolist()), ds
    else:
        assert FIT_SOURCE == 'manual' and RECALIBRATION == 'prior_shift', (FIT_SOURCE, RECALIBRATION)
        fit_idx = np.array([], dtype=np.int64)
    pi_te = (float(_PARAMS['datasets'][ds]['pi_te_estimate']) if FIT_SOURCE == 'manual'
             else float(labels[fit_idx].mean()))
    if not 0 < pi_te < 1:
        raise ValueError(f'{ds}: fit rows are single-class (label rate {pi_te}); pick another fit_seed or fit_n')
    print(f'  {ds}: fit rows {len(fit_idx)} ({FIT_SOURCE}), pi_te / fit label rate = {pi_te:.4f} '
          f'| validated test rows {len(val_pos)}/{len(test_idx)} test rows (human valid rate {val_labels.mean():.4f}, '
          f'LLM+matching rate on them {labels[val_pos].mean():.4f}) '
          f'| pool rows {len(pool_idx)} (LLM+matching rate {labels[pool_idx].mean():.4f})')

    test_data[ds] = {
        'real_df': real_df,
        'labels': labels,
        'pool_docs': pool_docs,
        'test_idx': test_idx,
        'fit_idx': fit_idx,
        'pi_te': pi_te,
        'val_pos': val_pos,
        'val_labels': val_labels,
        'val_sha256': hashlib.sha256(_vbytes).hexdigest(),
    }


def _score_rows(train_ds, test_ds, mids, raw_ntp_probs, act_dir):
    """Unrecalibrated probe and NTP-calibrator probabilities for some rows.

    Args:
        train_ds: Dataset whose probe and calibrator are applied.
        test_ds: Dataset being scored (for error messages).
        mids: Measurement ids (activation keys).
        raw_ntp_probs: Judge p(true) per row.
        act_dir: Judge run directory holding the activations.

    Returns:
        ``(probe_probs, ntp_probs)``, one per row.
    """
    pd_data = probe_cache[train_ds]
    top = pd_data['top_layer'] if PROBE_TYPE == 'layer' else pd_data['top_k_heads']
    ntp_probs = ntp_cal_cache[train_ds]['calibrator'].predict_proba(raw_ntp_probs.reshape(-1, 1))[:, 1]
    if PROBE_TYPE == "layer":
        lo = np.load(act_dir / 'layer_outputs.npz')
        X = np.stack([np.array(lo[str(mid)], dtype=np.float32)[top] for mid in mids], axis=0)
    else:
        X = _HEAD_ACTS.features(act_dir, mids, top)
    assert X.shape[0] == len(mids), (X.shape, len(mids))
    probe_probs = pd_data['probe'].predict_proba(X)[:, 1]
    assert probe_probs.shape == ntp_probs.shape == (len(mids),), (train_ds, test_ds)
    assert np.isfinite(probe_probs).all() and np.isfinite(ntp_probs).all(), (train_ds, test_ds)
    return probe_probs, ntp_probs


def compute_predictions():
    """Score every (dataset type, train, test) cell, fit real-cell maps on LLM+matching
    labels, and save them.

    Writes predictions.pkl, recalibration_maps.csv and human_vs_llm_matching.csv.
    Real cells hold the human-validated rows only.

    Returns:
        ``{dtype: {judge: {train_ds: {test_ds: cell}}}}``. Each cell holds
        ``probe_probs`` / ``ntp_probs`` (evaluated), ``probe_raw`` / ``ntp_raw``,
        ``labels`` (human for real cells), ``recal_map`` (None for syn), ids, and row
        provenance and ``validation_sha256`` for real cells.
    """
    judge_model = JUDGE_MODEL
    setting_results = {dtype: {judge_model: {}} for dtype in _DTYPES}
    map_rows = []
    agree_rows = []

    for train_ds in TRAIN_DATASETS:
        pd_data = probe_cache[train_ds]
        for dataset_type in _DTYPES:
            setting_results[dataset_type][judge_model][train_ds] = {}
            for test_ds in DATASETS:
                if dataset_type == 'syn':
                    syn_dir = _INPUTS['datasets'][test_ds]['syn_test_dir']
                    with open(syn_dir / 'responses.json') as f:
                        syn_df_s = pd.DataFrame(json.load(f))
                    mids    = syn_df_s['measurement_id'].tolist()
                    labels  = (syn_df_s['label'] == 'valid').to_numpy(dtype=bool)
                    doc_ids = syn_df_s['document_id'].to_numpy()
                    assert pd.notna(doc_ids).all(), f'{test_ds}: synthetic rows without document_id'
                    probe_raw, ntp_raw = _score_rows(
                        train_ds, test_ds, mids, syn_df_s['judgement_p_true'].to_numpy(), syn_dir)
                    probe_probs, ntp_probs = probe_raw, ntp_raw
                    cell = {'recal_map': None,
                            'measurement_ids': np.asarray(mids), 'document_ids': doc_ids}
                else:  # real
                    td      = test_data[test_ds]
                    real_df = td['real_df']
                    col     = f'judgement_p_true_{judge_model}'
                    act_dir = _INPUTS['datasets'][test_ds]['judge_interp_dir']

                    # Also exclude train_ds's training documents; asserted to remove nothing
                    # extra, so test rows depend only on test_ds.
                    exclude = td['pool_docs'] | set(pd_data['syn_document_ids'])
                    test_rows = np.where(~real_df['document_id'].isin(exclude).to_numpy())[0]
                    assert np.array_equal(test_rows, td['test_idx']), (
                        f'{train_ds} probe -> {test_ds}: excluding the probe\'s training documents '
                        f'changed the test rows ({len(test_rows)} vs {len(td["test_idx"])})')
                    # Scored rows = every validated row. None is dropped: all are test rows
                    # for this train dataset too, and none is a fit row.
                    idx = td['val_pos']
                    assert np.isin(idx, test_rows).all(), (
                        f'{train_ds} probe -> {test_ds}: validated rows fall in the excluded documents')
                    assert not real_df['document_id'].iloc[idx].isin(exclude).any(), (train_ds, test_ds)
                    assert not set(idx.tolist()) & set(td['fit_idx'].tolist()), (train_ds, test_ds)
                    mids   = real_df['measurement_id'].iloc[idx].tolist()
                    labels = td['val_labels']
                    auto_labels = td['labels'][idx]
                    assert len(idx) == len(labels) == len(auto_labels), (train_ds, test_ds)
                    print(f'  real test {train_ds} probe -> {test_ds}: {len(idx)} validated rows scored, 0 dropped '
                          f'({len(test_rows)}/{len(real_df)} real rows are test rows)')
                    agree_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds, 'N': len(idx),
                                       'Human valid rate': float(labels.mean()),
                                       'LLM+matching valid rate': float(auto_labels.mean()),
                                       'LLM+matching TP': int((auto_labels & labels).sum()),
                                       'LLM+matching FP': int((auto_labels & ~labels).sum()),
                                       'LLM+matching FN': int((~auto_labels & labels).sum()),
                                       'LLM+matching TN': int((~auto_labels & ~labels).sum())})
                    probe_raw, ntp_raw = _score_rows(train_ds, test_ds, mids, real_df[col].iloc[idx].to_numpy(), act_dir)
                    raw = {'probe': probe_raw, 'ntp': ntp_raw}

                    fit_idx = td['fit_idx']
                    fit_labels = td['labels'][fit_idx]  # LLM + matching: the pool has no human labels
                    if RECALIBRATION in ('intercept_fit', 'platt_fit'):
                        f_probe, f_ntp = _score_rows(
                            train_ds, test_ds, real_df['measurement_id'].iloc[fit_idx].tolist(),
                            real_df[col].iloc[fit_idx].to_numpy(), act_dir)
                        fit_raw = {'probe': f_probe, 'ntp': f_ntp}

                    pi_tr = {'probe': pd_data['train_prevalence'],
                             'ntp': ntp_cal_cache[train_ds]['train_prevalence']}
                    maps = {}
                    for meth in ('probe', 'ntp'):
                        if RECALIBRATION == 'prior_shift':
                            coef, icpt = prior_shift_map(td['pi_te'], pi_tr[meth])
                            if FIT_SOURCE != 'manual':
                                # Known answer: the library's prior_shift fit on the same rows.
                                ref = fit_prior_shift(fit_labels, pi_tr[meth])
                                assert ref == (coef, icpt), (train_ds, test_ds, meth, ref, (coef, icpt))
                        elif RECALIBRATION == 'intercept_fit':
                            # Asserts its score equation: mapped fit rows average to their label rate.
                            coef, icpt = intercept_fit_map(fit_raw[meth], fit_labels)
                        else:
                            assert RECALIBRATION == 'platt_fit', RECALIBRATION
                            # Asserts both Platt score equations; non-convergence is an error.
                            coef, icpt = platt_fit_map(fit_raw[meth], fit_labels)
                        maps[meth] = (coef, icpt)
                        mapped = apply_platt(raw[meth], coef, icpt)
                        map_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds, 'Method': meth,
                                         'Recalibration': RECALIBRATION, 'Fit source': FIT_SOURCE,
                                         'Fit n': len(fit_idx), 'Fit seed': FIT_SEED, 'pi_te': td['pi_te'],
                                         'Train prevalence': float(pi_tr[meth]), 'coef': coef, 'intercept': icpt,
                                         'N test': len(idx), 'Test label rate': float(labels.mean()),
                                         'LLM+matching label rate on test rows': float(auto_labels.mean()),
                                         'Raw mean prob': float(raw[meth].mean()),
                                         'Mapped mean prob': float(mapped.mean())})
                    probe_probs = apply_platt(probe_raw, *maps['probe'])
                    ntp_probs   = apply_platt(ntp_raw, *maps['ntp'])
                    cell = {'recal_map': maps, 'validation_sha256': td['val_sha256']}
                    cell.update(real_cell_provenance(
                        real_df, idx, fit_idx, exclude,
                        _INPUTS['datasets'][test_ds]['extraction_dir'] / 'final.json',
                        _INPUTS['datasets'][test_ds]['judge_combine_dir'] / 'combined.json', CONFIG_ID, SEED))

                assert len(mids) == len(labels) > 0, (dataset_type, train_ds, test_ds, len(mids), len(labels))
                assert probe_probs.shape == ntp_probs.shape == labels.shape, (
                    probe_probs.shape, ntp_probs.shape, labels.shape)
                assert np.isfinite(probe_probs).all() and np.isfinite(ntp_probs).all(), (train_ds, test_ds)
                cell.update(probe_probs=probe_probs, ntp_probs=ntp_probs, labels=labels,
                            probe_raw=probe_raw, ntp_raw=ntp_raw)
                setting_results[dataset_type][judge_model][train_ds][test_ds] = cell

    maps_df = pd.DataFrame(map_rows)
    assert len(maps_df) == len(TRAIN_DATASETS) * len(DATASETS) * 2, len(maps_df)
    maps_df.to_csv(OUT_DIR / 'recalibration_maps.csv', index=False)
    print(maps_df.to_string(index=False, float_format='{:.4f}'.format))

    agree_df = pd.DataFrame(agree_rows)
    assert len(agree_df) == len(TRAIN_DATASETS) * len(DATASETS), len(agree_df)
    agree_df.to_csv(OUT_DIR / 'human_vs_llm_matching.csv', index=False)
    print('\nLLM+matching labels vs human labels on the scored rows:')
    print(agree_df.to_string(index=False, float_format='{:.3f}'.format))

    with open(OUT_DIR / 'predictions.pkl', 'wb') as f:
        pickle.dump(setting_results, f)
    return setting_results


# (display name, method key, linestyle). Probe and NTP are plotted and tabled separately.
_METHODS = [
    ('Probe', 'probe', '-'),
    ('NTP',   'ntp',   '--'),
]


def plot_calibration_curves(boot, dtype):
    """Save one reliability diagram per (method, train dataset), one curve per test dataset.

    Args:
        boot: Output of ``bootstrap_calibration``.
        dtype: ``"syn"`` or ``"real"``.
    """
    for method, key, linestyle in _METHODS:
        for train_ds in DATASETS:
            train_dict = boot[dtype][JUDGE_MODEL][train_ds]
            assert set(train_dict) == set(DATASETS), (dtype, train_ds, sorted(train_dict))

            fig_cal, ax_cal = plt.subplots(figsize=(4.0, 3.8))
            ax_cal.plot([0, 1], [0, 1], 'k:', lw=1.0, alpha=0.5, zorder=1)
            for test_ds in DATASETS:
                draw_reliability_curve(
                    ax_cal, train_dict[test_ds][key], _DS_COLORS[test_ds],
                    linestyle=linestyle, lw=2.5, line_zorder=3, band_zorder=1,
                )
            ax_cal.set_xlim(-0.02, 1.02)
            ax_cal.set_ylim(-0.02, 1.02)
            ax_cal.set_xlabel('Predicted Probability')
            if method == 'NTP':
                ax_cal.set_ylabel('Observed Frequency')
            ax_cal.set_title(method, fontsize=15, style='italic')
            ax_cal.grid(alpha=0.25, linestyle='-', linewidth=0.4)
            ax_cal.set_axisbelow(True)
            fig_cal.tight_layout()
            fig_cal.savefig(
                FIGURES_DIR / f'cal_{dtype}_{method.lower()}_train-{train_ds}.pdf',
                bbox_inches='tight', dpi=200,
            )
            plt.close(fig_cal)


def bootstrap_calibration(setting_results):
    """Document-bootstrap calibration summaries for every cell and method.

    Resamples depend only on (dtype, test_ds), so cells on the same evaluation set are paired.

    Args:
        setting_results: Output of ``compute_predictions``.

    Returns:
        ``{dtype: {judge: {train_ds: {test_ds: {method: doc_bootstrap_calibration summary}}}}}``.
    """
    out, resamples = {}, {}
    for dtype, by_judge in setting_results.items():
        out[dtype] = {JUDGE_MODEL: {}}
        for train_ds, by_test in by_judge[JUDGE_MODEL].items():
            out[dtype][JUDGE_MODEL][train_ds] = {}
            for test_ds, cell in by_test.items():
                mids = np.asarray(cell['measurement_ids'])
                if (dtype, test_ds) not in resamples:
                    resamples[dtype, test_ds] = (mids, db.document_resamples(
                        cell['document_ids'], N_BOOT, db.resample_rng(SEED, dtype, test_ds)))
                # Resamples index rows, so every cell on this evaluation set must have the same rows.
                assert np.array_equal(resamples[dtype, test_ds][0], mids), (dtype, train_ds, test_ds)
                print(f'  document bootstrap {dtype} {train_ds} -> {test_ds}: {N_BOOT} resamples')
                out[dtype][JUDGE_MODEL][train_ds][test_ds] = {
                    key: db.doc_bootstrap_calibration(cell[f'{key}_probs'], cell['labels'], resamples[dtype, test_ds][1])
                    for _, key, _ in _METHODS
                }
    return out


def _threshold_metrics(probs, labels, threshold=0.5):
    """Classification metrics at a threshold, plus AUROC.

    Undefined metrics are NaN; ``validity`` is 0.0 when nothing is predicted positive.

    Args:
        probs: Predicted probabilities.
        labels: Binary labels.
        threshold: Predicted positive if prob > threshold.

    Returns:
        Dict with ``acc``, ``prec``, ``rec``, ``f1``, ``auroc``, ``validity``.
    """
    y = np.asarray(labels, dtype=bool)
    pred = np.asarray(probs) > threshold
    tp, fp = int((pred & y).sum()), int((pred & ~y).sum())
    fn, tn = int((~pred & y).sum()), int((~pred & ~y).sum())
    prec = tp / (tp + fp) if tp + fp else float('nan')
    rec = tp / (tp + fn) if tp + fn else float('nan')
    return dict(acc=(tp + tn) / len(y), prec=prec, rec=rec,
                f1=2 * prec * rec / (prec + rec) if prec + rec > 0 else float('nan'),
                auroc=roc_auc_score(y, probs) if 0 < y.sum() < len(y) else float('nan'),
                validity=validity_rate_from_labels(y, pred))


def compute_metrics(setting_results, boot):
    """Build the metrics table: one row per (dtype, train_ds, test_ds, method).

    Calibration errors carry document-bootstrap intervals; threshold metrics and
    AUROC are on the full evaluation set. Recalibration fields are None/NaN for syn.

    Args:
        setting_results: Output of ``compute_predictions``.
        boot: Output of ``bootstrap_calibration``.

    Returns:
        Metrics DataFrame.
    """
    rows = []
    for dtype in setting_results:
        for train_ds, by_test in setting_results[dtype][JUDGE_MODEL].items():
            for test_ds, rdict in by_test.items():
                for kind, key, _ in _METHODS:
                    t = _threshold_metrics(rdict[f'{key}_probs'], rdict['labels'])
                    b = boot[dtype][JUDGE_MODEL][train_ds][test_ds][key]
                    assert b['n_boot'] == N_BOOT, b['n_boot']
                    mapped = rdict['recal_map'] is not None
                    row = {
                        'Dataset type':  dtype,
                        'Judge model':   JUDGE_MODEL,
                        'Train dataset': train_ds,
                        'Test dataset':  test_ds,
                        'Type':          kind,
                        'N':             len(rdict['labels']),
                        'N docs':        len(np.unique(rdict['document_ids'])),
                        'Label rate':    float(np.mean(rdict['labels'])),
                        'Recalibration': RECALIBRATION if mapped else None,
                        'Fit source':    FIT_SOURCE if mapped else None,
                        'Fit n':         len(test_data[test_ds]['fit_idx']) if mapped else np.nan,
                        'Fit seed':      FIT_SEED if mapped else None,
                        'pi_te':         test_data[test_ds]['pi_te'] if mapped else np.nan,
                        'Intercept':     rdict['recal_map'][key][1] if mapped else np.nan,
                        'Mean prob':     float(np.mean(rdict[f'{key}_probs'])),
                        'Doc resamples': b['n_boot'],
                        'Accuracy':      t['acc'],
                        'Precision':     t['prec'],
                        'Recall':        t['rec'],
                        'F1':            t['f1'],
                        'AUROC':         t['auroc'],
                    }
                    for m in db.METRICS:
                        row.update({m: b['point'][m], f'{m}_lo': b['lo'][m], f'{m}_hi': b['hi'][m]})
                    row.update({'Validity': t['validity'], 'relplot sigma': b['sigma']})
                    rows.append(row)
    df = pd.DataFrame(rows)
    n_expected = len(_DTYPES) * len(TRAIN_DATASETS) * len(DATASETS) * len(_METHODS)
    assert len(df) == n_expected, (len(df), n_expected)
    assert not df.duplicated(['Dataset type', 'Train dataset', 'Test dataset', 'Type']).any()
    return df


if __name__ == "__main__":
    setting_results = compute_predictions()
    print('Document bootstrap...')
    boot = bootstrap_calibration(setting_results)
    with open(OUT_DIR / 'bootstrap.pkl', 'wb') as f:
        pickle.dump(boot, f)
    for _dt in _DTYPES:
        plot_calibration_curves(boot, dtype=_dt)
    metrics_df = compute_metrics(setting_results, boot)
    for _kind, _, _ in _METHODS:
        _sub = metrics_df[metrics_df['Type'] == _kind]
        print(f'\n=== {_kind} ===')
        print(_sub.to_string(index=False, float_format='{:.3f}'.format))
        _sub.to_csv(OUT_DIR / f'metrics_{_kind.lower()}.csv', index=False)

    # ── Standalone legend: curve color = TEST dataset ─────────────────────────
    _legend_handles = [
        mlines.Line2D([], [], color=_DS_COLORS[ds], lw=2, marker='o', ms=3.5, label=_DS_LABELS[ds])
        for ds in DATASETS
    ]
    _fig_leg, _ax_leg = plt.subplots(figsize=(6.0, 0.45))
    _ax_leg.axis('off')
    _ax_leg.legend(handles=_legend_handles, loc='center', ncol=len(DATASETS), fontsize=13,
                   frameon=False, handlelength=2.0, title='Test set', title_fontsize=13)
    _fig_leg.savefig(FIGURES_DIR / 'legend_calibration.pdf', bbox_inches='tight', dpi=200)
    plt.close(_fig_leg)
