"""Probe and NTP calibration with real test cells scored against human-validated labels.

Same cells as calibration.py, but uses the v3 scheme: real cells are recalibrated on
n_fit_samples resampled fit samples of platt_n rows (fit labels are LLM + matching,
since the pool has no human labels), and only human-validated test rows are scored.
CIs come from the nested bootstrap (common/nested_bootstrap.py). Runs at import time.

Outputs in analysis/results/calibration-validated/<config id>/: predictions.pkl,
nested_bootstrap.pkl, platt_fits.csv, human_vs_llm_matching.csv,
metrics_{probe,ntp}.csv, figures/.

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

from analysis.common.config import analysis_results_dir, load_calibration_validated_config, validations_path
from analysis.common import calibration_ids as cids
from analysis.common import matching
from analysis.common import nested_bootstrap as nb
from analysis.common.provenance import sha256_file
from analysis.common.prediction_store import PROVENANCE_KEYS, real_cell_provenance
from analysis.common.calibration_plot_utils import draw_reliability_curve
from analysis.common.head_activations import HeadActivationCache
from analysis.common.loaders import load_probe_artifact
from scholarlm.utils.calibration import (
    apply_platt, fit_recalibration, RECALIBRATION_METHODS,
)
from analysis.common.config import RECALIBRATION_METHODS as _CFG_RECALIBRATION_METHODS

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
def _parse_args():
    """Parse the single positional calibration-validated config path."""
    parser = argparse.ArgumentParser(
        description="Probe/NTP calibration analysis (v3 recalibration, but real test cells are scored against human-validated labels)."
    )
    parser.add_argument('config', type=Path,
                        help="analysis/analysis-configs/calibration-validated/<id>.yaml (see load_calibration_validated_config)")
    return parser.parse_args()


_CFG = load_calibration_validated_config(_parse_args().config)
CONFIG_ID = _CFG['id']
SEED = _CFG['seed']
_PARAMS = _CFG['params']

PROBE_TYPE   = _PARAMS['probe_type']
PROBE_VARIANT = _PARAMS['probe_variant']
SYN_SPLIT    = _PARAMS['syn_split']
# Rows per recalibration fit sample (LLM + matching labels).
PLATT_N = _PARAMS['platt_n']
# Nested bootstrap: N_FIT_SAMPLES fit samples x N_DOC_BOOT test-document resamples for
# real cells; N_SYN_BOOT document resamples for synthetic cells.
N_FIT_SAMPLES = _PARAMS['n_fit_samples']
N_DOC_BOOT    = _PARAMS['n_doc_boot']
N_SYN_BOOT    = _PARAMS['n_syn_boot']
# Recalibration method (platt_fit / intercept_fit / prior_shift), as
# expit(coef * logit(p) + intercept); see common/recalibration.py.
RECALIBRATION = _PARAMS['recalibration']
assert tuple(RECALIBRATION_METHODS) == tuple(_CFG_RECALIBRATION_METHODS), (RECALIBRATION_METHODS, _CFG_RECALIBRATION_METHODS)
assert RECALIBRATION in RECALIBRATION_METHODS, RECALIBRATION
DATASETS = list(_PARAMS['datasets'])
TRAIN_DATASETS = list(_PARAMS['datasets'])  # every validated dataset has its own synthetic-probe config

_INPUTS = cids.resolve_calibration_inputs(_CFG)
JUDGE_MODEL  = _INPUTS['judge_model']
JUDGE_MODELS = [JUDGE_MODEL]  # kept as a list: every plot/metrics loop below is judge_model-indexed

OUT_DIR = analysis_results_dir("calibration-validated") / CONFIG_ID
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# None selects the Platt-scaled filenames; 'noplatt' the suffixed ones.
_PROBE_VARIANT_KW = None if PROBE_VARIANT == 'platt' else PROBE_VARIANT

_DTYPES = ['syn', 'real']

print(f'[calibration validated] config: {CONFIG_ID} | probe type: {PROBE_TYPE} | probe variant: {PROBE_VARIANT} '
      f'| syn split: {SYN_SPLIT} | datasets: {DATASETS} | platt_n: {PLATT_N} | recalibration: {RECALIBRATION} '
      f'| fit samples: {N_FIT_SAMPLES} x doc resamples: {N_DOC_BOOT} | syn doc resamples: {N_SYN_BOOT} '
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
    ntp_cal_cache[_train_ds] = {JUDGE_MODEL: load_probe_artifact(_INPUTS['datasets'][_train_ds]['probe_dir'], _ntp_cal_filename, _train_ds, JUDGE_MODEL)}
    probe_cache[_train_ds]   = {JUDGE_MODEL: load_probe_artifact(_INPUTS['datasets'][_train_ds]['probe_dir'], _probe_filename, _train_ds, JUDGE_MODEL)}

# Decompress each activation row once, keeping the union of all probes' top heads.
_HEAD_ACTS = (HeadActivationCache([lh for _tr in TRAIN_DATASETS for lh in probe_cache[_tr][JUDGE_MODEL]['top_k_heads']])
              if PROBE_TYPE == 'head' else None)


# Load each test dataset once: rows, labels, human validations and fit samples.
test_data = {}
for ds in DATASETS:
    print(f'Loading test data for {ds}...')
    ds_inputs = _INPUTS['datasets'][ds]

    with open(ds_inputs['judge_combine_dir'] / 'combined.json') as f:
        real_df = pd.DataFrame(json.load(f))
    assert f'judgement_p_true_{JUDGE_MODEL}' in real_df.columns, (ds, JUDGE_MODEL)

    # Scored on final.json rows; combined.json must match them row for row.
    with open(ds_inputs['extraction_dir'] / 'final.json') as f:
        final_records = json.load(f)
    final_df = pd.DataFrame(final_records)
    for col in ('measurement_id', 'document_id', 'attribute'):
        assert final_df[col].tolist() == real_df[col].tolist(), (
            f'{ds}: final.json and combined.json disagree on {col}')

    jlabels = real_df['judgement_combined'].to_numpy(dtype=bool)
    # Label = judge, OR ground-truth match if use_matching_labels (from the verified
    # match cache; a row matches if any of its postprocessed children did).
    if _PARAMS['datasets'][ds]['use_matching_labels']:
        _gt_df, ext_df, cached_edges = matching.load_cached_matching(
            _PARAMS['datasets'][ds]['extraction_id'], ds_inputs['ground_truth_path'])
        judged_edges = matching.edges_to_judged_rows(cached_edges, ext_df, real_df)
        ex_edge_exists = np.zeros(len(real_df), dtype=bool)
        for _gt_idx, ex_idx in judged_edges:
            ex_edge_exists[ex_idx] = True
        print(f'  {ds}: matching labels on -- {int(ex_edge_exists.sum())}/{len(real_df)} judged rows matched')
        combined_labels = jlabels | ex_edge_exists
    else:
        print(f'  {ds}: matching labels off -- labels are judgement_combined alone')
        combined_labels = jlabels.copy()

    # Human validations: test labels only. Each must match its final.json row on
    # document_id and every validated field (except `sampled`).
    _vpath = validations_path(ds)
    _vart = json.loads(_vpath.read_bytes())
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

    # Fit pool = real rows from the probe's training documents, excluded from every
    # test cell. Each fit sample resamples the pool's documents, then draws PLATT_N
    # rows (see nb.pool_resampled_fit_sample). predictions.pkl stores sample 0.
    pool_docs = set(probe_cache[ds][JUDGE_MODEL]['syn_document_ids'])
    pool_idx = np.where(real_df['document_id'].isin(pool_docs).to_numpy())[0]
    assert len(pool_idx) >= PLATT_N, f'{ds}: Platt pool has {len(pool_idx)} rows < platt_n={PLATT_N}'
    fit_idx = []
    for r in range(N_FIT_SAMPLES):
        sample = pool_idx[nb.pool_resampled_fit_sample(
            real_df['document_id'].to_numpy()[pool_idx], PLATT_N, nb.fit_sample_rng(SEED, r))]
        assert len(sample) == PLATT_N and set(sample.tolist()) <= set(pool_idx.tolist())
        assert 0 < combined_labels[sample].sum() < PLATT_N, (
            f'{ds}: Platt sample {r} is single-class ({int(combined_labels[sample].sum())}/{PLATT_N} valid)')
        fit_idx.append(sample)
    platt_idx = fit_idx[0]
    _rates = np.array([combined_labels[i].mean() for i in fit_idx])
    print(f'  {ds}: {N_FIT_SAMPLES} Platt samples of {PLATT_N} rows from a resampled pool of '
          f'{len(pool_idx)} rows / {len(pool_docs & set(real_df["document_id"]))} docs; label rate over samples '
          f'mean {_rates.mean():.3f}, range {_rates.min():.2f}-{_rates.max():.2f} '
          f'(pool row rate {combined_labels[pool_idx].mean():.3f})')

    test_data[ds] = {
        'real_df': real_df,
        'labels': combined_labels,
        'judge_labels': jlabels,
        'pool_docs': pool_docs,
        'platt_idx': platt_idx,
        'fit_idx': fit_idx,
        'val_pos': val_pos,
        'val_labels': val_labels,
        'val_sha256': hashlib.sha256(_vpath.read_bytes()).hexdigest(),
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
    pd_data = probe_cache[train_ds][JUDGE_MODEL]
    ntp_cal_data = ntp_cal_cache[train_ds][JUDGE_MODEL]
    top = pd_data['top_layer'] if PROBE_TYPE == 'layer' else pd_data['top_k_heads']
    ntp_probs = ntp_cal_data['calibrator'].predict_proba(raw_ntp_probs.reshape(-1, 1))[:, 1]
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


def compute_predictions(load_from_precomputed=False):
    """Score every cell, fit one map per real fit sample, and save predictions.pkl.

    Also writes platt_fits.csv and human_vs_llm_matching.csv.

    Args:
        load_from_precomputed: Reuse an existing predictions.pkl after checking it
            still matches the inputs and config.

    Returns:
        ``{dtype: {judge: {train_ds: {test_ds: cell}}}}``. Real cells hold
        ``fit_maps[method][r]`` (one per fit sample), ``platt`` (sample 0's map),
        evaluated probs from sample 0, raw scores, human labels and provenance.
        Syn cells have ``platt`` and ``fit_maps`` = None.
    """

    cache_file = OUT_DIR / 'predictions.pkl'

    if load_from_precomputed and cache_file.exists():
        print(f'Loading precomputed predictions from {cache_file}...')
        with open(cache_file, 'rb') as f:
            loaded = pickle.load(f)
        for _tr, _by_test in loaded['real'][JUDGE_MODEL].items():
            for _te, _cell in _by_test.items():
                missing = [k for k in PROVENANCE_KEYS if k not in _cell]
                assert not missing, f'{cache_file} predates row provenance (cell {_tr}->{_te} lacks {missing}); rerun'
                assert _cell['final_sha256'] == sha256_file(_INPUTS['datasets'][_te]['extraction_dir'] / 'final.json'), (
                    f'{cache_file}: final.json changed since it was built ({_tr}->{_te})')
                assert _cell['calibration_config_id'] == CONFIG_ID, (_tr, _te)
                assert _cell['recalibration'] == RECALIBRATION, (
                    f'{cache_file}: built with recalibration {_cell["recalibration"]!r} != {RECALIBRATION!r}')
                assert 'fit_maps' in _cell, f'{cache_file} predates the nested bootstrap; rerun'
                assert all(len(v) == N_FIT_SAMPLES for v in _cell['fit_maps'].values()), (
                    f'{cache_file}: built with a different n_fit_samples ({_tr}->{_te})')
                assert _cell['validation_sha256'] == test_data[_te]['val_sha256'], (
                    f'{cache_file}: validations changed since it was built ({_tr}->{_te})')
        return loaded

    judge_model = JUDGE_MODEL
    setting_results = {dtype: {judge_model: {}} for dtype in _DTYPES}
    fit_rows = []
    agree_rows = []

    for train_ds in TRAIN_DATASETS:
        pd_data = probe_cache[train_ds][judge_model]

        for dataset_type in _DTYPES:
            setting_results[dataset_type][judge_model][train_ds] = {}

            for test_ds in DATASETS:
                platt = fit_maps = None
                if dataset_type == 'syn':
                    syn_dir = _INPUTS['datasets'][test_ds]['syn_test_dir']
                    with open(syn_dir / 'responses.json') as f:
                        syn_resp = json.load(f)
                    syn_df_s = pd.DataFrame(syn_resp)
                    mids     = syn_df_s['measurement_id'].tolist()
                    labels   = (syn_df_s['label'] == 'valid').to_numpy(dtype=bool)
                    raw_ntp_probs = syn_df_s['judgement_p_true'].to_numpy()
                    doc_ids  = syn_df_s['document_id'].to_numpy()
                    assert pd.notna(doc_ids).all(), f'{test_ds}: synthetic rows without document_id'
                    probe_raw, ntp_raw = _score_rows(train_ds, test_ds, mids, raw_ntp_probs, syn_dir)
                    probe_probs, ntp_probs = probe_raw, ntp_raw
                else:  # real
                    td       = test_data[test_ds]
                    real_df  = td['real_df']
                    act_dir  = _INPUTS['datasets'][test_ds]['judge_interp_dir']
                    col      = f'judgement_p_true_{judge_model}'

                    # One map per fit sample and method. pi_tr (prior_shift only) is the
                    # scorer's synthetic training prevalence. Fit rows are scored once.
                    pi = td['platt_idx']
                    fit_union = np.unique(np.concatenate(td['fit_idx']))
                    u_probe, u_ntp = _score_rows(
                        train_ds, test_ds, real_df['measurement_id'].iloc[fit_union].tolist(),
                        real_df[col].iloc[fit_union].to_numpy(), act_dir)
                    pi_tr = {'probe': pd_data['train_prevalence'],
                             'ntp': ntp_cal_cache[train_ds][judge_model]['train_prevalence']}
                    fit_maps = {'probe': [], 'ntp': []}
                    for r, sample in enumerate(td['fit_idx']):
                        at = np.searchsorted(fit_union, sample)
                        assert np.array_equal(fit_union[at], sample), (train_ds, test_ds, r)
                        p_raw = {'probe': u_probe[at], 'ntp': u_ntp[at]}
                        p_labels = td['labels'][sample]
                        for meth in ('probe', 'ntp'):
                            coef, icpt = fit_recalibration(RECALIBRATION, p_raw[meth], p_labels, pi_tr[meth])
                            p_scaled_mean = float(apply_platt(p_raw[meth], coef, icpt).mean())
                            if RECALIBRATION != 'platt_fit':
                                assert coef == 1.0, (RECALIBRATION, meth, coef)
                            if RECALIBRATION == 'intercept_fit':
                                # Score equation of the intercept MLE (known answer, solved to ~1e-12).
                                assert abs(p_scaled_mean - p_labels.mean()) < 1e-8, (
                                    train_ds, test_ds, meth, r, p_scaled_mean, p_labels.mean())
                            fit_maps[meth].append((coef, icpt))
                            fit_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds, 'Method': meth,
                                             'Fit sample': r, 'Recalibration': RECALIBRATION,
                                             'coef': coef, 'intercept': icpt, 'N': len(sample),
                                             'Label rate': float(p_labels.mean()),
                                             'Train prevalence': float(pi_tr[meth]),
                                             'Raw mean prob': float(p_raw[meth].mean()),
                                             'Scaled mean prob': p_scaled_mean})
                    platt = {meth: fit_maps[meth][0] for meth in fit_maps}

                    # Test rows: outside test_ds's Platt pool and train_ds's probe-training docs.
                    exclude = td['pool_docs'] | set(pd_data['syn_document_ids'])
                    idx_all = np.where(~real_df['document_id'].isin(exclude).to_numpy())[0]
                    assert len(idx_all) > 0, f'{train_ds} probe -> {test_ds}: no real test rows'
                    assert not set(idx_all.tolist()) & set(fit_union.tolist()), (train_ds, test_ds)

                    # Score only validated rows that are test rows; the rest are dropped
                    # (their documents were seen in training) and the count is printed.
                    in_test = np.isin(td['val_pos'], idx_all)
                    idx = td['val_pos'][in_test]
                    n_dropped = int((~in_test).sum())
                    assert len(idx) > 0, f'{train_ds} probe -> {test_ds}: no validated test rows'
                    assert len(idx) + n_dropped == len(td['val_pos'])
                    assert (np.diff(idx) > 0).all() and not set(idx.tolist()) & set(fit_union.tolist()), (train_ds, test_ds)
                    assert not real_df['document_id'].iloc[idx].isin(exclude).any(), (train_ds, test_ds)
                    print(f'  real test split {train_ds} probe -> {test_ds}: {len(idx)} validated test rows '
                          f'({n_dropped}/{len(td["val_pos"])} validated rows dropped: in Platt pool or probe-training docs; '
                          f'{len(idx_all)}/{len(real_df)} real rows are test rows)')

                    mids     = real_df['measurement_id'].iloc[idx].tolist()
                    labels   = td['val_labels'][in_test]
                    auto_labels = td['labels'][idx]
                    agree_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds,
                                       'N validated': len(td['val_pos']), 'N dropped': n_dropped, 'N': len(idx),
                                       'Human valid rate': float(labels.mean()),
                                       'LLM+matching valid rate': float(auto_labels.mean()),
                                       'LLM+matching TP': int((auto_labels & labels).sum()),
                                       'LLM+matching FP': int((auto_labels & ~labels).sum()),
                                       'LLM+matching FN': int((~auto_labels & labels).sum()),
                                       'LLM+matching TN': int((~auto_labels & ~labels).sum())})
                    probe_raw, ntp_raw = _score_rows(
                        train_ds, test_ds, mids, real_df[col].iloc[idx].to_numpy(), act_dir)
                    probe_probs = apply_platt(probe_raw, *platt['probe'])
                    ntp_probs   = apply_platt(ntp_raw, *platt['ntp'])

                assert len(mids) == len(labels) > 0, (dataset_type, train_ds, test_ds, len(mids), len(labels))
                assert probe_probs.shape == ntp_probs.shape == labels.shape, (
                    probe_probs.shape, ntp_probs.shape, labels.shape)
                assert np.isfinite(probe_probs).all() and np.isfinite(ntp_probs).all(), (train_ds, test_ds)
                setting_results[dataset_type][judge_model][train_ds][test_ds] = {
                    'probe_probs': probe_probs, 'ntp_probs': ntp_probs, 'labels': labels,
                    'probe_raw': probe_raw, 'ntp_raw': ntp_raw, 'fit_maps': fit_maps,
                    'platt': platt, 'recalibration': None if platt is None else RECALIBRATION,
                }
                if dataset_type == 'syn':
                    setting_results[dataset_type][judge_model][train_ds][test_ds].update(
                        measurement_ids=np.asarray(mids), document_ids=doc_ids)
                else:
                    setting_results[dataset_type][judge_model][train_ds][test_ds].update(real_cell_provenance(
                        real_df, idx, pi, exclude,
                        _INPUTS['datasets'][test_ds]['extraction_dir'] / 'final.json',
                        _INPUTS['datasets'][test_ds]['judge_combine_dir'] / 'combined.json', CONFIG_ID, SEED))
                    setting_results[dataset_type][judge_model][train_ds][test_ds]['validation_sha256'] = td['val_sha256']

    fits_df = pd.DataFrame(fit_rows)
    assert len(fits_df) == len(TRAIN_DATASETS) * len(DATASETS) * 2 * N_FIT_SAMPLES, len(fits_df)
    fits_df.to_csv(OUT_DIR / 'platt_fits.csv', index=False)
    print(fits_df[fits_df['Fit sample'] == 0].to_string(index=False, float_format='{:.3f}'.format))
    print(fits_df.groupby(['Train dataset', 'Test dataset', 'Method'])[['coef', 'intercept', 'Label rate']]
          .agg(['mean', 'std', 'min', 'max']).to_string(float_format='{:.3f}'.format))

    agree_df = pd.DataFrame(agree_rows)
    assert len(agree_df) == len(TRAIN_DATASETS) * len(DATASETS), len(agree_df)
    agree_df.to_csv(OUT_DIR / 'human_vs_llm_matching.csv', index=False)
    print('\nLLM+matching labels vs human labels on the scored rows:')
    print(agree_df.to_string(index=False, float_format='{:.3f}'.format))

    print(f'Saving predictions to {cache_file}...')
    with open(cache_file, 'wb') as f:
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
        boot: Output of ``nb.bootstrap_cells``.
        dtype: ``"syn"`` or ``"real"``.
    """
    for judge_model in JUDGE_MODELS:
        for method, key, linestyle in _METHODS:
            for train_ds in DATASETS:
                train_dict = boot[dtype][judge_model][train_ds]
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
                plt.show()


def compute_metrics(setting_results, boot):
    """Build the metrics table: one row per (dtype, train_ds, test_ds, method).

    Calibration errors are nested-bootstrap points with intervals; threshold metrics
    and AUROC are averaged over fit samples on the un-resampled test set.

    Args:
        setting_results: Output of ``compute_predictions``.
        boot: Output of ``nb.bootstrap_cells``.

    Returns:
        Metrics DataFrame.
    """
    rows = []
    for dtype in setting_results:
        for judge_model in setting_results[dtype]:
            for train_ds in setting_results[dtype][judge_model]:
                for test_ds, rdict in setting_results[dtype][judge_model][train_ds].items():
                    for kind, key, _ in _METHODS:
                        t = nb.threshold_metrics(nb.cell_prediction_sets(rdict, key), rdict['labels'])
                        b = boot[dtype][judge_model][train_ds][test_ds][key]
                        row = {
                            'Dataset type':   dtype,
                            'Judge model':    judge_model,
                            'Train dataset':  train_ds,
                            'Test dataset':   test_ds,
                            'Type':           kind,
                            'N':              len(rdict['labels']),
                            'N docs':         len(np.unique(rdict['document_ids'])),
                            'Label rate':     float(np.mean(rdict['labels'])),
                            'Platt N':        np.nan if rdict['platt'] is None else PLATT_N,
                            'Recalibration':  rdict['recalibration'],
                            'Fit samples':    b['n_fit_samples'],
                            'Doc resamples':  b['n_doc_boot'],
                            'Accuracy':       t['acc'],
                            'Precision':      t['prec'],
                            'Recall':         t['rec'],
                            'F1':             t['f1'],
                            'AUROC':          t['auroc'],
                        }
                        for m in nb.METRICS:
                            row.update({m: b['point'][m], f'{m}_lo': b['lo'][m], f'{m}_hi': b['hi'][m]})
                        row.update({'Validity': t['validity'], 'Curve sigma': b['sigma_curve']})
                        rows.append(row)
    df = pd.DataFrame(rows)
    n_expected = len(_DTYPES) * len(JUDGE_MODELS) * len(TRAIN_DATASETS) * len(DATASETS) * len(_METHODS)
    assert len(df) == n_expected, (len(df), n_expected)
    assert not df.duplicated(['Dataset type', 'Train dataset', 'Test dataset', 'Type']).any()
    return df


if __name__ == "__main__":
    # Set to True to load precomputed results if available, False to recompute from scratch.
    load_from_precomputed = False

    setting_results = compute_predictions(load_from_precomputed=load_from_precomputed)
    print('Nested bootstrap...')
    boot = nb.bootstrap_cells(setting_results, SEED, N_DOC_BOOT, N_SYN_BOOT)
    with open(OUT_DIR / 'nested_bootstrap.pkl', 'wb') as f:
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
    plt.show()
