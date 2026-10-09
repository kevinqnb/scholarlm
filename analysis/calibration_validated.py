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

from analysis.common.config import load_calibration_validated_config, validations_path
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

# One consistent color per dataset, reused across every plot (synthetic/real,
# within/cross) so a dataset is always the same color regardless of role.
_DS_COLORS = {
    'pond':     palette[7],
    'nfix':     palette[1],
    'supermat': palette[0],
}

_DS_LABELS = {'pond': 'PLW', 'nfix': 'NF', 'supermat': 'SM'}


# ── Parameters ───────────────────────────────────────────────────────────────
# One positional analysis config (analysis/analysis-configs/<id>.yaml, loaded by
# analysis_config.load_calibration_validated_config) names every run this script reads:
# per dataset, the real extraction + its qwen interp-judge run + judge_combine
# run, the synthetic-probe analysis config whose cached probe is applied, and
# the synthetic test runs. calibration_ids.resolve_calibration_inputs
# cross-checks all of those ids against each run's own committed config before
# anything is loaded. No flags or defaults (the one env var, SCHOLARLM_VALIDATIONS_DIR,
# only locates the human-validation files, whose sha256 the config pins): the config id is the
# run's identity, and every figure/CSV/pickle goes under
# analysis/results/calibration/<config id>/.
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Probe/NTP calibration analysis (v3 recalibration, but real test cells are scored against human-validated labels)."
    )
    parser.add_argument('config', type=Path,
                        help="analysis/analysis-configs/<id>.yaml (see load_calibration_validated_config)")
    return parser.parse_args()


_CFG = load_calibration_validated_config(_parse_args().config)
CONFIG_ID = _CFG['id']
SEED = _CFG['seed']
_PARAMS = _CFG['params']

PROBE_TYPE   = _PARAMS['probe_type']
PROBE_VARIANT = _PARAMS['probe_variant']
SYN_SPLIT    = _PARAMS['syn_split']
# Number of real rows per dataset used to fit each Platt scaler (config: platt_n);
# these stay labelled by LLM + matching (the training split has no human labels).
PLATT_N = _PARAMS['platt_n']
# Nested-bootstrap sizes (see analysis/common/nested_bootstrap.py): real cells are refit on
# N_FIT_SAMPLES independent platt_n-row samples, each crossed with N_DOC_BOOT test-document
# resamples; synthetic cells (never recalibrated) get N_SYN_BOOT document resamples.
N_FIT_SAMPLES = _PARAMS['n_fit_samples']
N_DOC_BOOT    = _PARAMS['n_doc_boot']
N_SYN_BOOT    = _PARAMS['n_syn_boot']
# How the real cells are recalibrated on that sample (config: recalibration), per method
# (probe / NTP), always as expit(coef * logit(p) + intercept):
#   platt_fit     -- coef and intercept by unregularized logistic MLE (Platt).
#   intercept_fit -- coef fixed at 1, intercept by MLE: the scaled sample probabilities
#                    average to the sample's label rate.
#   prior_shift   -- coef fixed at 1, intercept logit(sample label rate) - logit(pi_tr),
#                    pi_tr = the scorer's synthetic training prevalence (label-shift
#                    correction, as calibration_updated.py did with a hand-set pi_te).
RECALIBRATION = _PARAMS['recalibration']
assert tuple(RECALIBRATION_METHODS) == tuple(_CFG_RECALIBRATION_METHODS), (RECALIBRATION_METHODS, _CFG_RECALIBRATION_METHODS)
assert RECALIBRATION in RECALIBRATION_METHODS, RECALIBRATION
DATASETS = list(_PARAMS['datasets'])
TRAIN_DATASETS = list(_PARAMS['datasets'])  # every validated dataset has its own synthetic-probe config

_INPUTS = cids.resolve_calibration_inputs(_CFG)
JUDGE_MODEL  = _INPUTS['judge_model']
JUDGE_MODELS = [JUDGE_MODEL]  # kept as a list: every plot/metrics loop below is judge_model-indexed

OUT_DIR = REPO_ROOT / "analysis" / "results" / "calibration" / CONFIG_ID
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# None reproduces the Platt-scaled baseline filenames; only 'noplatt' picks the suffixed variant.
_PROBE_VARIANT_KW = None if PROBE_VARIANT == 'platt' else PROBE_VARIANT

_DTYPES = ['syn', 'real']  # both always available: every dataset has a synthetic probe + test set and real judge_interp data

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

# Head features: each activation row is decompressed once per run, keeping the union of
# every train probe's top heads (see analysis/common/head_activations.py).
_HEAD_ACTS = (HeadActivationCache([lh for _tr in TRAIN_DATASETS for lh in probe_cache[_tr][JUDGE_MODEL]['top_k_heads']])
              if PROBE_TYPE == 'head' else None)


# Pre-load all test data, including matching results, to avoid redundant loading and matching within the loop
test_data = {}
for ds in DATASETS:
    print(f'Loading test data for {ds}...')
    ds_inputs = _INPUTS['datasets'][ds]

    with open(ds_inputs['judge_combine_dir'] / 'combined.json') as f:
        real_df = pd.DataFrame(json.load(f))
    assert f'judgement_p_true_{JUDGE_MODEL}' in real_df.columns, (ds, JUDGE_MODEL)

    # Calibration is always scored on the final.json datapoints -- the rows the
    # judge saw. Labels and activations are per row, so combined.json must be
    # row-for-row the same run as final.json: same measurement_id, document_id
    # and attribute sequences, not just the same length.
    with open(ds_inputs['extraction_dir'] / 'final.json') as f:
        final_records = json.load(f)
    final_df = pd.DataFrame(final_records)
    for col in ('measurement_id', 'document_id', 'attribute'):
        assert final_df[col].tolist() == real_df[col].tolist(), (
            f'{ds}: final.json and combined.json disagree on {col}')

    jlabels = real_df['judgement_combined'].to_numpy(dtype=bool)
    # Per-dataset switch (config: use_matching_labels): True counts a row valid if
    # the judge said so OR it matched ground truth; False is the judge alone and
    # reads no matching at all. Matchings are never built here: when on, edges come
    # from the match_cache.pkl analysis/match_cache.py already built (validated
    # against this ground truth file and extraction file, thresholded at the
    # dataset config's own fuzzy_threshold). That cache indexes postprocessed.json's
    # rows -- every final.json measurement_id, with list-valued rows expanded into
    # several -- so a final.json row counts as matched if any of its expanded rows
    # did. Recovery is not computed here (see analysis/recovery_validity.py).
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

    # Human validations (pinned by sha256 in the config; the loader already checked it).
    # Used only as the *test* labels. They are joined to final.json rows by
    # measurement_id, and every validated row must be the same extraction row the
    # humans saw: all of its `fields` equal the final.json record (bar the site's own
    # `sampled` key), plus document_id agreement. Platt below stays fit on the LLM +
    # matching labels, since the training split has no human labels.
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

    # Platt training pool: this dataset's real rows whose document was in its own
    # synthetic-probe training set (the "training split" of the real data). Each of
    # the N_FIT_SAMPLES Platt samples resamples the pool's documents with replacement,
    # then draws PLATT_N rows uniformly from that resampled pool
    # (nb.pool_resampled_fit_sample): its label rate targets the per-row rate the
    # test metrics weight by, and the spread over samples covers which documents
    # happened to form the pool. A row can repeat within a sample (its document was
    # drawn twice); in the fit that is a weight. predictions.pkl stores sample 0.
    # Every real test cell for this dataset excludes the whole pool, so Platt-train
    # and test rows never overlap whichever sample or probe is used.
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
    """Raw (un-rescaled) probe and NTP-calibrator probabilities for these rows."""
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
    # Result format: {dataset_type: {judge_model: {train_ds: {test_ds: {probe_probs, ntp_probs, labels, platt, ...}}}}}
    # Every (train_ds, test_ds) pair is scored separately, for both 'syn' (test_ds's
    # own synthetic test set) and 'real' (test_ds's real extraction, excluding
    # test_ds's Platt pool and train_ds's probe-training documents). 'real' cells
    # are recalibrated by RECALIBRATION -- probe and NTP separately -- with one map per
    # fit sample (fit_maps[method][r], each fit on test_ds's PLATT_N-row sample r);
    # probe_probs/ntp_probs and 'platt' are fit sample 0's. 'syn' cells are not scaled
    # and carry platt=None, fit_maps=None ('platt' holds the (coef, intercept) pair
    # whatever the method; 'recalibration' names the method). probe_raw/ntp_raw are
    # the un-recalibrated scores, document_ids the rows' documents (the nested
    # bootstrap's clusters). Also writes platt_fits.csv (one row per fitted map).

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

                    # Recalibration maps: one per fit sample and method, each fit on that
                    # sample of test_ds's pool. pi_tr (used only by prior_shift) is train_ds's
                    # scorer's synthetic training prevalence. Every row in any fit sample is
                    # scored once.
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

                    # Scored rows: only the human-validated rows that are also test rows.
                    # Validated rows inside the Platt pool or the probe's training documents
                    # cannot be scored (the scaler/probe saw those documents) and are dropped
                    # here, loudly -- never silently.
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


# (display name, method key in setting_results / nested-bootstrap cells, linestyle).
# Probe and NTP are never drawn on the same axes or reported in the same table rows.
_METHODS = [
    ('Probe', 'probe', '-'),
    ('NTP',   'ntp',   '--'),
]


def plot_calibration_curves(boot, dtype):
    # One figure per (method, train_ds): the train_ds probe (or NTP calibrator)
    # evaluated on every test dataset, one curve per test_ds, colored by test_ds.
    # Line, band and drawn region are the cell's nested-bootstrap summary.
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
    # One row per (dtype, train_ds, test_ds, method). Calibration errors (ECE, ECE_em,
    # RMSCE_db, SmECE) are nested-bootstrap point estimates with percentile intervals;
    # threshold metrics and AUROC are averaged over fit samples on the un-resampled
    # test set (synthetic cells: the single raw set). 'Label rate' is the evaluated
    # labels' positive rate (diagnostic only); 'Platt N' the size of each real-data
    # fit sample (NaN for syn); 'Fit samples' x 'Doc resamples' the replicates behind
    # each interval; 'Curve sigma' the reliability curve's bandwidth.
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
