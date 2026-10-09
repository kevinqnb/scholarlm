import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import json
import pickle
import argparse
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.metrics import roc_auc_score

from analysis.common.config import load_calibration_v4_config
from analysis.common import calibration_ids as cids
from analysis.common import doc_bootstrap as db
from analysis.common.metrics import validity_rate_from_labels
from analysis.common.prediction_store import real_cell_provenance
from analysis.common.calibration_plot_utils import draw_reliability_curve
from analysis.common.head_activations import HeadActivationCache
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
# analysis_config.load_calibration_v4_config) names every run this script reads,
# exactly as in calibration_updated_v3.py, and every figure/CSV/pickle goes under
# analysis/results/calibration/<config id>/.
#
# v4 recalibrates each real cell with ONE slope-1 map, fit once on one draw of fit rows:
# no resampling of the fit data and no averaging over fit samples (how the results move
# with fit_seed is a separate experiment). The only resampling is a document-level
# bootstrap of the evaluation set (n_boot resamples, seeded by the envelope seed; see
# analysis/common/doc_bootstrap.py, where every SmECE / curve is relplot's own), for real and
# synthetic cells alike. Synthetic cells are never recalibrated.
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Probe/NTP calibration analysis (v4: one prior-shift or intercept-fit recalibration per cell, fit once, document-bootstrap CIs)."
    )
    parser.add_argument('config', type=Path,
                        help="analysis/analysis-configs/<id>.yaml (see load_calibration_v4_config)")
    return parser.parse_args()


_CFG = load_calibration_v4_config(_parse_args().config)
CONFIG_ID = _CFG['id']
SEED = _CFG['seed']
_PARAMS = _CFG['params']

PROBE_TYPE    = _PARAMS['probe_type']
PROBE_VARIANT = _PARAMS['probe_variant']
SYN_SPLIT     = _PARAMS['syn_split']
# Test-document resamples per cell (real and synthetic).
N_BOOT = _PARAMS['n_boot']
# Real-cell map (config: recalibration), expit(slope * logit(p) + intercept); slope 1 except platt_fit:
#   prior_shift   -- intercept logit(pi_te) - logit(pi_tr), pi_tr the scorer's synthetic
#                    training prevalence.
#   intercept_fit -- intercept by MLE on the fit rows (their mapped probabilities average
#                    to their label rate).
#   platt_fit     -- slope and intercept by unregularized logistic MLE on the fit rows
#                    (recalibration_maps.csv's coef column is the slope).
# What it is fit from (config: fit_source):
#   sample -- fit_n rows drawn uniformly without replacement, by fit_seed, from the test
#             dataset's probe-training pool; pi_te = their label rate.
#   manual -- prior_shift only: the config's per-dataset pi_te_estimate.
#   oracle -- the evaluated real test rows themselves (diagnostic only: a perfect pi_te,
#             or an intercept / Platt fit in-sample on the rows it is scored on).
RECALIBRATION = _PARAMS['recalibration']
FIT_SOURCE    = _PARAMS['fit_source']
FIT_N         = _PARAMS['fit_n']
FIT_SEED      = _PARAMS['fit_seed']
DATASETS = list(_PARAMS['datasets'])
TRAIN_DATASETS = list(_PARAMS['datasets'])  # every dataset has its own synthetic-probe config

_INPUTS = cids.resolve_calibration_inputs(_CFG)
JUDGE_MODEL = _INPUTS['judge_model']

OUT_DIR = REPO_ROOT / "analysis" / "results" / "calibration" / CONFIG_ID
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# None reproduces the Platt-scaled baseline filenames; only 'noplatt' picks the suffixed variant.
_PROBE_VARIANT_KW = None if PROBE_VARIANT == 'platt' else PROBE_VARIANT

_DTYPES = ['syn', 'real']

print(f'[calibration v4] config: {CONFIG_ID} | probe type: {PROBE_TYPE} | probe variant: {PROBE_VARIANT} '
      f'| syn split: {SYN_SPLIT} | datasets: {DATASETS} | recalibration: {RECALIBRATION} '
      f'| fit source: {FIT_SOURCE} | fit n: {FIT_N} | fit seed: {FIT_SEED} | doc resamples: {N_BOOT} '
      f'| judge: {JUDGE_MODEL} '
      f'| out dir: {OUT_DIR}')
if FIT_SOURCE == 'oracle':
    print('[calibration v4] ORACLE fit: maps are fit on the evaluated rows themselves -- diagnostic only.')


# ── Trained probe / NTP calibrator (id-addressed, one per TRAIN_DATASETS entry) ──
def _load_trained_artifact(train_ds, filename):
    path = _INPUTS['datasets'][train_ds]['probe_dir'] / filename
    if not path.exists():
        raise FileNotFoundError(
            f'{path} does not exist. Run analysis/synthetic_probe_train.py '
            f'on the {train_ds} synthetic-probe config first.'
        )
    artifact = joblib.load(path)
    assert artifact['judge_model'] == JUDGE_MODEL, (
        f'{path}: judge_model {artifact["judge_model"]!r} != {JUDGE_MODEL!r}')
    assert artifact['dataset'] == train_ds, f'{path}: dataset {artifact["dataset"]!r} != {train_ds!r}'
    return artifact


_ntp_cal_filename = 'ntp_calibrator.pkl' if _PROBE_VARIANT_KW is None else 'ntp_calibrator_noplatt.pkl'
_probe_filename = (
    'layer_probe.pkl' if PROBE_TYPE == 'layer'
    else ('head_probe.pkl' if _PROBE_VARIANT_KW is None else 'head_probe_noplatt.pkl')
)
ntp_cal_cache, probe_cache = {}, {}
for _train_ds in TRAIN_DATASETS:
    print(f'Loading trained probe/NTP calibrator ({_train_ds}, {JUDGE_MODEL}) '
          f'from {_INPUTS["datasets"][_train_ds]["syn_train_id"]}...')
    ntp_cal_cache[_train_ds] = _load_trained_artifact(_train_ds, _ntp_cal_filename)
    probe_cache[_train_ds]   = _load_trained_artifact(_train_ds, _probe_filename)

# Head features: each activation row is decompressed once per run, keeping the union of
# every train probe's top heads (see analysis/common/head_activations.py).
_HEAD_ACTS = (HeadActivationCache([lh for _tr in TRAIN_DATASETS for lh in probe_cache[_tr]['top_k_heads']])
              if PROBE_TYPE == 'head' else None)


# ── Real data, test rows and fit rows, per test dataset ──────────────────────
test_data = {}
for ds in DATASETS:
    print(f'Loading test data for {ds}...')
    ds_inputs = _INPUTS['datasets'][ds]

    with open(ds_inputs['judge_combine_dir'] / 'combined.json') as f:
        real_df = pd.DataFrame(json.load(f))
    assert f'judgement_p_true_{JUDGE_MODEL}' in real_df.columns, (ds, JUDGE_MODEL)

    # Scored on the final.json datapoints; combined.json must be row-for-row the same run.
    with open(ds_inputs['extraction_dir'] / 'final.json') as f:
        final_df = pd.DataFrame(json.load(f))
    for col in ('measurement_id', 'document_id', 'attribute'):
        assert final_df[col].tolist() == real_df[col].tolist(), (
            f'{ds}: final.json and combined.json disagree on {col}')

    # Labels exactly as in v3: judgement_combined, OR'd with a cached ground-truth
    # match when the dataset's use_matching_labels is on.
    jlabels = real_df['judgement_combined'].to_numpy(dtype=bool)
    if _PARAMS['datasets'][ds]['use_matching_labels']:
        _gt_df, ext_df, cached_edges = cids.load_cached_matching(
            _PARAMS['datasets'][ds]['extraction_id'], ds_inputs['ground_truth_path'])
        judged_edges = cids.edges_to_judged_rows(cached_edges, ext_df, real_df)
        ex_edge_exists = np.zeros(len(real_df), dtype=bool)
        for _gt_idx, ex_idx in judged_edges:
            ex_edge_exists[ex_idx] = True
        print(f'  {ds}: matching labels on -- {int(ex_edge_exists.sum())}/{len(real_df)} judged rows matched')
        labels = jlabels | ex_edge_exists
    else:
        print(f'  {ds}: matching labels off -- labels are judgement_combined alone')
        labels = jlabels.copy()

    # The fit pool is this dataset's real rows whose document was in its own
    # synthetic-probe training set. It is excluded from the test rows whatever the fit
    # source, so every config scores the same rows (and the same rows as v3).
    pool_docs = set(probe_cache[ds]['syn_document_ids'])
    in_pool = real_df['document_id'].isin(pool_docs).to_numpy()
    pool_idx = np.where(in_pool)[0]
    test_idx = np.where(~in_pool)[0]
    assert len(test_idx) > 0, f'{ds}: no real test rows outside the fit pool'

    # fit_idx: the real_df rows the maps are fit on (and pi_te is read from). One draw,
    # never resampled. Empty for manual (pi_te comes from the config).
    if FIT_SOURCE == 'sample':
        assert FIT_N <= len(pool_idx), f'{ds}: fit_n={FIT_N} > {len(pool_idx)} pool rows'
        fit_idx = pool_idx[uniform_fit_sample(len(pool_idx), FIT_N, FIT_SEED)]
        assert not set(fit_idx.tolist()) & set(test_idx.tolist()), ds
    elif FIT_SOURCE == 'oracle':
        fit_idx = test_idx
    else:
        assert FIT_SOURCE == 'manual' and RECALIBRATION == 'prior_shift', (FIT_SOURCE, RECALIBRATION)
        fit_idx = np.array([], dtype=np.int64)
    pi_te = (float(_PARAMS['datasets'][ds]['pi_te_estimate']) if FIT_SOURCE == 'manual'
             else float(labels[fit_idx].mean()))
    if not 0 < pi_te < 1:
        raise ValueError(f'{ds}: fit rows are single-class (label rate {pi_te}); pick another fit_seed or fit_n')
    print(f'  {ds}: fit rows {len(fit_idx)} ({FIT_SOURCE}), pi_te / fit label rate = {pi_te:.4f} '
          f'| test rows {len(test_idx)}/{len(real_df)} (label rate {labels[test_idx].mean():.4f}) '
          f'| pool rows {len(pool_idx)} (label rate {labels[pool_idx].mean():.4f})')

    test_data[ds] = {
        'real_df': real_df,
        'labels': labels,
        'pool_docs': pool_docs,
        'test_idx': test_idx,
        'fit_idx': fit_idx,
        'pi_te': pi_te,
    }


def _score_rows(train_ds, test_ds, mids, raw_ntp_probs, act_dir):
    """Raw (unmapped) probe and NTP-calibrator probabilities for these rows."""
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
    # Result format: {dataset_type: {judge_model: {train_ds: {test_ds: cell}}}}. 'syn' cells
    # are test_ds's synthetic test set, never recalibrated (recal_map None). 'real' cells are
    # test_ds's real rows outside its fit pool, recalibrated -- probe and NTP separately -- by
    # recal_map[method] = (coef, intercept). probe_probs/ntp_probs are the evaluated
    # predictions, probe_raw/ntp_raw the unmapped scores. Also writes recalibration_maps.csv.
    judge_model = JUDGE_MODEL
    setting_results = {dtype: {judge_model: {}} for dtype in _DTYPES}
    map_rows = []

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

                    # Test rows also exclude train_ds's probe-training documents; those belong
                    # to train_ds's own corpus, so for a cross-dataset probe this must remove
                    # nothing (the test rows, and so the oracle fit, are per test dataset).
                    exclude = td['pool_docs'] | set(pd_data['syn_document_ids'])
                    idx = np.where(~real_df['document_id'].isin(exclude).to_numpy())[0]
                    assert np.array_equal(idx, td['test_idx']), (
                        f'{train_ds} probe -> {test_ds}: excluding the probe\'s training documents '
                        f'changed the test rows ({len(idx)} vs {len(td["test_idx"])})')
                    mids   = real_df['measurement_id'].iloc[idx].tolist()
                    labels = td['labels'][idx]
                    probe_raw, ntp_raw = _score_rows(train_ds, test_ds, mids, real_df[col].iloc[idx].to_numpy(), act_dir)
                    raw = {'probe': probe_raw, 'ntp': ntp_raw}

                    fit_idx = td['fit_idx']
                    fit_labels = td['labels'][fit_idx]
                    if RECALIBRATION in ('intercept_fit', 'platt_fit'):
                        if FIT_SOURCE == 'oracle':
                            assert np.array_equal(fit_idx, idx)
                            fit_raw = raw
                        else:
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
                        if FIT_SOURCE == 'oracle':
                            assert td['pi_te'] == float(labels.mean()), (train_ds, test_ds, td['pi_te'])
                            if RECALIBRATION == 'intercept_fit':
                                assert abs(float(mapped.mean()) - float(labels.mean())) < 1e-8, (train_ds, test_ds, meth)
                        map_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds, 'Method': meth,
                                         'Recalibration': RECALIBRATION, 'Fit source': FIT_SOURCE,
                                         'Fit n': len(fit_idx), 'Fit seed': FIT_SEED, 'pi_te': td['pi_te'],
                                         'Train prevalence': float(pi_tr[meth]), 'coef': coef, 'intercept': icpt,
                                         'N test': len(idx), 'Test label rate': float(labels.mean()),
                                         'Raw mean prob': float(raw[meth].mean()),
                                         'Mapped mean prob': float(mapped.mean())})
                    probe_probs = apply_platt(probe_raw, *maps['probe'])
                    ntp_probs   = apply_platt(ntp_raw, *maps['ntp'])
                    cell = {'recal_map': maps,
                            'fit_on_test_rows': FIT_SOURCE == 'oracle'}
                    # Provenance's fit-row field must not overlap the test rows, so an oracle
                    # fit (on the test rows themselves) records none; fit_on_test_rows says so.
                    cell.update(real_cell_provenance(
                        real_df, idx, fit_idx if FIT_SOURCE == 'sample' else np.array([], dtype=np.int64), exclude,
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

    with open(OUT_DIR / 'predictions.pkl', 'wb') as f:
        pickle.dump(setting_results, f)
    return setting_results


# (display name, method key in setting_results / calibration summaries, linestyle).
# Probe and NTP are never drawn on the same axes or reported in the same table rows.
_METHODS = [
    ('Probe', 'probe', '-'),
    ('NTP',   'ntp',   '--'),
]


def plot_calibration_curves(boot, dtype):
    # One figure per (method, train_ds): the train_ds probe (or NTP calibrator)
    # evaluated on every test dataset, one curve per test_ds, colored by test_ds.
    # Line = relplot's curve on the full evaluation set; band = pointwise percentiles of
    # relplot's curves over the document resamples; drawn only inside the data's support.
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
    # Same nesting as setting_results, {dtype: {judge: {train_ds: {test_ds: {method: summary}}}}},
    # each summary from db.doc_bootstrap_calibration. Resamples depend only on (dtype, test_ds):
    # every probe and method scored on an evaluation set sees the same N_BOOT resamples.
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
    # One row per (dtype, train_ds, test_ds, method). Calibration errors are the full
    # evaluation set's value with a percentile interval over N_BOOT document resamples;
    # threshold metrics and AUROC are on the full evaluation set.
    # Recalibration / Fit source / pi_te / Intercept are None / NaN for syn (never recalibrated);
    # pi_te is the fit rows' label rate (or the manual estimate).
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
