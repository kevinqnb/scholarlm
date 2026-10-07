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
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
from matplotlib.collections import LineCollection
import seaborn as sns
import relplot
from sklearn.metrics import precision_recall_curve, roc_auc_score, brier_score_loss

from analysis.analysis_config import load_calibration_v3_config
from analysis.metrics import validity_rate_from_labels
from analysis import calibration_ids as cids
from analysis.match_cache import sha256_file
from analysis.prediction_store import PROVENANCE_KEYS, real_cell_provenance
from analysis.calibration_plot_utils import support_mask
from analysis.head_activations import HeadActivationCache
from scholarlm.utils.calibration import (
    bootstrap_ece, apply_platt, fit_recalibration, RECALIBRATION_METHODS,
)
from analysis.analysis_config import RECALIBRATION_METHODS as _CFG_RECALIBRATION_METHODS

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
# analysis_config.load_calibration_v3_config) names every run this script reads:
# per dataset, the real extraction + its qwen interp-judge run + judge_combine
# run, the synthetic-probe analysis config whose cached probe is applied, and
# the synthetic test runs. calibration_ids.resolve_calibration_inputs
# cross-checks all of those ids against each run's own committed config before
# anything is loaded. No flags, env vars, or defaults: the config id is the
# run's identity, and every figure/CSV/pickle goes under
# analysis/results/calibration/<config id>/.
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Probe/NTP calibration analysis (v3: per-test-dataset recalibration -- Platt, intercept-only, or prior shift -- on a small labelled real sample)."
    )
    parser.add_argument('config', type=Path,
                        help="analysis/analysis-configs/<id>.yaml (see load_calibration_v3_config)")
    return parser.parse_args()


_CFG = load_calibration_v3_config(_parse_args().config)
CONFIG_ID = _CFG['id']
SEED = _CFG['seed']
_PARAMS = _CFG['params']

PROBE_TYPE   = _PARAMS['probe_type']
PROBE_VARIANT = _PARAMS['probe_variant']
SYN_SPLIT    = _PARAMS['syn_split']
# Number of real rows per dataset used to fit each Platt scaler (config: platt_n).
PLATT_N = _PARAMS['platt_n']
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
TRAIN_DATASETS = list(_PARAMS['datasets'])  # every dataset has its own synthetic-probe config

_INPUTS = cids.resolve_calibration_inputs(_CFG)
JUDGE_MODEL  = _INPUTS['judge_model']
JUDGE_MODELS = [JUDGE_MODEL]  # kept as a list: every plot/metrics loop below is judge_model-indexed

OUT_DIR = REPO_ROOT / "analysis" / "results" / "calibration" / CONFIG_ID
FIGURES_DIR = OUT_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# None reproduces the Platt-scaled baseline filenames; only 'noplatt' picks the suffixed variant.
_PROBE_VARIANT_KW = None if PROBE_VARIANT == 'platt' else PROBE_VARIANT

_DTYPES = ['syn', 'real']  # both always available: every dataset has a synthetic probe + test set and real judge_interp data

print(f'[calibration v3] config: {CONFIG_ID} | probe type: {PROBE_TYPE} | probe variant: {PROBE_VARIANT} '
      f'| syn split: {SYN_SPLIT} | datasets: {DATASETS} | platt_n: {PLATT_N} | recalibration: {RECALIBRATION} '
      f'| judge: {JUDGE_MODEL} '
      f'| out dir: {OUT_DIR}')



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
    ntp_cal_cache[_train_ds] = {JUDGE_MODEL: _load_trained_artifact(_train_ds, _ntp_cal_filename)}
    probe_cache[_train_ds]   = {JUDGE_MODEL: _load_trained_artifact(_train_ds, _probe_filename)}

# Head features: each activation row is decompressed once per run, keeping the union of
# every train probe's top heads (see analysis/head_activations.py).
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
        final_df = pd.DataFrame(json.load(f))
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
        _gt_df, ext_df, cached_edges = cids.load_cached_matching(
            _PARAMS['datasets'][ds]['extraction_id'], ds_inputs['ground_truth_path'])
        judged_edges = cids.edges_to_judged_rows(cached_edges, ext_df, real_df)
        ex_edge_exists = np.zeros(len(real_df), dtype=bool)
        for _gt_idx, ex_idx in judged_edges:
            ex_edge_exists[ex_idx] = True
        print(f'  {ds}: matching labels on -- {int(ex_edge_exists.sum())}/{len(real_df)} judged rows matched')
        combined_labels = jlabels | ex_edge_exists
    else:
        print(f'  {ds}: matching labels off -- labels are judgement_combined alone')
        combined_labels = jlabels.copy()

    # Platt training pool: this dataset's real rows whose document was in its own
    # synthetic-probe training set (the "training split" of the real data). The
    # Platt sample is PLATT_N of these rows, spread evenly over documents (random
    # document order, one random unchosen row per document per pass; see
    # cids.document_balanced_order), seeded with the config seed; every real test
    # cell for this dataset excludes the whole pool, so Platt-train and test rows
    # never overlap whichever probe is evaluated.
    pool_docs = set(probe_cache[ds][JUDGE_MODEL]['syn_document_ids'])
    pool_idx = np.where(real_df['document_id'].isin(pool_docs).to_numpy())[0]
    assert len(pool_idx) >= PLATT_N, f'{ds}: Platt pool has {len(pool_idx)} rows < platt_n={PLATT_N}'
    pool_order = cids.document_balanced_order(
        real_df['document_id'].to_numpy()[pool_idx], np.random.default_rng(SEED))
    platt_idx = np.sort(pool_idx[pool_order[:PLATT_N]])
    assert len(set(platt_idx.tolist())) == PLATT_N and set(platt_idx) <= set(pool_idx.tolist())
    assert 0 < combined_labels[platt_idx].sum() < PLATT_N, (
        f'{ds}: Platt sample is single-class ({int(combined_labels[platt_idx].sum())}/{PLATT_N} valid)')
    print(f'  {ds}: Platt sample {PLATT_N}/{len(pool_idx)} pool rows from '
          f'{real_df["document_id"].iloc[platt_idx].nunique()}/{len(pool_docs & set(real_df["document_id"]))} pool docs, '
          f'{int(combined_labels[platt_idx].sum())} valid')

    test_data[ds] = {
        'real_df': real_df,
        'labels': combined_labels,
        'judge_labels': jlabels,
        'pool_docs': pool_docs,
        'platt_idx': platt_idx,
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
    # Result format: {dataset_type: {judge_model: {train_ds: {test_ds: {probe_probs, ntp_probs, labels, platt}}}}}
    # Every (train_ds, test_ds) pair is scored separately, for both 'syn' (test_ds's
    # own synthetic test set) and 'real' (test_ds's real extraction, excluding
    # test_ds's Platt pool and train_ds's probe-training documents). 'real' cells
    # are recalibrated by RECALIBRATION -- probe and NTP separately -- with maps fit on
    # test_ds's PLATT_N-row sample; 'syn' cells are not scaled and carry platt=None
    # ('platt' holds the (coef, intercept) pair whatever the method; 'recalibration'
    # names the method). Also writes platt_fits.csv (one row per fitted map).

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
        return loaded

    judge_model = JUDGE_MODEL
    setting_results = {dtype: {judge_model: {}} for dtype in _DTYPES}
    fit_rows = []

    for train_ds in TRAIN_DATASETS:
        pd_data = probe_cache[train_ds][judge_model]

        for dataset_type in _DTYPES:
            setting_results[dataset_type][judge_model][train_ds] = {}

            for test_ds in DATASETS:
                platt = None
                if dataset_type == 'syn':
                    syn_dir = _INPUTS['datasets'][test_ds]['syn_test_dir']
                    with open(syn_dir / 'responses.json') as f:
                        syn_resp = json.load(f)
                    syn_df_s = pd.DataFrame(syn_resp)
                    mids     = syn_df_s['measurement_id'].tolist()
                    labels   = (syn_df_s['label'] == 'valid').to_numpy(dtype=bool)
                    raw_ntp_probs = syn_df_s['judgement_p_true'].to_numpy()
                    probe_probs, ntp_probs = _score_rows(train_ds, test_ds, mids, raw_ntp_probs, syn_dir)
                else:  # real
                    td       = test_data[test_ds]
                    real_df  = td['real_df']
                    act_dir  = _INPUTS['datasets'][test_ds]['judge_interp_dir']
                    col      = f'judgement_p_true_{judge_model}'

                    # Recalibration maps: fit on test_ds's own sample, per method. pi_tr (used
                    # only by prior_shift) is train_ds's scorer's synthetic training prevalence.
                    pi = td['platt_idx']
                    p_probe, p_ntp = _score_rows(
                        train_ds, test_ds, real_df['measurement_id'].iloc[pi].tolist(),
                        real_df[col].iloc[pi].to_numpy(), act_dir)
                    p_labels = td['labels'][pi]
                    p_raw = {'probe': p_probe, 'ntp': p_ntp}
                    pi_tr = {'probe': pd_data['train_prevalence'],
                             'ntp': ntp_cal_cache[train_ds][judge_model]['train_prevalence']}
                    platt = {meth: fit_recalibration(RECALIBRATION, p_raw[meth], p_labels, pi_tr[meth])
                             for meth in ('probe', 'ntp')}
                    for meth, (coef, icpt) in platt.items():
                        p_scaled_mean = float(apply_platt(p_raw[meth], coef, icpt).mean())
                        if RECALIBRATION != 'platt_fit':
                            assert coef == 1.0, (RECALIBRATION, meth, coef)
                        if RECALIBRATION == 'intercept_fit':
                            # Score equation of the intercept MLE (known answer, solved to ~1e-12).
                            assert abs(p_scaled_mean - p_labels.mean()) < 1e-8, (
                                train_ds, test_ds, meth, p_scaled_mean, p_labels.mean())
                        fit_rows.append({'Train dataset': train_ds, 'Test dataset': test_ds, 'Method': meth,
                                         'Recalibration': RECALIBRATION,
                                         'coef': coef, 'intercept': icpt, 'N': len(pi),
                                         'Label rate': float(p_labels.mean()),
                                         'Train prevalence': float(pi_tr[meth]),
                                         'Raw mean prob': float(p_raw[meth].mean()),
                                         'Scaled mean prob': p_scaled_mean})

                    # Test rows: outside test_ds's Platt pool and train_ds's probe-training docs.
                    exclude = td['pool_docs'] | set(pd_data['syn_document_ids'])
                    idx = np.where(~real_df['document_id'].isin(exclude).to_numpy())[0]
                    assert len(idx) > 0, f'{train_ds} probe -> {test_ds}: no real test rows'
                    assert not set(idx.tolist()) & set(pi.tolist()), (train_ds, test_ds)
                    print(f'  real test split {train_ds} probe -> {test_ds}: {len(idx)}/{len(real_df)} rows')

                    mids     = real_df['measurement_id'].iloc[idx].tolist()
                    labels   = td['labels'][idx]
                    probe_probs, ntp_probs = _score_rows(
                        train_ds, test_ds, mids, real_df[col].iloc[idx].to_numpy(), act_dir)
                    probe_probs = apply_platt(probe_probs, *platt['probe'])
                    ntp_probs   = apply_platt(ntp_probs, *platt['ntp'])

                assert len(mids) == len(labels) > 0, (dataset_type, train_ds, test_ds, len(mids), len(labels))
                assert probe_probs.shape == ntp_probs.shape == labels.shape, (
                    probe_probs.shape, ntp_probs.shape, labels.shape)
                assert np.isfinite(probe_probs).all() and np.isfinite(ntp_probs).all(), (train_ds, test_ds)
                setting_results[dataset_type][judge_model][train_ds][test_ds] = {
                    'probe_probs': probe_probs, 'ntp_probs': ntp_probs, 'labels': labels,
                    'platt': platt, 'recalibration': None if platt is None else RECALIBRATION,
                }
                if dataset_type == 'syn':
                    setting_results[dataset_type][judge_model][train_ds][test_ds]['measurement_ids'] = np.asarray(mids)
                else:
                    setting_results[dataset_type][judge_model][train_ds][test_ds].update(real_cell_provenance(
                        real_df, idx, pi, exclude,
                        _INPUTS['datasets'][test_ds]['extraction_dir'] / 'final.json',
                        _INPUTS['datasets'][test_ds]['judge_combine_dir'] / 'combined.json', CONFIG_ID, SEED))

    fits_df = pd.DataFrame(fit_rows)
    assert len(fits_df) == len(TRAIN_DATASETS) * len(DATASETS) * 2, len(fits_df)
    fits_df.to_csv(OUT_DIR / 'platt_fits.csv', index=False)
    print(fits_df.to_string(index=False, float_format='{:.3f}'.format))

    print(f'Saving predictions to {cache_file}...')
    with open(cache_file, 'wb') as f:
        pickle.dump(setting_results, f)

    return setting_results


# (display name, key in each setting_results cell, linestyle). Probe and NTP are
# never drawn on the same axes or reported in the same table rows.
_METHODS = [
    ('Probe', 'probe_probs', '-'),
    ('NTP',   'ntp_probs',   '--'),
]


# Bootstrap settings for ECE confidence intervals (also reused to seed
# relplot's own internal bootstraps, which take no seed argument of their
# own — see the seeding note in _plot_relplot_curve/_probe_metrics below).
ECE_N_BOOT = 2000
ECE_CI     = 0.95
ECE_SEED   = SEED


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
    d = relplot.prepare_rel_diagram(np.asarray(probs), np.asarray(labels), num_bootstrap=ECE_N_BOOT)
    mesh, mu, density = d['mesh'], d['mu'], d['density']

    density_norm = density / density.max() if density.max() > 0 else np.ones_like(density)
    alpha = _CURVE_DENSITY_ALPHA_FLOOR + (1 - _CURVE_DENSITY_ALPHA_FLOOR) * density_norm

    # Draw only where the data support the curve (see support_mask); outside it the
    # smoother returns 0/eps (or an upward extrapolation), not an estimate.
    in_support = support_mask(probs, mesh, d['sigma'])

    points = np.array([mesh, mu]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    seg_colors = np.tile(mcolors.to_rgba(color), (len(segments), 1))
    seg_colors[:, 3] = (alpha[:-1] + alpha[1:]) / 2
    seg_keep = in_support[:-1] & in_support[1:]
    segments, seg_colors = segments[seg_keep], seg_colors[seg_keep]
    assert len(segments) > 0

    if linestyle == '--':
        seg_idx = np.arange(len(segments))
        dash_mask = (seg_idx % _NTP_DASH_PERIOD) < _NTP_DASH_ON
        segments = segments[dash_mask]
        seg_colors = seg_colors[dash_mask]

    lc = LineCollection(
        segments, colors=seg_colors, lw=lw,
        capstyle='round', zorder=line_zorder,
    )
    ax.add_collection(lc)

    ax.fill_between(
        mesh, d['lower'], d['upper'], where=in_support,
        color=color, alpha=0.20, linewidth=0, zorder=band_zorder,
    )


def plot_calibration_curves(setting_results, dtype):
    # One figure per (method, train_ds): the train_ds probe (or NTP calibrator)
    # evaluated on every test dataset, one curve per test_ds, colored by test_ds.
    for judge_model in JUDGE_MODELS:
        for method, key, linestyle in _METHODS:
            for train_ds in DATASETS:
                train_dict = setting_results[dtype][judge_model][train_ds]
                assert set(train_dict) == set(DATASETS), (dtype, train_ds, sorted(train_dict))

                fig_cal, ax_cal = plt.subplots(figsize=(4.0, 3.8))
                ax_cal.plot([0, 1], [0, 1], 'k:', lw=1.0, alpha=0.5, zorder=1)

                for test_ds in DATASETS:
                    rdict = train_dict[test_ds]
                    _plot_relplot_curve(
                        ax_cal, rdict[key], rdict['labels'], _DS_COLORS[test_ds],
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


def _probe_metrics(probs, y_true, threshold=0.5):
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
        probs, y_true, num_bootstrap=ECE_N_BOOT, plot_confidence_band=False, plot_bag_lines=False,
    )

    bs    = float(brier_score_loss(y_true, probs))
    p_pos = float(y_true.mean())
    bss   = 1.0 - bs / (p_pos * (1 - p_pos)) if p_pos not in (0.0, 1.0) else float('nan')
    validity = validity_rate_from_labels(y_true, preds)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, auroc=auroc,
                ece=ece_ew['ece'],         ece_lo=ece_ew['ci_low'],    ece_hi=ece_ew['ci_high'],
                ece_em=ece_em['ece'],      ece_em_lo=ece_em['ci_low'], ece_em_hi=ece_em['ci_high'],
                rmsce_db=rmsce['ece'],     rmsce_db_lo=rmsce['ci_low'], rmsce_db_hi=rmsce['ci_high'],
                smece=smece_d['ce'],
                smece_lo=smece_d['ce'] - smece_d['ce_ci_width'],
                smece_hi=smece_d['ce'] + smece_d['ce_ci_width'],
                bs=bs, bss=bss, n=n, validity=validity)


def compute_metrics(setting_results):
    # One row per (dtype, train_ds, test_ds, method). 'Label rate' is the
    # empirical positive rate of the evaluated labels (diagnostic only); 'Platt N'
    # is the sample size the real-data recalibration maps were fit on (NaN for syn) and
    # 'Recalibration' the method (None for syn).
    rows = []
    for dtype in setting_results:
        for judge_model in setting_results[dtype]:
            for train_ds in setting_results[dtype][judge_model]:
                for test_ds, rdict in setting_results[dtype][judge_model][train_ds].items():
                    for kind, key, _ in _METHODS:
                        m = _probe_metrics(rdict[key], rdict['labels'])
                        rows.append({
                            'Dataset type':   dtype,
                            'Judge model':    judge_model,
                            'Train dataset':  train_ds,
                            'Test dataset':   test_ds,
                            'Type':           kind,
                            'N':              m['n'],
                            'Label rate':     float(np.mean(rdict['labels'])),
                            'Platt N':        np.nan if rdict['platt'] is None else PLATT_N,
                            'Recalibration':  rdict['recalibration'],
                            'Accuracy':       m['acc'],
                            'Precision':      m['prec'],
                            'Recall':         m['rec'],
                            'F1':             m['f1'],
                            'AUROC':          m['auroc'],
                            'ECE':            m['ece'],
                            'ECE_lo':         m['ece_lo'],
                            'ECE_hi':         m['ece_hi'],
                            'ECE_em':         m['ece_em'],
                            'ECE_em_lo':      m['ece_em_lo'],
                            'ECE_em_hi':      m['ece_em_hi'],
                            'RMSCE_db':       m['rmsce_db'],
                            'RMSCE_db_lo':    m['rmsce_db_lo'],
                            'RMSCE_db_hi':    m['rmsce_db_hi'],
                            'SmECE':          m['smece'],
                            'SmECE_lo':       m['smece_lo'],
                            'SmECE_hi':       m['smece_hi'],
                            'Validity':       m['validity'],
                        })
    df = pd.DataFrame(rows)
    n_expected = len(_DTYPES) * len(JUDGE_MODELS) * len(TRAIN_DATASETS) * len(DATASETS) * len(_METHODS)
    assert len(df) == n_expected, (len(df), n_expected)
    assert not df.duplicated(['Dataset type', 'Train dataset', 'Test dataset', 'Type']).any()
    return df


if __name__ == "__main__":
    # Set to True to load precomputed results if available, False to recompute from scratch.
    load_from_precomputed = False

    setting_results = compute_predictions(load_from_precomputed=load_from_precomputed)
    for _dt in _DTYPES:
        plot_calibration_curves(setting_results, dtype=_dt)
    metrics_df = compute_metrics(setting_results)
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
