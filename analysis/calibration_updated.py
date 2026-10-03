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

from analysis.analysis_config import load_calibration_config
from analysis.metrics import validity_rate_from_labels
from analysis import calibration_ids as cids
from scholarlm.utils.calibration import (
    rescale_probabilities_em, bootstrap_ece, intercept_adjustment,
)

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
# analysis_config.load_calibration_config) names every run this script reads:
# per dataset, the real extraction + its qwen interp-judge run + judge_combine
# run, the synthetic-probe analysis config whose cached probe is applied, and
# the synthetic test runs. calibration_ids.resolve_calibration_inputs
# cross-checks all of those ids against each run's own committed config before
# anything is loaded. No flags, env vars, or defaults: the config id is the
# run's identity, and every figure/CSV/pickle goes under
# analysis/results/calibration/<config id>/.
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Probe/NTP calibration analysis from one analysis config."
    )
    parser.add_argument('config', type=Path,
                        help="analysis/analysis-configs/<id>.yaml (see load_calibration_config)")
    return parser.parse_args()


_CFG = load_calibration_config(_parse_args().config)
CONFIG_ID = _CFG['id']
SEED = _CFG['seed']
_PARAMS = _CFG['params']

PROBE_TYPE   = _PARAMS['probe_type']
PROBE_VARIANT = _PARAMS['probe_variant']
SYN_SPLIT    = _PARAMS['syn_split']
PI_TE_ESTIMATE = _PARAMS['pi_te_estimate']  # test prevalence for label-shift rescaling; None → off
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

print(f'[calibration] config: {CONFIG_ID} | probe type: {PROBE_TYPE} | probe variant: {PROBE_VARIANT} '
      f'| syn split: {SYN_SPLIT} | datasets: {DATASETS} | judge: {JUDGE_MODEL} '
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

    test_data[ds] = {
        'real_df': real_df,
        'labels': combined_labels,
        'judge_labels': jlabels,
    }


def compute_predictions(load_from_precomputed=False):
    # ── Collect data for each test setting ────────────────────────────────
    # Result format: {dataset_type: {judge_model: {train_ds: {test_ds: {probe_probs: x, ntp_probs: y, labels: z}}}}}
    # train_ds ranges over TRAIN_DATASETS -- every dataset with a migrated
    # synthetic-probe train run (see calibration_ids.py).

    cache_file = OUT_DIR / 'predictions.pkl'

    if load_from_precomputed and cache_file.exists():
        print(f'Loading precomputed predictions from {cache_file}...')
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    judge_model = JUDGE_MODEL
    setting_results = {dtype: {judge_model: {}} for dtype in _DTYPES}

    for train_ds in TRAIN_DATASETS:
        pd_data = probe_cache[train_ds][judge_model]
        ntp_cal_data = ntp_cal_cache[train_ds][judge_model]
        top = pd_data['top_layer'] if PROBE_TYPE == 'layer' else pd_data['top_k_heads']

        for dataset_type in _DTYPES:
            setting_results[dataset_type][judge_model][train_ds] = {}

            # 'real' covers every dataset DATASETS was narrowed to, same as
            # always. 'syn' now does too (restored 2026-09-17 -- collapsed to
            # [train_ds] by the 7f3492a id-addressed-contract migration,
            # which made sense in the moment since pond was the only dataset
            # with a migrated synthetic probe/test set, but silently dropped
            # cross-domain synthetic evaluation as a capability once nfix/
            # supermat got their own): restricted to TRAIN_DATASETS since
            # only those have a synthetic test set to evaluate against at
            # all -- train_ds's own entry is always included, since train_ds
            # is itself drawn from TRAIN_DATASETS.
            test_datasets = (
                DATASETS if dataset_type == 'real'
                else [ds for ds in DATASETS if ds in TRAIN_DATASETS]
            )

            for test_ds in test_datasets:
                if dataset_type == 'syn':
                    # test_ds's own synthetic test set, scored with train_ds's
                    # trained probe/calibrator (pd_data/ntp_cal_data below) --
                    # this is what makes the cross-domain case meaningful when
                    # test_ds != train_ds.
                    syn_dir = _INPUTS['datasets'][test_ds]['syn_test_dir']
                    with open(syn_dir / 'responses.json') as f:
                        syn_resp = json.load(f)
                    syn_df_s = pd.DataFrame(syn_resp)
                    mids     = syn_df_s['measurement_id'].tolist()
                    labels   = (syn_df_s['label'] == 'valid').to_numpy(dtype=bool)
                    raw_ntp_probs = syn_df_s['judgement_p_true'].to_numpy()
                    ntp_probs = ntp_cal_data['calibrator'].predict_proba(
                        raw_ntp_probs.reshape(-1, 1)
                    )[:, 1]

                    if PROBE_TYPE == "layer":
                        syn_lo  = np.load(syn_dir / 'layer_outputs.npz')
                        X = np.stack([
                            np.array(syn_lo[str(mid)], dtype=np.float32)[top]
                            for mid in mids
                        ], axis=0)
                        probe_probs = pd_data['probe'].predict_proba(X)[:, 1]

                    else:
                        syn_act  = np.load(syn_dir / 'attention_outputs.npz')
                        X = np.concatenate([
                            np.stack([
                                np.array(syn_act[str(mid)], dtype=np.float32)[l, h, :]
                                for mid in mids
                            ], axis=0)
                            for l, h in top
                        ], axis=1)
                        probe_probs = pd_data['probe'].predict_proba(X)[:, 1]


                else:  # real
                    td       = test_data[test_ds]
                    real_df  = td['real_df']
                    syn_docs = set(pd_data['syn_document_ids'])

                    # Filter extractions to test documents (those not used in probe training).
                    # idx: positional indices into real_df/ext_df for the test split.
                    mask     = ~real_df['document_id'].isin(syn_docs)
                    idx      = np.where(mask.to_numpy())[0]
                    assert len(idx) > 0, f'{train_ds} probe -> {test_ds}: no real rows outside probe training docs'
                    print(f'  real test split {train_ds} probe -> {test_ds}: {len(idx)}/{len(real_df)} rows')

                    mids     = real_df['measurement_id'].iloc[idx].tolist()
                    labels   = td['labels'][idx]

                    raw_ntp_probs = real_df[f'judgement_p_true_{judge_model}'].iloc[idx].to_numpy()
                    ntp_probs = ntp_cal_data['calibrator'].predict_proba(
                        raw_ntp_probs.reshape(-1, 1)
                    )[:, 1]

                    judge_dir = _INPUTS['datasets'][test_ds]['judge_interp_dir']
                    if PROBE_TYPE == "layer":
                        real_lo  = np.load(judge_dir / 'layer_outputs.npz')
                        X = np.stack([
                            np.array(real_lo[str(mid)], dtype=np.float32)[top]
                            for mid in mids
                        ], axis=0)
                        probe_probs = pd_data['probe'].predict_proba(X)[:, 1]
                    else:
                        real_act = np.load(judge_dir / 'attention_outputs.npz')
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
                }

    # Save to cache for future use
    print(f'Saving predictions to {cache_file}...')
    with open(cache_file, 'wb') as f:
        pickle.dump(setting_results, f)

    return setting_results


def _pool_cross_domain(train_dict, train_ds):
    """Pool every test_ds != train_ds in train_dict into one cross-domain cell.

    ``train_dict`` is ``setting_results[dtype][judge_model][train_ds]``, keyed by
    test_ds. Concatenates ``probe_probs``/``ntp_probs``/``labels``
    across the other test_ds's. Returns None when train_ds has no other dataset to pool
    against -- for 'syn' dtype, that only happens when TRAIN_DATASETS (via
    DATASETS) covers just train_ds itself, e.g. a single dataset has a
    migrated synthetic probe/test set (was the case for every dataset but
    pond before 2026-09-16). With multiple TRAIN_DATASETS entries, 'syn'
    pools the same way 'real' always has: test_ds's own synthetic test set
    scored with train_ds's trained probe (see compute_predictions).
    """
    other_test_ds = [ds for ds in DATASETS if ds != train_ds and ds in train_dict]
    if not other_test_ds:
        return None

    probe_probs, ntp_probs, labels = [], [], []
    for test_ds in other_test_ds:
        cell = train_dict[test_ds]
        probe_probs.append(cell['probe_probs'])
        ntp_probs.append(cell['ntp_probs'])
        labels.append(cell['labels'])

    return {
        'probe_probs': np.concatenate(probe_probs),
        'ntp_probs': np.concatenate(ntp_probs),
        'labels': np.concatenate(labels),
        'test_ds': other_test_ds,
    }


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
        capstyle='round', zorder=line_zorder,
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
    # has activations for (see _pool_cross_domain).
    for judge_model in JUDGE_MODELS:
        # train_datasets is derived from setting_results itself (not just
        # DATASETS): 'syn' dtype only ever has TRAIN_DATASETS entries as keys,
        # while 'real' dtype has every dataset DATASETS was narrowed to.
        train_datasets = [ds for ds in DATASETS if ds in setting_results[dtype][judge_model]]
        if not train_datasets:
            continue

        subfigure_dir = FIGURES_DIR

        for ctype in ['in-domain', 'cross-domain']:
            # Base figure
            fig_cal, ax_cal = plt.subplots(figsize=(4.0, 3.8))
            ax_cal.plot([0, 1], [0, 1], 'k:', lw=1.0, alpha=0.5, zorder=1)

            for train_ds in train_datasets:
                train_dict = setting_results[dtype][judge_model][train_ds]

                if ctype == 'in-domain':
                    rdict = train_dict[train_ds]
                else:
                    rdict = _pool_cross_domain(train_dict, train_ds)
                    if rdict is None:
                        continue

                color = _DS_COLORS[train_ds]

                # zorder layering: both CI bands sit at the bottom (order between
                # them is irrelevant — probe and NTP share `color`, so the bands
                # are visually identical), then NTP line, then probe line on top
                # — so neither line is ever dimmed by a translucent band, and
                # probe still has final drawing priority.

                # Probe — solid, density-weighted smoothed curve + bootstrap CI band
                _plot_relplot_curve(
                    ax_cal, rdict['probe_probs'], rdict['labels'], color,
                    linestyle='-', lw=2.5, line_zorder=3, band_zorder=1,
                )

                # NTP baseline — dashed, density-weighted smoothed curve + bootstrap CI band
                _plot_relplot_curve(
                    ax_cal, rdict['ntp_probs'], rdict['labels'], color,
                    linestyle='--', lw=2.0, line_zorder=2, band_zorder=1,
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
                        m = _probe_metrics(probs, rdict['labels'])
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
        subfigure_dir = FIGURES_DIR

        for train_ds in DATASETS:
            for ctype in ['in-domain', 'cross-domain']:
                for test_ds in DATASETS:
                    if (train_ds == test_ds) != (ctype == 'in-domain'):
                        continue
                    if train_ds not in setting_results[dtype][judge_model] \
                            or test_ds not in setting_results[dtype][judge_model].get(train_ds, {}):
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


if __name__ == "__main__":
    # Set to True to load precomputed results if available, False to recompute from scratch.
    load_from_precomputed = False

    setting_results = compute_predictions(load_from_precomputed=load_from_precomputed)
    for _dt in _DTYPES:
        plot_calibration_curves(setting_results, dtype=_dt)
    metrics_df = compute_metrics(setting_results)
    print(metrics_df.to_string(index=False, float_format='{:.3f}'.format))
    metrics_df.to_csv(OUT_DIR / 'metrics_pooled.csv', index=False)

    #plot_pr_curves(setting_results, dtype='syn')
    #plot_pr_curves(setting_results, dtype='real')

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
