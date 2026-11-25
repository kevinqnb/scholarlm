"""Calibration and predictive performance of judge confidence baselines on real extractions.

Scores each baseline (verbalized, sampling, ntp) on the real test rows and labels of one
finished calibration run, so every method is evaluated on the same rows as the probe.
Baseline scores are used as produced and are never recalibrated. Every metric gets a
document-bootstrap 95% CI. Resamples are drawn with the calibration run's seed, so they are
identical to its resamples when ``n_boot`` is also equal.

Labels are the majority vote of LLM judges (or a ground-truth match). A baseline whose
judge model is also one of the voting judges is scored against labels it helped produce;
the ``Judge votes on labels`` column marks these.

Outputs in analysis/results/calibration-baseline/<config id>/: predictions.pkl,
bootstrap.pkl, metrics.csv, manifest.json, figures/ (one reliability diagram per method).

Usage
-----
    python analysis/calibration_baseline.py analysis/analysis-configs/calibration-baseline/<id>.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils as paths
from analysis.common import calibration_ids as cids
from analysis.common import doc_bootstrap as db
from analysis.common.calibration_plot_utils import draw_reliability_curve
from analysis.common.config import (
    CALIBRATION_BASELINE_METHODS, CALIBRATION_DATASETS, analysis_results_dir, load_calibration_baseline_config,
)
from analysis.common.loaders import load_probe_artifact
from analysis.common.prediction_store import check_real_cell
from analysis.common.provenance import repo_relative, sha256_file
from run_judge_combine import VOTING_JUDGE_KEYS

RCPARAMS = {
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
}

# One fixed color and label per dataset.
_PALETTE = sns.color_palette("husl", 10)
DS_COLORS = {'pond': _PALETTE[7], 'nfix': _PALETTE[1], 'supermat': _PALETTE[0]}
DS_LABELS = {'pond': 'PLW', 'nfix': 'NF', 'supermat': 'SM'}
METHOD_LABELS = {'verbalized': 'Verbalized', 'sampling': 'Sampling', 'ntp': 'NTP'}
assert set(DS_COLORS) == set(DS_LABELS) == set(CALIBRATION_DATASETS)
assert set(METHOD_LABELS) == set(CALIBRATION_BASELINE_METHODS)

# Extraction fields a response row must share with its final.json row.
ROW_COLS = ('measurement_id', 'document_id', 'attribute', 'name', 'value', 'units')


def baseline_scores(responses: list[dict], final: list[dict], measurement_ids) -> np.ndarray:
    """Baseline scores of the given rows, after checking the run covers the extraction exactly.

    Args:
        responses: Rows of a baseline run's responses.json.
        final: Rows of the extraction's final.json.
        measurement_ids: Ids to score, in the order wanted.

    Returns:
        Score per id, in [0, 1].

    Raises:
        ValueError: The run's rows differ from final.json, a row failed, or a score is
            missing or outside [0, 1].
    """
    if len(responses) != len(final):
        raise ValueError(f'{len(responses)} response rows, {len(final)} final.json rows')
    for i, (r, f) in enumerate(zip(responses, final)):
        for col in ROW_COLS:
            if r[col] != f[col] and not (pd.isna(r[col]) and pd.isna(f[col])):
                raise ValueError(f'row {i}: {col} {r[col]!r} in responses.json, {f[col]!r} in final.json')
        if r['judgement_failed'] is not False:
            raise ValueError(f'row {i}: judgement_failed is {r["judgement_failed"]!r}')
    by_id = pd.Series([r['judgement_p_true'] for r in responses], dtype=float,
                      index=[r['measurement_id'] for r in responses])
    if not by_id.index.is_unique:
        raise ValueError('measurement_id is not unique')
    if not by_id.between(0, 1).all():  # also False for NaN
        raise ValueError(f'{int((~by_id.between(0, 1)).sum())} scores missing or outside [0, 1]')
    return by_id.loc[list(measurement_ids)].to_numpy()


def load_test_rows(ds: str, calibration: dict) -> tuple[pd.DataFrame, list[dict], dict]:
    """Load one dataset's real test rows and labels from the calibration run.

    Args:
        ds: Dataset.
        calibration: ``calibration`` entry of ``resolve_calibration_baseline_inputs``.

    Returns:
        Tuple of:
            - verified test rows (measurement_id, document_id, label, ...) in stored order
            - every row of final.json
            - input-file hashes for the manifest

    Raises:
        ValueError: The stored rows do not match the current final.json or combined.json.
    """
    params = calibration['cfg']['params']
    inputs = calibration['inputs']['datasets'][ds]
    judge = calibration['inputs']['judge_model']
    with open(calibration['predictions_path'], 'rb') as f:
        cell = pickle.load(f)['real'][judge][ds][ds]

    final_path = inputs['extraction_dir'] / 'final.json'
    combined_path = inputs['judge_combine_dir'] / 'combined.json'
    if cell['combined_sha256'] != sha256_file(combined_path):
        raise ValueError(f'{combined_path} changed since {calibration["id"]} was run (sha256 mismatch)')
    if params['probe_type'] == 'layer':
        probe_filename = 'layer_probe.pkl'
    else:
        probe_filename = {'platt': 'head_probe.pkl', 'noplatt': 'head_probe_noplatt.pkl'}[params['probe_variant']]
    # The test rows are the rows outside the probe's training documents.
    train_docs = set(load_probe_artifact(inputs['probe_dir'], probe_filename, ds, judge)['syn_document_ids'])

    final = json.loads(final_path.read_text())
    hashes = {repo_relative(final_path): sha256_file(final_path),
              repo_relative(combined_path): cell['combined_sha256']}
    rows = check_real_cell(cell, pd.DataFrame(final), train_docs, hashes[repo_relative(final_path)], calibration['id'])
    assert len(rows) > 0, ds
    return rows, final, hashes


def compute_predictions(inputs: dict) -> tuple[dict, dict]:
    """Score every (dataset, method) cell.

    Args:
        inputs: Output of ``resolve_calibration_baseline_inputs``.

    Returns:
        Tuple of:
            - ``{dataset: {'measurement_ids', 'document_ids', 'labels', 'probs': {method: scores}}}``
            - input file -> sha256
    """
    preds, hashes = {}, {repo_relative(inputs['calibration']['predictions_path']):
                         sha256_file(inputs['calibration']['predictions_path'])}
    for ds, runs in inputs['runs'].items():
        rows, final, ds_hashes = load_test_rows(ds, inputs['calibration'])
        hashes |= ds_hashes
        probs = {}
        for method in CALIBRATION_BASELINE_METHODS:
            path = runs[method] / 'responses.json'
            hashes[repo_relative(path)] = sha256_file(path)
            probs[method] = baseline_scores(json.loads(path.read_text()), final, rows['measurement_id'])
            assert probs[method].shape == (len(rows),), (ds, method, probs[method].shape)
        preds[ds] = {
            'measurement_ids': rows['measurement_id'].to_numpy(),
            'document_ids': rows['document_id'].to_numpy(),
            'labels': rows['label'].to_numpy(dtype=bool),
            'probs': probs,
        }
        print(f'{ds}: {len(rows)}/{len(final)} test rows, {rows["document_id"].nunique()} documents, '
              f'label rate {rows["label"].mean():.4f}')
    return preds, hashes


def bootstrap_predictions(preds: dict, n_boot: int, seed: int) -> dict:
    """Document-bootstrap summaries for every cell, paired across methods.

    Args:
        preds: First output of ``compute_predictions``.
        n_boot: Number of document resamples.
        seed: Seed of the resample stream.

    Returns:
        ``{dataset: {method: doc_bootstrap_calibration summary}}``.
    """
    out = {}
    for ds, cell in preds.items():
        resamples = db.document_resamples(cell['document_ids'], n_boot, db.resample_rng(seed, 'real', ds))
        out[ds] = {}
        for method, probs in cell['probs'].items():
            print(f'  document bootstrap {ds} {method}: {n_boot} resamples', flush=True)
            out[ds][method] = db.doc_bootstrap_calibration(probs, cell['labels'], resamples)
    return out


def compute_metrics(preds: dict, boot: dict, judges: dict, voters: dict) -> pd.DataFrame:
    """Build the metrics table, one row per (dataset, method).

    Args:
        preds: First output of ``compute_predictions``.
        boot: Output of ``bootstrap_predictions``.
        judges: Method -> judge name.
        voters: Dataset -> set of judges whose votes make the labels.

    Returns:
        DataFrame with, per metric in ``doc_bootstrap.METRICS``, the point value and its
        ``_lo`` / ``_hi`` 95% interval.
    """
    rows = []
    for ds, cell in preds.items():
        for method, probs in cell['probs'].items():
            b = boot[ds][method]
            row = {
                'Test dataset': ds, 'Method': method, 'Judge model': judges[method],
                'Judge votes on labels': judges[method] in voters[ds],
                'N': len(probs), 'N docs': len(np.unique(cell['document_ids'])),
                'Label rate': float(cell['labels'].mean()), 'Mean prob': float(probs.mean()),
                'N distinct probs': len(np.unique(probs)), 'Doc resamples': b['n_boot'],
            }
            for m in db.METRICS:
                row.update({m: b['point'][m], f'{m}_lo': b['lo'][m], f'{m}_hi': b['hi'][m]})
            row['relplot sigma'] = b['sigma']
            rows.append(row)
    df = pd.DataFrame(rows)
    assert len(df) == len(CALIBRATION_DATASETS) * len(CALIBRATION_BASELINE_METHODS), len(df)
    assert not df.duplicated(['Test dataset', 'Method']).any()
    return df


def label_voters(calibration: dict) -> dict:
    """Judges whose votes make each dataset's labels.

    Args:
        calibration: ``calibration`` entry of ``resolve_calibration_baseline_inputs``.

    Returns:
        Dataset -> set of judge names that vote.
    """
    out = {}
    for ds, block in calibration['cfg']['params']['datasets'].items():
        combine = paths.load_experiment_config(paths.find_experiment_config(block['judge_combine_id']))
        judges = {paths.load_experiment_config(paths.find_experiment_config(i))['params']['judge']
                  for i in combine['params']['judge_ids']}
        out[ds] = judges & set(VOTING_JUDGE_KEYS)
        assert out[ds], f'{ds}: no voting judge among {sorted(judges)}'
    return out


def plot_reliability(boot: dict, judges: dict, figures_dir: Path) -> None:
    """Save one reliability diagram per method, with one curve per dataset, and the legend.

    Args:
        boot: Output of ``bootstrap_predictions``.
        judges: Method -> judge name.
        figures_dir: Output directory.
    """
    assert set(boot) == set(CALIBRATION_DATASETS), sorted(boot)
    for i, method in enumerate(CALIBRATION_BASELINE_METHODS):
        fig, ax = plt.subplots(figsize=(4.0, 3.8))
        ax.plot([0, 1], [0, 1], 'k:', lw=1.0, alpha=0.5, zorder=1)
        for ds in CALIBRATION_DATASETS:
            draw_reliability_curve(ax, boot[ds][method], DS_COLORS[ds], linestyle='-', lw=2.5,
                                   line_zorder=3, band_zorder=1)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel('Predicted Probability')
        if i == 0:
            ax.set_ylabel('Observed Frequency')
        ax.set_title(f'{METHOD_LABELS[method]} ({judges[method]})', fontsize=15, style='italic')
        ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(figures_dir / f'cal_real_{method}.pdf', bbox_inches='tight', dpi=200)
        plt.close(fig)

    handles = [mlines.Line2D([], [], color=DS_COLORS[ds], lw=2, marker='o', ms=3.5, label=DS_LABELS[ds])
               for ds in CALIBRATION_DATASETS]
    fig, ax = plt.subplots(figsize=(6.0, 0.45))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(handles), fontsize=13, frameon=False,
              handlelength=2.0, title='Test set', title_fontsize=13)
    fig.savefig(figures_dir / 'legend_calibration.pdf', bbox_inches='tight', dpi=200)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    """Run the analysis described by a config file.

    Args:
        argv: Command-line arguments; None reads ``sys.argv``.
    """
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('config', type=Path, help='analysis/analysis-configs/calibration-baseline/<id>.yaml')
    cfg = load_calibration_baseline_config(parser.parse_args(argv).config)
    n_boot, seed = cfg['params']['n_boot'], cfg['seed']
    inputs = cids.resolve_calibration_baseline_inputs(cfg)
    voters = label_voters(inputs['calibration'])

    out_dir = analysis_results_dir('calibration-baseline') / cfg['id']
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams.update(RCPARAMS)
    print(f'[calibration baseline] config: {cfg["id"]} | rows and labels from: {inputs["calibration"]["id"]} '
          f'| judges: {inputs["judges"]} | doc resamples: {n_boot} | seed: {seed} | out dir: {out_dir}')

    preds, hashes = compute_predictions(inputs)
    with open(out_dir / 'predictions.pkl', 'wb') as f:
        pickle.dump(preds, f)
    boot = bootstrap_predictions(preds, n_boot, seed)
    with open(out_dir / 'bootstrap.pkl', 'wb') as f:
        pickle.dump(boot, f)
    metrics = compute_metrics(preds, boot, inputs['judges'], voters)
    metrics.to_csv(out_dir / 'metrics.csv', index=False)
    print(metrics.to_string(index=False, float_format='{:.3f}'.format))
    plot_reliability(boot, inputs['judges'], figures_dir)
    manifest = {
        'config_id': cfg['id'], 'calibration_config_id': inputs['calibration']['id'],
        'seed': seed, 'n_boot': n_boot, 'judges': inputs['judges'],
        'runs': cfg['params']['datasets'], 'label_voters': {ds: sorted(v) for ds, v in voters.items()},
        'input_sha256': hashes,
    }
    (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
