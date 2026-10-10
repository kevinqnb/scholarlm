"""Does weighting extracted pond entities by probe / NTP confidence make KMeans recover the ground truth's clusters?

Usage: python analysis/pond_clustering.py analysis/analysis-configs/clustering/<id>.yaml

1. Rows and confidences come from pond_meta.load_data (held-out documents only), with
   pond_meta_analysis.py's unit table (UNIT_CONVERSION_V2) and therefore the same
   PHYSICAL_BOUNDS filter. Each extracted row gets outlier_weight.py's non-outlier
   factor, computed on all rows exactly as in pond_meta_analysis.py.
2. Both sides are restricted to rows whose ecosystem bucket is pond, lake or wetland
   (ECOSYSTEMS), and non-positive values of LOG_SCALE_ATTRIBUTES are dropped (they have
   no log10).
3. Build entity x attribute matrices: one cell per (entity, attribute), holding the
   median value and mean confidence of its rows. GT entities are distinct (document_id,
   name, ecosystem); extracted entities are entity_ids.
4. Pick the attributes: a fixed list, or the subset (of the given sizes) whose densified
   GT matrix maximises n_rows * d.
5. Densify; log10 the LOG_SCALE_ATTRIBUTES; standardize both sides with one scaler fit
   on the GT (NaN-aware), so both live in GT-standard-deviation coordinates; KNN-impute
   each side on its own. Entity confidence = mean of its observed cells' confidences.
6. n_init single-start KMeans fits on the GT give n_init reference centroid sets. For
   each reference fit i, each arm and each gamma, n_runs single-start KMeans fits on the
   extraction with weights conf ** gamma are scored by the mean Hungarian-matched
   centroid distance to reference i. Every (i, r) trial uses the same KMeans seed in
   every arm.

Arms: ``ntp_conf`` / ``probe_conf`` (confidence alone), ``ntp`` / ``probe`` (confidence
x non-outlier factor), and ``random`` (a fresh U(0, 1) confidence per entity per trial).

Each curve is the mean over the n_init x n_runs trials. Its band is a 95% CI over KMeans
initializations only -- a t-interval over the n_init per-reference-fit means -- and says
nothing about document / entity sampling. Asserted known answers: each GT refit with its
own seed is at distance 0, and at gamma = 0 every arm gives identical fits.

Outputs in analysis/results/clustering/<config id>/: attribute_sets.csv,
centroid_distance.csv (mean, CI, Kish n_eff per arm and gamma), distances.npz,
meta.json, figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / 'src'))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))
sys.path.insert(0, str(REPO_ROOT))

import argparse
import json
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import seaborn as sns
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from analysis.common.config import _load_envelope, analysis_results_dir, get_section, is_int
from analysis.common.outlier_weight import add_outlier_columns
from analysis.common.meta_inputs import CALIBRATION_LOADERS, CONFIDENCE_CHOICES, ROWS_CHOICES, resolve_meta_inputs
from analysis.common.pond_meta import (
    DATASET, ECOSYSTEMS, LOG_SCALE_ATTRIBUTES, PAPER_RCPARAMS, UNIT_CONVERSION_V2, load_data,
)

mpl.rcParams.update(PAPER_RCPARAMS)

SECTION = 'clustering'
SECTION_KEYS = (
    # inputs (ids are also declared here and cross-checked against the calibration config)
    'calibration_config_id', 'calibration_version', 'extraction_id', 'judge_combine_id', 'judge_model',
    'probe_train_dataset', 'rows', 'deduplication_config_id', 'confidence',
    # method
    'missing_threshold', 'attributes', 'attribute_set_sizes', 'n_clusters', 'knn_neighbors', 'gammas',
    'n_init', 'n_runs',
)
GAMMA_KEYS = ('start', 'stop', 'num')
# Only pond -> pond is supported; load_data reads that cell only.
PROBE_TRAIN_DATASETS = (DATASET,)
GT_ENTITY_COLS = ['document_id', 'name', 'ecosystem']
# Confidence column per arm: *_raw is the confidence alone, the plain column is
# confidence x non-outlier factor (see outlier_weight.add_outlier_columns).
ARM_PROB_COL = {'ntp_conf': 'ntp_prob_raw', 'probe_conf': 'probe_prob_raw', 'ntp': 'ntp_prob', 'probe': 'probe_prob'}
RANDOM_ARM = 'random'
ARMS = (*ARM_PROB_COL, RANDOM_ARM)
# Second seed word per RNG use (seeds are [seed, stream, ...]).
GT_STREAM, EXT_STREAM, RANDOM_STREAM = 0, 1, 100
CI_LEVEL = 0.95
CLUSTERING_ROOT = analysis_results_dir('clustering')

# Curves, as pond_meta_analysis.py's W1-vs-threshold curves: pastel tab10 (NTP blue,
# probe green); confidence x factor solid, confidence only dashed, the random baseline
# grey dotted.
PASTEL_WHITE_FRACTION = 0.35


def _pastel(color) -> tuple:
    """Lighten a color by mixing in PASTEL_WHITE_FRACTION white.

    Args:
        color: Any matplotlib color.

    Returns:
        RGB tuple.
    """
    rgb = np.array(mcolors.to_rgb(color))
    return tuple((1 - PASTEL_WHITE_FRACTION) * rgb + PASTEL_WHITE_FRACTION)


_TAB10 = sns.color_palette('tab10', 10)
CURVE_STYLE = {
    'ntp':        dict(color=_pastel(_TAB10[0]), ls='-', lw=2.5, label='NTP'),
    'probe':      dict(color=_pastel(_TAB10[2]), ls='-', lw=2.5, label='Probe'),
    'ntp_conf':   dict(color=_pastel(_TAB10[0]), ls='--', lw=1.8, label='NTP (confidence only)'),
    'probe_conf': dict(color=_pastel(_TAB10[2]), ls='--', lw=1.8, label='Probe (confidence only)'),
    'random':     dict(color=_pastel(_TAB10[7]), ls=':', lw=2.0, label='Random'),
}
assert set(CURVE_STYLE) == set(ARMS)


# ── Config ──────────────────────────────────────────────────────────────────


def _is_pos_int(v) -> bool:
    """True for a non-bool int > 0.

    Args:
        v: Any value.

    Returns:
        Whether ``v`` is a positive int.
    """
    return is_int(v) and v > 0


def load_clustering_config(path: Path) -> dict:
    """Load and validate a clustering config.

    Exactly one of ``attributes`` and ``attribute_set_sizes`` is set. ``gammas`` is
    {start, stop, num} for np.linspace with start == 0 (needed for the gamma = 0 check).
    ``n_init`` (GT reference fits) must be >= 2 for the CI.

    Args:
        path: Config path.

    Returns:
        The config dict.

    Raises:
        ValueError, KeyError: A key is missing, extra or malformed.
    """
    cfg = _load_envelope(path, 'clustering')
    unexpected = set(cfg['params']) - {SECTION}
    if unexpected:
        raise ValueError(f"{path}: unexpected params key(s) {sorted(unexpected)}")
    sec = get_section(cfg, SECTION, SECTION_KEYS)

    if not is_int(cfg['seed']) or cfg['seed'] < 0:
        raise ValueError(f"{path}: seed must be a non-negative int, got {cfg['seed']!r}")
    for k in ('calibration_config_id', 'extraction_id', 'judge_combine_id', 'judge_model'):
        if not isinstance(sec[k], str) or not sec[k]:
            raise ValueError(f"{path}: {SECTION}.{k} must be a non-empty string, got {sec[k]!r}")
    if sec['calibration_version'] not in CALIBRATION_LOADERS:
        raise ValueError(f"{path}: {SECTION}.calibration_version must be one of {sorted(CALIBRATION_LOADERS)}, "
                         f"got {sec['calibration_version']!r}")
    if sec['probe_train_dataset'] not in PROBE_TRAIN_DATASETS:
        raise ValueError(f"{path}: {SECTION}.probe_train_dataset must be one of {PROBE_TRAIN_DATASETS}, "
                         f"got {sec['probe_train_dataset']!r} (cross-domain cells are not wired up)")
    if sec['rows'] not in ROWS_CHOICES:
        raise ValueError(f"{path}: {SECTION}.rows must be one of {ROWS_CHOICES}, got {sec['rows']!r}")
    if sec['rows'] == 'deduplicated':
        if not isinstance(sec['deduplication_config_id'], str) or not sec['deduplication_config_id']:
            raise ValueError(f"{path}: {SECTION}.deduplication_config_id must be a non-empty string when rows is 'deduplicated'")
        if sec['confidence'] not in CONFIDENCE_CHOICES:
            raise ValueError(f"{path}: {SECTION}.confidence must be one of {CONFIDENCE_CHOICES} when rows is "
                             f"'deduplicated', got {sec['confidence']!r}")
    else:
        for k in ('deduplication_config_id', 'confidence'):
            if sec[k] is not None:
                raise ValueError(f"{path}: {SECTION}.{k} must be null unless rows is 'deduplicated', got {sec[k]!r}")

    mt = sec['missing_threshold']
    if isinstance(mt, bool) or not isinstance(mt, (int, float)) or not 0 <= mt < 1:
        raise ValueError(f"{path}: {SECTION}.missing_threshold must be a number in [0, 1), got {mt!r}")
    attrs, sizes = sec['attributes'], sec['attribute_set_sizes']
    if (attrs is None) == (sizes is None):
        raise ValueError(f"{path}: exactly one of {SECTION}.attributes and {SECTION}.attribute_set_sizes must be "
                         f"set, the other null; got attributes={attrs!r}, attribute_set_sizes={sizes!r}")
    if attrs is not None:
        if (not isinstance(attrs, list) or not attrs or not all(isinstance(a, str) and a for a in attrs)
                or len(set(attrs)) != len(attrs)):
            raise ValueError(f"{path}: {SECTION}.attributes must be a non-empty list of distinct non-empty "
                             f"strings, got {attrs!r}")
    elif (not isinstance(sizes, list) or not sizes or not all(_is_pos_int(s) for s in sizes)
            or any(a >= b for a, b in zip(sizes, sizes[1:]))):
        raise ValueError(f"{path}: {SECTION}.attribute_set_sizes must be a non-empty strictly increasing list of "
                         f"positive ints, got {sizes!r}")
    if not is_int(sec['n_clusters']) or sec['n_clusters'] < 2:
        raise ValueError(f"{path}: {SECTION}.n_clusters must be an int >= 2, got {sec['n_clusters']!r}")
    for k in ('knn_neighbors', 'n_runs'):
        if not _is_pos_int(sec[k]):
            raise ValueError(f"{path}: {SECTION}.{k} must be a positive int, got {sec[k]!r}")
    if not is_int(sec['n_init']) or sec['n_init'] < 2:
        raise ValueError(f"{path}: {SECTION}.n_init must be an int >= 2 (the CI needs two GT fits), "
                         f"got {sec['n_init']!r}")
    g = sec['gammas']
    if not isinstance(g, dict) or set(g) != set(GAMMA_KEYS):
        raise ValueError(f"{path}: {SECTION}.gammas must have exactly the keys {list(GAMMA_KEYS)}, got {g!r}")
    if (any(isinstance(g[k], bool) or not isinstance(g[k], (int, float)) for k in ('start', 'stop'))
            or g['start'] != 0 or not g['stop'] > g['start'] or not is_int(g['num']) or g['num'] < 2):
        raise ValueError(f"{path}: {SECTION}.gammas must have start == 0, stop > start, int num >= 2, got {g!r}")
    return cfg


def gamma_grid(sec: dict) -> np.ndarray:
    """The gamma sweep from the config.

    Args:
        sec: The ``clustering`` section.

    Returns:
        ``np.linspace(start, stop, num)``.
    """
    g = sec['gammas']
    return np.linspace(float(g['start']), float(g['stop']), g['num'])


def resolve_clustering_inputs(cfg: dict) -> dict:
    """Resolve inputs and check the declared extraction, judge_combine and judge match them.

    Args:
        cfg: Loaded clustering config.

    Returns:
        Output of ``resolve_meta_inputs``.

    Raises:
        ValueError: A declared id or judge differs from the calibration config's.
    """
    sec = cfg['params'][SECTION]
    inputs = resolve_meta_inputs(sec, DATASET)
    got = {'extraction_id': inputs['extraction_id'], 'judge_combine_id': inputs['judge_combine_dir'].name,
           'judge_model': inputs['judge_model']}
    bad = {k: (sec[k], v) for k, v in got.items() if sec[k] != v}
    if bad:
        raise ValueError(f"{cfg['id']}: declared != calibration config {sec['calibration_config_id']} "
                         f"(declared, actual): {bad}")
    return inputs


# ── Row filters ─────────────────────────────────────────────────────────────

def filter_ecosystems(df: pd.DataFrame, entity_cols: list[str]) -> pd.DataFrame:
    """Keep rows whose ecosystem bucket is in ECOSYSTEMS (drops 'other').

    Args:
        df: Rows with ``ecosystem_bucket``.
        entity_cols: Columns identifying an entity; each entity must have one bucket,
            so the filter keeps or drops whole entities.

    Returns:
        The kept rows (index reset).

    Raises:
        ValueError: An entity's rows span several buckets.
    """
    n_buckets = df.groupby(entity_cols, dropna=False)['ecosystem_bucket'].nunique()
    if (n_buckets > 1).any():
        raise ValueError(f'{int((n_buckets > 1).sum())} entities span several ecosystem buckets')
    return df[df['ecosystem_bucket'].isin(ECOSYSTEMS)].reset_index(drop=True)


def drop_nonpositive_log(df: pd.DataFrame) -> pd.DataFrame:
    """Drop valued rows of LOG_SCALE_ATTRIBUTES with converted_value <= 0 (no log10).

    Args:
        df: Rows with ``attribute`` and ``converted_value``.

    Returns:
        The kept rows (index reset).
    """
    bad = df['attribute'].isin(LOG_SCALE_ATTRIBUTES) & (df['converted_value'] <= 0)
    return df[~bad].reset_index(drop=True)


# ── Matrices ────────────────────────────────────────────────────────────────

def cell_matrix(df: pd.DataFrame, entity_col: str, conf_cols: list[str]):
    """Pivot long rows into entity x attribute matrices.

    Rows without ``converted_value`` are dropped. A cell's value is the median of its
    rows and each confidence the mean over the same rows.

    Args:
        df: Long rows.
        entity_col: Entity id column.
        conf_cols: Confidence columns to aggregate.

    Returns:
        Tuple of:
            - value matrix (sorted entities x sorted attributes)
            - ``{col: confidence matrix}``, NaN exactly where value is NaN
            - row count per (entity, attribute) cell

    Raises:
        ValueError: NaN confidence or missing entity id on a valued row.
    """
    d = df.dropna(subset=['converted_value'])
    if d[conf_cols].isna().any().any():
        raise ValueError(f'NaN confidence in rows with a value: {d[conf_cols].isna().sum().to_dict()}')
    if d[entity_col].isna().any():
        raise ValueError(f'{int(d[entity_col].isna().sum())} rows have no {entity_col!r}')
    g = d.groupby([entity_col, 'attribute'], sort=True)
    agg = g['converted_value'].median().to_frame('value')
    for c in conf_cols:
        agg[c] = g[c].mean()
    agg['n_rows'] = g.size()
    assert int(agg['n_rows'].sum()) == len(d), 'cell aggregation dropped or duplicated rows'
    value = agg['value'].unstack('attribute')
    value.columns.name = None
    confs = {}
    for c in conf_cols:
        m = agg[c].unstack('attribute').reindex(index=value.index, columns=value.columns)
        m.columns.name = None
        assert (m.isna().to_numpy() == value.isna().to_numpy()).all(), f'{c} and value cells misaligned'
        confs[c] = m
    return value, confs, agg['n_rows']


def dense_submatrix(m: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Greedily drop the row with the most NaNs until the missing fraction is <= threshold.

    Args:
        m: Matrix with NaNs.
        threshold: Maximum missing fraction.

    Returns:
        Remaining rows (may be empty).
    """
    out = m
    while len(out) and out.isna().to_numpy().mean() > threshold:
        out = out.drop(index=out.isna().sum(axis=1).idxmax())
    return out


def enumerate_attribute_sets(value: pd.DataFrame, sizes: list[int], threshold: float) -> pd.DataFrame:
    """Score every attribute subset of the given sizes by its densified size.

    Args:
        value: GT value matrix.
        sizes: Subset sizes to try.
        threshold: Missing-fraction threshold for ``dense_submatrix``.

    Returns:
        One row per subset (combinations order): attributes, d, n_rows, missing_frac,
        score = n_rows * d.

    Raises:
        ValueError: A size exceeds the number of attributes.
    """
    if max(sizes) > value.shape[1]:
        raise ValueError(f'attribute_set_sizes {sizes} exceed the {value.shape[1]} attributes {list(value.columns)}')
    rows = []
    for d in sizes:
        for cols in combinations(value.columns, d):
            sub = dense_submatrix(value[list(cols)], threshold)
            rows.append(dict(attributes='|'.join(cols), d=d, n_rows=len(sub),
                             missing_frac=float(sub.isna().to_numpy().mean()) if len(sub) else np.nan,
                             score=len(sub) * d))
    return pd.DataFrame(rows)


def fixed_attribute_set(value: pd.DataFrame, attrs: list[str], threshold: float) -> pd.DataFrame:
    """Like ``enumerate_attribute_sets`` but for one given attribute list.

    Args:
        value: GT value matrix.
        attrs: Attributes to use.
        threshold: Missing-fraction threshold.

    Returns:
        One-row DataFrame with the same columns.

    Raises:
        ValueError: An attribute has no valued rows.
    """
    missing = [a for a in attrs if a not in value.columns]
    if missing:
        raise ValueError(f'attributes {missing} have no valued rows; available: {list(value.columns)}')
    sub = dense_submatrix(value[attrs], threshold)
    return pd.DataFrame([dict(attributes='|'.join(attrs), d=len(attrs), n_rows=len(sub),
                              missing_frac=float(sub.isna().to_numpy().mean()) if len(sub) else np.nan,
                              score=len(sub) * len(attrs))])


def log_scale(m: pd.DataFrame) -> pd.DataFrame:
    """log10 the LOG_SCALE_ATTRIBUTES columns; other columns unchanged.

    Args:
        m: Matrix with NaNs; log columns must be > 0 where observed.

    Returns:
        Copy with log columns replaced by their log10.

    Raises:
        ValueError: A non-positive value in a log column.
    """
    x = m.copy()
    for c in x.columns:
        if c in LOG_SCALE_ATTRIBUTES:
            if (x[c] <= 0).any():
                raise ValueError(f'{int((x[c] <= 0).sum())} non-positive values in log-scaled column {c!r}')
            x[c] = np.log10(x[c])
    return x


def _impute(scaled: np.ndarray, knn_neighbors: int) -> np.ndarray:
    """KNN-impute (distance-weighted) a scaled matrix, fit on that matrix alone.

    Args:
        scaled: Standardized matrix with NaNs.
        knn_neighbors: Neighbours for imputation.

    Returns:
        Finite array of the same shape.
    """
    X = KNNImputer(n_neighbors=knn_neighbors, weights='distance').fit_transform(scaled)
    assert X.shape == scaled.shape, f'KNNImputer changed shape {scaled.shape} -> {X.shape} (an all-NaN column?)'
    assert np.isfinite(X).all(), 'non-finite values after scaling / imputation'
    return X


def process_matrices(gt: pd.DataFrame, ext: pd.DataFrame, knn_neighbors: int):
    """log10, then standardize both sides with a scaler fit on the GT alone, then KNN-impute each side.

    One GT-fitted transform puts both sides (and so both centroid sets) in the same
    coordinates -- GT standard deviations from the GT mean -- so a systematic shift or
    inflated spread in the extraction shows up as distance instead of being scaled away.
    The scaler ignores NaNs, so imputation runs on scaled features; each side's imputer
    is fit on that side only.

    Args:
        gt: GT dense matrix with NaNs.
        ext: Extracted dense matrix, same columns in the same order.
        knn_neighbors: Neighbours for imputation.

    Returns:
        Tuple of:
            - X_gt, X_ext: finite arrays of the input shapes
            - the scaler's per-column ``mean`` and ``scale`` (log10 units for log columns)

    Raises:
        ValueError: Column mismatch, a non-positive value in a log column, or a GT column
            with fewer than two distinct observed values.
    """
    if list(gt.columns) != list(ext.columns):
        raise ValueError(f'GT columns {list(gt.columns)} != extracted columns {list(ext.columns)}')
    gt_log, ext_log = log_scale(gt), log_scale(ext)
    for c in gt_log.columns:
        if gt_log[c].nunique() < 2:
            raise ValueError(f'GT column {c!r} has fewer than two distinct observed values: cannot standardize')
    scaler = StandardScaler().fit(gt_log)
    out = []
    for x in (gt_log, ext_log):
        scaled = scaler.transform(x)
        assert (np.isnan(scaled) == x.isna().to_numpy()).all(), 'scaling changed the NaN pattern'
        out.append(_impute(scaled, knn_neighbors))
    params = {c: dict(mean=float(mu), scale=float(s)) for c, mu, s in zip(gt.columns, scaler.mean_, scaler.scale_)}
    return out[0], out[1], params


def entity_confidence(conf: pd.DataFrame) -> np.ndarray:
    """Entity confidence: mean of its observed cells' confidences.

    Args:
        conf: Entity x attribute confidence matrix (NaN = unobserved).

    Returns:
        One confidence per entity, in [0, 1].

    Raises:
        ValueError: An entity has no observed cell.
    """
    if conf.isna().all(axis=1).any():
        raise ValueError(f'{int(conf.isna().all(axis=1).sum())} entities have no observed cell')
    p = conf.mean(axis=1, skipna=True).to_numpy(dtype=float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'entity confidence outside [0, 1]'
    return p


# ── Clustering ──────────────────────────────────────────────────────────────

def kmeans_seed(*words: int) -> int:
    """A KMeans random_state derived from integer seed words.

    Args:
        words: Seed words, e.g. (seed, stream, i, r).

    Returns:
        A uint32-range int.
    """
    return int(np.random.SeedSequence([int(w) for w in words]).generate_state(1)[0])


def fit_kmeans(X: np.ndarray, n_clusters: int, random_state: int, sample_weight=None) -> KMeans:
    """One single-start (n_init=1) KMeans fit; a ConvergenceWarning is an error.

    Args:
        X: Feature matrix.
        n_clusters: Number of clusters.
        random_state: KMeans seed.
        sample_weight: Optional per-row weights.

    Returns:
        The fitted KMeans.
    """
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=1)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        km.fit(X, sample_weight=sample_weight)
    return km


def centroid_matching_distance(A: np.ndarray, B: np.ndarray, metric: str = 'euclidean') -> float:
    """Mean distance between two centroid sets under the optimal (Hungarian) matching.

    Args:
        A: Centroids, (k, d).
        B: Centroids, (k, d).
        metric: cdist metric.

    Returns:
        Mean matched distance.
    """
    D = cdist(A, B, metric=metric)
    row_ind, col_ind = linear_sum_assignment(D, maximize=False)
    return float(D[row_ind, col_ind].mean())


def distance_curve(X: np.ndarray, conf: np.ndarray, gt_centers: np.ndarray, n_clusters: int,
                   gammas: np.ndarray, random_state: int) -> np.ndarray:
    """Centroid distance to the GT of weighted KMeans, for each gamma.

    Args:
        X: Extracted feature matrix.
        conf: Entity confidences (weights are conf ** gamma).
        gt_centers: Reference centroids.
        n_clusters: Number of clusters.
        gammas: Weight exponents.
        random_state: KMeans seed, the same for every gamma.

    Returns:
        One distance per gamma.
    """
    out = np.empty(len(gammas))
    for i, gamma in enumerate(gammas):
        km = fit_kmeans(X, n_clusters, random_state, sample_weight=conf ** gamma)
        out[i] = centroid_matching_distance(gt_centers, km.cluster_centers_)
    return out


def random_confidence(seed: int, i: int, r: int, n: int) -> np.ndarray:
    """The random arm's U(0, 1) entity confidences for trial (i, r).

    Args:
        seed: Global seed.
        i: GT reference fit.
        r: Extraction run.
        n: Number of entities.

    Returns:
        ``n`` draws from U(0, 1).
    """
    return np.random.default_rng([seed, RANDOM_STREAM, i, r]).uniform(0.0, 1.0, n)


def kish_n_eff(w: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2.

    Args:
        w: Nonnegative weights with positive sum.

    Returns:
        Effective sample size.
    """
    assert (w >= 0).all() and w.sum() > 0
    return float(w.sum() ** 2 / (w ** 2).sum())


def block_mean_ci(d: np.ndarray, level: float = CI_LEVEL) -> dict:
    """Grand mean and t-interval over the per-reference-fit (block) means.

    Trials sharing a GT reference fit are correlated, so the n_init block means, not
    the n_init x n_runs trials, are treated as the independent units.

    Args:
        d: Distances, (n_init, n_runs).
        level: Confidence level.

    Returns:
        Dict with ``mean``, ``ci_lo``, ``ci_hi``, ``se`` (SE of the grand mean).
    """
    assert d.ndim == 2 and d.shape[0] >= 2 and np.isfinite(d).all(), d.shape
    blocks = d.mean(axis=1)
    mean = float(blocks.mean())
    se = float(blocks.std(ddof=1) / np.sqrt(blocks.size))
    half = float(stats.t.ppf(0.5 + level / 2, blocks.size - 1)) * se
    return dict(mean=mean, ci_lo=mean - half, ci_hi=mean + half, se=se)


# ── Plots ───────────────────────────────────────────────────────────────────

def plot_legend(out_path: Path) -> None:
    """Save the standalone legend for the five arms (pond_meta_analysis.py style; 3 columns pair solid / dashed / random).

    Args:
        out_path: Figure path.
    """
    order = ('probe', 'ntp', 'probe_conf', 'ntp_conf', 'random')
    handles = [mlines.Line2D([], [], color=CURVE_STYLE[a]['color'], lw=4 if CURVE_STYLE[a]['ls'] == '-' else 2.5,
                             linestyle=CURVE_STYLE[a]['ls'], label=CURVE_STYLE[a]['label']) for a in order]
    fig, ax = plt.subplots(figsize=(10.0, 0.7))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=3, fontsize=12, frameon=False, handlelength=2.0)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[clustering] wrote {out_path}")


def plot_center_dist(summary: pd.DataFrame, gammas: np.ndarray, out_path: Path) -> None:
    """Plot mean centroid distance vs gamma for every arm, with its 95% CI over KMeans initializations.

    Args:
        summary: centroid_distance.csv frame.
        gammas: Gamma grid.
        out_path: Figure path.
    """
    fig, ax = plt.subplots(figsize=(3.0, 2.8))
    for arm in ARMS:
        s = summary[summary['arm'] == arm].sort_values('gamma')
        assert np.allclose(s['gamma'].to_numpy(), gammas)
        style = CURVE_STYLE[arm]
        ax.plot(gammas, s['mean'].to_numpy(), **style)
        ax.fill_between(gammas, s['ci_lo'].to_numpy(), s['ci_hi'].to_numpy(), color=style['color'], alpha=0.2,
                        linewidth=0)
    ax.set_xlabel('$\\gamma$', fontsize=11)
    ax.set_ylabel('Mean Centroid Distance', fontsize=11)
    ax.set_xlim(gammas[0], gammas[-1])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)
    print(f"[clustering] wrote {out_path}")


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    """CLI: build matrices, run all sweeps and known-answer checks, write outputs."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('config', type=Path, help='analysis/analysis-configs/clustering/<id>.yaml')
    args = ap.parse_args()

    cfg = load_clustering_config(args.config)
    sec = cfg['params'][SECTION]
    seed = cfg['seed']
    gammas = gamma_grid(sec)
    n_clusters, threshold = sec['n_clusters'], sec['missing_threshold']
    n_init, n_runs = sec['n_init'], sec['n_runs']
    out_dir = CLUSTERING_ROOT / cfg['id']
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)

    inputs = resolve_clustering_inputs(cfg)
    gt_df, ext_df, manifest = load_data(sec, inputs, unit_conversion=UNIT_CONVERSION_V2)
    # Per-row factor on all rows, as in pond_meta_analysis.py (it is per (bucket, attribute),
    # so the ecosystem filter below doesn't change kept rows' factors).
    ext_df, moments = add_outlier_columns(ext_df, True)
    print(f"[clustering] non-outlier factor: exp(-z^2/2), robust z per (ecosystem, attribute)\n{moments}")

    # ── Row filters ──
    n_rows_loaded = {'gt': len(gt_df), 'ext': len(ext_df)}
    gt_df = filter_ecosystems(gt_df, GT_ENTITY_COLS)
    ext_df = filter_ecosystems(ext_df, ['entity_id'])
    n_rows_ecosystem = {'gt': len(gt_df), 'ext': len(ext_df)}
    gt_df, ext_df = drop_nonpositive_log(gt_df), drop_nonpositive_log(ext_df)
    n_rows_kept = {'gt': len(gt_df), 'ext': len(ext_df)}
    print(f"[clustering] rows: loaded {n_rows_loaded} -> ecosystems {ECOSYSTEMS} {n_rows_ecosystem} "
          f"-> positive log-scale values {n_rows_kept}")

    # ── Entity x attribute matrices ──
    span = ext_df.groupby('entity_id')['document_id'].nunique()
    assert (span == 1).all(), f'{int((span > 1).sum())} extracted entity_ids span several documents'
    gt_df = gt_df.copy()
    gt_df['gt_entity'] = gt_df.groupby(GT_ENTITY_COLS, dropna=False).ngroup()
    gt_val, _, gt_cell_rows = cell_matrix(gt_df, 'gt_entity', [])
    ext_val, ext_conf, ext_cell_rows = cell_matrix(ext_df, 'entity_id', list(ARM_PROB_COL.values()))

    # ── Attribute subset, chosen on the GT ──
    if sec['attributes'] is not None:
        sets = fixed_attribute_set(gt_val, sec['attributes'], threshold)
    else:
        sets = enumerate_attribute_sets(gt_val, sec['attribute_set_sizes'], threshold)
    best_i = int(sets['score'].idxmax())  # first in enumeration order among ties
    sets['chosen'] = sets.index == best_i
    keep_attrs = sets.loc[best_i, 'attributes'].split('|')
    if sec['attributes'] is not None:
        assert keep_attrs == sec['attributes'] and len(sets) == 1
    n_ties = int((sets['score'] == sets.loc[best_i, 'score']).sum())
    print(f"[clustering] keep_attrs={keep_attrs} (score {sets.loc[best_i, 'score']}, {n_ties} tied)")
    missing_ext = sorted(set(keep_attrs) - set(ext_val.columns))
    assert not missing_ext, f'extraction has no valued rows for {missing_ext}'

    gt_dense = dense_submatrix(gt_val[keep_attrs], threshold)
    ext_dense = dense_submatrix(ext_val[keep_attrs], threshold)
    for name, m in (('gt', gt_dense), ('ext', ext_dense)):
        assert len(m) >= n_clusters, f'{name} dense matrix has {len(m)} entities < n_clusters={n_clusters}'
    X_gt, X_ext, scaler_params = process_matrices(gt_dense, ext_dense, sec['knn_neighbors'])
    conf = {arm: entity_confidence(ext_conf[col].loc[ext_dense.index, keep_attrs])
            for arm, col in ARM_PROB_COL.items()}
    gt_bucket = gt_df.groupby('gt_entity')['ecosystem_bucket'].first()
    ext_bucket = ext_df.groupby('entity_id')['ecosystem_bucket'].first()
    dense_by_eco = {'gt': gt_bucket.loc[gt_dense.index].value_counts().to_dict(),
                    'ext': ext_bucket.loc[ext_dense.index].value_counts().to_dict()}
    print(f"[clustering] dense matrices: gt {X_gt.shape}, ext {X_ext.shape}; by ecosystem {dense_by_eco}")

    # ── Sweeps ──
    gt_centers = []
    for i in range(n_init):
        rs = kmeans_seed(seed, GT_STREAM, i)
        centers = fit_kmeans(X_gt, n_clusters, rs).cluster_centers_
        # Known answer: refitting the GT with the same seed recovers its centroids.
        self_dist = centroid_matching_distance(centers, fit_kmeans(X_gt, n_clusters, rs).cluster_centers_)
        assert self_dist < 1e-9, f'GT fit {i}: refit is {self_dist} from its own centroids'
        gt_centers.append(centers)
    gt_spread = [centroid_matching_distance(gt_centers[0], c) for c in gt_centers[1:]]
    print(f"[clustering] GT reference fits: mean distance of fits 1.. to fit 0 = {np.mean(gt_spread):.3f}")

    dist = {arm: np.empty((n_init, n_runs, len(gammas))) for arm in ARMS}
    n_eff_random = np.empty((n_init, n_runs, len(gammas)))
    for i in range(n_init):
        for r in range(n_runs):
            rs = kmeans_seed(seed, EXT_STREAM, i, r)   # shared by every arm
            rconf = random_confidence(seed, i, r, len(X_ext))
            for arm in ARMS:
                c = rconf if arm == RANDOM_ARM else conf[arm]
                dist[arm][i, r] = distance_curve(X_ext, c, gt_centers[i], n_clusters, gammas, rs)
            n_eff_random[i, r] = [kish_n_eff(rconf ** g) for g in gammas]
        print(f"[clustering] GT fit {i + 1}/{n_init} done")

    # Known answer: gamma = 0 makes every weight 1, so every arm runs the same KMeans fits.
    assert gammas[0] == 0
    for arm in ARMS:
        assert np.array_equal(dist[arm][:, :, 0], dist[ARMS[0]][:, :, 0]), f'gamma=0: {arm} differs from {ARMS[0]}'

    # Kish n_eff of conf**gamma; the random arm's is the mean over its draws.
    n_eff = {arm: [kish_n_eff(conf[arm] ** g) for g in gammas] for arm in ARM_PROB_COL}
    n_eff[RANDOM_ARM] = n_eff_random.mean(axis=(0, 1)).tolist()
    summary = pd.DataFrame([
        dict(arm=arm, gamma=float(g), n_init=n_init, n_runs=n_runs, **block_mean_ci(v[:, :, j]), n_eff=n_eff[arm][j])
        for arm, v in dist.items() for j, g in enumerate(gammas)
    ])

    # ── Outputs ──
    sets.to_csv(out_dir / 'attribute_sets.csv', index=False)
    summary.to_csv(out_dir / 'centroid_distance.csv', index=False)
    np.savez(out_dir / 'distances.npz', gammas=gammas, gt_centers=np.stack(gt_centers), **dist)
    plot_center_dist(summary, gammas, figures_dir / 'center_dist.pdf')
    plot_legend(figures_dir / 'legend.pdf')
    manifest.update(
        analysis_config_id=cfg['id'], script='analysis/pond_clustering.py', seed=seed, params=sec,
        inputs={k: str(v) for k, v in inputs.items()},
        arms={**ARM_PROB_COL, RANDOM_ARM: 'U(0, 1) per entity per (i, r) trial'},
        ci=f'{CI_LEVEL:.0%} t-interval over the n_init per-GT-fit mean distances (KMeans initializations only)',
        outlier_moments=moments.to_dict('records'),
        ecosystems=ECOSYSTEMS, log_scale_attributes=sorted(set(keep_attrs) & LOG_SCALE_ATTRIBUTES),
        gt_fitted_scaler=scaler_params,
        n_rows_loaded=n_rows_loaded, n_rows_after_ecosystem_filter=n_rows_ecosystem,
        n_rows_after_nonpositive_log_drop=n_rows_kept,
        keep_attrs=keep_attrs, n_tied_attribute_sets=n_ties,
        n_gt_entities=int(len(gt_val)), n_ext_entities=int(len(ext_val)),
        n_gt_multi_row_cells=int((gt_cell_rows > 1).sum()), n_ext_multi_row_cells=int((ext_cell_rows > 1).sum()),
        n_gt_dense=int(len(gt_dense)), n_ext_dense=int(len(ext_dense)), n_dense_by_ecosystem=dense_by_eco,
        gt_dense_missing_frac=float(gt_dense.isna().to_numpy().mean()),
        ext_dense_missing_frac=float(ext_dense.isna().to_numpy().mean()),
        gt_reference_fit_spread=float(np.mean(gt_spread)),
        ext_confidence_mean={arm: float(c.mean()) for arm, c in conf.items()},
    )
    (out_dir / 'meta.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[clustering] wrote {out_dir}")


if __name__ == '__main__':
    main()
