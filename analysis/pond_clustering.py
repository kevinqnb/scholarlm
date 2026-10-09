"""Does weighting extracted pond entities by probe / NTP confidence make KMeans recover the ground truth's clusters?

Usage: python analysis/pond_clustering.py analysis/analysis-configs/clustering/<id>.yaml

1. Rows and confidences come from pond_meta.load_data (held-out documents only).
2. Build entity x attribute matrices: one cell per (entity, attribute), holding the
   median value and mean confidence of its rows. GT entities are distinct (document_id,
   name, ecosystem); extracted entities are entity_ids.
3. Pick the attributes: a fixed list, or the subset (of the given sizes) whose densified
   GT matrix maximises n_rows * d.
4. Densify, KNN-impute and standardize the GT and extracted matrices separately. Entity
   confidence = product of its observed cells' confidences.
5. KMeans on the GT gives reference centroids. For each gamma, run KMeans on the
   extraction with weights conf ** gamma and score the mean Hungarian-matched centroid
   distance. A shuffled-confidence arm per method is the permutation control.

With outlier_adjust, row confidences are first scaled by outlier_weight.py, and the
shuffled arms permute raw row confidences (RowShuffler). Asserted known answers: the
GT refit is at distance 0, and at gamma = 0 all arms give identical fits.

Outputs in analysis/results/clustering/<config id>/: attribute_sets.csv,
centroid_distance.csv (mean, se, Kish n_eff per arm and gamma), distances.npz,
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
import matplotlib.lines as mlines
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from analysis.common.config import _load_envelope, analysis_results_dir, get_section, is_int
from analysis.common.outlier_weight import add_outlier_columns
from analysis.common.meta_inputs import CALIBRATION_LOADERS, CONFIDENCE_CHOICES, ROWS_CHOICES, resolve_meta_inputs
from analysis.common.pond_meta import DATASET, PAPER_RCPARAMS, UNIT_CONVERSION, load_data

# Base style; _apply_style() overrides part of it before plotting.
mpl.rcParams.update(PAPER_RCPARAMS)

SECTION = 'clustering'
SECTION_KEYS = (
    # inputs (ids are also declared here and cross-checked against the calibration config)
    'calibration_config_id', 'calibration_version', 'extraction_id', 'judge_combine_id', 'judge_model',
    'probe_train_dataset', 'rows', 'deduplication_config_id', 'confidence',
    # method
    'missing_threshold', 'attributes', 'attribute_set_sizes', 'n_clusters', 'knn_neighbors', 'gammas',
    'n_runs', 'n_shuffle_samples', 'outlier_adjust',
)
GAMMA_KEYS = ('start', 'stop', 'num')
# Only pond -> pond is supported; load_data reads that cell only.
PROBE_TRAIN_DATASETS = (DATASET,)
GT_ENTITY_COLS = ['document_id', 'name', 'ecosystem']
CONF_COLS = {'ntp': 'ntp_prob', 'probe': 'probe_prob'}
ARMS = ('ntp', 'probe', 'ntp_shuffled', 'probe_shuffled')
# Third rng seed word per shuffled arm (rng [seed, sample, stream]).
SHUFFLE_STREAMS = {'ntp': 1, 'probe': 2}
CLUSTERING_ROOT = analysis_results_dir('clustering')

# blue: 7, orange: 1, red: 0, green: 4
palette = sns.color_palette('husl', 10)
ARM_STYLE = {
    'ntp':    dict(color=palette[2], ls='-', lw=3.0, alpha=0.85, label='NTP'),
    'probe':  dict(color=palette[7], ls='-', lw=3.0, label='Probe'),
    'ntp_shuffled':   dict(color=palette[2], ls=':', lw=2.0, alpha=0.85, label='NTP (shuffled)'),
    'probe_shuffled': dict(color=palette[7], ls=':', lw=2.0, alpha=0.85, label='Probe (shuffled)'),
}


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
    for k in ('knn_neighbors', 'n_runs', 'n_shuffle_samples'):
        if not _is_pos_int(sec[k]):
            raise ValueError(f"{path}: {SECTION}.{k} must be a positive int, got {sec[k]!r}")
    if not isinstance(sec['outlier_adjust'], bool):
        raise ValueError(f"{path}: {SECTION}.outlier_adjust must be a bool, got {sec['outlier_adjust']!r}")
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


def process_matrix(m: pd.DataFrame, knn_neighbors: int) -> np.ndarray:
    """KNN-impute (distance-weighted) and standardize, both fit on ``m`` alone.

    Args:
        m: Dense-ish matrix with NaNs.
        knn_neighbors: Neighbours for imputation.

    Returns:
        Finite standardized array of the same shape.
    """
    imputed = KNNImputer(n_neighbors=knn_neighbors, weights='distance').fit_transform(m)
    assert imputed.shape == m.shape, f'KNNImputer changed shape {m.shape} -> {imputed.shape} (an all-NaN column?)'
    X = StandardScaler().fit_transform(imputed)
    assert np.isfinite(X).all(), 'non-finite values after imputation / scaling (a constant column?)'
    return X


def entity_confidence(conf: pd.DataFrame) -> np.ndarray:
    """Entity confidence: product of its observed cells' confidences.

    Args:
        conf: Entity x attribute confidence matrix (NaN = unobserved).

    Returns:
        One confidence per entity, in [0, 1].

    Raises:
        ValueError: An entity has no observed cell.
    """
    if conf.isna().all(axis=1).any():
        raise ValueError(f'{int(conf.isna().all(axis=1).sum())} entities have no observed cell')
    p = conf.prod(axis=1, min_count=1).to_numpy(dtype=float)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'entity confidence outside [0, 1]'
    return p


class RowShuffler:
    """Row-level permutation control for one method, used when outlier_adjust is on.

    Each draw permutes raw row confidences within each attribute (over all valued rows),
    reapplies each row's outlier factor, and re-aggregates like cell_matrix /
    entity_confidence.

    Args:
        ext_df: Extracted rows with ``{prob_col}_raw`` and ``outlier_factor``.
        prob_col: Confidence column.
        dense_index: Entity ids of the dense matrix.
        keep_attrs: Attributes clustered on.
    """

    def __init__(self, ext_df: pd.DataFrame, prob_col: str, dense_index: pd.Index, keep_attrs: list[str]):
        """Precompute row-to-cell and cell-to-matrix indices (see class docstring for args)."""
        d = ext_df.dropna(subset=['converted_value'])
        d = d[d['attribute'].isin(keep_attrs)]
        self.raw = d[f'{prob_col}_raw'].to_numpy(dtype=float)
        self.factor = d['outlier_factor'].to_numpy(dtype=float)
        assert np.isfinite(self.raw).all() and np.isfinite(self.factor).all()
        attr_pos = pd.Categorical(d['attribute'], categories=keep_attrs).codes
        self.attr_rows = [np.flatnonzero(attr_pos == j) for j in range(len(keep_attrs))]
        assert all(len(r) > 0 for r in self.attr_rows), 'a kept attribute has no valued extracted rows'
        code = d.groupby(['entity_id', 'attribute'], sort=False).ngroup().to_numpy()
        self.cell_code, self.n_cells = code, int(code.max()) + 1
        self.cell_count = np.bincount(code, minlength=self.n_cells)
        first = pd.DataFrame({'code': code, 'entity_id': d['entity_id'].to_numpy(),
                              'attr_pos': attr_pos}).drop_duplicates('code').sort_values('code')
        assert len(first) == self.n_cells
        self.cell_ent = dense_index.get_indexer(first['entity_id'])
        self.cell_attr = first['attr_pos'].to_numpy()
        self.in_dense = self.cell_ent >= 0
        self.shape = (len(dense_index), len(keep_attrs))

    def confidence(self, raw: np.ndarray) -> np.ndarray:
        """Entity confidences from per-row raw confidences.

        Args:
            raw: Raw confidence per valued row (same order as ``self.raw``).

        Returns:
            One confidence per dense entity.
        """
        cell_mean = np.bincount(self.cell_code, weights=raw * self.factor, minlength=self.n_cells) / self.cell_count
        m = np.full(self.shape, np.nan)
        m[self.cell_ent[self.in_dense], self.cell_attr[self.in_dense]] = cell_mean[self.in_dense]
        return entity_confidence(pd.DataFrame(m))

    def shuffled(self, rng: np.random.Generator) -> np.ndarray:
        """Entity confidences after permuting raw row confidences within each attribute.

        Args:
            rng: Random generator.

        Returns:
            One confidence per dense entity.
        """
        raw = self.raw.copy()
        for rows in self.attr_rows:
            raw[rows] = rng.permutation(self.raw[rows])
        return self.confidence(raw)


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
                   gammas: np.ndarray, kmeans_seed: int) -> np.ndarray:
    """Centroid distance to the GT of weighted KMeans, for each gamma.

    Args:
        X: Extracted feature matrix.
        conf: Entity confidences (weights are conf ** gamma).
        gt_centers: Reference centroids.
        n_clusters: Number of clusters.
        gammas: Weight exponents.
        kmeans_seed: KMeans random_state.

    Returns:
        One distance per gamma.

    Raises:
        ConvergenceWarning: KMeans did not converge (promoted to an error).
    """
    out = np.empty(len(gammas))
    for i, gamma in enumerate(gammas):
        km = KMeans(n_clusters=n_clusters, random_state=kmeans_seed, n_init='auto')
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            km.fit(X, sample_weight=conf ** gamma)
        out[i] = centroid_matching_distance(gt_centers, km.cluster_centers_)
    return out


def kish_n_eff(w: np.ndarray) -> float:
    """Kish effective sample size (sum w)^2 / sum w^2.

    Args:
        w: Nonnegative weights with positive sum.

    Returns:
        Effective sample size.
    """
    assert (w >= 0).all() and w.sum() > 0
    return float(w.sum() ** 2 / (w ** 2).sum())


# ── Plots ───────────────────────────────────────────────────────────────────

def _apply_style() -> None:
    """Switch matplotlib to this script's smaller figure style."""
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


def plot_legend(out_path: Path) -> None:
    """Save the standalone legend for the four arms.

    Args:
        out_path: Figure path.
    """
    order =('probe', 'ntp', 'probe_shuffled', 'ntp_shuffled')
    handles = [mlines.Line2D([], [], color=ARM_STYLE[a]['color'], lw=4 if ARM_STYLE[a]['ls'] == '-' else 2.5,
                             linestyle=ARM_STYLE[a]['ls'], label=ARM_STYLE[a]['label'])
               for a in order]
    fig, ax = plt.subplots(figsize=(10.0, 0.45))
    ax.axis('off')
    ax.legend(handles=handles, loc='center', ncol=len(order), fontsize=13, frameon=False, handlelength=2.0)
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


def plot_center_dist(summary: pd.DataFrame, gammas: np.ndarray, out_path: Path) -> None:
    """Plot mean centroid distance (± se) vs gamma for every arm.

    Args:
        summary: centroid_distance.csv frame.
        gammas: Gamma grid.
        out_path: Figure path.
    """
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for arm in ARMS:
        s = summary[summary['arm'] == arm].sort_values('gamma')
        assert np.allclose(s['gamma'].to_numpy(), gammas)
        mean, se = s['mean'].to_numpy(), s['se'].to_numpy()
        ax.plot(gammas, mean, **ARM_STYLE[arm])
        ax.fill_between(gammas, mean - se, mean + se, color=ARM_STYLE[arm]['color'], alpha=0.2)
    ax.grid(alpha=0.25, linestyle='-', linewidth=0.4)
    ax.set_xlabel('$\\gamma$', fontsize=11)
    ax.set_ylabel('Mean Centroid Distance', fontsize=11)
    ax.set_xlim(gammas[0], gammas[-1])
    ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%.2f'))
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', dpi=200)
    plt.close(fig)


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
    out_dir = CLUSTERING_ROOT / cfg['id']
    figures_dir = out_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    _apply_style()

    inputs = resolve_clustering_inputs(cfg)
    gt_df, ext_df, manifest = load_data(sec, inputs, unit_conversion=UNIT_CONVERSION)
    # Adjust per row, before aggregation into cells and entities.
    ext_df, moments = add_outlier_columns(ext_df, sec['outlier_adjust'])
    if moments is not None:
        print(f"[clustering] outlier_adjust: confidences x exp(-(x-mu)^2/2sigma^2)\n{moments}")

    # ── Entity x attribute matrices ──
    span = ext_df.groupby('entity_id')['document_id'].nunique()
    assert (span == 1).all(), f'{int((span > 1).sum())} extracted entity_ids span several documents'
    gt_df = gt_df.copy()
    gt_df['gt_entity'] = gt_df.groupby(GT_ENTITY_COLS, dropna=False).ngroup()
    gt_val, _, gt_cell_rows = cell_matrix(gt_df, 'gt_entity', [])
    conf_cols = list(CONF_COLS.values())
    ext_val, ext_conf, ext_cell_rows = cell_matrix(ext_df, 'entity_id', conf_cols)

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
    X_gt = process_matrix(gt_dense, sec['knn_neighbors'])
    X_ext = process_matrix(ext_dense, sec['knn_neighbors'])
    conf = {arm: entity_confidence(ext_conf[col].loc[ext_dense.index, keep_attrs]) for arm, col in CONF_COLS.items()}
    print(f"[clustering] dense matrices: gt {X_gt.shape}, ext {X_ext.shape}")

    # ── Sweeps ──
    gt_centers = KMeans(n_clusters=n_clusters, random_state=seed, n_init='auto').fit(X_gt).cluster_centers_
    # Known answer: refitting the GT itself with the reference seed recovers gt_centers.
    self_dist = distance_curve(X_gt, np.ones(len(X_gt)), gt_centers, n_clusters, np.array([0.0]), seed)[0]
    assert self_dist < 1e-9, f'GT refit is {self_dist} from its own centroids (reference fit / matching broken)'
    dist = {arm: np.empty((sec['n_runs'], len(gammas))) for arm in CONF_COLS}
    for r in range(sec['n_runs']):
        for arm in CONF_COLS:
            dist[arm][r] = distance_curve(X_ext, conf[arm], gt_centers, n_clusters, gammas, seed + r)
    # Shuffled arms: permute entity confidences, or with outlier_adjust use RowShuffler
    # (checked to reproduce the real confidences when unpermuted).
    shufflers = {}
    if sec['outlier_adjust']:
        for arm, col in CONF_COLS.items():
            shufflers[arm] = RowShuffler(ext_df, col, ext_dense.index, keep_attrs)
            assert np.allclose(shufflers[arm].confidence(shufflers[arm].raw), conf[arm], rtol=1e-12, atol=0), (
                f'{arm}: RowShuffler does not reproduce the real entity confidences')
    n_eff_shuf = {}
    for arm in CONF_COLS:
        dist[f'{arm}_shuffled'] = np.empty((sec['n_shuffle_samples'], len(gammas)))
        neff = np.empty((sec['n_shuffle_samples'], len(gammas)))
        for s in range(sec['n_shuffle_samples']):
            rng = np.random.default_rng([seed, s, SHUFFLE_STREAMS[arm]])
            sconf = shufflers[arm].shuffled(rng) if sec['outlier_adjust'] else rng.permutation(conf[arm])
            dist[f'{arm}_shuffled'][s] = distance_curve(X_ext, sconf, gt_centers, n_clusters, gammas, seed + s)
            neff[s] = [kish_n_eff(sconf ** g) for g in gammas]
        n_eff_shuf[arm] = neff
    assert set(dist) == set(ARMS), sorted(dist)

    # Known answer: gamma = 0 makes every weight 1, so the arms are the same KMeans fits.
    assert gammas[0] == 0
    assert np.array_equal(dist['ntp'][:, 0], dist['probe'][:, 0]), 'gamma=0: ntp and probe arms differ'
    k = min(sec['n_runs'], sec['n_shuffle_samples'])
    for arm in CONF_COLS:
        assert np.array_equal(dist[f'{arm}_shuffled'][:k, 0], dist['ntp'][:k, 0]), (
            f'gamma=0: {arm}_shuffled arm differs on shared seeds')

    # Kish n_eff of conf**gamma. Entity permutation leaves it unchanged; the row-level
    # shuffle doesn't, so that arm reports the mean over draws.
    n_eff = {arm: [kish_n_eff(conf[arm] ** g) for g in gammas] for arm in CONF_COLS}
    for arm in CONF_COLS:
        n_eff[f'{arm}_shuffled'] = n_eff_shuf[arm].mean(axis=0).tolist() if sec['outlier_adjust'] else n_eff[arm]
    summary = pd.DataFrame([
        dict(arm=arm, gamma=float(g), n=v.shape[0], mean=float(v[:, i].mean()),
             se=float(v[:, i].std() / np.sqrt(v.shape[0])), n_eff=n_eff[arm][i])
        for arm, v in dist.items() for i, g in enumerate(gammas)
    ])

    # ── Outputs ──
    sets.to_csv(out_dir / 'attribute_sets.csv', index=False)
    summary.to_csv(out_dir / 'centroid_distance.csv', index=False)
    np.savez(out_dir / 'distances.npz', gammas=gammas, **dist)
    plot_center_dist(summary, gammas, figures_dir / 'center_dist.pdf')
    plot_legend(figures_dir / 'legend.pdf')
    manifest.update(
        analysis_config_id=cfg['id'], script='analysis/pond_clustering.py', seed=seed, params=sec,
        inputs={k: str(v) for k, v in inputs.items()},
        outlier_adjust=sec['outlier_adjust'],
        outlier_moments=None if moments is None else moments.to_dict('index'),
        keep_attrs=keep_attrs, n_tied_attribute_sets=n_ties,
        n_gt_entities=int(len(gt_val)), n_ext_entities=int(len(ext_val)),
        n_gt_multi_row_cells=int((gt_cell_rows > 1).sum()), n_ext_multi_row_cells=int((ext_cell_rows > 1).sum()),
        n_gt_dense=int(len(gt_dense)), n_ext_dense=int(len(ext_dense)),
        gt_dense_missing_frac=float(gt_dense.isna().to_numpy().mean()),
        ext_dense_missing_frac=float(ext_dense.isna().to_numpy().mean()),
        ext_confidence_mean={arm: float(c.mean()) for arm, c in conf.items()},
    )
    (out_dir / 'meta.json').write_text(json.dumps(manifest, indent=2, default=str))
    print(f"[clustering] wrote {out_dir}")


if __name__ == '__main__':
    main()
