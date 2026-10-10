"""Unit tests for analysis/pond_clustering.py's pure helpers and config loader.

Known answers (stated before running):
  - centroid_matching_distance of a centroid set with a row-permutation of itself is 0;
    with every centroid shifted by (3, 4) it is exactly 5.
  - cell_matrix: entity 1 / attribute a has rows 1, 2, 10 with confidences 0.2, 0.4, 0.9
    -> value median 2, confidence mean 0.5; a row with NaN converted_value contributes
    neither value nor confidence (entity 2 / b stays NaN although its row has conf 0.7).
  - dense_submatrix on a 4x2 matrix whose rows have 0, 1, 2, 0 NaNs (missing 3/8):
    threshold 0.2 drops the 2-NaN row (-> 1/6 <= 0.2); threshold 0.1 also drops the 1-NaN
    row (-> 0); an all-NaN matrix ends empty.
  - entity_confidence: mean of observed cells only ([0.5, NaN] -> 0.5, [0.2, 0.5] -> 0.35);
    an all-NaN entity is an error.
  - process_matrices: GT surface_area (log-scaled) 1, 10, 100, 1000 -> log10 0..3 -> standardized
    +-1.3416, +-0.4472; with knn_neighbors=1 the missing ph of the 1000 row is imputed from its
    nearest row (100, ph 3) in the *scaled* space: (3 - 2) / std(1, 2, 3) = 1.2247, not 3.
    The extraction is the GT with surface_area x10 (log +1) and ph +1. The scaler is fit on the
    GT only, so the extraction sits at GT + 1 / 1.1180 = +0.8944 on surface_area and
    + 1 / 0.8165 = +1.2247 on ph (its missing ph imputed from its own row 2: 2.4495). Separate
    per-side scaling would have made the two matrices identical.
  - block_mean_ci on [[1, 3], [3, 5], [5, 7]]: block means 2, 4, 6 -> mean 4, se 2 / sqrt(3),
    half-width t(0.975, 2) * se = 4.3027 * 1.1547 = 4.9683.
"""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.pond_clustering import (
    block_mean_ci, cell_matrix, centroid_matching_distance, dense_submatrix, drop_nonpositive_log, entity_confidence,
    enumerate_attribute_sets, filter_ecosystems, fit_kmeans, fixed_attribute_set, kmeans_seed, load_clustering_config,
    process_matrices, random_confidence,
)


def test_centroid_matching_distance_permutation_and_shift():
    A = np.array([[0.0, 0.0], [1.0, 2.0], [5.0, -1.0]])
    assert centroid_matching_distance(A, A[[2, 0, 1]]) == 0.0
    assert centroid_matching_distance(A, A[[1, 2, 0]] + np.array([3.0, 4.0])) == pytest.approx(5.0)


def test_cell_matrix_median_value_mean_conf_same_rows():
    df = pd.DataFrame({
        'eid':             [1,   1,   1,    1,   2,      2],
        'attribute':       ['a', 'a', 'a',  'b', 'a',    'b'],
        'converted_value': [1.0, 2.0, 10.0, 7.0, 3.0,    np.nan],
        'p':               [0.2, 0.4, 0.9,  0.6, 0.8,    0.7],
    })
    value, confs, n = cell_matrix(df, 'eid', ['p'])
    assert list(value.index) == [1, 2] and list(value.columns) == ['a', 'b']
    assert value.loc[1, 'a'] == 2.0 and confs['p'].loc[1, 'a'] == pytest.approx(0.5)
    assert value.loc[1, 'b'] == 7.0 and confs['p'].loc[1, 'b'] == pytest.approx(0.6)
    assert np.isnan(value.loc[2, 'b']) and np.isnan(confs['p'].loc[2, 'b'])
    assert n.loc[(1, 'a')] == 3 and n.sum() == 5


def test_cell_matrix_rejects_nan_confidence_on_valued_row():
    df = pd.DataFrame({'eid': [1], 'attribute': ['a'], 'converted_value': [1.0], 'p': [np.nan]})
    with pytest.raises(ValueError, match='NaN confidence'):
        cell_matrix(df, 'eid', ['p'])


def test_dense_submatrix():
    m = pd.DataFrame({'a': [1.0, np.nan, np.nan, 4.0], 'b': [1.0, 2.0, np.nan, 4.0]}, index=[10, 11, 12, 13])
    assert list(dense_submatrix(m, 0.2).index) == [10, 11, 13]
    assert list(dense_submatrix(m, 0.1).index) == [10, 13]
    assert len(dense_submatrix(m, 0.5)) == 4
    assert len(dense_submatrix(pd.DataFrame({'a': [np.nan, np.nan]}), 0.2)) == 0


def test_enumerate_attribute_sets_scores():
    # a, b dense on 4 rows; c only on row 0 -> {a,b} scores 8, any set with c scores less.
    m = pd.DataFrame({'a': [1.0, 2, 3, 4], 'b': [1.0, 2, 3, 4], 'c': [1.0, np.nan, np.nan, np.nan]})
    sets = enumerate_attribute_sets(m, [2, 3], 0.2)
    assert len(sets) == 4  # C(3,2) + C(3,3)
    best = sets.loc[sets['score'].idxmax()]
    assert best['attributes'] == 'a|b' and best['n_rows'] == 4 and best['score'] == 8
    with pytest.raises(ValueError, match='exceed'):
        enumerate_attribute_sets(m, [4], 0.2)


def test_fixed_attribute_set():
    m = pd.DataFrame({'a': [1.0, 2, 3, 4], 'b': [1.0, 2, 3, 4], 'c': [1.0, np.nan, np.nan, np.nan]})
    s = fixed_attribute_set(m, ['c', 'a'], 0.2)  # order preserved, c forces row drops
    assert len(s) == 1 and s.loc[0, 'attributes'] == 'c|a' and s.loc[0, 'd'] == 2
    assert s.loc[0, 'n_rows'] == 1 and s.loc[0, 'score'] == 2
    with pytest.raises(ValueError, match='no valued rows'):
        fixed_attribute_set(m, ['a', 'zzz'], 0.2)


def test_entity_confidence_observed_cells_only():
    conf = pd.DataFrame({'a': [0.5, 0.2], 'b': [np.nan, 0.5]})
    assert entity_confidence(conf) == pytest.approx([0.5, 0.35])
    with pytest.raises(ValueError, match='no observed cell'):
        entity_confidence(pd.DataFrame({'a': [np.nan], 'b': [np.nan]}))


def test_process_matrices_gt_fitted_scaler_then_impute():
    gt = pd.DataFrame({'surface_area': [1.0, 10.0, 100.0, 1000.0], 'ph': [1.0, 2.0, 3.0, np.nan]})
    ext = pd.DataFrame({'surface_area': gt['surface_area'] * 10, 'ph': gt['ph'] + 1})
    X_gt, X_ext, params = process_matrices(gt, ext, 1)
    assert X_gt[:, 0] == pytest.approx([-1.3416408, -0.4472136, 0.4472136, 1.3416408])
    assert X_gt[:3, 1] == pytest.approx([-1.2247449, 0.0, 1.2247449])
    assert X_gt[3, 1] == pytest.approx(1.2247449)
    assert X_ext[:, 0] - X_gt[:, 0] == pytest.approx([0.8944272] * 4)
    assert X_ext[:3, 1] - X_gt[:3, 1] == pytest.approx([1.2247449] * 3)
    assert X_ext[3, 1] == pytest.approx(2.4494897)
    assert params['surface_area'] == pytest.approx(dict(mean=1.5, scale=1.1180340))
    assert params['ph'] == pytest.approx(dict(mean=2.0, scale=0.8164966))


def test_process_matrices_rejects_bad_inputs():
    ok = pd.DataFrame({'max_depth': [1.0, 2.0, 3.0], 'ph': [1.0, 2.0, 3.0]})
    with pytest.raises(ValueError, match='non-positive'):
        process_matrices(ok, ok.assign(max_depth=[0.0, 1.0, 2.0]), 1)
    with pytest.raises(ValueError, match='distinct'):
        process_matrices(ok.assign(ph=[7.0, 7.0, np.nan]), ok, 1)
    with pytest.raises(ValueError, match='columns'):
        process_matrices(ok, ok[['ph', 'max_depth']], 1)


def test_block_mean_ci():
    r = block_mean_ci(np.array([[1.0, 3.0], [3.0, 5.0], [5.0, 7.0]]))
    assert r['mean'] == pytest.approx(4.0) and r['se'] == pytest.approx(2 / np.sqrt(3))
    assert r['ci_hi'] - r['mean'] == pytest.approx(4.302653 * 2 / np.sqrt(3), rel=1e-5)
    assert r['mean'] - r['ci_lo'] == pytest.approx(r['ci_hi'] - r['mean'])


def test_filter_ecosystems_drops_other_and_rejects_mixed_entities():
    df = pd.DataFrame({'eid': [1, 1, 2, 3, 4], 'ecosystem_bucket': ['pond', 'pond', 'other', 'lake', 'wetland']})
    assert list(filter_ecosystems(df, ['eid'])['eid']) == [1, 1, 3, 4]
    with pytest.raises(ValueError, match='several ecosystem buckets'):
        filter_ecosystems(df.assign(ecosystem_bucket=['pond', 'lake', 'other', 'lake', 'wetland']), ['eid'])


def test_drop_nonpositive_log_only_touches_log_attributes():
    df = pd.DataFrame({'attribute': ['surface_area', 'max_depth', 'ph', 'vegetation_cover', 'surface_area', 'ph'],
                       'converted_value': [0.0, -1.0, 0.0, 0.0, 5.0, np.nan]})
    out = drop_nonpositive_log(df)
    assert list(out['attribute']) == ['ph', 'vegetation_cover', 'surface_area', 'ph']


def test_random_confidence_deterministic_and_distinct_per_trial():
    a = random_confidence(0, 1, 2, 50)
    assert np.array_equal(a, random_confidence(0, 1, 2, 50)) and ((a >= 0) & (a < 1)).all()
    assert not np.array_equal(a, random_confidence(0, 2, 1, 50))


def test_fit_kmeans_single_start_and_seeded():
    X = np.random.default_rng(0).normal(size=(30, 2))
    a, b = fit_kmeans(X, 3, 7), fit_kmeans(X, 3, 7)
    assert a.n_init == 1 and np.array_equal(a.cluster_centers_, b.cluster_centers_)
    assert kmeans_seed(0, 1, 2, 3) == kmeans_seed(0, 1, 2, 3) != kmeans_seed(0, 1, 3, 2)


# ── Config loader ──

BASE = {
    'id': 'x', 'project': 'scholarlm', 'description': 'd', 'seed': 42,
    'params': {'clustering': {
        'calibration_config_id': 'cal', 'calibration_version': 'v4', 'extraction_id': 'ext',
        'judge_combine_id': 'jc', 'judge_model': 'qwen-2.5-7b', 'probe_train_dataset': 'pond',
        'rows': 'deduplicated', 'deduplication_config_id': 'dd', 'confidence': 'center',
        'missing_threshold': 0.2, 'attributes': None, 'attribute_set_sizes': [2, 3], 'n_clusters': 5, 'knn_neighbors': 5,
        'gammas': {'start': 0.0, 'stop': 5.0, 'num': 3}, 'n_init': 2, 'n_runs': 2,
    }},
}


def _write(tmp_path, cfg):
    p = tmp_path / "clustering" / f"{cfg['id']}.yaml"
    p.parent.mkdir(exist_ok=True)
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_config_valid(tmp_path):
    load_clustering_config(_write(tmp_path, BASE))


def test_config_valid_fixed_attributes(tmp_path):
    cfg = copy.deepcopy(BASE)
    cfg['params']['clustering'].update(attributes=['surface_area', 'max_depth', 'ph'], attribute_set_sizes=None)
    load_clustering_config(_write(tmp_path, cfg))


@pytest.mark.parametrize('mutate, match', [
    (lambda s: s.pop('n_runs'), 'missing required'),
    (lambda s: s.pop('n_init'), 'missing required'),
    (lambda s: s.update(n_init=1), 'n_init'),
    (lambda s: s.update(n_runs=0), 'n_runs'),
    (lambda s: s.update(outlier_adjust=True), 'unexpected'),
    (lambda s: s.update(n_shuffle_samples=2), 'unexpected'),
    (lambda s: s.update(extra=1), 'unexpected'),
    (lambda s: s.update(probe_train_dataset='nfix'), 'probe_train_dataset'),
    (lambda s: s.update(rows='final'), 'must be null'),
    (lambda s: s.update(gammas={'start': 0.5, 'stop': 5.0, 'num': 3}), 'gammas'),
    (lambda s: s.update(attribute_set_sizes=[3, 2]), 'attribute_set_sizes'),
    (lambda s: s.pop('attributes'), 'missing required'),
    (lambda s: s.update(attributes=['a', 'b']), 'exactly one'),
    (lambda s: s.update(attribute_set_sizes=None), 'exactly one'),
    (lambda s: s.update(attributes=[], attribute_set_sizes=None), 'attributes'),
    (lambda s: s.update(attributes=['a', 'a'], attribute_set_sizes=None), 'attributes'),
    (lambda s: s.update(n_clusters=1), 'n_clusters'),
    (lambda s: s.update(missing_threshold=1.0), 'missing_threshold'),
])
def test_config_rejects(tmp_path, mutate, match):
    cfg = copy.deepcopy(BASE)
    mutate(cfg['params']['clustering'])
    with pytest.raises((ValueError, KeyError), match=match):
        load_clustering_config(_write(tmp_path, cfg))


def test_kish_n_eff():
    from analysis.pond_clustering import kish_n_eff
    assert kish_n_eff(np.ones(7)) == pytest.approx(7)
    assert kish_n_eff(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)


