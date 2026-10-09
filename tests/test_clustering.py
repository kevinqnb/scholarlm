"""Unit tests for analysis/clustering.py's pure helpers and config loader.

Known answers (stated before running):
  - centroid_matching_distance of a centroid set with a row-permutation of itself is 0;
    with every centroid shifted by (3, 4) it is exactly 5.
  - cell_matrix: entity 1 / attribute a has rows 1, 2, 10 with confidences 0.2, 0.4, 0.9
    -> value median 2, confidence mean 0.5; a row with NaN converted_value contributes
    neither value nor confidence (entity 2 / b stays NaN although its row has conf 0.7).
  - dense_submatrix on a 4x2 matrix whose rows have 0, 1, 2, 0 NaNs (missing 3/8):
    threshold 0.2 drops the 2-NaN row (-> 1/6 <= 0.2); threshold 0.1 also drops the 1-NaN
    row (-> 0); an all-NaN matrix ends empty.
  - entity_confidence: product of observed cells only ([0.5, NaN] -> 0.5); an all-NaN
    entity is an error.
"""
import copy

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.clustering import (
    cell_matrix, centroid_matching_distance, dense_submatrix, entity_confidence,
    enumerate_attribute_sets, fixed_attribute_set, load_clustering_config, process_matrix,
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
    assert entity_confidence(conf) == pytest.approx([0.5, 0.1])
    with pytest.raises(ValueError, match='no observed cell'):
        entity_confidence(pd.DataFrame({'a': [np.nan], 'b': [np.nan]}))


def test_process_matrix_standardizes():
    m = pd.DataFrame({'a': [1.0, 2.0, 3.0, np.nan], 'b': [2.0, 4.0, 6.0, 8.0]})
    X = process_matrix(m, 2)
    assert X.shape == (4, 2) and np.allclose(X.mean(axis=0), 0) and np.allclose(X.std(axis=0), 1)


# ── Config loader ──

BASE = {
    'id': 'x', 'project': 'scholarlm', 'description': 'd', 'seed': 42,
    'params': {'clustering': {
        'calibration_config_id': 'cal', 'calibration_version': 'v4', 'extraction_id': 'ext',
        'judge_combine_id': 'jc', 'judge_model': 'qwen-2.5-7b', 'probe_train_dataset': 'pond',
        'rows': 'deduplicated', 'deduplication_config_id': 'dd', 'confidence': 'center',
        'missing_threshold': 0.2, 'attributes': None, 'attribute_set_sizes': [2, 3], 'n_clusters': 5, 'knn_neighbors': 5,
        'gammas': {'start': 0.0, 'stop': 5.0, 'num': 3}, 'n_runs': 2, 'n_random_samples': 2, 'n_shuffle_samples': 2,
    }},
}


def _write(tmp_path, cfg):
    p = tmp_path / f"{cfg['id']}.yaml"
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
    (lambda s: s.pop('n_shuffle_samples'), 'missing required'),
    (lambda s: s.update(n_shuffle_samples=0), 'n_shuffle_samples'),
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
    from analysis.clustering import kish_n_eff
    assert kish_n_eff(np.ones(7)) == pytest.approx(7)
    assert kish_n_eff(np.array([1.0, 0.0, 0.0])) == pytest.approx(1)
