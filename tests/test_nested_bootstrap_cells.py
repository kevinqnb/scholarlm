"""Rung-1 tests for nested_bootstrap's cell-level plumbing: prediction sets from stored
recalibration maps, and resamples shared across the cells of one test set."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.common import nested_bootstrap as nb
from scholarlm.utils.calibration import apply_platt


def _cell(rng, n=120, maps=True, mids=None):
    raw = rng.uniform(0.05, 0.95, n)
    fit_maps = {'probe': [(1.0, 0.0), (1.0, 0.7)], 'ntp': [(1.0, -0.4), (0.8, 0.1)]} if maps else None
    cell = dict(probe_raw=raw, ntp_raw=raw[::-1].copy(), labels=rng.random(n) < raw, fit_maps=fit_maps,
                document_ids=np.repeat(np.arange(n // 10), 10),
                measurement_ids=np.arange(n) if mids is None else mids)
    for m in ('probe', 'ntp'):
        cell[f'{m}_probs'] = apply_platt(cell[f'{m}_raw'], *fit_maps[m][0]) if maps else cell[f'{m}_raw']
    return cell


def test_prediction_sets_follow_the_stored_maps():
    cell = _cell(np.random.default_rng(0))
    sets = nb.cell_prediction_sets(cell, 'ntp')
    assert len(sets) == 2
    assert np.array_equal(sets[1], apply_platt(cell['ntp_raw'], 0.8, 0.1))


def test_unrecalibrated_cell_is_one_raw_set():
    cell = _cell(np.random.default_rng(0), maps=False)
    sets = nb.cell_prediction_sets(cell, 'probe')
    assert len(sets) == 1 and sets[0] is cell['probe_raw']


def test_fit_sample_zero_must_match_stored_predictions():
    cell = _cell(np.random.default_rng(0))
    cell['probe_probs'] = cell['probe_probs'] + 1e-9
    with pytest.raises(AssertionError):
        nb.cell_prediction_sets(cell, 'probe')


def test_cells_on_one_test_set_share_resamples():
    rng = np.random.default_rng(1)
    real = {'a': {'t': _cell(rng)}, 'b': {'t': _cell(rng)}}
    syn = {'a': {'t': _cell(rng, maps=False)}}
    # the same rows (mids) under both training datasets, as in the real pipeline
    real['b']['t']['measurement_ids'] = real['a']['t']['measurement_ids']
    out = nb.bootstrap_cells({'real': {'j': real}, 'syn': {'j': syn}}, seed=0, n_doc_boot=3, n_syn_boot=4)
    ra, rb = out['real']['j']['a']['t']['probe'], out['real']['j']['b']['t']['probe']
    assert ra['replicates']['SmECE'].shape == (2, 3)
    assert out['syn']['j']['a']['t']['ntp']['replicates']['SmECE'].shape == (1, 4)
    # identical cells -> identical results only if the resamples are shared
    real['b']['t'] = real['a']['t']
    out2 = nb.bootstrap_cells({'real': {'j': real}}, seed=0, n_doc_boot=3, n_syn_boot=4)
    assert np.array_equal(out2['real']['j']['a']['t']['probe']['replicates']['SmECE'],
                          out2['real']['j']['b']['t']['probe']['replicates']['SmECE'])


def test_cells_on_one_test_set_must_have_the_same_rows():
    rng = np.random.default_rng(2)
    real = {'a': {'t': _cell(rng)}, 'b': {'t': _cell(rng, mids=np.arange(1, 121))}}
    with pytest.raises(AssertionError):
        nb.bootstrap_cells({'real': {'j': real}}, seed=0, n_doc_boot=2, n_syn_boot=2)
