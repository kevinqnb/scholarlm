"""analysis/head_activations.py: the cached head-feature reader must reproduce the
per-head npz slicing it replaced, exactly."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from analysis.head_activations import HeadActivationCache  # noqa: E402


def _old_features(act_dir, mids, top):
    # Verbatim from calibration_updated_v3._score_rows before the cache.
    act = np.load(act_dir / 'attention_outputs.npz')
    return np.concatenate([
        np.stack([np.array(act[str(mid)], dtype=np.float32)[l, h, :] for mid in mids], axis=0)
        for l, h in top
    ], axis=1)


@pytest.fixture
def act_dir(tmp_path):
    rng = np.random.default_rng(0)
    # float16 on disk, like a real run might store; 4 layers x 3 heads x d_head 5.
    np.savez_compressed(tmp_path / 'attention_outputs.npz',
                        **{str(m): rng.normal(size=(4, 3, 5)).astype(np.float16) for m in range(10)})
    return tmp_path


def test_hand_computed_layout(tmp_path):
    # Row m: a[l, h, :] = [100*l + 10*h + d for d in 0..1]; top = [(2, 1), (0, 0)].
    a = np.array([[[100 * l + 10 * h + d for d in range(2)] for h in range(2)] for l in range(3)], dtype=np.float32)
    np.savez_compressed(tmp_path / 'attention_outputs.npz', **{'7': a})
    X = HeadActivationCache([(0, 0), (2, 1)]).features(tmp_path, [7], [(2, 1), (0, 0)])
    assert X.tolist() == [[210.0, 211.0, 0.0, 1.0]]


def test_matches_old_reader_across_probes(act_dir):
    probes = [[(3, 2), (0, 1), (1, 1)], [(0, 1), (2, 0)], [(1, 1)]]  # overlapping, unsorted
    cache = HeadActivationCache([lh for top in probes for lh in top])
    mids = [5, 2, 9, 2, 0]  # out of order, with a repeat
    for top in probes:
        new = cache.features(act_dir, mids, top)
        old = _old_features(act_dir, mids, top)
        assert new.dtype == old.dtype == np.float32
        assert np.array_equal(new, old)


def test_rows_decompressed_once(act_dir, monkeypatch):
    reads = []
    real_load = np.load

    class Counting:
        def __init__(self, z): self.z = z
        def __enter__(self): return self
        def __exit__(self, *a): self.z.close()
        def __getitem__(self, k): reads.append(k); return self.z[k]

    monkeypatch.setattr(np, 'load', lambda p: Counting(real_load(p)))
    cache = HeadActivationCache([(0, 0), (1, 1), (2, 2)])
    cache.features(act_dir, [1, 2, 3], [(0, 0), (1, 1)])
    cache.features(act_dir, [2, 3, 4], [(2, 2)])
    assert sorted(reads) == ['1', '2', '3', '4']


def test_head_outside_union_fails_loud(act_dir):
    with pytest.raises(KeyError):
        HeadActivationCache([(0, 0)]).features(act_dir, [1], [(1, 1)])


def test_missing_row_fails_loud(act_dir):
    with pytest.raises(KeyError):
        HeadActivationCache([(0, 0)]).features(act_dir, [999], [(0, 0)])
