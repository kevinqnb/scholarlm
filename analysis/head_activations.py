"""Head-probe feature matrices read from a judge run's attention_outputs.npz, each row
decompressed at most once.

attention_outputs.npz holds one compressed (n_layers, n_heads, d_head) array per
measurement_id. A head probe's features for a row are its top heads' d_head vectors,
concatenated in the probe's top_k_heads order. Slicing that per head re-decompresses the
whole array once per head (and again for every probe scored on the same rows); this cache
keeps only the union of the heads any probe needs, per row, the first time the row is read.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class HeadActivationCache:
    """Per-row cache of the ``union_heads`` slices of every attention_outputs.npz it reads.

    Args:
        union_heads: every (layer, head) any caller will request; a request for a head
            outside it is a hard error.
    """

    def __init__(self, union_heads):
        heads = sorted({(int(l), int(h)) for l, h in union_heads})
        if not heads:
            raise ValueError("union_heads is empty")
        self._layers = np.array([l for l, _ in heads])
        self._heads = np.array([h for _, h in heads])
        self._pos = {lh: i for i, lh in enumerate(heads)}
        self._rows: dict[Path, dict[str, np.ndarray]] = {}

    def features(self, act_dir: Path, mids, top_k_heads) -> np.ndarray:
        """``(len(mids), len(top_k_heads) * d_head)`` float32: for each row, the d_head
        vectors of ``top_k_heads`` concatenated in that order (same layout as stacking per
        head and concatenating along axis 1)."""
        sel = [self._pos[(int(l), int(h))] for l, h in top_k_heads]
        rows = self._rows.setdefault(Path(act_dir), {})
        missing = [str(m) for m in mids if str(m) not in rows]
        if missing:
            with np.load(Path(act_dir) / 'attention_outputs.npz') as act:
                for key in dict.fromkeys(missing):
                    a = np.array(act[key], dtype=np.float32)
                    assert a.ndim == 3, (act_dir, key, a.shape)
                    rows[key] = a[self._layers, self._heads, :]
        X = np.stack([rows[str(m)][sel].reshape(-1) for m in mids], axis=0)
        assert X.shape[0] == len(mids) and X.dtype == np.float32, (X.shape, X.dtype)
        return X
