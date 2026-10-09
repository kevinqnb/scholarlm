"""Cached head-probe feature matrices from a judge run's attention_outputs.npz.

The npz holds one compressed (n_layers, n_heads, d_head) array per measurement_id.
Slicing it per head re-decompresses the whole array each time, so this cache
decompresses each row once and keeps only the heads any probe needs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class HeadActivationCache:
    """Per-row cache of the ``union_heads`` slices of each attention_outputs.npz read.

    Args:
        union_heads: Every (layer, head) any caller will request. Requesting a head
            outside this set raises KeyError.
    """

    def __init__(self, union_heads):
        """Index the heads to keep.

        Args:
            union_heads: Iterable of (layer, head) pairs.

        Raises:
            ValueError: ``union_heads`` is empty.
        """
        heads = sorted({(int(l), int(h)) for l, h in union_heads})
        if not heads:
            raise ValueError("union_heads is empty")
        self._layers = np.array([l for l, _ in heads])
        self._heads = np.array([h for _, h in heads])
        self._pos = {lh: i for i, lh in enumerate(heads)}
        self._rows: dict[Path, dict[str, np.ndarray]] = {}

    def features(self, act_dir: Path, mids, top_k_heads) -> np.ndarray:
        """Feature matrix for a head probe: each row's top-head vectors, concatenated.

        Args:
            act_dir: Judge run directory containing attention_outputs.npz.
            mids: Measurement ids (npz keys), one per output row.
            top_k_heads: The probe's (layer, head) pairs, in feature order.

        Returns:
            float32 array of shape ``(len(mids), len(top_k_heads) * d_head)``.
        """
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
