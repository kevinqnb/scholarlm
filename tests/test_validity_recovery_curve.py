"""Rung-1 tests for the validity/recovery operating-curve helpers ported into
analysis/recovery_validity.py. Hand-built 4-row fixture, expected values worked
out by inspection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))
sys.path.insert(0, str(_REPO / "src"))

from analysis import recovery_validity as rv  # noqa: E402

# rows 0,1 valid; rows 2,3 invalid. 3 GT rows: gt0<-row0, gt1<-row1, gt2<-row2 (an invalid row).
PROBS = np.array([0.9, 0.6, 0.4, 0.1])
LABELS = np.array([True, True, False, False])
EDGES = [(0, 0), (1, 1), (2, 2)]


def test_curve_known_answer():
    v, r, ts = rv.validity_recovery_curve(PROBS, LABELS, 3, EDGES, [0.0, 0.5, 0.95])
    # t=0: keep all -> validity 2/4, recovery 3/3. t=0.5: keep rows 0,1 -> validity 1, recovery 2/3.
    # t=0.95: nothing kept -> skipped.
    assert ts.tolist() == [0.0, 0.5]
    assert np.allclose(v, [0.5, 1.0])
    assert np.allclose(r, [1.0, 2 / 3])


def test_curve_all_thresholds_empty():
    v, r, ts = rv.validity_recovery_curve(PROBS, LABELS, 3, EDGES, [0.99])
    assert len(v) == len(r) == len(ts) == 0


def test_plot_and_helpers_write_files(tmp_path):
    out = tmp_path / "vr.pdf"
    rv.plot_validity_recovery(
        [(PROBS[::-1], "--"), (PROBS, "-")], LABELS, 3, EDGES, out,
        thresholds=np.linspace(0.0, 0.95, 20), n_random=5, seed=0,
    )
    rv.save_validity_recovery_colorbar(tmp_path / "cb.pdf")
    rv.save_validity_recovery_legend(tmp_path / "leg.pdf")
    assert out.stat().st_size > 0
    assert (tmp_path / "cb.pdf").stat().st_size > 0 and (tmp_path / "leg.pdf").stat().st_size > 0
