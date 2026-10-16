import sys
from pathlib import Path

import numpy as np
import pytest
import relplot

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.common.calibration_plot_utils import support_mask


def test_support_mask_hand_built():
    mesh = np.linspace(0, 1, 11)  # 0.0, 0.1, ..., 1.0
    probs = np.array([0.3, 0.4, 0.5])
    mask = support_mask(probs, mesh, sigma=0.15)
    # support [0.15, 0.65] -> mesh points 0.2..0.6 (no float ties)
    assert mesh[mask].min() == pytest.approx(0.2) and mesh[mask].max() == pytest.approx(0.6)
    assert mask.sum() == 5


def test_support_mask_removes_zero_artifact():
    # Perfectly calibrated, no predictions above 0.75 (toy case from the investigation):
    # the unmasked relplot curve collapses to ~0 at high x; the masked part must not.
    rng = np.random.default_rng(0)
    f = rng.uniform(0, 0.75, 3000)
    y = rng.random(3000) < f
    np.random.seed(0)
    d = relplot.prepare_rel_diagram(f, y, plot_confidence_band=False, num_bootstrap=10)
    mask = support_mask(f, d['mesh'], d['sigma'])
    assert d['mu'][-1] < 0.01  # the artifact exists
    assert not mask[-1] and d['mesh'][mask].max() <= f.max() + d['sigma'] + 1e-9
    # Inside the retained region the curve tracks the truth (y = x).
    kept = mask & (d['mesh'] > 0.1) & (d['mesh'] < 0.7)
    assert np.abs(d['mu'][kept] - d['mesh'][kept]).max() < 0.1


def test_support_mask_fails_loud():
    with pytest.raises(AssertionError):
        support_mask(np.array([]), np.linspace(0, 1, 5), 0.1)
    with pytest.raises(AssertionError):
        support_mask(np.array([0.5]), np.linspace(0, 1, 5), 0.0)
