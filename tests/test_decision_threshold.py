"""Unit tests for analysis/decision_threshold.py on a hand-built fixture.

Ground truth (4 rows):  g0, g1 in doc A; g2 in doc B; g3 in doc C.
Postprocessed extraction (5 rows):
  e0  doc A  measurement_id 0
  e1  doc A  measurement_id 1  (split child)
  e2  doc A  measurement_id 1  (split child)
  e3  doc B  measurement_id 2
  e4  doc C  measurement_id 3
Edges (already at the dataset fuzzy threshold): (g0,e0) (g1,e2) (g2,e3) (g3,e4).
Doc C is excluded (no predictions), so after restriction: 3 GT rows, 4 extraction
rows, edges (0,0) (1,2) (2,3).
Probabilities by measurement_id: 0 -> 0.9, 1 -> 0.4, 2 -> 0.6.
Judged: e0..e3 = [T, F, F, F]; matched = [T, F, T, T]; validity labels = [T, F, T, T].

Known answers (stated before running):
  t=0.0   keep e0..e3   recovery 3/3 = 1.0   validity 3/4
  t=0.5   keep e0, e3   recovery 2/3          validity 2/2 = 1.0
  t=0.95  keep none     recovery 0.0          validity NaN, n_kept 0
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

from analysis.decision_threshold import (  # noqa: E402
    decision_threshold_curve, load_decision_threshold_config, permutation_control,
    restrict_to_scored_documents, row_probabilities,
)


def fixture():
    gt = pd.DataFrame({"document_id": ["A", "A", "B", "C"]})
    ext = pd.DataFrame({"document_id": ["A", "A", "A", "B", "C"], "measurement_id": [0, 1, 1, 2, 3]})
    edges = [(0, 0), (1, 2), (2, 3), (3, 4)]
    validity = np.array([True, False, True, True, True])
    scored = pd.DataFrame({"measurement_id": [0, 1, 2], "document_id": ["A", "A", "B"],
                           "probe_prob": [0.9, 0.4, 0.6]})
    return gt, ext, edges, validity, scored


def restricted():
    gt, ext, edges, validity, scored = fixture()
    gt_keep, ext_keep, sub_edges = restrict_to_scored_documents(gt, ext, edges, excluded_documents={"C"})
    probs = row_probabilities(ext[ext_keep].reset_index(drop=True), scored, "probe_prob")
    return int(gt_keep.sum()), sub_edges, validity[ext_keep], probs


def test_restriction_drops_excluded_doc_and_reindexes_edges():
    gt, ext, edges, _validity, _scored = fixture()
    gt_keep, ext_keep, sub_edges = restrict_to_scored_documents(gt, ext, edges, excluded_documents={"C"})
    assert gt_keep.tolist() == [True, True, True, False]
    assert ext_keep.tolist() == [True, True, True, True, False]
    assert sub_edges == [(0, 0), (1, 2), (2, 3)]


def test_restriction_rejects_edge_into_excluded_doc():
    gt, ext, _edges, _validity, _scored = fixture()
    with pytest.raises(AssertionError, match="excluded"):
        restrict_to_scored_documents(gt, ext, [(0, 4)], excluded_documents={"C"})


def test_split_children_inherit_parent_probability():
    _n_gt, _edges, _validity, probs = restricted()
    assert probs.tolist() == [0.9, 0.4, 0.4, 0.6]


def test_row_without_probability_is_an_error():
    gt, ext, edges, validity, scored = fixture()
    with pytest.raises(ValueError, match="measurement_id"):
        row_probabilities(ext.iloc[:4], scored[scored["measurement_id"] != 2], "probe_prob")


def test_curve_known_answers():
    n_gt, edges, validity, probs = restricted()
    curve = decision_threshold_curve(n_gt, edges, validity, probs, [0.0, 0.5, 0.95])
    assert curve["n_kept"].tolist() == [4, 2, 0]
    assert curve["recovery"].tolist() == pytest.approx([1.0, 2 / 3, 0.0])
    assert curve["validity"].iloc[0] == pytest.approx(0.75)
    assert curve["validity"].iloc[1] == pytest.approx(1.0)
    assert np.isnan(curve["validity"].iloc[2])


def test_threshold_is_strict():
    n_gt, edges, validity, probs = restricted()
    curve = decision_threshold_curve(n_gt, edges, validity, probs, [0.6])
    # Only e0 (0.9) survives > 0.6; e3 (exactly 0.6) is dropped.
    assert curve["n_kept"].tolist() == [1]


def test_permutation_control_is_seed_deterministic():
    gt, ext, edges, validity, scored = fixture()
    gt_keep, ext_keep, sub_edges = restrict_to_scored_documents(gt, ext, edges, excluded_documents={"C"})
    args = (int(gt_keep.sum()), sub_edges, validity[ext_keep], ext[ext_keep].reset_index(drop=True),
            scored, "probe_prob", [0.0, 0.5])
    a = permutation_control(*args, n_permutations=5, seed=0)
    b = permutation_control(*args, n_permutations=5, seed=0)
    pd.testing.assert_frame_equal(a, b)
    # At t=0 every row is kept whatever the permutation: the unfiltered values.
    assert a["recovery"].iloc[0] == pytest.approx(1.0)
    assert a["validity"].iloc[0] == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

CALIBRATION_ID = "2026-01-01-calibration-v4-test-01"


@pytest.fixture(autouse=True)
def _calibration_config(tmp_path, monkeypatch):
    """A minimal valid v4 calibration config (pond, nfix, supermat) under
    tmp_path/analysis-configs/calibration/, which the loader resolves CALIBRATION_ID in."""
    from analysis.common import config as ac
    root = tmp_path / "analysis-configs"
    (root / "calibration").mkdir(parents=True)
    monkeypatch.setattr(ac, "ANALYSIS_CONFIGS_ROOT", root)
    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    block = {"extraction_id": "e", "judge_interp_id": "j", "judge_combine_id": "c",
             "ground_truth_file": str(gt), "synthetic_probe_config": "p", "use_matching_labels": True,
             "pi_te_estimate": None, "syn_test_ids": {"primary": "tp", "diag": "td"}}
    (root / "calibration" / f"{CALIBRATION_ID}.yaml").write_text(yaml.safe_dump({
        "id": CALIBRATION_ID, "project": "scholarlm", "description": "t", "seed": 0,
        "params": {"probe_type": "head", "probe_variant": "platt", "syn_split": "primary", "n_boot": 10,
                   "recalibration": "intercept_fit", "fit_source": "sample", "fit_n": 5, "fit_seed": 0,
                   "datasets": {ds: dict(block) for ds in ac.CALIBRATION_DATASETS}}}))


def _config(**overrides):
    params = {
        "calibration_config_id": CALIBRATION_ID,
        "cells": [{"train": "pond", "test": "pond"}],
        "thresholds": [0.0, 0.25, 0.5, 0.75],
        "highlight_threshold": 0.5,
        "n_permutations": 3,
        "invalid_datasets": {},
    }
    params.update(overrides)
    return {"id": "x", "project": "scholarlm", "description": "d", "seed": 0, "params": params}


def _write(tmp_path, cfg):
    cfg = copy.deepcopy(cfg)
    path = tmp_path / "decision-threshold" / f"{cfg['id']}.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_config_loads(tmp_path):
    cfg = load_decision_threshold_config(_write(tmp_path, _config()))
    assert cfg["params"]["thresholds"] == [0.0, 0.25, 0.5, 0.75]


@pytest.mark.parametrize("overrides, match", [
    ({"thresholds": [0.25, 0.5]}, "0.0"),
    ({"thresholds": [0.0, 0.5, 0.5]}, "strictly increasing"),
    ({"thresholds": [0.0, 0.5, 1.5]}, r"\[0, 1\]"),
    ({"highlight_threshold": 0.4}, "highlight_threshold"),
    ({"cells": [{"train": "pond", "test": "lakes"}]}, "cells"),
    ({"cells": [{"train": "pond", "test": "pond"}, {"train": "pond", "test": "pond"}]}, "duplicate"),
    ({"invalid_datasets": {"lakes": "x"}}, "invalid_datasets"),
    ({"n_permutations": 0}, "n_permutations"),
])
def test_config_rejects(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        load_decision_threshold_config(_write(tmp_path, _config(**overrides)))


def test_config_rejects_missing_calibration_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibration config"):
        load_decision_threshold_config(_write(tmp_path, _config(calibration_config_id="2026-01-01-nope-01")))


def test_config_rejects_unknown_key(tmp_path):
    cfg = _config()
    cfg["params"]["prob_source"] = "raw"
    with pytest.raises(ValueError, match="params keys"):
        load_decision_threshold_config(_write(tmp_path, cfg))
