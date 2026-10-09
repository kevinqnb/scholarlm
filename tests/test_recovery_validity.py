"""Unit tests for analysis/recovery_validity.py.

The bootstrap/mask functions are pure and tested on hand-built fixtures where
the expected point estimate (and, for the degenerate all-true/all-false
cases, the CI) can be verified by inspection. find_judge_combine_id and
load_validity_labels are tested against a tmp_path fixture tree, monkeypatching
utils.EXPERIMENT_CONFIGS_ROOT/RESULTS_ROOT (same pattern as
tests/test_calibration_ids.py) so no real repo data is touched.
test_compute_metrics_for_id_* builds a full tiny end-to-end fixture (ground
truth + extraction + a hand-crafted match_cache.pkl + judge_combine) under
the same monkeypatched roots, so compute_metrics_for_id itself -- including
the --skip-validity path -- runs against something other than real repo data.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import BaseModel

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))
sys.path.insert(0, str(_REPO / "src"))

import utils as paths  # noqa: E402
from scholarlm.config import DatasetConfig  # noqa: E402
from analysis import recovery_validity as rv  # noqa: E402
from analysis.common import matching, provenance, recovery  # noqa: E402


# ---------------------------------------------------------------------------
# gt_recovered_mask / ext_matched_mask
# ---------------------------------------------------------------------------


def test_masks_mark_gt_and_ext_indices_present_in_edges():
    # edges is already threshold-filtered (matching.load_match_cache(id,
    # fuzzy_threshold=...) -- see module docstring), so the masks just mark
    # presence, no weight/threshold argument.
    edges = [(0, 0), (0, 1)]  # gt row 1 has no surviving edge

    recovered = recovery.gt_recovered_mask(2, edges)
    matched = recovery.ext_matched_mask(3, edges)

    assert recovered.tolist() == [True, False]
    assert matched.tolist() == [True, True, False]


def test_masks_all_zero_when_no_edges():
    assert recovery.gt_recovered_mask(2, []).tolist() == [False, False]
    assert recovery.ext_matched_mask(1, []).tolist() == [False]


# ---------------------------------------------------------------------------
# assert_matching_columns_present
# ---------------------------------------------------------------------------

_CFG = {
    "strict": {"document_id": "document_id", "attribute": "attribute", "point_value": "point_value", "units": "units"},
    "fuzzy": {"name": "name", "ecosystem": "ecosystem"},
}


def test_matching_columns_present_passes_when_all_columns_exist():
    gt = pd.DataFrame(columns=["document_id", "attribute", "point_value", "units", "name", "ecosystem"])
    ext = pd.DataFrame(columns=["document_id", "attribute", "point_value", "units", "name", "ecosystem"])
    matching.assert_matching_columns_present(gt, ext, _CFG)  # must not raise


def test_matching_columns_present_raises_on_legacy_schema():
    # Simulates a cache built under ablation.py's rules (value/converted_value)
    # sitting where a point_value-keyed cache is now expected.
    gt = pd.DataFrame(columns=["document_id", "attribute", "value", "units", "name", "ecosystem", "location"])
    ext = pd.DataFrame(columns=["document_id", "attribute", "converted_value", "units", "name", "ecosystem", "location"])
    with pytest.raises(KeyError, match="point_value"):
        matching.assert_matching_columns_present(gt, ext, _CFG)


# ---------------------------------------------------------------------------
# bootstrap_cluster_rate
# ---------------------------------------------------------------------------


def test_bootstrap_point_estimate_matches_plain_mean():
    labels = np.array([True, True, False, False, True])
    clusters = np.array(["a", "a", "b", "b", "c"])
    point, lo, hi = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=500, seed=0)
    assert point == pytest.approx(3 / 5)
    assert 0.0 <= lo <= point <= hi <= 1.0


def test_bootstrap_all_true_gives_degenerate_ci():
    labels = np.array([True, True, True, True])
    clusters = np.array(["a", "a", "b", "b"])
    point, lo, hi = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=200, seed=1)
    assert (point, lo, hi) == (1.0, 1.0, 1.0)


def test_bootstrap_all_false_gives_degenerate_ci():
    labels = np.array([False, False, False])
    clusters = np.array(["a", "b", "c"])
    point, lo, hi = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=200, seed=1)
    assert (point, lo, hi) == (0.0, 0.0, 0.0)


def test_bootstrap_seed_determinism():
    labels = np.array([True, False, True, False, True, True, False])
    clusters = np.array(["p1", "p1", "p2", "p2", "p3", "p4", "p4"])
    r1 = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=1000, seed=7)
    r2 = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=1000, seed=7)
    assert r1 == r2


def test_bootstrap_different_seeds_can_differ():
    # 20 papers of varied size/label-rate so the bootstrap distribution isn't
    # so coarse that different seeds coincidentally land on the same
    # percentile value.
    rng = np.random.default_rng(123)
    n_papers = 20
    clusters = np.repeat([f"p{i}" for i in range(n_papers)], rng.integers(1, 6, size=n_papers))
    labels = rng.random(len(clusters)) < 0.6
    _, lo1, hi1 = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=1000, seed=1)
    _, lo2, hi2 = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=1000, seed=2)
    assert (lo1, hi1) != (lo2, hi2)


def test_bootstrap_single_cluster_collapses_to_its_own_rate():
    # Only one paper -- every resample just redraws that same paper some
    # number of times, so the rate is identical to the point estimate.
    labels = np.array([True, True, False])
    clusters = np.array(["only"] * 3)
    point, lo, hi = rv.bootstrap_cluster_rate(labels, clusters, n_resamples=100, seed=0)
    assert (point, lo, hi) == pytest.approx((2 / 3, 2 / 3, 2 / 3))


def test_bootstrap_length_mismatch_raises():
    with pytest.raises(ValueError):
        rv.bootstrap_cluster_rate(np.array([True]), np.array(["a", "b"]), n_resamples=10, seed=0)


def test_bootstrap_zero_rows_raises():
    with pytest.raises(ValueError):
        rv.bootstrap_cluster_rate(np.array([], dtype=bool), np.array([]), n_resamples=10, seed=0)


# ---------------------------------------------------------------------------
# find_judge_combine_id / load_validity_labels — fixture tree
# ---------------------------------------------------------------------------


def _write_experiment_config(root: Path, dataset: str, exp_type: str, exp_id: str, params: dict) -> Path:
    path = root / dataset / exp_type / exp_id / f"{exp_id}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 0, "params": params}
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def _write_run_metadata(root: Path, dataset: str, exp_type: str, run_id: str, meta: dict) -> Path:
    run_dir = root / dataset / exp_type / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f)
    return run_dir


def _write_judge_config(root: Path, dataset: str, judge_id: str, extraction_id: str | None) -> Path:
    """A judge_local/judge_interp-shaped committed config -- find_judge_combine_id
    resolves each judge_id's extraction via this, not via run_metadata."""
    params = {"dataset": dataset, "judge": "some-judge"}
    if extraction_id is not None:
        params["extraction_id"] = extraction_id
    return _write_experiment_config(root, dataset, "judge_local", judge_id, params)


@pytest.fixture
def fixture_roots(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment-configs"
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)
    return exp_root, results_root


def test_find_judge_combine_id_happy_path(fixture_roots):
    exp_root, _ = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_ids = ["2026-01-02-pond-model-judgea-judge-local-01", "2026-01-02-pond-model-judgeb-judge-local-01"]
    combine_id = "2026-01-03-pond-model-judge-combine-01"

    for jid in judge_ids:
        _write_judge_config(exp_root, "pond", jid, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": judge_ids})

    found_id, found_judge_ids = recovery.find_judge_combine_id("pond", extraction_id)
    assert found_id == combine_id
    assert found_judge_ids == judge_ids


def test_find_judge_combine_id_no_match_raises(fixture_roots):
    exp_root, _ = fixture_roots
    _write_judge_config(exp_root, "pond", "2026-01-02-pond-model-j1-01", "2026-01-01-pond-model-other-01")
    _write_experiment_config(exp_root, "pond", "judge_combine", "2026-01-03-pond-model-c1-01", {"judge_ids": ["2026-01-02-pond-model-j1-01"]})

    with pytest.raises(FileNotFoundError):
        recovery.find_judge_combine_id("pond", "2026-01-01-pond-model-extraction-01")


def test_find_judge_combine_id_ambiguous_raises(fixture_roots):
    exp_root, _ = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    for combine_id, jid in [
        ("2026-01-03-pond-model-c1-01", "2026-01-02-pond-model-j1-01"),
        ("2026-01-03-pond-model-c2-01", "2026-01-02-pond-model-j2-01"),
    ]:
        _write_judge_config(exp_root, "pond", jid, extraction_id)
        _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [jid]})

    with pytest.raises(ValueError, match="more than one"):
        recovery.find_judge_combine_id("pond", extraction_id)


def test_find_judge_combine_id_skips_candidate_with_unresolvable_judge(fixture_roots):
    # An unrelated combine config names a judge_id with no committed config at
    # all -- must be skipped as a candidate, not crash the search for a
    # different, unrelated extraction_id.
    exp_root, _ = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})

    unrelated_combine_id = "2026-01-04-pond-other-judge-combine-01"
    _write_experiment_config(
        exp_root, "pond", "judge_combine", unrelated_combine_id,
        {"judge_ids": ["2026-01-04-pond-other-nosuchconfig-judge-local-01"]},
    )

    found_id, found_judge_ids = recovery.find_judge_combine_id("pond", extraction_id)
    assert found_id == combine_id
    assert found_judge_ids == [judge_id]


def test_find_judge_combine_id_skips_synthetic_judge_with_no_extraction_id(fixture_roots):
    exp_root, _ = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})

    synthetic_judge_id = "2026-01-04-pond-synthetic-judgea-judge-local-01"
    _write_judge_config(exp_root, "pond", synthetic_judge_id, None)
    _write_experiment_config(
        exp_root, "pond", "judge_combine", "2026-01-05-pond-synthetic-combine-01",
        {"judge_ids": [synthetic_judge_id]},
    )

    found_id, found_judge_ids = recovery.find_judge_combine_id("pond", extraction_id)
    assert found_id == combine_id
    assert found_judge_ids == [judge_id]


def test_find_judge_combine_id_internally_inconsistent_combine_raises(fixture_roots):
    exp_root, _ = fixture_roots
    ext_a = "2026-01-01-pond-model-exta-01"
    ext_b = "2026-01-01-pond-model-extb-01"
    _write_judge_config(exp_root, "pond", "2026-01-02-pond-model-j1-01", ext_a)
    _write_judge_config(exp_root, "pond", "2026-01-02-pond-model-j2-01", ext_b)
    _write_experiment_config(
        exp_root, "pond", "judge_combine", "2026-01-03-pond-model-c1-01",
        {"judge_ids": ["2026-01-02-pond-model-j1-01", "2026-01-02-pond-model-j2-01"]},
    )

    with pytest.raises(ValueError, match="disagree"):
        recovery.find_judge_combine_id("pond", ext_a)


def test_find_judge_combine_id_run_metadata_disagreement_raises(fixture_roots):
    # The winning combine's judge ran, but against a different extraction_id
    # than its config currently declares (config edited after the run) --
    # must be caught, not silently trusted.
    exp_root, results_root = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})
    _write_run_metadata(results_root, "pond", "judge_local", judge_id, {"extraction_id": "some-other-extraction"})

    with pytest.raises(ValueError, match="not run against"):
        recovery.find_judge_combine_id("pond", extraction_id)


def test_find_judge_combine_id_run_metadata_agreement_passes(fixture_roots):
    exp_root, results_root = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})
    _write_run_metadata(results_root, "pond", "judge_local", judge_id, {"extraction_id": extraction_id})

    found_id, found_judge_ids = recovery.find_judge_combine_id("pond", extraction_id)
    assert found_id == combine_id
    assert found_judge_ids == [judge_id]


# ---------------------------------------------------------------------------
# verify_judge_combine_id -- the config-declared-id path (checks the exact
# same per-candidate rule find_judge_combine_id applies while scanning,
# against a single named candidate instead)
# ---------------------------------------------------------------------------


def test_verify_judge_combine_id_happy_path(fixture_roots):
    exp_root, _ = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})

    assert recovery.verify_judge_combine_id("pond", combine_id, extraction_id) == [judge_id]


def test_verify_judge_combine_id_disambiguates_what_scanning_would_refuse(fixture_roots):
    # Two judge_combine runs judge the same extraction -- find_judge_combine_id
    # itself would raise "more than one" here; a declared id should let the
    # caller pick one directly instead.
    exp_root, _ = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    for combine_id, jid in [
        ("2026-01-03-pond-model-c1-01", "2026-01-02-pond-model-j1-01"),
        ("2026-01-03-pond-model-c2-01", "2026-01-02-pond-model-j2-01"),
    ]:
        _write_judge_config(exp_root, "pond", jid, extraction_id)
        _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [jid]})

    with pytest.raises(ValueError, match="more than one"):
        recovery.find_judge_combine_id("pond", extraction_id)

    assert recovery.verify_judge_combine_id("pond", "2026-01-03-pond-model-c1-01", extraction_id) == [
        "2026-01-02-pond-model-j1-01"
    ]


def test_verify_judge_combine_id_wrong_extraction_raises(fixture_roots):
    exp_root, _ = fixture_roots
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, "2026-01-01-pond-model-actual-01")
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})

    with pytest.raises(ValueError, match="not the declared extraction_id"):
        recovery.verify_judge_combine_id("pond", combine_id, "2026-01-01-pond-model-wrong-01")


def test_verify_judge_combine_id_wrong_dataset_raises(fixture_roots):
    exp_root, _ = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})

    with pytest.raises(ValueError, match="not under experiment-configs/nfix/judge_combine"):
        recovery.verify_judge_combine_id("nfix", combine_id, extraction_id)


def test_verify_judge_combine_id_unresolvable_judge_raises(fixture_roots):
    exp_root, _ = fixture_roots
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_experiment_config(
        exp_root, "pond", "judge_combine", combine_id,
        {"judge_ids": ["2026-01-02-pond-model-nosuchconfig-judge-local-01"]},
    )

    with pytest.raises(FileNotFoundError):
        recovery.verify_judge_combine_id("pond", combine_id, "2026-01-01-pond-model-extraction-01")


def test_verify_judge_combine_id_run_metadata_disagreement_raises(fixture_roots):
    exp_root, results_root = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})
    _write_run_metadata(results_root, "pond", "judge_local", judge_id, {"extraction_id": "some-other-extraction"})

    with pytest.raises(ValueError, match="not run against"):
        recovery.verify_judge_combine_id("pond", combine_id, extraction_id)


def _final_json_row(mid, doc="d1", attr="ph"):
    return {"measurement_id": mid, "document_id": doc, "attribute": attr}


def test_load_validity_labels_happy_path(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0), _final_json_row(1, attr="tn")])
    combined = [
        {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True},
        {"measurement_id": 1, "document_id": "d1", "attribute": "tn", "judgement_combined": False},
    ]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)

    labels = recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)
    assert labels.tolist() == [True, False]


def test_load_validity_labels_out_of_order_still_aligns(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0), _final_json_row(1, attr="tn")])
    # combined.json rows out of order relative to measurement_id.
    combined = [
        {"measurement_id": 1, "document_id": "d1", "attribute": "tn", "judgement_combined": False},
        {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True},
    ]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)

    labels = recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)
    assert labels.tolist() == [True, False]


def test_load_validity_labels_split_children_inherit_parent_judgement(fixture_roots):
    _, results_root = fixture_roots
    # postprocessed.json split measurement 1 (doc d1) into two rows sharing its id.
    extraction_df = pd.DataFrame([
        _final_json_row(0), _final_json_row(1, attr="tn"), _final_json_row(1, attr="tn"), _final_json_row(2),
    ])
    combined = [
        {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True},
        {"measurement_id": 1, "document_id": "d1", "attribute": "tn", "judgement_combined": False},
        {"measurement_id": 2, "document_id": "d1", "attribute": "ph", "judgement_combined": True},
    ]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)

    labels = recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)
    assert labels.tolist() == [True, False, False, True]
    assert rv.count_split_rows(extraction_df) == 2
    assert rv.count_split_rows(extraction_df.drop_duplicates(["document_id", "measurement_id"])) == 0


def test_load_validity_labels_same_measurement_id_in_different_documents_is_distinct(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0, doc="d1"), _final_json_row(0, doc="d2")])
    combined = [
        {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True},
        {"measurement_id": 0, "document_id": "d2", "attribute": "ph", "judgement_combined": False},
    ]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)
    labels = recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)
    assert labels.tolist() == [True, False]


def test_load_validity_labels_unjudged_extraction_key_raises(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0), _final_json_row(1, attr="tn")])
    combined = [{"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True}]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)
    with pytest.raises(ValueError, match="no judgement"):
        recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


def test_load_validity_labels_attribute_mismatch_at_shared_key_raises(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0, attr="tn")])
    combined = [{"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True}]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)
    with pytest.raises(ValueError, match="attribute"):
        recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


def test_load_validity_labels_duplicate_combined_key_raises(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0)])
    rec = {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True}
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump([rec, rec], f)
    with pytest.raises(ValueError, match="distinct"):
        recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


def test_load_validity_labels_row_count_mismatch_raises(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0)])
    combined = [
        {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True},
        {"measurement_id": 1, "document_id": "d1", "attribute": "tn", "judgement_combined": False},
    ]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)

    with pytest.raises(ValueError, match="not the same run"):
        recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


def test_load_validity_labels_document_id_mismatch_raises(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0, doc="d1")])
    combined = [
        {"measurement_id": 0, "document_id": "d2", "attribute": "ph", "judgement_combined": True},
    ]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)

    with pytest.raises(ValueError, match="disagree"):
        recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


def test_load_validity_labels_non_bool_judgement_raises(fixture_roots):
    _, results_root = fixture_roots
    extraction_df = pd.DataFrame([_final_json_row(0)])
    combined = [
        {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": None},
    ]
    combine_dir = results_root / "pond" / "judge_combine" / "2026-01-03-pond-model-combo-01"
    combine_dir.mkdir(parents=True)
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)

    with pytest.raises(ValueError, match="not a bool"):
        recovery.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


# ---------------------------------------------------------------------------
# compute_metrics_for_id -- tiny end-to-end fixture (ground truth +
# extraction + a hand-crafted match_cache.pkl + judge_combine)
# ---------------------------------------------------------------------------


class _FixtureEntity(BaseModel):
    name: str


def _write_match_cache(cache_path: Path, edges: list[tuple[int, int]], edge_weights: list[float]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump((None, edges, edge_weights), f)


@pytest.fixture
def e2e_fixture(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment-configs"
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)
    monkeypatch.setattr(matching, "MATCH_CACHE_ROOT", tmp_path / "match_cache_root")

    dataset = "testset"
    extraction_id = "2026-01-01-testset-model-extraction-01"
    judge_id = "2026-01-02-testset-model-judge-local-01"
    combine_id = "2026-01-03-testset-model-judge-combine-01"

    # 3 ground-truth rows across 2 papers; row 2 (d2) has no fuzzy-matching
    # candidate above threshold below, so it's neither recovered nor matched.
    gt_rows = [
        {"document_id": "d1", "attribute": "ph", "point_value": 7.0, "units": None, "name": "Lake A", "ecosystem": "pond"},
        {"document_id": "d1", "attribute": "tn", "point_value": 1.0, "units": "mg/L", "name": "Lake A", "ecosystem": "pond"},
        {"document_id": "d2", "attribute": "ph", "point_value": 6.5, "units": None, "name": "Lake B", "ecosystem": "pond"},
    ]
    ext_rows = [
        {"document_id": "d1", "attribute": "ph", "point_value": 7.0, "units": None, "name": "Lake A", "ecosystem": "pond", "measurement_id": 0},
        {"document_id": "d1", "attribute": "tn", "point_value": 1.0, "units": "mg/L", "name": "Lake A", "ecosystem": "pond", "measurement_id": 1},
        {"document_id": "d2", "attribute": "ph", "point_value": 6.5, "units": None, "name": "Lake B", "ecosystem": "pond", "measurement_id": 2},
    ]

    gt_path = tmp_path / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt_rows, f)

    extraction_dir = results_root / dataset / "extraction" / extraction_id
    extraction_dir.mkdir(parents=True)
    final_path = extraction_dir / "final.json"
    with open(final_path, "w") as f:
        json.dump(ext_rows, f)

    dataset_config = DatasetConfig(
        name=dataset,
        data_dir=f"data/{dataset}",
        metadata_file=f"data/{dataset}/directory.json",
        entity_schema=_FixtureEntity,
        entity_identification_prompt="prompt",
        entity_type_description="a thing",
        attribute_info_dict={},
        strict_matching={"document_id": "document_id", "attribute": "attribute", "point_value": "point_value", "units": "units"},
        fuzzy_matching={"name": "name", "ecosystem": "ecosystem"},
        fuzzy_threshold=0.5,
        numeric_coerce=["point_value"],
    )
    monkeypatch.setattr(matching, "load_dataset_config", lambda ds: dataset_config)

    # Cache built at threshold 0.0: gt row 2 / ext row 2's candidate edge
    # scores 0.2, below the 0.5 fuzzy_threshold above -- excluded once
    # load_match_cache(..., fuzzy_threshold=0.5) filters it.
    cache_path = matching.match_cache_path(extraction_id)
    edges = [(0, 0), (1, 1), (2, 2)]
    edge_weights = [1.0, 1.0, 0.2]
    _write_match_cache(cache_path, edges, edge_weights)
    # match_cache.meta.json sidecar -- assert_ground_truth_matches_cache/
    # assert_extraction_matches_cache require this to confirm the cache was
    # built against gt_path/final_path (no postprocessed.json in this fixture,
    # so extraction_path() falls back to final.json).
    meta = {
        "ground_truth_file": provenance.repo_relative(gt_path),
        "ground_truth_sha256": provenance.sha256_file(gt_path),
        "n_gt": len(gt_rows),
        "extraction_file": provenance.repo_relative(final_path),
        "extraction_sha256": provenance.sha256_file(final_path),
    }
    with open(cache_path.with_name("match_cache.meta.json"), "w") as f:
        json.dump(meta, f)
    # Cache must be >= final.json/ground-truth mtime (freshness guard).
    import os
    import time
    future = time.time() + 10
    os.utime(cache_path, (future, future))

    judge_ids = [judge_id]
    combine_dir = results_root / dataset / "judge_combine" / combine_id
    combine_dir.mkdir(parents=True)
    combined = [
        {"measurement_id": 0, "document_id": "d1", "attribute": "ph", "judgement_combined": True},
        {"measurement_id": 1, "document_id": "d1", "attribute": "tn", "judgement_combined": False},
        {"measurement_id": 2, "document_id": "d2", "attribute": "ph", "judgement_combined": True},
    ]
    with open(combine_dir / "combined.json", "w") as f:
        json.dump(combined, f)
    _write_judge_config(exp_root, dataset, judge_id, extraction_id)
    _write_experiment_config(exp_root, dataset, "judge_combine", combine_id, {"judge_ids": judge_ids})

    return extraction_id, combine_id, judge_ids, gt_path


def test_compute_metrics_for_id_with_validity(e2e_fixture):
    extraction_id, combine_id, judge_ids, gt_path = e2e_fixture
    row = rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)

    assert row["n_gt"] == 3
    assert row["n_ext"] == 3
    assert row["ground_truth_file"] == provenance.repo_relative(gt_path)
    extraction_dir = paths.find_result_dir(extraction_id)
    assert row["extraction_file"] == provenance.repo_relative(extraction_dir / "final.json")
    # gt rows 0,1 recovered (weight 1.0 > 0.5); row 2 not (weight 0.2).
    assert row["recovery"] == pytest.approx(2 / 3)
    assert row["judge_combine_id"] == combine_id
    assert row["judge_ids"] == ";".join(judge_ids)
    # matched = [True, True, False]; judged = [True, False, True] -> OR = [True, True, True].
    assert row["validity"] == pytest.approx(1.0)
    assert row["validity_ci_lo"] == pytest.approx(1.0)
    assert row["validity_ci_hi"] == pytest.approx(1.0)
    assert 0.0 <= row["recovery_ci_lo"] <= row["recovery"] <= row["recovery_ci_hi"] <= 1.0


# ---------------------------------------------------------------------------
# assert_ground_truth_matches_cache -- a cache passing every OTHER guard
# (fresh by mtime, n_gt in range) must still be refused if it wasn't built
# against the ground_truth_path given now. Without this guard these three
# scenarios would silently misalign the cached (gt_idx, ex_idx) edges.
# ---------------------------------------------------------------------------


def test_compute_metrics_for_id_different_ground_truth_file_raises(tmp_path, e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture

    # Same shape as the fixture's ground truth (so n_gt still matches, and
    # this file is freshly written so it's newer than nothing the freshness
    # check would catch), but one value changed -- only the sidecar's sha256
    # can tell this apart from the cache's real ground truth.
    gt_b_rows = json.loads(gt_path.read_text())
    gt_b_rows[0]["point_value"] = 8.0
    gt_b_path = tmp_path / "gt_b.json"
    with open(gt_b_path, "w") as f:
        json.dump(gt_b_rows, f)

    with pytest.raises(RuntimeError, match="different ground truth"):
        rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_b_path, n_resamples=200, seed=0)


def test_compute_metrics_for_id_ground_truth_edited_in_place_raises(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture

    # Same path, contents changed after the cache was built -- the sha256
    # check must catch this even though the path string is identical.
    rows = json.loads(gt_path.read_text())
    rows[0]["point_value"] = 8.0
    gt_path.write_text(json.dumps(rows))

    with pytest.raises(RuntimeError, match="sha256"):
        rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)


def test_compute_metrics_for_id_missing_sidecar_raises(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture

    cache_path = matching.match_cache_path(extraction_id)
    cache_path.with_name("match_cache.meta.json").unlink()

    with pytest.raises(FileNotFoundError, match="match_cache.meta.json"):
        rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)


def test_compute_metrics_for_id_extraction_edited_in_place_raises(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture

    # Same path, contents changed after the cache was built (e.g. a later
    # analysis/postprocessing.py run) -- the sha256 check must catch this
    # even though final.json is still the file extraction_path() resolves to,
    # and even though row count/order (and so measurement_id alignment) is
    # unchanged.
    final_path = paths.find_result_dir(extraction_id) / "final.json"
    rows = json.loads(final_path.read_text())
    rows[0]["point_value"] = 99.0
    final_path.write_text(json.dumps(rows))

    with pytest.raises(RuntimeError, match="different extraction file"):
        rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)


def test_compute_metrics_for_id_sidecar_missing_extraction_tracking_raises(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture

    # A match_cache.meta.json written before extraction-file tracking existed
    # (ground_truth_file/sha256/n_gt only) must not be trusted, exactly like
    # a missing sidecar -- it says nothing about which extraction file
    # produced the cache sitting next to it.
    meta_path = matching.match_cache_path(extraction_id).with_name("match_cache.meta.json")
    meta = json.loads(meta_path.read_text())
    del meta["extraction_file"]
    del meta["extraction_sha256"]
    meta_path.write_text(json.dumps(meta))

    with pytest.raises(FileNotFoundError, match="predates extraction-file tracking"):
        rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)


def test_compute_metrics_for_id_prefers_postprocessed_json(e2e_fixture):
    # A postprocessed.json appearing after the cache was built (and the
    # sidecar still pointing at final.json) must be caught as a mismatch --
    # load_frames now resolves extraction_path() itself, so this is exactly
    # the scenario assert_extraction_matches_cache exists for.
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    extraction_dir = paths.find_result_dir(extraction_id)
    rows = json.loads((extraction_dir / "final.json").read_text())
    (extraction_dir / "postprocessed.json").write_text(json.dumps(rows))

    with pytest.raises(RuntimeError, match="different extraction file"):
        rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)


def test_compute_metrics_for_id_skip_validity_never_touches_judge_combine(e2e_fixture, monkeypatch):
    extraction_id, combine_id, _judge_ids, gt_path = e2e_fixture

    def _boom(*args, **kwargs):
        raise AssertionError("find_judge_combine_id must not be called when compute_validity=False")

    monkeypatch.setattr(recovery, "find_judge_combine_id", _boom)

    row = rv.compute_metrics_for_id(
        extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0, compute_validity=False,
    )

    assert row["recovery"] == pytest.approx(2 / 3)
    assert row["judge_combine_id"] is None
    assert row["judge_ids"] is None
    assert row["validity"] is None
    assert row["validity_ci_lo"] is None
    assert row["validity_ci_hi"] is None


def test_compute_metrics_for_id_with_declared_judge_combine_id_skips_scan(e2e_fixture, monkeypatch):
    extraction_id, combine_id, judge_ids, gt_path = e2e_fixture

    def _boom(*args, **kwargs):
        raise AssertionError("find_judge_combine_id must not be called when judge_combine_id is given")

    monkeypatch.setattr(recovery, "find_judge_combine_id", _boom)

    row = rv.compute_metrics_for_id(
        extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0, judge_combine_id=combine_id,
    )
    assert row["judge_combine_id"] == combine_id
    assert row["judge_ids"] == ";".join(judge_ids)
    assert row["validity"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# max_weight_matching_recovered / edge filter / any-edge vs matching recovery
# ---------------------------------------------------------------------------


def test_matching_one_extraction_edged_to_two_gt_recovers_only_one():
    # ex 0 can serve only one gt row -- any-edge counting would say 2/2.
    recovered, matching = rv.max_weight_matching_recovered(2, 1, [(0, 0), (1, 0)], [1.0, 1.0])
    assert recovered.sum() == 1
    assert len(matching) == 1


def test_matching_heavy_edge_beats_two_light_edges():
    # gt0-ex0=0.9 alone (0.9) beats gt0-ex1 + gt1-ex0 (0.4 + 0.4 = 0.8), even
    # though the latter recovers 2 gt rows: weight first, cardinality second.
    edges = [(0, 0), (0, 1), (1, 0)]
    recovered, matching = rv.max_weight_matching_recovered(2, 2, edges, [0.9, 0.4, 0.4])
    assert matching == [(0, 0)]
    assert recovered.tolist() == [True, False]


def test_matching_light_edges_win_when_their_total_is_larger():
    edges = [(0, 0), (0, 1), (1, 0)]
    recovered, matching = rv.max_weight_matching_recovered(2, 2, edges, [0.7, 0.4, 0.4])
    assert matching == [(0, 1), (1, 0)]
    assert recovered.tolist() == [True, True]


def test_matching_zero_weight_lone_edge_is_recovered():
    recovered, matching = rv.max_weight_matching_recovered(1, 1, [(0, 0)], [0.0])
    assert recovered.tolist() == [True]


def test_matching_zero_weight_edge_never_displaces_positive_edge():
    recovered, matching = rv.max_weight_matching_recovered(2, 1, [(0, 0), (1, 0)], [0.0, 0.5])
    assert matching == [(1, 0)]


def test_matching_cardinality_breaks_exact_weight_ties_only():
    # gt0-ex0=0.5 alone vs gt0-ex1=0.5 + gt1-ex0=0.0: equal total weight, the
    # 2-edge matching wins the tie.
    edges = [(0, 0), (0, 1), (1, 0)]
    recovered, _ = rv.max_weight_matching_recovered(2, 2, edges, [0.5, 0.5, 0.0])
    assert recovered.tolist() == [True, True]
    # ...but a tiny real-weight deficit loses to the single heavier edge.
    recovered, matching = rv.max_weight_matching_recovered(2, 2, edges, [0.502, 0.5, 0.0])
    assert matching == [(0, 0)]


def test_matching_gt_and_ext_index_spaces_do_not_collide():
    # gt row 0 and extraction row 0 are different nodes; both can be matched.
    recovered, matching = rv.max_weight_matching_recovered(2, 2, [(0, 1), (1, 0)], [1.0, 1.0])
    assert matching == [(0, 1), (1, 0)]


def test_matching_no_edges_recovers_nothing():
    recovered, matching = rv.max_weight_matching_recovered(3, 3, [], [])
    assert recovered.tolist() == [False] * 3 and matching == []


def test_matching_is_deterministic():
    edges = [(i % 4, (i * 3) % 5) for i in range(12)]
    edges = sorted(set(edges))
    w = [0.1 * ((i * 7) % 10) for i in range(len(edges))]
    a = rv.max_weight_matching_recovered(4, 5, edges, w)
    b = rv.max_weight_matching_recovered(4, 5, edges, w)
    assert a[1] == b[1] and a[0].tolist() == b[0].tolist()


def test_matching_rejects_duplicate_edges_bad_weights_and_out_of_range():
    with pytest.raises(ValueError, match="duplicate"):
        rv.max_weight_matching_recovered(1, 1, [(0, 0), (0, 0)], [1.0, 1.0])
    with pytest.raises(ValueError, match="outside"):
        rv.max_weight_matching_recovered(1, 1, [(0, 0)], [1.5])
    with pytest.raises(ValueError, match="out of range"):
        rv.max_weight_matching_recovered(1, 1, [(0, 3)], [1.0])


def test_filter_edges_by_threshold_is_inclusive_and_keeps_weights_aligned():
    edges, weights = recovery.filter_edges_by_threshold([(0, 0), (1, 1), (2, 2)], [0.5, 0.49, 1.0], 0.5)
    assert edges == [(0, 0), (2, 2)] and weights == [0.5, 1.0]


def test_verify_matching_rejects_cross_paper_edge():
    gt = pd.DataFrame({"document_id": ["d1"]})
    ex = pd.DataFrame({"document_id": ["d2"]})
    with pytest.raises(AssertionError, match="different document_ids"):
        rv._verify_matching([(0, 0)], [(0, 0)], np.array([True]), np.array([True]), gt, ex)


def _rewrite_fixture_cache(extraction_id, edges, edge_weights):
    """Swap the e2e fixture's cached edges, keeping its sidecar and future mtime."""
    import os
    import time
    cache_path = matching.match_cache_path(extraction_id)
    _write_match_cache(cache_path, edges, edge_weights)
    future = time.time() + 10
    os.utime(cache_path, (future, future))


def test_compute_metrics_reports_any_edge_recovery_and_matching_separately(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    # ext 0 has threshold edges to gt 0 and gt 1 (both d1); gt 2's edge is
    # 0.2 < 0.5. Any-edge: gt 0, 1 recovered -> 2/3. Matching: ext 0 serves
    # only one gt row -> 1/3.
    _rewrite_fixture_cache(extraction_id, [(0, 0), (1, 0), (2, 2)], [1.0, 1.0, 0.2])
    row = rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)
    assert row["fuzzy_threshold"] == 0.5
    assert row["n_surviving_edges"] == 2
    assert row["recovery"] == pytest.approx(2 / 3)
    assert row["recovery_max_weight_matching"] == pytest.approx(1 / 3)
    assert 0.0 <= row["recovery_max_weight_matching_ci_lo"] <= row["recovery_max_weight_matching"]
    assert row["recovery_max_weight_matching"] <= row["recovery_max_weight_matching_ci_hi"] <= 1.0
    assert "recovery_any_edge" not in row and "edge_filter" not in row
    # matched ext = [T, F, F] | judged [T, F, T] -> 2/3.
    assert row["validity"] == pytest.approx(2 / 3)


def test_validity_counts_every_extraction_with_a_threshold_edge_not_just_matched_ones(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    # gt 0 has threshold edges to ext 0 (1.0) and ext 1 (0.6); the 1-1
    # matching keeps only the heavier, (0, 0). ext 1 is judged invalid, so
    # validity is 1.0 only because ext 1 counts as matched by having an edge;
    # matching-membership labels would give [T, F, F] | [T, F, T] = 2/3.
    _rewrite_fixture_cache(extraction_id, [(0, 0), (0, 1), (2, 2)], [1.0, 0.6, 0.2])
    assert rv.max_weight_matching_recovered(3, 3, [(0, 0), (0, 1)], [1.0, 0.6])[1] == [(0, 0)]
    row = rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=200, seed=0)
    assert row["recovery"] == pytest.approx(1 / 3)
    assert row["recovery_max_weight_matching"] == pytest.approx(1 / 3)
    assert row["validity"] == pytest.approx(1.0)


def test_calibration_labels_count_every_extraction_with_a_threshold_edge(e2e_fixture):
    # Same cache as above, through the calibration path
    # (calibration_updated_v4 / platt_scaling_v2 / meta via predictions.pkl):
    # both ext 0 and ext 1 must come back as having an edge.
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    _rewrite_fixture_cache(extraction_id, [(0, 0), (0, 1), (2, 2)], [1.0, 0.6, 0.2])
    _gt_df, ext_df, edges = matching.load_cached_matching(extraction_id, gt_path)
    judged_df = ext_df[["measurement_id", "document_id", "attribute"]].reset_index(drop=True)
    judged_edges = matching.edges_to_judged_rows(edges, ext_df, judged_df)
    assert judged_edges == [(0, 0), (0, 1)]


def test_compute_metrics_seed_determinism(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    kw = dict(ground_truth_path=gt_path, n_resamples=200, seed=7)
    assert rv.compute_metrics_for_id(extraction_id, **kw) == rv.compute_metrics_for_id(extraction_id, **kw)


# ---------------------------------------------------------------------------
# main(): --config vs. ad-hoc CLI, mutually exclusive
# ---------------------------------------------------------------------------


def test_main_rejects_neither_ids_nor_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["recovery_validity.py"])
    with pytest.raises(SystemExit):
        rv.main([])


def test_main_rejects_config_plus_cli_flags(tmp_path):
    with pytest.raises(SystemExit):
        rv.main(["some-id", "--config", str(tmp_path / "x.yaml"), "--n-resamples", "10"])


def test_main_rejects_ids_without_n_resamples_or_seed():
    with pytest.raises(SystemExit):
        rv.main(["some-id", "--ground-truth-file", "data/pond/ground_truth_review.json"])


def test_main_rejects_ids_without_ground_truth_file():
    with pytest.raises(SystemExit):
        rv.main(["some-id", "--n-resamples", "10", "--seed", "0"])


def test_main_rejects_removed_edge_filter_flag(tmp_path):
    gt_path = tmp_path / "gt.json"
    gt_path.write_text("[]")
    with pytest.raises(SystemExit):
        rv.main([
            "some-id", "--n-resamples", "10", "--seed", "0", "--ground-truth-file", str(gt_path),
            "--edge-filter", "threshold",
        ])


def test_main_config_mode_rejects_removed_edge_filter_key(tmp_path):
    with pytest.raises(KeyError, match="edge_filter"):
        rv.main(["--config", str(_rv_config(tmp_path, edge_filter="threshold"))])


def test_main_ad_hoc_cli_calls_compute_metrics_for_id(tmp_path, monkeypatch):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")

    calls = []

    def _fake_compute(experiment_id, *, ground_truth_path, n_resamples, seed, alpha, compute_validity, judge_combine_id):
        calls.append((experiment_id, ground_truth_path, n_resamples, seed, alpha, compute_validity, judge_combine_id))
        return {
            "experiment_id": experiment_id, "recovery": 0.5, "recovery_ci_lo": 0.4, "recovery_ci_hi": 0.6,
            "recovery_max_weight_matching": 0.5,
            "validity": None, "validity_ci_lo": None, "validity_ci_hi": None, "judge_combine_id": None,
            "judge_ids": None,
        }

    monkeypatch.setattr(rv, "compute_metrics_for_id", _fake_compute)
    output = tmp_path / "out.csv"
    rv.main([
        "id-a", "--n-resamples", "10", "--seed", "0", "--ground-truth-file", str(gt_path),
        "--skip-validity", "--output", str(output),
    ])

    assert calls == [("id-a", gt_path, 10, 0, 0.05, False, None)]
    df = pd.read_csv(output)
    assert pd.isna(df.loc[0, "analysis_config_id"])  # None round-tripped through CSV as NaN


def test_main_config_mode_reads_params_and_applies_judge_combine_override(tmp_path, monkeypatch):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")

    config_path = tmp_path / "2026-09-23-test-rv-01.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "id": "2026-09-23-test-rv-01",
                "project": "scholarlm",
                "description": "test",
                "seed": 342,
                "params": {
                    "experiment_ids": ["id-a", "id-b"],
                    "ground_truth_file": str(gt_path),
                    "recovery_validity": {
                        "n_resamples": 500,
                        "alpha": 0.1,
                        "compute_validity": True,
                        "output": str(tmp_path / "out.csv"),
                        "judge_combine_ids": {"id-a": "declared-combine-id"},
                    },
                },
            },
            f,
        )

    calls = []

    def _fake_compute(experiment_id, *, ground_truth_path, n_resamples, seed, alpha, compute_validity, judge_combine_id):
        calls.append((experiment_id, ground_truth_path, n_resamples, seed, alpha, compute_validity, judge_combine_id))
        return {
            "experiment_id": experiment_id, "recovery": 0.5, "recovery_ci_lo": 0.4, "recovery_ci_hi": 0.6,
            "recovery_max_weight_matching": 0.5,
            "validity": None, "validity_ci_lo": None, "validity_ci_hi": None, "judge_combine_id": None,
            "judge_ids": None,
        }

    monkeypatch.setattr(rv, "compute_metrics_for_id", _fake_compute)
    rv.main(["--config", str(config_path)])

    assert calls == [
        ("id-a", gt_path, 500, 342, 0.1, True, "declared-combine-id"),
        ("id-b", gt_path, 500, 342, 0.1, True, None),
    ]
    df = pd.read_csv(tmp_path / "out.csv")
    assert (df["analysis_config_id"] == "2026-09-23-test-rv-01").all()


def test_main_config_mode_unknown_judge_combine_override_key_raises(tmp_path):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")

    config_path = tmp_path / "2026-09-23-test-rv-01.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "id": "2026-09-23-test-rv-01",
                "project": "scholarlm",
                "description": "test",
                "seed": 342,
                "params": {
                    "experiment_ids": ["id-a"],
                    "ground_truth_file": str(gt_path),
                    "recovery_validity": {
                        "n_resamples": 500,
                        "alpha": 0.1,
                        "compute_validity": True,
                        "output": str(tmp_path / "out.csv"),
                        "judge_combine_ids": {"id-not-in-experiment-ids": "declared-combine-id"},
                    },
                },
            },
            f,
        )
    with pytest.raises(ValueError, match="not in params.experiment_ids"):
        rv.main(["--config", str(config_path)])


def _rv_config(tmp_path, **rv_overrides):
    section = {
        "n_resamples": 500,
        "alpha": 0.1,
        "compute_validity": True,
        "output": str(tmp_path / "out.csv"),
    }
    section.update(rv_overrides)
    gt_path = tmp_path / "ground_truth.json"
    if not gt_path.exists():
        gt_path.write_text("[]")
    config_path = tmp_path / "2026-09-23-test-rv-01.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "id": "2026-09-23-test-rv-01",
                "project": "scholarlm",
                "description": "test",
                "seed": 342,
                "params": {
                    "experiment_ids": ["id-a"],
                    "ground_truth_file": str(gt_path),
                    "recovery_validity": section,
                },
            },
            f,
        )
    return config_path


def test_main_config_mode_non_bool_compute_validity_raises(tmp_path):
    config_path = _rv_config(tmp_path, compute_validity="false")  # YAML string, not a bool
    with pytest.raises(ValueError, match="compute_validity must be a bool"):
        rv.main(["--config", str(config_path)])


def test_main_config_mode_null_judge_combine_override_value_raises(tmp_path):
    config_path = _rv_config(tmp_path, judge_combine_ids={"id-a": None})  # "id-a:" with nothing after it
    with pytest.raises(ValueError, match="string-to-string mapping"):
        rv.main(["--config", str(config_path)])


def test_main_config_mode_judge_combine_override_with_compute_validity_false_raises(tmp_path):
    config_path = _rv_config(
        tmp_path, compute_validity=False, judge_combine_ids={"id-a": "declared-combine-id"}
    )
    with pytest.raises(ValueError, match="would never be used"):
        rv.main(["--config", str(config_path)])


# ---------------------------------------------------------------------------
# fuzzy_threshold_curve -- recovery/validity swept over the fuzzy threshold
# ---------------------------------------------------------------------------

# Edge weights 1.0 / 0.6 / 0.2 on gt-ext pairs 0/1/2; judged [T, F, T];
# the fixture's dataset threshold is 0.5. Edges are kept at w >= t, so:
#   t=0.0, 0.2: all 3 edges -> recovery 3/3; matched all -> validity 3/3
#   t=0.5, 0.6: edges 0,1   -> recovery 2/3; matched [T,T,F] | judged [T,F,T] -> 3/3
#   t=0.7, 1.0: edge 0      -> recovery 1/3; matched [T,F,F] | judged [T,F,T] -> 2/3
_CURVE_THRESHOLDS = [0.0, 0.2, 0.5, 0.6, 0.7, 1.0]
_CURVE_WEIGHTS = [1.0, 0.6, 0.2]


def test_fuzzy_threshold_curve_known_answer(e2e_fixture):
    extraction_id, combine_id, _judge_ids, gt_path = e2e_fixture
    _rewrite_fixture_cache(extraction_id, [(0, 0), (1, 1), (2, 2)], _CURVE_WEIGHTS)

    curve = rv.fuzzy_threshold_curve(extraction_id, ground_truth_path=gt_path, thresholds=_CURVE_THRESHOLDS)

    assert curve["fuzzy_threshold"].tolist() == _CURVE_THRESHOLDS
    assert curve["recovery"].tolist() == pytest.approx([1, 1, 2 / 3, 2 / 3, 1 / 3, 1 / 3])
    assert curve["validity"].tolist() == pytest.approx([1, 1, 1, 1, 2 / 3, 2 / 3])
    assert curve["n_surviving_edges"].tolist() == [3, 3, 2, 2, 1, 1]
    assert curve["is_dataset_threshold"].tolist() == [False, False, True, False, False, False]
    assert (curve["judge_combine_id"] == combine_id).all()

    # Known-answer control: the dataset-threshold point is the headline row.
    row = rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=50, seed=0)
    rv._assert_curve_matches_row(curve, row)


def test_assert_curve_matches_row_catches_disagreement(e2e_fixture):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    curve = rv.fuzzy_threshold_curve(extraction_id, ground_truth_path=gt_path, thresholds=[0.0, 0.5])
    row = rv.compute_metrics_for_id(extraction_id, ground_truth_path=gt_path, n_resamples=50, seed=0)
    row["recovery"] = 0.0
    with pytest.raises(AssertionError, match="disagrees"):
        rv._assert_curve_matches_row(curve, row)


@pytest.mark.parametrize("thresholds, match", [
    ([0.0, 0.6], "must include the dataset's own fuzzy_threshold"),
    ([0.5, 0.2], "strictly increasing"),
    ([0.5, 0.5], "strictly increasing"),
    ([0.5, 1.5], r"\[0, 1\]"),
    ([], "non-empty list"),
    ([True, 0.5], "non-empty list of numbers"),
])
def test_fuzzy_threshold_curve_rejects_bad_grid(e2e_fixture, thresholds, match):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    with pytest.raises(ValueError, match=match):
        rv.fuzzy_threshold_curve(extraction_id, ground_truth_path=gt_path, thresholds=thresholds)


def _curve_config(tmp_path, gt_path, experiment_ids, curve_section, compute_validity=True):
    config_path = tmp_path / "2026-10-07-test-curve-01.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "id": "2026-10-07-test-curve-01",
                "project": "scholarlm",
                "description": "test",
                "seed": 342,
                "params": {
                    "experiment_ids": experiment_ids,
                    "ground_truth_file": str(gt_path),
                    "recovery_validity": {
                        "n_resamples": 50,
                        "alpha": 0.05,
                        "compute_validity": compute_validity,
                        "output": str(tmp_path / "out.csv"),
                        "fuzzy_threshold_curve": curve_section,
                    },
                },
            },
            f,
        )
    return config_path


def test_main_config_mode_writes_curve_csv_and_figures(tmp_path, e2e_fixture, monkeypatch):
    extraction_id, _combine_id, _judge_ids, gt_path = e2e_fixture
    _rewrite_fixture_cache(extraction_id, [(0, 0), (1, 1), (2, 2)], _CURVE_WEIGHTS)
    figures_dir = tmp_path / "rv-out" / "2026-10-07-test-curve-01" / "figures"
    monkeypatch.setattr(rv, "fuzzy_threshold_figures_dir", lambda config_id: figures_dir)

    config_path = _curve_config(
        tmp_path, gt_path, [extraction_id],
        {"experiment_ids": [extraction_id], "thresholds": _CURVE_THRESHOLDS},
    )
    rv.main(["--config", str(config_path)])

    curve = pd.read_csv(figures_dir.parent / f"{extraction_id}-fuzzy-threshold-curve.csv")
    assert curve["recovery"].tolist() == pytest.approx([1, 1, 2 / 3, 2 / 3, 1 / 3, 1 / 3])
    assert (curve["analysis_config_id"] == "2026-10-07-test-curve-01").all()
    assert (figures_dir / f"{extraction_id}.pdf").stat().st_size > 0
    assert (figures_dir / "fuzzy_threshold_colorbar.pdf").stat().st_size > 0


def test_main_config_mode_curve_id_not_in_experiment_ids_raises(tmp_path):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")
    config_path = _curve_config(tmp_path, gt_path, ["id-a"], {"experiment_ids": ["id-b"], "thresholds": [0.5]})
    with pytest.raises(ValueError, match="not in params.experiment_ids"):
        rv.main(["--config", str(config_path)])


def test_main_config_mode_curve_without_validity_raises(tmp_path):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")
    config_path = _curve_config(
        tmp_path, gt_path, ["id-a"], {"experiment_ids": ["id-a"], "thresholds": [0.5]}, compute_validity=False,
    )
    with pytest.raises(ValueError, match="compute_validity is false"):
        rv.main(["--config", str(config_path)])


def test_main_config_mode_curve_section_missing_key_raises(tmp_path):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")
    config_path = _curve_config(tmp_path, gt_path, ["id-a"], {"experiment_ids": ["id-a"]})
    with pytest.raises(ValueError, match="exactly the keys"):
        rv.main(["--config", str(config_path)])


def test_plot_fuzzy_threshold_curve_requires_exactly_one_selected_threshold(tmp_path):
    curve = pd.DataFrame({
        "fuzzy_threshold": [0.0, 0.5, 1.0], "recovery": [1.0, 0.5, 0.0], "validity": [1.0, 0.8, 0.6],
        "is_dataset_threshold": [False, True, False],
    })
    rv.plot_fuzzy_threshold_curve(curve, tmp_path / "ok.pdf")
    assert (tmp_path / "ok.pdf").stat().st_size > 0
    for flags in ([False, False, False], [True, True, False]):
        with pytest.raises(ValueError, match="exactly one is_dataset_threshold"):
            rv.plot_fuzzy_threshold_curve(curve.assign(is_dataset_threshold=flags), tmp_path / "bad.pdf")
