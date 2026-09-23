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


# ---------------------------------------------------------------------------
# gt_recovered_mask / ext_matched_mask
# ---------------------------------------------------------------------------


def test_masks_mark_gt_and_ext_indices_present_in_edges():
    # edges is already threshold-filtered (match_cache.load_match_cache(id,
    # fuzzy_threshold=...) -- see module docstring), so the masks just mark
    # presence, no weight/threshold argument.
    edges = [(0, 0), (0, 1)]  # gt row 1 has no surviving edge

    recovered = rv.gt_recovered_mask(2, edges)
    matched = rv.ext_matched_mask(3, edges)

    assert recovered.tolist() == [True, False]
    assert matched.tolist() == [True, True, False]


def test_masks_all_zero_when_no_edges():
    assert rv.gt_recovered_mask(2, []).tolist() == [False, False]
    assert rv.ext_matched_mask(1, []).tolist() == [False]


# ---------------------------------------------------------------------------
# _assert_matching_columns_present
# ---------------------------------------------------------------------------

_CFG = {
    "strict": {"document_id": "document_id", "attribute": "attribute", "point_value": "point_value", "units": "units"},
    "fuzzy": {"name": "name", "ecosystem": "ecosystem"},
}


def test_matching_columns_present_passes_when_all_columns_exist():
    gt = pd.DataFrame(columns=["document_id", "attribute", "point_value", "units", "name", "ecosystem"])
    ext = pd.DataFrame(columns=["document_id", "attribute", "point_value", "units", "name", "ecosystem"])
    rv._assert_matching_columns_present(gt, ext, _CFG)  # must not raise


def test_matching_columns_present_raises_on_legacy_schema():
    # Simulates a cache built under ablation.py's rules (value/converted_value)
    # sitting where a point_value-keyed cache is now expected.
    gt = pd.DataFrame(columns=["document_id", "attribute", "value", "units", "name", "ecosystem", "location"])
    ext = pd.DataFrame(columns=["document_id", "attribute", "converted_value", "units", "name", "ecosystem", "location"])
    with pytest.raises(KeyError, match="point_value"):
        rv._assert_matching_columns_present(gt, ext, _CFG)


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

    found_id, found_judge_ids = rv.find_judge_combine_id("pond", extraction_id)
    assert found_id == combine_id
    assert found_judge_ids == judge_ids


def test_find_judge_combine_id_no_match_raises(fixture_roots):
    exp_root, _ = fixture_roots
    _write_judge_config(exp_root, "pond", "2026-01-02-pond-model-j1-01", "2026-01-01-pond-model-other-01")
    _write_experiment_config(exp_root, "pond", "judge_combine", "2026-01-03-pond-model-c1-01", {"judge_ids": ["2026-01-02-pond-model-j1-01"]})

    with pytest.raises(FileNotFoundError):
        rv.find_judge_combine_id("pond", "2026-01-01-pond-model-extraction-01")


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
        rv.find_judge_combine_id("pond", extraction_id)


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

    found_id, found_judge_ids = rv.find_judge_combine_id("pond", extraction_id)
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

    found_id, found_judge_ids = rv.find_judge_combine_id("pond", extraction_id)
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
        rv.find_judge_combine_id("pond", ext_a)


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
        rv.find_judge_combine_id("pond", extraction_id)


def test_find_judge_combine_id_run_metadata_agreement_passes(fixture_roots):
    exp_root, results_root = fixture_roots
    extraction_id = "2026-01-01-pond-model-extraction-01"
    judge_id = "2026-01-02-pond-model-judgea-judge-local-01"
    combine_id = "2026-01-03-pond-model-judge-combine-01"
    _write_judge_config(exp_root, "pond", judge_id, extraction_id)
    _write_experiment_config(exp_root, "pond", "judge_combine", combine_id, {"judge_ids": [judge_id]})
    _write_run_metadata(results_root, "pond", "judge_local", judge_id, {"extraction_id": extraction_id})

    found_id, found_judge_ids = rv.find_judge_combine_id("pond", extraction_id)
    assert found_id == combine_id
    assert found_judge_ids == [judge_id]


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

    labels = rv.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)
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

    labels = rv.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)
    assert labels.tolist() == [True, False]


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
        rv.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


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
        rv.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


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
        rv.load_validity_labels("2026-01-03-pond-model-combo-01", extraction_df)


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
        ground_truth_file=str(gt_path),
        strict_matching={"document_id": "document_id", "attribute": "attribute", "point_value": "point_value", "units": "units"},
        fuzzy_matching={"name": "name", "ecosystem": "ecosystem"},
        fuzzy_threshold=0.5,
        numeric_coerce=["point_value"],
    )
    monkeypatch.setattr(rv, "load_dataset_config", lambda ds: dataset_config)

    # Cache built at threshold 0.0: gt row 2 / ext row 2's candidate edge
    # scores 0.2, below the 0.5 fuzzy_threshold above -- excluded once
    # load_match_cache(..., fuzzy_threshold=0.5) filters it.
    cache_path = extraction_dir / "match_cache.pkl"
    edges = [(0, 0), (1, 1), (2, 2)]
    edge_weights = [1.0, 1.0, 0.2]
    _write_match_cache(cache_path, edges, edge_weights)
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

    return extraction_id, combine_id, judge_ids


def test_compute_metrics_for_id_with_validity(e2e_fixture):
    extraction_id, combine_id, judge_ids = e2e_fixture
    row = rv.compute_metrics_for_id(extraction_id, n_resamples=200, seed=0)

    assert row["n_gt"] == 3
    assert row["n_ext"] == 3
    # gt rows 0,1 recovered (weight 1.0 > 0.5); row 2 not (weight 0.2).
    assert row["recovery"] == pytest.approx(2 / 3)
    assert row["judge_combine_id"] == combine_id
    assert row["judge_ids"] == ";".join(judge_ids)
    # matched = [True, True, False]; judged = [True, False, True] -> OR = [True, True, True].
    assert row["validity"] == pytest.approx(1.0)
    assert row["validity_ci_lo"] == pytest.approx(1.0)
    assert row["validity_ci_hi"] == pytest.approx(1.0)
    assert 0.0 <= row["recovery_ci_lo"] <= row["recovery"] <= row["recovery_ci_hi"] <= 1.0


def test_compute_metrics_for_id_skip_validity_never_touches_judge_combine(e2e_fixture, monkeypatch):
    extraction_id, combine_id, _judge_ids = e2e_fixture

    def _boom(*args, **kwargs):
        raise AssertionError("find_judge_combine_id must not be called when compute_validity=False")

    monkeypatch.setattr(rv, "find_judge_combine_id", _boom)

    row = rv.compute_metrics_for_id(extraction_id, n_resamples=200, seed=0, compute_validity=False)

    assert row["recovery"] == pytest.approx(2 / 3)
    assert row["judge_combine_id"] is None
    assert row["judge_ids"] is None
    assert row["validity"] is None
    assert row["validity_ci_lo"] is None
    assert row["validity_ci_hi"] is None
