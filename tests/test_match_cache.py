"""Unit tests for analysis/match_cache.py: get_matching_config (the bridge
from a DatasetConfig's strict_matching/fuzzy_matching/fuzzy_threshold/
numeric_coerce fields into the dict shape build_match_cache and
analysis/recovery_validity.py use), main()'s --config/positional-ids
argument resolution (build_match_cache itself is monkeypatched out for those
-- rung-1 tests of the CLI plumbing, not a match_cache.pkl build), the
repo_relative/sha256_file sidecar helpers, and a tiny end-to-end
build_match_cache fixture covering the match_cache.meta.json sidecar it
writes and the document_id-overlap guard.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments"))
sys.path.insert(0, str(_REPO))

import utils as paths  # noqa: E402
from scholarlm.config import DatasetConfig  # noqa: E402
from analysis import match_cache  # noqa: E402


class _Entity(BaseModel):
    name: str


def _minimal_dataset_config(**overrides) -> DatasetConfig:
    kwargs = dict(
        name="testset",
        data_dir="data/testset",
        metadata_file="data/testset/directory.json",
        entity_schema=_Entity,
        entity_identification_prompt="prompt",
        entity_type_description="a thing",
        attribute_info_dict={},
    )
    kwargs.update(overrides)
    return DatasetConfig(**kwargs)


def test_get_matching_config_happy_path():
    cfg = _minimal_dataset_config(
        strict_matching={"document_id": "document_id"},
        fuzzy_matching={"name": "name"},
        fuzzy_threshold=1 / 3,
        numeric_coerce=["point_value"],
    )
    resolved = match_cache.get_matching_config(cfg)
    assert resolved == {
        "strict": {"document_id": "document_id"},
        "fuzzy": {"name": "name"},
        "fuzzy_threshold": 1 / 3,
        "numeric_coerce": ["point_value"],
    }


def test_get_matching_config_numeric_coerce_defaults_to_empty_list():
    cfg = _minimal_dataset_config(
        strict_matching={"document_id": "document_id"},
        fuzzy_matching={"name": "name"},
        fuzzy_threshold=0.5,
    )
    assert match_cache.get_matching_config(cfg)["numeric_coerce"] == []


def test_get_matching_config_raises_when_unset():
    cfg = _minimal_dataset_config()  # no matching fields set
    with pytest.raises(KeyError, match="strict_matching"):
        match_cache.get_matching_config(cfg)


def test_get_matching_config_raises_when_partially_set():
    cfg = _minimal_dataset_config(strict_matching={"document_id": "document_id"})
    with pytest.raises(KeyError, match="fuzzy_matching"):
        match_cache.get_matching_config(cfg)


# ---------------------------------------------------------------------------
# main(): --config vs. positional experiment_ids, mutually exclusive
# ---------------------------------------------------------------------------


def test_main_rejects_neither_ids_nor_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["match_cache.py"])
    with pytest.raises(SystemExit):
        match_cache.main()


def test_main_rejects_both_ids_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["match_cache.py", "some-id", "--config", str(tmp_path / "x.yaml")]
    )
    with pytest.raises(SystemExit):
        match_cache.main()


def test_main_rejects_ids_without_ground_truth_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["match_cache.py", "id-a"])
    with pytest.raises(SystemExit):
        match_cache.main()


def test_main_rejects_config_plus_ground_truth_file(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv",
        ["match_cache.py", "--config", str(tmp_path / "x.yaml"), "--ground-truth-file", "data/pond/ground_truth_review.json"],
    )
    with pytest.raises(SystemExit):
        match_cache.main()


def test_main_positional_ids_call_build_match_cache(tmp_path, monkeypatch):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")

    seen = []
    monkeypatch.setattr(match_cache, "build_match_cache", lambda eid, gt: seen.append((eid, gt)))
    monkeypatch.setattr(sys, "argv", ["match_cache.py", "id-a", "id-b", "--ground-truth-file", str(gt_path)])
    match_cache.main()
    assert seen == [("id-a", gt_path), ("id-b", gt_path)]


def test_main_config_reads_experiment_ids_and_calls_build_match_cache(tmp_path, monkeypatch):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")

    config_path = tmp_path / "2026-09-23-test-analysis-01.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "id": "2026-09-23-test-analysis-01",
                "project": "scholarlm",
                "description": "test",
                "seed": 342,
                "params": {"experiment_ids": ["id-a", "id-b"], "ground_truth_file": str(gt_path)},
            },
            f,
        )

    seen = []
    monkeypatch.setattr(match_cache, "build_match_cache", lambda eid, gt: seen.append((eid, gt)))
    monkeypatch.setattr(sys, "argv", ["match_cache.py", "--config", str(config_path)])
    match_cache.main()
    assert seen == [("id-a", gt_path), ("id-b", gt_path)]


# ---------------------------------------------------------------------------
# repo_relative / sha256_file -- the match_cache.meta.json sidecar helpers
# ---------------------------------------------------------------------------


def test_repo_relative_under_repo_root():
    path = match_cache._REPO_ROOT / "data" / "pond" / "ground_truth_review.json"
    assert match_cache.repo_relative(path) == "data/pond/ground_truth_review.json"


def test_repo_relative_outside_repo_root_returns_plain_string(tmp_path):
    path = tmp_path / "ground_truth.json"
    assert match_cache.repo_relative(path) == str(path)


def test_sha256_file_matches_known_digest(tmp_path):
    import hashlib

    path = tmp_path / "gt.json"
    path.write_text("hello world")
    assert match_cache.sha256_file(path) == hashlib.sha256(b"hello world").hexdigest()


def test_sha256_file_differs_when_contents_differ(tmp_path):
    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    path_a.write_text("[1, 2, 3]")
    path_b.write_text("[1, 2, 4]")
    assert match_cache.sha256_file(path_a) != match_cache.sha256_file(path_b)


# ---------------------------------------------------------------------------
# build_match_cache -- tiny end-to-end fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def build_cache_fixture(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)

    dataset = "testset"
    experiment_id = "2026-01-01-testset-model-extraction-01"

    ext_rows = [{"document_id": "d1", "attribute": "ph"}, {"document_id": "d2", "attribute": "tn"}]
    extraction_dir = results_root / dataset / "extraction" / experiment_id
    extraction_dir.mkdir(parents=True)
    with open(extraction_dir / "final.json", "w") as f:
        json.dump(ext_rows, f)

    dataset_config = DatasetConfig(
        name=dataset,
        data_dir=f"data/{dataset}",
        metadata_file=f"data/{dataset}/directory.json",
        entity_schema=_Entity,
        entity_identification_prompt="prompt",
        entity_type_description="a thing",
        attribute_info_dict={},
        strict_matching={"document_id": "document_id", "attribute": "attribute"},
        fuzzy_matching={},
        fuzzy_threshold=0.0,
    )
    monkeypatch.setattr(match_cache, "load_dataset_config", lambda ds: dataset_config)

    return experiment_id, extraction_dir


def test_build_match_cache_writes_pkl_and_sidecar(tmp_path, build_cache_fixture):
    experiment_id, extraction_dir = build_cache_fixture
    gt_rows = [{"document_id": "d1", "attribute": "ph"}, {"document_id": "d2", "attribute": "tn"}]
    gt_path = tmp_path / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt_rows, f)

    cache_path = match_cache.build_match_cache(experiment_id, gt_path)

    assert cache_path == extraction_dir / "match_cache.pkl"
    assert cache_path.exists()

    meta_path = extraction_dir / "match_cache.meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    assert meta["ground_truth_file"] == match_cache.repo_relative(gt_path)
    assert meta["ground_truth_sha256"] == match_cache.sha256_file(gt_path)
    assert meta["n_gt"] == 2


def test_build_match_cache_raises_on_zero_document_id_overlap(tmp_path, build_cache_fixture):
    experiment_id, _extraction_dir = build_cache_fixture
    gt_rows = [{"document_id": "unrelated-paper", "attribute": "ph"}]
    gt_path = tmp_path / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump(gt_rows, f)

    with pytest.raises(ValueError, match="zero document_id"):
        match_cache.build_match_cache(experiment_id, gt_path)


def test_build_match_cache_rebuild_updates_sidecar_to_new_ground_truth(tmp_path, build_cache_fixture):
    # Rebuilding an existing cache against a DIFFERENT ground truth file must
    # leave the sidecar describing the new file, not a stale mix of the two.
    experiment_id, extraction_dir = build_cache_fixture
    gt_a_path = tmp_path / "gt_a.json"
    with open(gt_a_path, "w") as f:
        json.dump([{"document_id": "d1", "attribute": "ph"}, {"document_id": "d2", "attribute": "tn"}], f)
    match_cache.build_match_cache(experiment_id, gt_a_path)

    gt_b_path = tmp_path / "gt_b.json"
    with open(gt_b_path, "w") as f:
        json.dump([{"document_id": "d1", "attribute": "ph"}, {"document_id": "d2", "attribute": "tn"}, {"document_id": "d1", "attribute": "extra"}], f)
    match_cache.build_match_cache(experiment_id, gt_b_path)

    with open(extraction_dir / "match_cache.meta.json") as f:
        meta = json.load(f)
    assert meta["ground_truth_file"] == match_cache.repo_relative(gt_b_path)
    assert meta["ground_truth_sha256"] == match_cache.sha256_file(gt_b_path)
    assert meta["n_gt"] == 3


def test_build_match_cache_deletes_stale_sidecar_when_rebuild_fails(tmp_path, build_cache_fixture, monkeypatch):
    # If a rebuild fails after the stale sidecar is deleted but before the new
    # one is written, the old pkl must be left with NO sidecar -- not the
    # previous (now-wrong) one -- so a later read fails loud instead of
    # silently trusting metadata for a ground truth file that never actually
    # produced the pkl sitting next to it.
    experiment_id, extraction_dir = build_cache_fixture
    gt_a_path = tmp_path / "gt_a.json"
    with open(gt_a_path, "w") as f:
        json.dump([{"document_id": "d1", "attribute": "ph"}, {"document_id": "d2", "attribute": "tn"}], f)
    match_cache.build_match_cache(experiment_id, gt_a_path)
    assert (extraction_dir / "match_cache.meta.json").exists()

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure after sidecar unlink")

    monkeypatch.setattr(match_cache, "cached_match", _boom)
    gt_b_path = tmp_path / "gt_b.json"
    with open(gt_b_path, "w") as f:
        json.dump([{"document_id": "d1", "attribute": "ph"}], f)

    with pytest.raises(RuntimeError, match="simulated failure"):
        match_cache.build_match_cache(experiment_id, gt_b_path)

    assert not (extraction_dir / "match_cache.meta.json").exists()
