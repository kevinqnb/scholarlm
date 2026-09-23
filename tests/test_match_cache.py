"""Unit tests for analysis/match_cache.py: get_matching_config (the bridge
from a DatasetConfig's strict_matching/fuzzy_matching/fuzzy_threshold/
numeric_coerce fields into the dict shape build_match_cache and
analysis/recovery_validity.py use) and main()'s --config/positional-ids
argument resolution (build_match_cache itself is monkeypatched out --
these are rung-1 tests of the CLI plumbing, not a match_cache.pkl build).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

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


def test_main_positional_ids_call_build_match_cache(monkeypatch):
    seen = []
    monkeypatch.setattr(match_cache, "build_match_cache", lambda eid: seen.append(eid))
    monkeypatch.setattr(sys, "argv", ["match_cache.py", "id-a", "id-b"])
    match_cache.main()
    assert seen == ["id-a", "id-b"]


def test_main_config_reads_experiment_ids_and_calls_build_match_cache(tmp_path, monkeypatch):
    config_path = tmp_path / "2026-09-23-test-analysis-01.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(
            {
                "id": "2026-09-23-test-analysis-01",
                "project": "scholarlm",
                "description": "test",
                "seed": 342,
                "params": {"experiment_ids": ["id-a", "id-b"]},
            },
            f,
        )

    seen = []
    monkeypatch.setattr(match_cache, "build_match_cache", lambda eid: seen.append(eid))
    monkeypatch.setattr(sys, "argv", ["match_cache.py", "--config", str(config_path)])
    match_cache.main()
    assert seen == ["id-a", "id-b"]
