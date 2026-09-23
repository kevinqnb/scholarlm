"""Unit tests for analysis/match_cache.py's get_matching_config -- the
bridge from a DatasetConfig's strict_matching/fuzzy_matching/fuzzy_threshold/
numeric_coerce fields into the dict shape build_match_cache and
analysis/recovery_validity.py use.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
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
