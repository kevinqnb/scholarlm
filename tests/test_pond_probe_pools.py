"""Model-free unit tests for the pond fabricated-name pools (ladder rung 1).

Guards the 2026-09-07 fix: axis-2 ``pos_entity`` swaps must preserve the source
ecosystem type, and ~11% of pond GT valids are wetlands, for which the original
``_MADE_UP_NAMES`` list carried no type match (every wetland source fell back to
a pond/lake/pool name). ``_WETLAND_NAMES`` closes that gap and must stay OUT of
``_MADE_UP_NAMES`` so the default (non-augment) path stays byte-identical.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "data" / "pond" / "create_probe_dataset.py"

_spec = importlib.util.spec_from_file_location("pond_create_probe_dataset", _SCRIPT)
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)


def test_wetland_names_are_separate_from_made_up_names():
    # Appending to _MADE_UP_NAMES would shift create_noise_record's rng.choice
    # and break the byte-identical baseline gate -- the pools must be disjoint.
    assert set(cpd._WETLAND_NAMES).isdisjoint(cpd._MADE_UP_NAMES)
    assert len(cpd._WETLAND_NAMES) >= 5


def test_fabricated_names_by_type_has_a_populated_wetland_bucket():
    by_type = cpd._fabricated_names_by_type()
    assert by_type.get("wetland"), "wetland source rows would fall back to the flat pool"
    # and the other buckets are still populated
    for tok in ("pond", "lake", "pool", "reservoir"):
        assert by_type.get(tok)


def test_every_bucketed_name_suffix_matches_its_token():
    by_type = cpd._fabricated_names_by_type()
    for tok, names in by_type.items():
        for n in names:
            suffix = n.rsplit(" ", 1)[-1].lower()
            assert cpd._NAME_SUFFIX_TO_TYPE[suffix] == tok, (n, tok)


def test_wetland_bucket_covers_the_curated_wetland_names():
    by_type = cpd._fabricated_names_by_type()
    assert set(cpd._WETLAND_NAMES) <= set(by_type["wetland"])


def test_pond_type_token_prioritises_wetland_in_messy_ecosystem_strings():
    for eco in ("wetland", "Wetland", "wetland vs. lake", "wetland/lake", "wetland; pond"):
        assert cpd._pond_type_token({"ecosystem": eco}) == "wetland"
    # non-wetland strings still resolve as before
    assert cpd._pond_type_token({"ecosystem": "shallow pond"}) == "pond"
    assert cpd._pond_type_token({"ecosystem": "reservoir"}) == "reservoir"
    assert cpd._pond_type_token({"ecosystem": ""}) is None
