"""Model-free unit tests for the supermat augmentation rules (rung 1).

Guards the supermat-specific choices: pos_entity swaps the material formula in
``name`` (not ``sample_details``) and nulls the stale ``identifiers``
abbreviation catalogue; there is no attribute error type (single measurand);
pos_event uses ``pressure`` and never injects one (unstated pressure means
"ambient").
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "data" / "supermat" / "create_probe_dataset.py"

_spec = importlib.util.spec_from_file_location("supermat_create_probe_dataset", _SCRIPT)
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)

from scholarlm.utils import probe_augment as pa  # noqa: E402


_GT = [
    {"document_id": "D1", "name": "YBa2Cu3O7", "attribute": "tc", "value": "92",
     "units": "K", "pressure": None},
    {"document_id": "D1", "name": "MgB2", "attribute": "tc", "value": "39",
     "units": "K", "pressure": "2 GPa"},
]


def _rules():
    return cpd._build_augment_rules(_GT)


def test_augment_rules_target_name_not_sample_details():
    rules = _rules()
    assert rules.entity_name_field == "name"
    assert rules.entity_noun == "material formula"
    assert rules.fabricated_names_any == list(cpd._MADE_UP_NAMES)
    assert rules.entity_swap_clear_fields == ("identifiers",)
    assert rules.name_suffix_to_type == {}
    assert not hasattr(cpd, "_SUPERMAT_SAMPLE_DETAILS")


def test_no_attribute_error_type_and_pressure_event():
    rules = _rules()
    assert len(rules.attribute_pool) < 2            # single measurand (tc)
    assert rules.event_field == "pressure"
    assert rules.event_allow_inject is False
    assert "2 GPa" in rules.event_pool and "ambient" in rules.event_pool


def test_preserve_clause_is_supermat_specific():
    clause = _rules().entity_preserve_clause.lower()
    assert "ecosystem" not in clause and "temperature" in clause


def test_fabricated_name_draws_from_made_up_formulas_and_avoids_the_source():
    rules = _rules()
    rng = random.Random(0)
    for src_name in ("YBa2Cu3O7", cpd._MADE_UP_NAMES[0], "MgB2"):
        got = {rules.fabricated_name({"name": src_name}, rng) for _ in range(50)}
        assert got <= set(cpd._MADE_UP_NAMES) and src_name not in got


def _stub_src(name="YBa2Cu3O7", identifiers="YBCO; Y-123"):
    rules = _rules()
    src = pa._prep_base_valid(
        {
            "document_id": "D1", "name": name, "identifiers": identifiers,
            "sample_details": None, "pressure": None, "attribute": "tc",
            "value": "92", "units": "K", "page_number": 1, "gt_row_index": 3,
            "donor_gt_row_index": None, "_orig_idx": 3, "_paper_code": "D1",
            "_page_numbers": [1],
            pa._CTX_ORIG_KEY: f'The compound "{name}" (also {identifiers}) has a '
                              f'critical temperature of "92" K.',
        },
        rules,
    )
    return rules, src


def test_pos_entity_swaps_name_edits_page_and_nulls_identifiers():
    rules, src = _stub_src()
    row = pa.make_axis2_positive(src, "pos_entity", rules, pa.StubAugmentClient(),
                                 random.Random(0))
    assert row is not None and row["augment_axis"] == "pos_entity"
    assert row["name"] in cpd._MADE_UP_NAMES
    assert row["identifiers"] is None
    assert row["name"] in row[pa._CTX_EDIT_KEY] and "YBa2Cu3O7" not in row[pa._CTX_EDIT_KEY]
    assert row["value"] == "92" and row["units"] == "K"


def test_typed_negative_entity_fires_on_name_and_nulls_identifiers():
    rules, src = _stub_src(identifiers="YBCO; Y-123")
    neg = pa.make_typed_negative(src, "entity", rules, random.Random(0))
    assert neg is not None and neg["modification_type"] == "bad_entity"
    assert neg["name"] in cpd._MADE_UP_NAMES and neg["identifiers"] is None


def test_typed_negative_attribute_is_none_for_supermat():
    rules, src = _stub_src()
    assert pa.make_typed_negative(src, "attribute", rules, random.Random(0)) is None


def test_matched_negative_on_synthetic_positive_carries_edited_page():
    rules, src = _stub_src()
    pos = pa.make_axis2_positive(src, "pos_entity", rules, pa.StubAugmentClient(),
                                 random.Random(0))
    neg = pa.make_typed_negative(pos, "value", rules, random.Random(1))
    assert neg[pa._CTX_EDIT_KEY] == pos[pa._CTX_EDIT_KEY]
    assert neg["identifiers"] is None
