"""Model-free unit tests for the supermat axis-2 ``pos_entity`` rules (rung 1).

Guards the change from editing ``sample_details`` to swapping the material
formula in ``name``: the augmenter now renames the compound (from
``_MADE_UP_NAMES``, the fake-formula pool the default path already uses) and
rewrites the page to match, nulling the stale ``identifiers`` abbreviation
catalogue on the positive row.
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "data" / "supermat" / "create_probe_dataset.py"

_spec = importlib.util.spec_from_file_location("supermat_create_probe_dataset", _SCRIPT)
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)

from scholarlm.utils import probe_augment as pa  # noqa: E402


def test_augment_rules_swap_the_formula_not_sample_details():
    rules = cpd._build_augment_rules()
    assert rules.entity_name_field == "name"
    assert rules.fabricated_names_any == list(cpd._MADE_UP_NAMES)
    assert rules.entity_swap_clear_fields == ("identifiers",)
    assert not rules.entity_field_locked
    # the old sample_details pool is gone
    assert not hasattr(cpd, "_SUPERMAT_SAMPLE_DETAILS")


def test_preserve_clause_is_supermat_specific():
    clause = cpd._build_augment_rules().entity_swap_preserve_clause
    assert "ecosystem" not in clause.lower()
    assert "temperature" in clause.lower()


def test_fabricated_name_draws_from_made_up_formulas_and_avoids_the_source():
    rules = cpd._build_augment_rules()
    rng = random.Random(0)
    for src_name in ("YBa2Cu3O7", cpd._MADE_UP_NAMES[0], "MgB2"):
        got = {rules.fabricated_name({"name": src_name}, rng) for _ in range(50)}
        assert got <= set(cpd._MADE_UP_NAMES)
        assert src_name not in got


def _stub_src(name="YBa2Cu3O7", identifiers="YBCO; Y-123"):
    rules = cpd._build_augment_rules()
    src = pa._prep_base_valid(
        {
            "document_id": "D1", "name": name, "identifiers": identifiers,
            "sample_details": None, "attribute": "tc", "value": "92", "units": "K",
            "page_number": 1, "gt_row_index": 3, "donor_gt_row_index": None,
            "_orig_idx": 3, "_paper_code": "D1", "_page_numbers": [1],
            pa._CTX_ORIG_KEY: f"The compound {name} (also {identifiers}) has a "
                              f"critical temperature of 92 K.",
        },
        rules,
    )
    return rules, src


def test_pos_entity_swaps_name_edits_page_and_nulls_identifiers():
    rules, src = _stub_src()
    row = pa.make_axis2_positive(src, "pos_entity", rules, pa.StubAugmentClient(),
                                 random.Random(0))
    assert row is not None
    assert row["label"] == "valid" and row["augment_axis"] == "pos_entity"
    assert row["name"] in cpd._MADE_UP_NAMES
    assert row["identifiers"] is None                     # stale catalogue cleared
    assert row["name"] in row[pa._CTX_EDIT_KEY]           # page rewritten
    assert "YBa2Cu3O7" not in row[pa._CTX_EDIT_KEY]
    assert row["value"] == "92" and row["units"] == "K"   # measurement preserved


def test_matched_hard_negative_carries_the_cleared_identifiers():
    rules, src = _stub_src()
    pos = pa.make_axis2_positive(src, "pos_entity", rules, pa.StubAugmentClient(),
                                 random.Random(0))
    neg = pa.make_hard_negative(pos, "hard_value", rules, random.Random(1),
                                on_edited_context=True)
    assert neg["label"] == "invalid"
    assert neg["identifiers"] is None
    assert neg[pa._CTX_EDIT_KEY] == pos[pa._CTX_EDIT_KEY]


def test_hard_entity_now_fires_on_supermat_names():
    # Previously entity_name_field was sample_details (null ~82% of rows) so
    # hard_entity mostly skipped; on `name` it fires.
    rules, src = _stub_src(identifiers=None)
    neg = pa.make_hard_negative(src, "hard_entity", rules, random.Random(0))
    assert neg is not None
    assert neg["name"] in cpd._MADE_UP_NAMES
    assert neg["label"] == "invalid" and neg["modification_type"] == "hard_entity"


def test_hard_entity_nulls_stale_identifiers_on_gt_context():
    # The GT page is unedited and still says "YBCO"; a fabricated `name` with the
    # real abbreviation catalogue left in place would read as a valid claim on an
    # invalid-labelled row. entity_swap_clear_fields must reach this branch too.
    rules, src = _stub_src(identifiers="YBCO; Y-123")
    neg = pa.make_hard_negative(src, "hard_entity", rules, random.Random(0))
    assert neg is not None
    assert neg["identifiers"] is None
