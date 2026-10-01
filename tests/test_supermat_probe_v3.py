"""Rung-1 (model-free) tests for the supermat v3 probe-dataset changes: judge-visible
columns, K/mK unit table, prepare checks, and the required --ocr-dir / qualifier
GT-file guards."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
_spec = importlib.util.spec_from_file_location(
    "supermat_create_probe_dataset_v3", _REPO_ROOT / "data" / "supermat" / "create_probe_dataset.py")
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)

from scholarlm.utils import probe_augment as pa

_QGT = _REPO_ROOT / "data" / "supermat" / "ground_truth_qualifiers.json"


def test_augment_columns_are_exactly_the_judge_visible_fields():
    filt = set(cpd.CONFIG.judge_filter_fields)
    entity = set(cpd.CONFIG.entity_schema.model_fields) - filt
    event = set(cpd.CONFIG.measurement_event_schema.model_fields) - filt
    assert set(cpd._AUGMENT_GT_COLS) == entity | event | {"attribute", "value", "units", "document_id"}
    assert cpd._AUGMENT_GT_COLS == ["document_id", "name", "attribute", "value", "units"]


def test_config_ground_truth_is_the_qualifier_file():
    assert cpd.CONFIG.ground_truth_file == "data/supermat/ground_truth_qualifiers.json"


def test_units_equivalence_k_and_mk():
    assert cpd.units_equivalent("tc", "K", "K") is True
    assert cpd.units_equivalent("tc", "K", "mK") is False
    assert cpd.units_equivalent("tc", "mk", "MK") is True
    with pytest.raises(pa.UnknownUnit):
        cpd.units_equivalent("tc", "K/GPa", "K")


def test_every_gt_and_catalogue_unit_is_known():
    for r in json.load(open(_QGT)):
        assert r["units"].strip().lower() in cpd._UNIT_SCALE, r["units"]
    for info in cpd._ATTR_DICT.values():
        for u in info["units"]:
            assert u.strip().lower() in cpd._UNIT_SCALE


def test_bad_units_for_k_is_always_mk_and_never_equivalent():
    rules = cpd._build_augment_rules([])
    src = {"document_id": "D", "name": "MgB2", "attribute": "tc", "value": "39", "units": "K",
           "gt_row_index": 0, "source_group_id": 0, pa._CTX_ORIG_KEY: "paper"}
    got = {pa.make_typed_negative(src, "units", rules, random.Random(s))["units"] for s in range(50)}
    assert got == {"mK"}


def _rec(i, code, value, units="K"):
    return {"document_id": code, "_paper_code": code, "name": "MgB2", "attribute": "tc",
            "value": value, "units": units, "gt_row_index": i, "_orig_idx": i}


def test_prepare_augment_records_reports_and_keeps_values_verbatim(tmp_path):
    (tmp_path / "A.txt").write_text("text")
    recs = [_rec(0, "A", "39"), _rec(1, "A", "7-8"), _rec(2, "A", "~1.3"), _rec(3, "A", "5", units="K/GPa")]
    rep = cpd.prepare_augment_records(recs, tmp_path)
    assert rep == {"unknown_unit_rows": [3], "n_non_plain_value": 2}
    assert [r["value"] for r in recs] == ["39", "7-8", "~1.3", "5"]


def test_prepare_fails_on_missing_or_empty_ocr(tmp_path):
    with pytest.raises(FileNotFoundError):
        cpd.prepare_augment_records([_rec(0, "MISSING", "1")], tmp_path)
    (tmp_path / "E.txt").write_text(" ")
    with pytest.raises(AssertionError, match="empty"):
        cpd.prepare_augment_records([_rec(0, "E", "1")], tmp_path)


def test_qualifier_string_values_survive_into_base_valids():
    """A range value from the qualifier GT is carried verbatim by _prep_base_valid
    (no span substitution, no float coercion) and pos_value skips it cleanly."""
    rules = cpd._build_augment_rules([_rec(0, "A", "7-8")])
    src = {**_rec(0, "A", "7-8"), "gt_row_index": 0,
           pa._CTX_ORIG_KEY: 'MgB2 has a Tc of "7-8" K.', "donor_gt_row_index": None}
    row = pa._prep_base_valid(src, rules, use_value_spans=True)
    assert row["value"] == "7-8"
    assert pa.make_axis2_positive(row, "pos_value", rules, pa.StubAugmentClient(),
                                  random.Random(0)) is None


def test_augment_requires_ocr_dir():
    with pytest.raises(SystemExit):
        cpd.main(["--augment", "--augment-stub", "--qualifiers"])


def test_augment_requires_the_qualifier_ground_truth():
    with pytest.raises(SystemExit):
        cpd.main(["--augment", "--augment-stub", "--ocr-dir", "x"])          # would read the float-only GT


def test_reviewed_and_qualifiers_conflict():
    with pytest.raises(SystemExit):
        cpd.main(["--reviewed", "--qualifiers"])
