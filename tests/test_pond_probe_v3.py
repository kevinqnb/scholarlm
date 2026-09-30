"""Rung-1 (model-free) tests for the pond v3 probe-dataset changes: judge-visible
column set, extraction-vocabulary ecosystem, unit equivalence, and the
GT-record preparation (name casing, value spans, non-unit strings).

Fixtures are hand-built; expected values are readable off them. The two tests
that read the committed GT file check the maps *cover* it -- a GT value nobody
mapped must fail here, not at generation time.
"""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "src"))
_SCRIPT = _REPO_ROOT / "data" / "pond" / "create_probe_dataset.py"
_spec = importlib.util.spec_from_file_location("pond_create_probe_dataset_v3", _SCRIPT)
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)

from scholarlm.utils import probe_augment as pa

_GT_REVIEW = _REPO_ROOT / "data" / "pond" / "ground_truth_review.json"


# ─── columns ─────────────────────────────────────────────────────────────────


def test_augment_columns_are_exactly_the_judge_visible_fields():
    judge_filter = set(cpd.CONFIG.judge_filter_fields)
    entity = set(cpd.CONFIG.entity_schema.model_fields) - judge_filter
    event = set(cpd.CONFIG.measurement_event_schema.model_fields) - judge_filter
    judge_visible = entity | event | {"attribute", "value", "units"}
    assert set(cpd._AUGMENT_GT_COLS) == judge_visible | {"document_id"}
    for hidden in ("location", "identifiers", "additional_details", "page_number"):
        assert hidden not in cpd._AUGMENT_GT_COLS


def test_rules_use_the_augment_columns_and_the_unit_hook():
    rules = cpd._build_augment_rules([])
    assert rules.gt_cols == cpd._AUGMENT_GT_COLS
    assert rules.units_equivalent is cpd.units_equivalent


# ─── ecosystem -> extraction vocabulary ──────────────────────────────────────


def test_ecosystem_vocab_is_the_extraction_prompt_vocab():
    assert cpd.ECOSYSTEM_VOCAB == ("pond", "lake", "wetland", "other")
    for w in cpd.ECOSYSTEM_VOCAB:
        assert f'"{w}"' in cpd.CONFIG.direct_extraction_prompt


@pytest.mark.parametrize("raw, expect", [
    ("pond", "pond"), ("karst ponds", "pond"), ("shallow lakes", "lake"),
    ("mediterranean wetland", "wetland"), ("wetland vs. lake", "other"),
    ("pothole", "other"), ("lake/pond", "other"), ("small acidic bog lake", "lake"),
])
def test_map_ecosystem_known_cases(raw, expect):
    assert cpd.map_ecosystem(raw) == expect


def test_map_ecosystem_raises_on_unmapped_and_none():
    with pytest.raises(KeyError):
        cpd.map_ecosystem("swamp forest")
    with pytest.raises(KeyError):
        cpd.map_ecosystem(None)


def test_every_gt_ecosystem_is_mapped():
    gt = json.load(open(_GT_REVIEW))
    unmapped = {r["ecosystem"] for r in gt} - set(cpd._ECOSYSTEM_MAP)
    assert not unmapped, unmapped


def test_two_canonical_type_strings_map_to_other():
    # the stated rule: a string offering alternative canonical types -> "other"
    for raw in ("wetland vs. lake", "wetland/lake", "wetland; pond", "lake/pond",
                "pond or small lake", "temporary lake/seasonal pond"):
        assert cpd._ECOSYSTEM_MAP[raw] == "other", raw


# ─── unit equivalence ────────────────────────────────────────────────────────


@pytest.mark.parametrize("a, b", [
    ("µg/L", "ppb"), ("mg/L", "ppm"), ("µg/L", "mg/m^3"), ("mg/m^3", "ppb"),
    ("ha", "x10^-2 km^2"), ("m^2", "x10^-6 km^2"), ("ft", "feet"),
    ("µg/L", "μg/L"),                                   # U+00B5 vs U+03BC micro signs
    ("MG/L", "mg/l"),
])
def test_equivalent_units(a, b):
    assert cpd.units_equivalent("tp", a, b) is True


@pytest.mark.parametrize("a, b", [
    ("µg/L", "mg/L"), ("ppb", "ppm"), ("mg/L", "mg/m^3"), ("ha", "km^2"),
    ("m^2", "x10^-2 km^2"), ("percent", "fraction"), ("m", "km"), ("cm", "m"),
    ("µg/L", "μmol/L"), ("µg/L", "µg/cm^2"), ("ha", "acres"),
])
def test_non_equivalent_units(a, b):
    assert cpd.units_equivalent("tp", a, b) is False


def test_unit_table_covers_every_catalogue_unit():
    for attr, info in cpd._ATTR_DICT.items():
        for u in info["units"]:
            assert cpd._unit_key(u) in cpd._UNIT_CLASS, (attr, u)


def test_unknown_unit_raises():
    with pytest.raises(pa.UnknownUnit):
        cpd.units_equivalent("surface_area", "4.00", "ha")


def test_no_bad_units_negative_is_ever_equivalent_to_its_source():
    """Every (attribute, source unit) in the catalogue x many seeds: whatever
    make_typed_negative returns is a genuinely different unit."""
    rules = cpd._build_augment_rules([])
    for attr, info in cpd._ATTR_DICT.items():
        for src_u in info["units"]:
            src = {"document_id": "D", "name": "X Pond", "ecosystem": "pond",
                   "attribute": attr, "value": "1.0", "units": src_u, "date": None,
                   "gt_row_index": 0, "source_group_id": 0,
                   pa._CTX_ORIG_KEY: "paper text with no units at all"}
            for seed in range(40):
                neg = pa.make_typed_negative(src, "units", rules, random.Random(seed))
                if neg is None:
                    continue
                assert not cpd.units_equivalent(attr, src_u, neg["units"]), (attr, src_u, neg["units"])


# ─── prepare_augment_records ─────────────────────────────────────────────────


def _rec(i, code, name, eco, value, units, attr="tp"):
    return {"document_id": code, "_paper_code": code, "name": name, "ecosystem": eco,
            "attribute": attr, "value": str(value), "units": units, "date": None,
            "gt_row_index": i, "_orig_idx": i}


def test_prepare_augment_records(tmp_path):
    (tmp_path / "A.txt").write_text(
        "Big Sky Lake is deep. Big Sky Lake again. big sky lake once.\n"
        "<td>15.3 ± 6.1</td> total phosphorus of Big Sky Lake\n"
        "Site Pool 1 had 4.5 and the abstract says 4.5 m.\n")
    (tmp_path / "B.txt").write_text("Nothing about the named site. TP was 9.0\n")
    recs = [
        _rec(0, "A", "big sky lake", "shallow lakes", 15.3, "µg/L"),   # casing + span
        _rec(1, "A", "pool1", "pond", 4.5, "m", attr="max_depth"),     # flexible name; 4.5 bare elsewhere
        _rec(2, "B", "site x9", "pothole", 9.0, "4.00"),               # name absent; non-unit units
        {**_rec(3, "B", None, "wetland vs. lake", 9.0, "mg/L")},       # null name stays null
    ]
    rep = cpd.prepare_augment_records(recs, tmp_path)
    assert [r["name"] for r in recs] == ["Big Sky Lake", "Pool 1", "Site X9", None]
    assert [r["ecosystem"] for r in recs] == ["lake", "pond", "other", "other"]
    assert recs[0][pa._VALUE_SPAN_KEY] == "15.3 ± 6.1"
    assert pa._VALUE_SPAN_KEY not in recs[1]          # 4.5 appears bare -> ambiguous
    assert pa._VALUE_SPAN_KEY not in recs[2]
    assert recs[0]["value"] == "15.3"                 # value itself untouched here
    assert rep["name_how"] == {"verbatim": 1, "flexible": 1, "fallback_title": 1, "null": 1}
    assert rep["value_span_rows"] == [0]
    assert rep["unknown_unit_rows"] == [2]


def test_prepare_augment_records_fails_on_unmapped_ecosystem(tmp_path):
    (tmp_path / "A.txt").write_text("text")
    with pytest.raises(KeyError):
        cpd.prepare_augment_records([_rec(0, "A", "x", "swamp forest", 1, "m")], tmp_path)


def test_prepare_augment_records_fails_on_empty_ocr_file(tmp_path):
    (tmp_path / "A.txt").write_text("   \n")
    with pytest.raises(AssertionError, match="empty"):
        cpd.prepare_augment_records([_rec(0, "A", "x", "pond", 1, "m")], tmp_path)


def test_prepare_augment_records_missing_paper_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        cpd.prepare_augment_records([_rec(0, "MISSING", "x", "pond", 1, "m")], tmp_path)


# ─── CLI: --ocr-dir is required, no default ──────────────────────────────────


def test_augment_requires_ocr_dir():
    with pytest.raises(SystemExit):
        cpd.main(["--augment", "--augment-stub", "--reviewed"])


# ─── ecosystem mapping must not break fabricated-name TYPE preservation ──────


def test_wetland_named_source_still_gets_wetland_type_names_after_mapping(tmp_path):
    """Pipeline order: prepare_augment_records maps ecosystem -> "other", but the
    rules' type token must still see the raw "wetland vs. lake" -> wetland."""
    (tmp_path / "A.txt").write_text("Big Sky Marsh is shallow. Big Sky Marsh again.")
    rec = _rec(0, "A", "big sky marsh", "wetland vs. lake", 1.0, "m", attr="max_depth")
    cpd.prepare_augment_records([rec], tmp_path)
    assert rec["ecosystem"] == "other"                     # what the judge sees
    rules = cpd._build_augment_rules([rec])
    assert rules.entity_type_token(rec) == "wetland"       # what name selection sees
    cands = rules.entity_candidates(rec)
    assert cands and set(cands) <= set(cpd._fabricated_names_by_type()["wetland"])
    assert rules.entity_type_ok(rec, "Sedgemoor Marsh") is True
    assert rules.entity_type_ok(rec, "Bluebell Pond") is False   # would be a type-changing rename


def test_type_token_reads_raw_ecosystem_for_every_mapped_group():
    for raw, mapped in cpd._ECOSYSTEM_MAP.items():
        rec = {"ecosystem": mapped, cpd._ECO_RAW_KEY: raw}
        assert cpd._augment_type_token(rec) == cpd._pond_type_token({"ecosystem": raw}), raw


def test_pond_rules_turn_the_numeric_guard_on():
    assert cpd._build_augment_rules([]).numeric_value_guard is True
