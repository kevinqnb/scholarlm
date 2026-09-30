"""Rung-1 (model-free) tests for the nfix v3 probe-dataset changes: judge-visible
column set, extraction-format fields (substrate "wc" -> "water column", ISO date
-> prompt format), dimensional unit equivalence, GT-record preparation, and the
required --ocr-dir / GT-file guards. Hand-built fixtures; the two GT-reading
tests check the maps cover the committed GT."""
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
    "nfix_create_probe_dataset_v3", _REPO_ROOT / "data" / "nfix" / "create_probe_dataset.py")
cpd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cpd)

from scholarlm.utils import probe_augment as pa

_GT = _REPO_ROOT / "data" / "nfix" / "ground_truth_review.json"


# ─── columns ─────────────────────────────────────────────────────────────────


def test_augment_columns_are_exactly_the_judge_visible_fields():
    filt = set(cpd.CONFIG.judge_filter_fields)
    entity = set(cpd.CONFIG.entity_schema.model_fields) - filt
    event = set(cpd.CONFIG.measurement_event_schema.model_fields) - filt
    assert set(cpd._AUGMENT_GT_COLS) == entity | event | {"attribute", "value", "units", "document_id"}
    for hidden in ("location", "nfix_method", "sample_depth", "identifiers",
                   "additional_details", "page_number"):
        assert hidden not in cpd._AUGMENT_GT_COLS


def test_rules_use_augment_columns_and_hooks():
    r = cpd._build_augment_rules([])
    assert r.gt_cols == cpd._AUGMENT_GT_COLS
    assert r.units_equivalent is cpd.units_equivalent and r.numeric_value_guard is True


# ─── substrate / date ────────────────────────────────────────────────────────


def test_substrate_wc_becomes_water_column():
    assert cpd.map_substrate("wc") == "water column"
    assert cpd.map_substrate("benthos") == "benthos" and cpd.map_substrate("other") == "other"
    for bad in ("sediment", None):
        with pytest.raises(KeyError):
            cpd.map_substrate(bad)


def test_every_gt_substrate_is_mapped():
    assert {r["substrate_type"] for r in json.load(open(_GT))} <= set(cpd._SUBSTRATE_MAP)


@pytest.mark.parametrize("raw, expect", [
    ("2003-10", "10-2003"), ("2003-10-05", "05-10-2003"), ("1998", "1998"), (None, None),
])
def test_to_prompt_date(raw, expect):
    assert cpd.to_prompt_date(raw) == expect


@pytest.mark.parametrize("bad", ["2003-13", "2003-10-32", "Oct 2003", "10-2003", "03"])
def test_to_prompt_date_rejects_other_shapes(bad):
    with pytest.raises(ValueError):
        cpd.to_prompt_date(bad)


def test_every_gt_date_converts():
    for r in json.load(open(_GT)):
        cpd.to_prompt_date(r["date"])


# ─── unit equivalence ────────────────────────────────────────────────────────


@pytest.mark.parametrize("a, b", [
    ("nmol C2H4 mL⁻¹ h⁻¹", "nmol C2H4 cm⁻³ h⁻¹"),          # 1 mL == 1 cm^3
    ("nmol N g⁻¹ h⁻¹", "µmol N kg⁻¹ h⁻¹"),                 # nmol/g == umol/kg
    ("mg N m⁻³ d⁻¹", "µg N L⁻¹ d⁻¹"),                      # mg/m^3 == ug/L
])
def test_truly_equivalent_units(a, b):
    assert cpd.units_equivalent("x", a, b) is True


@pytest.mark.parametrize("a, b", [
    ("nmol N L⁻¹ d⁻¹", "nmol N2 L⁻¹ d⁻¹"),                 # N vs N2: judge-ambiguous -> excluded
    ("mg N m⁻² h⁻¹", "mg N2 m⁻² h⁻¹"),
    ("µmol N g⁻¹ h⁻¹", "µmol N g⁻¹ h⁻¹"),                  # identical
    ("µmol N g⁻¹ h⁻¹", "μmol N g⁻¹ h⁻¹"),                  # U+00B5 vs U+03BC
])
def test_ambiguous_or_identical_counts_as_equivalent(a, b):
    assert cpd.units_equivalent("x", a, b) is True


@pytest.mark.parametrize("a, b", [
    ("nmol N L⁻¹ d⁻¹", "nmol N L⁻¹ h⁻¹"),                  # per day vs per hour
    ("nmol N g⁻¹ h⁻¹", "nmol C2H4 g⁻¹ h⁻¹"),               # N2 fixation vs ethylene proxy
    ("nmol N g⁻¹ h⁻¹", "µg N g⁻¹ h⁻¹"),                    # mol vs mass
    ("µmol N m⁻² d⁻¹", "µmol N m⁻³ d⁻¹"),                  # area vs volume
    ("nmol C2H4 cm⁻² h⁻¹", "nmol C2H4 m⁻² h⁻¹"),           # 1e4 apart
    ("g N m⁻² yr⁻¹", "mg N m⁻² d⁻¹"),
    ("nmol N L⁻¹ d⁻¹", "nmol N cm⁻³ d⁻¹"),                 # L vs cm^3: 1000x apart
])
def test_non_equivalent_units(a, b):
    assert cpd.units_equivalent("x", a, b) is False


def test_unparseable_unit_raises():
    for bad in ("4.00", "µmol m⁻² h⁻¹", "nmol N per g"):
        with pytest.raises(pa.UnknownUnit):
            cpd.units_equivalent("x", bad, "nmol N g⁻¹ h⁻¹")


def test_every_catalogue_and_gt_unit_parses():
    for info in cpd._ATTR_DICT.values():
        for u in info["units"]:
            cpd._parse_unit(u)
    for r in json.load(open(_GT)):
        cpd._parse_unit(r["units"])


def test_no_bad_units_negative_is_ever_equivalent_to_its_source():
    rules = cpd._build_augment_rules([])
    for attr, info in cpd._ATTR_DICT.items():
        for src_u in info["units"]:
            src = {"document_id": "D", "name": "X", "attribute": attr, "value": "1.0",
                   "units": src_u, "date": None, "gt_row_index": 0, "source_group_id": 0,
                   pa._CTX_ORIG_KEY: "paper text with no units at all"}
            for seed in range(40):
                neg = pa.make_typed_negative(src, "units", rules, random.Random(seed))
                if neg is not None:
                    assert not cpd.units_equivalent(attr, src_u, neg["units"]), (src_u, neg["units"])


# ─── prepare_augment_records ─────────────────────────────────────────────────


def _rec(i, code, value, *, attr="nfix_rate_volumetric", units="nmol N L⁻¹ d⁻¹",
         sub="wc", date="2003-10"):
    return {"document_id": code, "_paper_code": code, "name": "Site A", "ecosystem_type": "lakes",
            "date": date, "substrate_type": sub, "attribute": attr, "_gt_attribute": attr,
            "value": str(value), "units": units, "gt_row_index": i, "_orig_idx": i}


def test_prepare_augment_records(tmp_path):
    (tmp_path / "A.txt").write_text("Rate was 245 ± 127 nmol; elsewhere 3.5 and 3.5 (SE 0.2) and 9 mean 9.")
    recs = [_rec(0, "A", 245.0, sub="wc", date="2003-10-05"),
            _rec(1, "A", 9.0, sub="benthos", date="1998", units="4.00"),
            _rec(2, "A", 7.0, sub="other", date=None)]
    rep = cpd.prepare_augment_records(recs, tmp_path)
    assert [r["substrate_type"] for r in recs] == ["water column", "benthos", "other"]
    assert [r["date"] for r in recs] == ["05-10-2003", "1998", None]
    assert recs[0][pa._VALUE_SPAN_KEY] == "245 ± 127"
    assert pa._VALUE_SPAN_KEY not in recs[1] and pa._VALUE_SPAN_KEY not in recs[2]
    assert rep == {"value_span_rows": [0], "unknown_unit_rows": [1]}
    assert recs[0]["value"] == "245.0"                       # value itself untouched here


def test_prepare_fails_when_units_inferred_attribute_disagrees_with_gt(tmp_path):
    (tmp_path / "A.txt").write_text("text")
    r = _rec(0, "A", 1.0); r["_gt_attribute"] = "nfix_rate_mass"
    with pytest.raises(AssertionError, match="units-inferred attribute"):
        cpd.prepare_augment_records([r], tmp_path)


def test_prepare_fails_on_missing_or_empty_ocr(tmp_path):
    with pytest.raises(FileNotFoundError):
        cpd.prepare_augment_records([_rec(0, "MISSING", 1.0)], tmp_path)
    (tmp_path / "E.txt").write_text("  \n")
    with pytest.raises(AssertionError, match="empty"):
        cpd.prepare_augment_records([_rec(0, "E", 1.0)], tmp_path)


def test_gt_attribute_agrees_with_units_inference_on_the_committed_gt():
    for r in json.load(open(_GT)):
        assert cpd._classify_attribute(r["units"]) == r["attribute"]


# ─── CLI guards ──────────────────────────────────────────────────────────────


def test_augment_requires_ocr_dir():
    with pytest.raises(SystemExit):
        cpd.main(["--augment", "--augment-stub", "--reviewed"])


def test_augment_rejects_a_ground_truth_file_other_than_the_configs():
    with pytest.raises(SystemExit):
        cpd.main(["--augment", "--augment-stub", "--ocr-dir", "x"])       # no --reviewed
