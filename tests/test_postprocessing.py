"""Unit tests for analysis/postprocessing.py: canonical_units_by_attribute,
postprocess_record (the per-row trigger/no-op logic), and a tiny end-to-end
postprocess_experiment fixture under tmp_path so no real repo data is
touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))
sys.path.insert(0, str(_REPO / "src"))

import utils as paths  # noqa: E402
from analysis import postprocessing as pp  # noqa: E402


# ---------------------------------------------------------------------------
# canonical_units_by_attribute
# ---------------------------------------------------------------------------


def test_canonical_units_by_attribute_groups_and_drops_null():
    gt = pd.DataFrame([
        {"attribute": "tc", "units": "K"},
        {"attribute": "tc", "units": "mK"},
        {"attribute": "tc", "units": "K"},
        {"attribute": "ph", "units": None},
    ])
    out = pp.canonical_units_by_attribute(gt)
    assert out["tc"] == frozenset({"K", "mK"})
    assert out["ph"] == frozenset()


def test_canonical_units_by_attribute_filters_numeric_looking_noise():
    # Real pond ground-truth artifact: surface_area has ~50 rows whose
    # `units` is a bare number. Must never enter the canonical set, or
    # split_value_and_unit_suffix could "recover" a second number out of a
    # value like "7.0 8.00" as if it were a unit.
    gt = pd.DataFrame([
        {"attribute": "surface_area", "units": "m^2"},
        {"attribute": "surface_area", "units": "0.75"},
        {"attribute": "surface_area", "units": "10.0"},
    ])
    out = pp.canonical_units_by_attribute(gt)
    assert out["surface_area"] == frozenset({"m^2"})


# ---------------------------------------------------------------------------
# postprocess_record
# ---------------------------------------------------------------------------

_BLANK_SHAPE = {
    "qualifiers": None, "point_value": None, "lower": None, "upper": None,
    "list_values": None, "tolerance": None, "standard_deviation": None,
}


def _record(**overrides) -> dict:
    r = {"document_id": "d1", "attribute": "tc", "value": "48", "units": "K", **_BLANK_SHAPE}
    r.update(overrides)
    return r


def test_fills_qualifiers_when_model_never_attempted_them():
    record, changed = pp.postprocess_record(_record(), canonical_units={"tc": frozenset({"K", "mK"})})
    assert changed == ["qualifiers"]
    assert record["qualifiers"] == []
    assert record["point_value"] == 48.0


def test_never_overwrites_a_row_the_model_partially_filled():
    # GLiNER-style garbage: upper is non-null even though point_value/
    # qualifiers are null -- must be left completely untouched.
    record = _record(value="17.2", units="C", upper="3.9")
    new_record, changed = pp.postprocess_record(record, canonical_units={"tc": frozenset({"K", "mK"})})
    assert changed == []
    assert new_record == record


def test_skips_qualifier_fill_when_already_tagged():
    record = _record(qualifiers=["IsApproximate"], point_value="0.26")
    new_record, changed = pp.postprocess_record(record, canonical_units={"tc": frozenset({"K", "mK"})})
    assert "qualifiers" not in changed
    assert new_record["point_value"] == "0.26"  # untouched, still a string


def test_standardizes_units_independent_of_qualifier_fill():
    record = _record(value="48", units="kelvin", point_value="48", qualifiers=[])
    new_record, changed = pp.postprocess_record(record, canonical_units={"tc": frozenset({"K", "mK"})})
    assert changed == ["units"]
    assert new_record["units"] == "K"


def test_recovers_units_from_embedded_percent_suffix():
    record = _record(attribute="vegetation_cover", value="34%", units=None)
    canonical = {"vegetation_cover": frozenset({"percent", "fraction"})}
    new_record, changed = pp.postprocess_record(record, canonical_units=canonical)
    assert set(changed) == {"qualifiers", "units"}
    assert new_record["units"] == "percent"
    assert new_record["point_value"] == 34.0


def test_unknown_attribute_gets_empty_canonical_set_and_is_a_no_op_for_units():
    record = _record(attribute="not_a_real_attribute", units="GPa")
    new_record, changed = pp.postprocess_record(record, canonical_units={"tc": frozenset({"K", "mK"})})
    # canonical_units.get("not_a_real_attribute", frozenset()) -> empty ->
    # only known dimensionless aliases would map to None; "GPa" isn't one.
    assert new_record["units"] == "GPa"


def test_fills_when_shape_fields_are_empty_string_or_empty_list_not_none():
    # Several baselines write "" instead of null for point_value/lower/
    # upper/tolerance/standard_deviation, and langextract writes
    # list_values=[] -- both must still count as "never attempted", not
    # "already filled".
    record = _record(point_value="", lower="", upper="", tolerance="",
                      standard_deviation="", list_values=[])
    new_record, changed = pp.postprocess_record(record, canonical_units={"tc": frozenset({"K", "mK"})})
    assert changed == ["qualifiers"]
    assert new_record["point_value"] == 48.0


def test_no_value_no_qualifier_fill_attempted():
    record = _record(value=None)
    new_record, changed = pp.postprocess_record(record, canonical_units={"tc": frozenset({"K", "mK"})})
    assert "qualifiers" not in changed
    assert new_record["qualifiers"] is None


# ---------------------------------------------------------------------------
# expand_list_values
# ---------------------------------------------------------------------------


def test_expand_list_values_one_row_per_entry():
    record = _record(
        value="1, 2, 3", qualifiers=["IsList"], list_values=["1", "2", "3"],
        document_id="d1", measurement_id=7,
    )
    expanded = pp.expand_list_values(record)
    assert [r["point_value"] for r in expanded] == [1.0, 2.0, 3.0]
    # every other field copied verbatim, list_values/qualifiers included
    for r in expanded:
        assert r["document_id"] == "d1"
        assert r["measurement_id"] == 7
        assert r["qualifiers"] == ["IsList"]
        assert r["list_values"] == ["1", "2", "3"]


def test_expand_list_values_keeps_unparseable_entry_as_raw_string():
    # A rare compact "value(uncertainty)" entry inside a list -- to_float
    # can't parse it; must be kept, not dropped or raised on.
    record = _record(qualifiers=["IsList"], list_values=["1", "2.05(5)"])
    expanded = pp.expand_list_values(record)
    assert [r["point_value"] for r in expanded] == [1.0, "2.05(5)"]


def test_expand_list_values_no_op_when_list_values_empty_or_absent():
    record = _record(list_values=None)
    assert pp.expand_list_values(record) == [record]
    record2 = _record(list_values=[])
    assert pp.expand_list_values(record2) == [record2]


def test_postprocess_experiment_expands_list_rows_and_grows_row_count(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)
    experiment_id = "2026-01-01-testset-model-baseline-chatextract-01"
    extraction_dir = results_root / "testset" / "baseline_chatextract" / experiment_id
    extraction_dir.mkdir(parents=True)

    final_rows = [
        {"document_id": "d1", "attribute": "tc", "value": "1, 2, 3", "units": "K", "measurement_id": 0, **_BLANK_SHAPE},
        {"document_id": "d1", "attribute": "tc", "value": "48", "units": "K", "measurement_id": 1, **_BLANK_SHAPE},
    ]
    with open(extraction_dir / "final.json", "w") as f:
        json.dump(final_rows, f)

    gt_path = tmp_path / "ground_truth.json"
    with open(gt_path, "w") as f:
        json.dump([{"document_id": "d1", "attribute": "tc", "point_value": 1.0, "units": "K"}], f)

    out_path = pp.postprocess_experiment(experiment_id, gt_path)
    with open(out_path) as f:
        rows = json.load(f)

    # row 0 ("1, 2, 3") expands to 3 rows; row 1 ("48") stays a single row.
    assert len(rows) == 4
    assert [r["point_value"] for r in rows] == pytest.approx([1.0, 2.0, 3.0, 48.0])
    assert all(r["document_id"] == "d1" and r["attribute"] == "tc" for r in rows)
    # duplicated measurement_id across the expanded rows -- deliberate, see
    # postprocess_experiment's docstring on the validity/judge_combine
    # consequence.
    assert [r["measurement_id"] for r in rows] == [0, 0, 0, 1]


# ---------------------------------------------------------------------------
# postprocess_experiment -- tiny end-to-end fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def experiment_fixture(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)

    dataset = "testset"
    experiment_id = "2026-01-01-testset-model-baseline-chatextract-01"
    extraction_dir = results_root / dataset / "baseline_chatextract" / experiment_id
    extraction_dir.mkdir(parents=True)

    final_rows = [
        # chatextract-style row: no qualifier fields filled at all.
        {"document_id": "d1", "attribute": "tc", "value": "48", "units": "K"},
        # already filled by the model -- must survive unchanged.
        {
            "document_id": "d1", "attribute": "tc", "value": "40 mK", "units": "mK",
            "qualifiers": [], "point_value": 40.0, "lower": None, "upper": None,
            "list_values": None, "tolerance": None, "standard_deviation": None,
        },
    ]
    with open(extraction_dir / "final.json", "w") as f:
        json.dump(final_rows, f)

    gt_path = tmp_path / "ground_truth.json"
    gt_rows = [
        {"document_id": "d1", "attribute": "tc", "point_value": 48.0, "units": "K"},
        {"document_id": "d1", "attribute": "tc", "point_value": 40.0, "units": "mK"},
    ]
    with open(gt_path, "w") as f:
        json.dump(gt_rows, f)

    return experiment_id, extraction_dir, gt_path


def test_postprocess_experiment_writes_postprocessed_json(experiment_fixture):
    experiment_id, extraction_dir, gt_path = experiment_fixture

    out_path = pp.postprocess_experiment(experiment_id, gt_path)

    assert out_path == extraction_dir / "postprocessed.json"
    with open(out_path) as f:
        rows = json.load(f)
    assert len(rows) == 2
    assert rows[0]["qualifiers"] == []
    assert rows[0]["point_value"] == 48.0
    assert rows[1]["point_value"] == 40.0  # untouched, already filled


def test_postprocess_experiment_asserts_document_id_untouched(experiment_fixture, monkeypatch):
    # postprocess_record must only ever touch qualifier/units fields -- guard
    # this at the postprocess_experiment boundary rather than trust it, since
    # a match_cache.pkl's cached edges are positions into this exact row
    # identity/order.
    experiment_id, _extraction_dir, gt_path = experiment_fixture

    def _bad_postprocess_record(record, *, canonical_units):
        record = dict(record)
        record["document_id"] = "mutated"
        return record, []

    monkeypatch.setattr(pp, "postprocess_record", _bad_postprocess_record)
    with pytest.raises(AssertionError, match="document_id"):
        pp.postprocess_experiment(experiment_id, gt_path)


def test_postprocess_experiment_raises_on_missing_final_json(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    monkeypatch.setattr(paths, "RESULTS_ROOT", results_root)
    experiment_id = "2026-01-01-testset-model-baseline-chatextract-01"
    (results_root / "testset" / "baseline_chatextract" / experiment_id).mkdir(parents=True)

    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")

    with pytest.raises(FileNotFoundError, match="no final.json"):
        pp.postprocess_experiment(experiment_id, gt_path)


# ---------------------------------------------------------------------------
# main(): --config vs. positional experiment_ids, mutually exclusive
# (same CLI contract as analysis/match_cache.py)
# ---------------------------------------------------------------------------


def test_main_rejects_neither_ids_nor_config(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["postprocessing.py"])
    with pytest.raises(SystemExit):
        pp.main()


def test_main_rejects_ids_without_ground_truth_file(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["postprocessing.py", "some-id"])
    with pytest.raises(SystemExit):
        pp.main()


def test_main_positional_ids_call_postprocess_experiment(tmp_path, monkeypatch):
    gt_path = tmp_path / "ground_truth.json"
    gt_path.write_text("[]")

    seen = []
    monkeypatch.setattr(pp, "postprocess_experiment", lambda eid, gt: seen.append((eid, gt)))
    monkeypatch.setattr(sys, "argv", ["postprocessing.py", "id-a", "id-b", "--ground-truth-file", str(gt_path)])
    pp.main()
    assert seen == [("id-a", gt_path), ("id-b", gt_path)]


def test_main_config_reads_experiment_ids(tmp_path, monkeypatch):
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
                "params": {"experiment_ids": ["id-a"], "ground_truth_file": str(gt_path)},
            },
            f,
        )

    seen = []
    monkeypatch.setattr(pp, "postprocess_experiment", lambda eid, gt: seen.append((eid, gt)))
    monkeypatch.setattr(sys, "argv", ["postprocessing.py", "--config", str(config_path)])
    pp.main()
    assert seen == [("id-a", gt_path)]
