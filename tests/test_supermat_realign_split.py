"""Rung-1 tests for data/supermat/realign_probe_split.py on a hand-built fixture.

Four papers, one per move case:

    A: v3 train -> train    B: v3 train -> test
    C: v3 test  -> test     D: v3 test  -> train

Row names: <paper><n>[s|n] -- "g" GT valid, "s" synthetic valid, "n" negative.
A negative's augment_axis matches its parent (None for a GT parent).
"""
from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "supermat_realign_probe_split", _REPO_ROOT / "data" / "supermat" / "realign_probe_split.py")
rps = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rps)


def _row(tag, doc, label, axis, mod=None, mid=0):
    return {"document_id": doc, "name": tag, "attribute": "tc", "value": "1", "units": "K",
            "label": label, "modification_type": mod, "gt_row_index": 0,
            "donor_gt_row_index": None, "measurement_id": mid, "source_group_id": 0,
            "augment_axis": axis, "augment_attempt": None,
            "context_override": f"paper {doc}" + (" (edited)" if axis else "")}


def _file(rows):
    for i, r in enumerate(rows):
        r["measurement_id"] = i
    return rows


def _fixture():
    train = _file([
        _row("Ag", "A", "valid", None), _row("As", "A", "valid", "pos_entity"),
        _row("Agn", "A", "invalid", None, "bad_value"),
        _row("Asn", "A", "invalid", "pos_entity", "bad_units"),
        _row("Bg", "B", "valid", None), _row("Bs", "B", "valid", "pos_value"),
        _row("Bgn", "B", "invalid", None, "bad_entity"),
        _row("Bsn", "B", "invalid", "pos_value", "bad_value"),
    ])
    primary = _file([
        _row("Cg", "C", "valid", None), _row("Cgn_p", "C", "invalid", None, "bad_units"),
        _row("Dg", "D", "valid", None), _row("Dgn_p", "D", "invalid", None, "bad_value"),
    ])
    diag = _file([
        _row("Cg", "C", "valid", None), _row("Cs", "C", "valid", "pos_entity"),
        _row("Cgn_d", "C", "invalid", None, "bad_entity"),
        _row("Csn", "C", "invalid", "pos_entity", "bad_value"),
        _row("Dg", "D", "valid", None), _row("Ds", "D", "valid", "pos_value"),
        _row("Dgn_d", "D", "invalid", None, "bad_units"),
        _row("Dsn", "D", "invalid", "pos_value", "bad_entity"),
    ])
    return {"train": train, "primary": primary, "diag": diag}


SPLIT = {"A": "train", "B": "test", "C": "test", "D": "train"}


def _names(rows):
    return sorted(r["name"] for r in rows)


def test_make_split_v1_assignment_and_new_papers_to_train():
    split = rps.make_split({"A", "B"}, {"C"}, {"A", "B", "C", "N1", "N2"})
    assert split == {"A": "train", "B": "train", "C": "test", "N1": "train", "N2": "train"}


def test_make_split_rejects_overlap_and_missing():
    with pytest.raises(AssertionError, match="overlaps"):
        rps.make_split({"A"}, {"A"}, {"A"})
    with pytest.raises(AssertionError, match="missing from the GT"):
        rps.make_split({"A"}, {"Z"}, {"A"})


def test_realign_moves_rows_per_case():
    out, log = rps.realign_rows(_fixture(), SPLIT, random.Random(0))
    # train: A stays (4 rows) + D's diag rows only (D's primary rows dropped)
    assert _names(out["train"]) == ["Ag", "Agn", "As", "Asn", "Dg", "Dgn_d", "Ds", "Dsn"]
    # primary: C's primary rows + B's GT valid and GT-parent negative only
    assert _names(out["primary"]) == ["Bg", "Bgn", "Cg", "Cgn_p"]
    # diag: C's diag rows + all of B's train rows
    assert _names(out["diag"]) == ["Bg", "Bgn", "Bs", "Bsn", "Cg", "Cgn_d", "Cs", "Csn"]
    assert log["moves"] == {"train->train": ["A"], "train->test": ["B"],
                            "test->train": ["D"], "test->test": ["C"]}
    for k in rps.FILES:
        assert [r["measurement_id"] for r in out[k]] == list(range(len(out[k])))
        assert log[k]["dropped"] == []


def test_provenance_points_at_the_source_row():
    fx = _fixture()
    out, log = rps.realign_rows(fx, SPLIT, random.Random(0))
    for k in rps.FILES:
        for p, row in zip(log[k]["provenance"], out[k]):
            src = fx[p["src_file"]][p["src_measurement_id"]]
            assert {**src, "measurement_id": row["measurement_id"]} == row


def test_seed_determinism():
    a, la = rps.realign_rows(_fixture(), SPLIT, random.Random(7))
    b, lb = rps.realign_rows(_fixture(), SPLIT, random.Random(7))
    assert a == b and la == lb


def test_inputs_not_mutated():
    fx = _fixture()
    snapshot = {k: [dict(r) for r in v] for k, v in fx.items()}
    rps.realign_rows(fx, SPLIT, random.Random(0))
    assert fx == snapshot


def test_trim_surplus_valids_drops_only_synthetic():
    fx = _fixture()
    # A gains two extra synthetic valids -> train has 2 surplus valids, both synthetic
    fx["train"] += [_row("As2", "A", "valid", "pos_value"), _row("As3", "A", "valid", "pos_entity")]
    _file(fx["train"])
    # train after the move: valid Ag As As2 As3 Dg Ds (6), invalid Agn Asn Dgn_d Dsn (4)
    out, log = rps.realign_rows(fx, SPLIT, random.Random(0))
    assert len(log["train"]["dropped"]) == 2
    assert all(d["label"] == "valid" and d["augment_axis"] is not None
               for d in log["train"]["dropped"])
    assert log["train"]["n_rows"] == 8 and log["train"]["n_valid"] == 4
    assert {"Ag", "Dg"} <= set(_names(out["train"]))          # GT valids always kept


def test_trim_surplus_valids_fails_without_enough_synthetic():
    fx = _fixture()
    fx["train"] += [_row("Ag2", "A", "valid", None), _row("Ag3", "A", "valid", None),
                    _row("Ag4", "A", "valid", None), _row("Ag5", "A", "valid", None),
                    _row("Ag6", "A", "valid", None)]
    _file(fx["train"])
    with pytest.raises(AssertionError, match="synthetic valids to drop"):
        rps.realign_rows(fx, SPLIT, random.Random(0))


def test_trim_surplus_negatives_drops_most_frequent_type():
    fx = _fixture()
    # primary for C gets 2 extra bad_units -> C/B primary: 2 valid, 4 invalid
    # types: bad_units 3 (Cgn_p + 2), bad_entity 1 (Bgn) -> both drops from bad_units
    fx["primary"] += [_row("Cu1", "C", "invalid", None, "bad_units"),
                      _row("Cu2", "C", "invalid", None, "bad_units")]
    _file(fx["primary"])
    out, log = rps.realign_rows(fx, SPLIT, random.Random(0))
    assert [d["modification_type"] for d in log["primary"]["dropped"]] == ["bad_units", "bad_units"]
    assert log["primary"]["n_rows"] == 4 and log["primary"]["n_valid"] == 2
    assert "Bgn" in _names(out["primary"])


def test_rejects_unassigned_or_unknown_paper():
    with pytest.raises(AssertionError, match="unassigned"):
        rps.realign_rows(_fixture(), {"A": "train", "B": "test", "C": "test"}, random.Random(0))
    with pytest.raises(AssertionError, match="not in v3"):
        rps.realign_rows(_fixture(), {**SPLIT, "Z": "train"}, random.Random(0))


def test_rejects_paper_on_both_v3_sides():
    fx = _fixture()
    fx["primary"].append(_row("Ax", "A", "valid", None, mid=len(fx["primary"])))
    fx["diag"].append(_row("Ax", "A", "valid", None, mid=len(fx["diag"])))
    with pytest.raises(AssertionError, match="both v3 train and test"):
        rps.realign_rows(fx, SPLIT, random.Random(0))


def test_rejects_duplicate_rows():
    fx = _fixture()
    fx["train"].append({**fx["train"][0], "measurement_id": len(fx["train"])})
    fx["train"].append(_row("Az", "A", "invalid", None, "bad_value", mid=len(fx["train"])))
    with pytest.raises(AssertionError, match="duplicate rows"):
        rps.realign_rows(fx, SPLIT, random.Random(0))


# --- judge transplant -------------------------------------------------------

def _judge_outputs(fx):
    """Fake responses + arrays per source file; each array encodes (file, mid)."""
    code = {"train": 1, "primary": 2, "diag": 3}
    resp, arrs = {}, {}
    for k, rows in fx.items():
        resp[k] = [{**r, "judgement": True, "judgement_prob": 0.5, "judgement_p_true": 0.5,
                    "judgement_p_false": 0.5, "judgement_logit_p_true": 0.0,
                    "judgement_logit_p_false": 0.0, "judgement_model": "m"} for r in rows]
        arrs[k] = {n: {str(r["measurement_id"]): np.array([code[k], r["measurement_id"], j])
                       for r in rows} for j, n in enumerate(rps.NPZ_NAMES)}
    return resp, arrs


def test_transplant_carries_response_and_arrays_of_source_row():
    fx = _fixture()
    out, log = rps.realign_rows(fx, SPLIT, random.Random(0))
    resp, arrs = _judge_outputs(fx)
    code = {"train": 1, "primary": 2, "diag": 3}
    for k in rps.FILES:
        new_resp, new_arr = rps.transplant_judge(log[k]["provenance"], out[k], resp, arrs)
        assert [r["measurement_id"] for r in new_resp] == list(range(len(out[k])))
        for p, r, row in zip(log[k]["provenance"], new_resp, out[k]):
            assert {kk: v for kk, v in r.items() if kk not in rps.JUDGE_FIELDS} == row
            for j, n in enumerate(rps.NPZ_NAMES):
                np.testing.assert_array_equal(
                    new_arr[n][str(row["measurement_id"])],
                    [code[p["src_file"]], p["src_measurement_id"], j])


def test_transplant_rejects_content_mismatch():
    fx = _fixture()
    out, log = rps.realign_rows(fx, SPLIT, random.Random(0))
    resp, arrs = _judge_outputs(fx)
    p = log["primary"]["provenance"][0]
    next(r for r in resp[p["src_file"]] if r["measurement_id"] == p["src_measurement_id"])["value"] = "99"
    with pytest.raises(AssertionError, match="response row != new row"):
        rps.transplant_judge(log["primary"]["provenance"], out["primary"], resp, arrs)


def test_transplant_rejects_npz_response_id_mismatch():
    fx = _fixture()
    out, log = rps.realign_rows(fx, SPLIT, random.Random(0))
    resp, arrs = _judge_outputs(fx)
    del arrs["diag"]["layer_outputs.npz"]["0"]
    with pytest.raises(AssertionError, match="measurement_id set"):
        rps.transplant_judge(log["diag"]["provenance"], out["diag"], resp, arrs)


# --- create_probe_dataset.py --split-file -------------------------------------

import json
import sys

sys.path.insert(0, str(_REPO_ROOT / "src"))
_cpd_spec = importlib.util.spec_from_file_location(
    "supermat_create_probe_dataset_split", _REPO_ROOT / "data" / "supermat" / "create_probe_dataset.py")
cpd = importlib.util.module_from_spec(_cpd_spec)
_cpd_spec.loader.exec_module(cpd)


def _recs(*codes):
    return [{"_paper_code": c, "i": i} for i, c in enumerate(codes)]


def test_pinned_split_assigns_whole_papers_in_input_order():
    tr, te = cpd.pinned_split(_recs("A", "B", "A", "C"), {"A": "train", "B": "test", "C": "train"})
    assert [r["i"] for r in tr] == [0, 2, 3] and [r["i"] for r in te] == [1]


def test_pinned_split_rejects_paper_set_mismatch():
    with pytest.raises(AssertionError, match="unassigned"):
        cpd.pinned_split(_recs("A", "B"), {"A": "train"})
    with pytest.raises(AssertionError, match="not in GT"):
        cpd.pinned_split(_recs("A"), {"A": "train", "Z": "test"})
    with pytest.raises(AssertionError, match="bad split values"):
        cpd.pinned_split(_recs("A"), {"A": "val"})


def _gt_records(gt_name, tmp_path):
    gt = json.loads((_REPO_ROOT / "data" / "supermat" / gt_name).read_text())
    for code in {str(r["document_id"]) for r in gt}:     # build_gt_records only lists names
        (tmp_path / f"{code}.txt").write_text("x")
    return cpd.build_gt_records(gt, tmp_path)


def test_committed_split_on_qualifier_gt_gives_v1_test_papers(tmp_path):
    split = json.loads((_REPO_ROOT / "data" / "supermat" / "probe_split_v3s.json").read_text())
    tr, te = cpd.pinned_split(_gt_records("ground_truth_qualifiers.json", tmp_path), split)
    v1_test = {r["document_id"] for r in json.loads(
        (_REPO_ROOT / "data" / "supermat" / "probe_dataset_test.json").read_text())}
    assert {r["_paper_code"] for r in te} == v1_test
    assert {r["_paper_code"] for r in tr}.isdisjoint(v1_test)
    assert len({r["_paper_code"] for r in tr}) == 70 and len(v1_test) == 72


def test_committed_split_fails_loud_on_v1_gt(tmp_path):
    split = json.loads((_REPO_ROOT / "data" / "supermat" / "probe_split_v3s.json").read_text())
    with pytest.raises(AssertionError, match="not in GT"):          # 6 qualifier-only papers
        cpd.pinned_split(_gt_records("ground_truth.json", tmp_path), split)


def test_judge_run_metadata_keeps_identity_top_level_and_checks_agreement():
    m = {"dataset": "supermat", "extraction_id": None, "judge_model": "j", "judge_model_id": "J/j",
         "runtime_seconds": 1.0}
    meta = rps.judge_run_metadata({"train": m, "primary": {**m, "runtime_seconds": 2.0}, "diag": m},
                                  {"train": "t"}, "log", "data")
    assert meta["judge_model"] == "j" and meta["dataset"] == "supermat"
    assert meta["source_run_metadata"]["primary"]["runtime_seconds"] == 2.0
    with pytest.raises(AssertionError, match="identity"):
        rps.judge_run_metadata({"train": m, "primary": {**m, "judge_model": "other"}, "diag": m},
                               {}, "log", "data")
