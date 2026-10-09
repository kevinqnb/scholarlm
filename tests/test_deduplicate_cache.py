"""Unit tests for analysis/deduplicate_cache.py on hand-built fixtures: config
validation, pre-normalisation (qualifier list joining, numeric coercion that
keeps unparseable strings raw), edge computation incl. the matcher's null rules
and parity with deduplicate_records' pairwise rule, component summary, and a
tiny end-to-end build_deduplicate_cache against a fake result dir.
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments"))
sys.path.insert(0, str(_REPO))

from analysis import deduplicate_cache as dc  # noqa: E402
from analysis.common import dedup  # noqa: E402
from scholarlm.utils.deduplication import deduplicate_records  # noqa: E402

SEC = {
    "extraction_file": "final.json",
    "strict_fields": ["document_id", "point_value", "units", "qualifiers"],
    "fuzzy_fields": ["name", "date"],
    "numeric_coerce": ["point_value"],
    "join_list_fields": ["qualifiers"],
    "summary_thresholds": [1.0, 0.5],
}


def _rec(name, *, doc="d1", pv="7.0", units="mg/L", q=(), date="2020", page=1):
    return {"document_id": doc, "point_value": pv, "units": units, "qualifiers": list(q),
            "name": name, "date": date, "page_number": [page], "context": [f"c-{name}-{page}"]}


def _cfg(tmp_path, **section_overrides):
    sec = copy.deepcopy(SEC)
    sec.update(section_overrides)
    gt = tmp_path / "gt.json"
    gt.write_text("[]")
    cfg = {"id": "cfg-01", "project": "p", "description": "d", "seed": 1,
           "params": {"experiment_ids": ["x"], "ground_truth_file": str(gt), "deduplicate_cache": sec}}
    path = tmp_path / "recovery-validity" / "cfg-01.yaml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(yaml.safe_dump(cfg))
    return path


def test_config_happy_path(tmp_path):
    assert dedup.load_deduplicate_cache_config(_cfg(tmp_path))["id"] == "cfg-01"


@pytest.mark.parametrize("override", [
    {"strict_fields": []},
    {"fuzzy_fields": ["document_id"]},             # strict and fuzzy overlap
    {"numeric_coerce": ["name"]},                  # not a strict field
    {"join_list_fields": ["page_number"]},         # not a match field
    {"summary_thresholds": [1.5]},
    {"summary_thresholds": [True]},
    {"extraction_file": "other.json"},
])
def test_config_rejects_bad_section(tmp_path, override):
    with pytest.raises(ValueError):
        dedup.load_deduplicate_cache_config(_cfg(tmp_path, **override))


def test_config_missing_and_extra_keys_fail_loud(tmp_path):
    path = _cfg(tmp_path)
    raw = yaml.safe_load(path.read_text())
    del raw["params"]["deduplicate_cache"]["summary_thresholds"]
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(KeyError):
        dedup.load_deduplicate_cache_config(path)
    raw["params"]["deduplicate_cache"]["summary_thresholds"] = [1.0]
    raw["params"]["stray"] = 1
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError):
        dedup.load_deduplicate_cache_config(path)


def test_config_outside_recovery_validity_dir_fails(tmp_path):
    path = _cfg(tmp_path)
    moved = tmp_path / "deduplicate-cache" / path.name
    moved.parent.mkdir()
    path.rename(moved)
    with pytest.raises(ValueError, match="analysis-configs/recovery-validity/"):
        dedup.load_deduplicate_cache_config(moved)


def test_prepare_frame_joins_qualifiers_sorted_and_coerces_numeric():
    recs = [_rec("a", q=["IsApproximate", "IsBound"], pv="7"), _rec("a", q=["IsBound", "IsApproximate"], pv=7.0),
            _rec("a", q=[], pv="n/a")]
    df = dc.prepare_frame(recs, SEC, "t")
    assert df["qualifiers"].tolist()[:2] == ["IsApproximate|IsBound"] * 2
    assert df["qualifiers"][2] == ""                    # empty list -> '' (null to the matcher)
    assert df["point_value"][0] == 7.0 and df["point_value"][1] == 7.0
    assert df["point_value"][2] == "n/a"                # unparseable kept raw, NOT NaN


def test_unparseable_garbage_does_not_strict_match_other_garbage():
    recs = [_rec("a", pv="pH"), _rec("a", pv="TN")]
    df = dc.prepare_frame(recs, SEC, "t")
    edges, _, _, _ = dc.compute_edges(df, SEC["strict_fields"], SEC["fuzzy_fields"])
    assert edges == []


def test_prepare_frame_fails_loud():
    with pytest.raises(KeyError):
        dc.prepare_frame([{"document_id": "d"}], SEC, "t")
    bad = [_rec("a")]
    bad[0]["qualifiers"] = 5
    with pytest.raises(TypeError):
        dc.prepare_frame(bad, SEC, "t")


def test_compute_edges_hand_verified():
    # rows: 0,1 same strict, names aaaa/aaab (0.75), date equal (1.0) -> 0.875
    #       2 differs on units -> no edges; 3 same strict as 0 but date null on one side -> (0.75... name aaaa==aaaa 1.0, date 0.0) = 0.5
    recs = [_rec("aaaa"), _rec("aaab"), _rec("aaaa", units="g/L"), _rec("aaaa", date=None)]
    df = dc.prepare_frame(recs, SEC, "t")
    edges, w, sizes, n_pairs = dc.compute_edges(df, SEC["strict_fields"], SEC["fuzzy_fields"])
    got = dict(zip(edges, w))
    assert set(got) == {(0, 1), (0, 3), (1, 3)}
    assert got[(0, 1)] == 0.875
    assert got[(0, 3)] == 0.5                      # name 1.0, date one-sided null 0.0
    assert got[(1, 3)] == pytest.approx((0.75 + 0.0) / 2)
    assert n_pairs == 6 - 3 and sizes[0] == 3 or n_pairs == 3   # block {0,1,3} -> 3 pairs
    assert edges == sorted(edges)


def test_edges_match_deduplicate_records_pairwise_rule():
    import pandas as pd
    recs = [_rec("pond a"), _rec("pond b"), _rec("pond a", date=None), _rec("lake", pv="8"), _rec("pond a", q=["IsBound"])]
    df = dc.prepare_frame(recs, SEC, "t")
    edges, w, _, _ = dc.compute_edges(df, SEC["strict_fields"], SEC["fuzzy_fields"])
    for thr in (0.0, 0.5, 0.9, 1.0):
        kept, dups = deduplicate_records(
            df, strict_fields=SEC["strict_fields"], fuzzy_fields=SEC["fuzzy_fields"],
            fuzzy_threshold=thr, provenance_fields=[],
        )
        cached = set(dc.edges_above_threshold(edges, w, thr))
        # every greedy drop is a cached edge at that threshold (kept_label is a kept row it matched)
        for _, r in dups.iterrows():
            a, b = sorted((int(r.dropped_label), int(r.kept_label)))
            assert (a, b) in cached


def test_summarise_chain_is_non_clique():
    # chain 0-1, 1-2 (no 0-2): one component of 3 with 2 of 3 possible edges
    s = dc.summarise(4, [(0, 1), (1, 2)], [1.0, 1.0], 1.0)
    assert s["n_components"] == 2 and s["n_components_size_gt1"] == 1
    assert s["largest_component"] == 3 and s["n_non_clique_components"] == 1
    assert s["rows_in_non_clique_components"] == 3
    # triangle is a clique; threshold filter is inclusive
    t = dc.summarise(3, [(0, 1), (1, 2), (0, 2)], [0.5, 0.5, 0.5], 0.5)
    assert t["n_non_clique_components"] == 0 and t["n_edges"] == 3
    assert dc.summarise(3, [(0, 1)], [0.5], 0.51)["n_edges"] == 0


def test_build_end_to_end_and_sidecar(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    recs = [_rec("aaaa"), _rec("aaab"), _rec("zzzz", doc="d2")]
    (run / "final.json").write_text(json.dumps(recs))
    monkeypatch.setattr(dc.paths, "find_result_dir", lambda _id: run)
    monkeypatch.setattr(dedup, "DEDUP_CACHE_ROOT", tmp_path / "cache")
    path = dc.build_deduplicate_cache("cfg-01", "x", SEC)
    data = dedup.load_deduplicate_cache("cfg-01", "x")
    assert data == {"edges": [(0, 1)], "edge_weights": [0.875], "n_rows": 3}
    assert dedup.load_deduplicate_cache("cfg-01", "x", 0.9) == []
    assert dedup.load_deduplicate_cache("cfg-01", "x", 0.875) == [(0, 1)]
    meta = json.loads(dedup.deduplicate_cache_meta_path("cfg-01", "x").read_text())
    assert meta["n_rows"] == 3 and meta["n_edges_at_threshold_0"] == 1 and meta["section"] == SEC
    assert meta["extraction_sha256"] == dc.sha256_file(run / "final.json")
    assert path.exists()


def test_build_fails_loud_without_extraction_file(tmp_path, monkeypatch):
    monkeypatch.setattr(dc.paths, "find_result_dir", lambda _id: tmp_path)
    with pytest.raises(FileNotFoundError):
        dc.build_deduplicate_cache("cfg-01", "x", SEC)


def test_load_cache_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup, "DEDUP_CACHE_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        dedup.load_deduplicate_cache("cfg-01", "x")
