"""Unit tests for analysis/deduplication.py (weighted correlation clustering + center
selection) on hand-built fixtures. The src-level pairwise utility has its own
tests/test_deduplication.py; the cache builder has tests/test_deduplicate_cache.py.
"""
from __future__ import annotations

import copy
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments"))
sys.path.insert(0, str(_REPO))

from analysis import deduplicate_cache as dc  # noqa: E402
from analysis.common import dedup  # noqa: E402
from analysis import deduplication as ded  # noqa: E402

TL = 60.0


def _brute_force_cost(members, W, tau):
    def parts(lst):
        if not lst:
            yield []
            return
        f, *rest = lst
        for p in parts(rest):
            for k in range(len(p)):
                yield p[:k] + [[f] + p[k]] + p[k + 1:]
            yield [[f]] + p

    best = None
    for p in parts(list(members)):
        cl = {v: k for k, g in enumerate(p) for v in g}
        if any(cl[a] == cl[b] and (a, b) not in W for a, b in itertools.combinations(sorted(members), 2)):
            continue
        c = ded.partition_cost(p, W, tau)
        best = c if best is None else min(best, c)
    return best


@pytest.mark.parametrize("seed", range(6))
def test_exact_solver_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    for _ in range(10):
        m = int(rng.integers(2, 7))
        members = list(range(m))
        tau = float(rng.random())
        W = {e: float(rng.random()) for e in itertools.combinations(members, 2) if rng.random() < 0.75}
        clusters, cost, _ = ded.solve_component(members, W, tau, TL)
        assert cost == pytest.approx(_brute_force_cost(members, W, tau), abs=1e-9)
        assert ded.partition_cost(clusters, W, tau) == pytest.approx(cost)


def test_pair_costs_threshold_centred():
    assert ded.pair_costs(0.9, 0.6) == (pytest.approx(0.3), 0.0)
    assert ded.pair_costs(0.4, 0.6) == (0.0, pytest.approx(0.2))
    assert ded.pair_costs(0.6, 0.6) == (0.0, 0.0)          # exactly tau: free either way


def test_triangle_all_above_tau_is_one_cluster():
    W = {(0, 1): 0.9, (0, 2): 0.8, (1, 2): 0.6}
    clusters, cost, _ = ded.solve_component([0, 1, 2], W, 0.5, TL)
    assert clusters == [[0, 1, 2]] and cost == 0.0


def test_forbidden_pair_breaks_a_chain_at_the_cheaper_cut():
    # 0-1 strong, 1-2 weaker, 0-2 not strict-equal (absent): can't join all three.
    W = {(0, 1): 0.95, (1, 2): 0.8}
    clusters, cost, _ = ded.solve_component([0, 1, 2], W, 0.5, TL)
    assert clusters == [[0, 1], [2]]
    assert cost == pytest.approx(0.8 - 0.5)               # cut the (1,2) edge


def test_cluster_rows_known_answer_and_sub_tau_pair_stays_apart():
    # rows 0,1,2 one clique; rows 3,4 strict-equal but scoring below tau; row 5 alone.
    edges = [(0, 1), (0, 2), (1, 2), (3, 4)]
    weights = [1.0, 0.9, 0.8, 0.3]
    res = ded.cluster_rows(6, edges, weights, 0.5, TL)
    assert res["clusters"] == [[0, 1, 2], [3], [4], [5]]
    assert res["cost"] == 0.0 and res["n_components"] == 4 and res["largest_component"] == 3


def test_cluster_rows_rejects_bad_cache():
    with pytest.raises(AssertionError):
        ded.cluster_rows(3, [(0, 1), (0, 1)], [1.0, 1.0], 0.5, TL)      # duplicate edge
    with pytest.raises(AssertionError):
        ded.cluster_rows(3, [(1, 0)], [1.0], 0.5, TL)                   # i < j violated


def test_solver_raises_when_not_proven_optimal(monkeypatch):
    class _R:
        status, message, x, fun = 1, "Time limit reached", None, None
    monkeypatch.setattr(ded, "milp", lambda *a, **k: _R())
    with pytest.raises(RuntimeError, match="not solved to proven optimality"):
        ded.solve_component([0, 1, 2], {(0, 1): 0.9, (0, 2): 0.9, (1, 2): 0.9}, 0.5, 1.0)


def test_center_is_highest_mean_raw_weight_not_w_minus_tau():
    W = {(0, 1): 0.9, (0, 2): 0.8, (1, 2): 0.6}
    center, mean, tied = ded.pick_center([0, 1, 2], W, [100, 101, 102])
    assert center == 0 and not tied
    assert mean[0] == pytest.approx(0.85) and mean[1] == pytest.approx(0.75) and mean[2] == pytest.approx(0.7)


def test_center_tie_goes_to_lowest_measurement_id_then_row_and_is_flagged():
    W = {(0, 1): 0.9}
    assert ded.pick_center([0, 1], W, [7, 7])[0] == 0              # equal ids -> lowest row
    center, _, tied = ded.pick_center([0, 1], W, [20, 10])
    assert center == 1 and tied                      # row 1 has measurement_id 10
    center, _, tied = ded.pick_center([0, 1], W, [10, 20])
    assert center == 0 and tied


def test_singleton_is_its_own_center():
    center, mean, tied = ded.pick_center([4], {}, [0, 1, 2, 3, 9])
    assert center == 4 and not tied and np.isnan(mean[4])


def test_center_missing_pair_is_a_bug_not_a_default():
    with pytest.raises(KeyError):
        ded.pick_center([0, 1, 2], {(0, 1): 0.9, (1, 2): 0.9}, [0, 1, 2])   # (0,2) absent


def _recs():
    return [
        {"measurement_id": 5, "name": "a", "page_number": [1], "context": ["c0"]},
        {"measurement_id": 3, "name": "b", "page_number": [2, 3], "context": ["c1", "c1b"]},
        {"measurement_id": 9, "name": "c", "page_number": [4], "context": ["c2"]},
        {"measurement_id": 1, "name": "d", "page_number": [7], "context": ["c3"]},
    ]


def test_merge_provenance_center_first_then_members_in_row_order():
    out = ded.merge_provenance(_recs(), [[0, 1, 2], [3]], [1, 3], ["page_number", "context"])
    assert [r["measurement_id"] for r in out] == [3, 1]               # centers, in row order
    assert out[0]["page_number"] == [2, 3, 1, 4] and out[0]["context"] == ["c1", "c1b", "c0", "c2"]
    assert out[1]["page_number"] == [7]
    assert out[0]["name"] == "b"                                      # non-provenance fields: center's own


def test_merge_provenance_fails_loud():
    recs = _recs()
    recs[2]["page_number"] = 4
    with pytest.raises(TypeError):
        ded.merge_provenance(recs, [[0], [1], [2], [3]], [0, 1, 2, 3], ["page_number"])
    recs = _recs()
    recs[0]["context"] = ["a", "b"]
    with pytest.raises(ValueError):
        ded.merge_provenance(recs, [[0], [1], [2], [3]], [0, 1, 2, 3], ["page_number", "context"])


# -- config ---------------------------------------------------------------------------

CACHE_SEC = {
    "extraction_file": "final.json",
    "strict_fields": ["document_id", "point_value", "units"],
    "fuzzy_fields": ["name"],
    "numeric_coerce": ["point_value"],
    "join_list_fields": [],
    "summary_thresholds": [0.5],
}
DEDUP_SEC = {"deduplicate_cache_config_id": "cache-01", "solver_time_limit_s": 60,
             "provenance_fields": ["page_number", "context"]}


def _write_configs(tmp_path, monkeypatch, dedup_sec=None, ids=("x",)):
    monkeypatch.setattr(dedup, "ANALYSIS_CONFIGS_ROOT", tmp_path)
    (tmp_path / "cache-01.yaml").write_text(yaml.safe_dump({
        "id": "cache-01", "project": "p", "description": "d", "seed": 1,
        "params": {"experiment_ids": ["x"], "deduplicate_cache": CACHE_SEC}}))
    path = tmp_path / "dedup-01.yaml"
    path.write_text(yaml.safe_dump({
        "id": "dedup-01", "project": "p", "description": "d", "seed": 1,
        "params": {"experiment_ids": list(ids), "deduplication": dedup_sec or DEDUP_SEC}}))
    return path


def test_config_happy_path(tmp_path, monkeypatch):
    assert dedup.load_deduplication_config(_write_configs(tmp_path, monkeypatch))["id"] == "dedup-01"


@pytest.mark.parametrize("override", [
    {"solver_time_limit_s": 0}, {"solver_time_limit_s": True}, {"provenance_fields": []},
    {"provenance_fields": ["name", "page_number"]},     # name is a match field of the cache config
    {"deduplicate_cache_config_id": "missing-cfg"},
])
def test_config_rejects_bad_section(tmp_path, monkeypatch, override):
    sec = {**DEDUP_SEC, **override}
    with pytest.raises((ValueError, FileNotFoundError)):
        dedup.load_deduplication_config(_write_configs(tmp_path, monkeypatch, sec))


def test_config_rejects_id_not_in_cache_config_and_missing_key(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="not in cache config"):
        dedup.load_deduplication_config(_write_configs(tmp_path, monkeypatch, ids=("x", "y")))
    sec = {k: v for k, v in DEDUP_SEC.items() if k != "solver_time_limit_s"}
    with pytest.raises(KeyError):
        dedup.load_deduplication_config(_write_configs(tmp_path, monkeypatch, sec))


# -- end to end on a fake run ----------------------------------------------------------

def _row(mid, name, *, doc="d1", pv="7.0", units="mg/L", page=1):
    return {"measurement_id": mid, "document_id": doc, "point_value": pv, "units": units, "name": name,
            "page_number": [page], "context": [f"ctx-{mid}"]}


@pytest.fixture
def built(tmp_path, monkeypatch):
    run = tmp_path / "run"
    run.mkdir()
    # rows 0,1,2: same strict key, names aaaa/aaab/aaaa (0<->2 exact: 1.0; 0<->1, 1<->2: 0.75)
    # row 3: different document (own cluster); row 4: same strict key as 0-2 but a very different name
    recs = [_row(10, "aaaa", page=1), _row(11, "aaab", page=2), _row(12, "aaaa", page=3),
            _row(13, "zzzz", doc="d2", page=4), _row(14, "qqqq", page=5)]
    (run / "final.json").write_text(json.dumps(recs))
    monkeypatch.setattr(dc.paths, "find_result_dir", lambda _id: run)
    monkeypatch.setattr(ded.paths, "find_result_dir", lambda _id: run)
    monkeypatch.setattr(dedup, "DEDUP_CACHE_ROOT", tmp_path / "cache")
    monkeypatch.setattr(dedup, "DEDUP_ROOT", tmp_path / "out")
    monkeypatch.setattr(ded, "get_threshold", lambda _id: 0.7)
    dc.build_deduplicate_cache("cache-01", "x", CACHE_SEC)
    return run, recs


def test_end_to_end_known_answer(built):
    run, recs = built
    ded.build_deduplication("dedup-01", "x", DEDUP_SEC, "cache-01", CACHE_SEC)
    out = ded.load_deduplicated("dedup-01", "x")
    # tau=0.7: edges 0-2 (1.0), 0-1 (0.75), 1-2 (0.75) all >= tau -> one clique {0,1,2}; row 4 scores
    # (0 + ...) low vs the others -> below tau, stays alone; row 3 alone. 5 rows -> 3 clusters.
    assert [r["measurement_id"] for r in out] == [10, 13, 14] or [r["measurement_id"] for r in out] == [12, 13, 14]
    center_mid = out[0]["measurement_id"]
    # means: row0 = (0.75+1.0)/2, row2 = (1.0+0.75)/2 (tied, 0.875) > row1 = 0.75 -> tie between rows 0 and 2 -> lowest mid 10
    assert center_mid == 10
    assert out[0]["page_number"] == [1, 2, 3] and out[0]["context"] == ["ctx-10", "ctx-11", "ctx-12"]
    meta = json.loads((dedup.deduplication_dir("dedup-01", "x") / "meta.json").read_text())
    assert (meta["rows_in"], meta["rows_out"], meta["n_clusters"], meta["n_singletons"]) == (5, 3, 3, 2)
    assert meta["n_centers_decided_by_tiebreak"] == 1 and meta["tau"] == 0.7
    assert meta["total_weighted_cost"] == 0.0     # joined pairs all >= tau, cut pairs all < tau: nothing to pay
    audit = pd.read_csv(dedup.deduplication_dir("dedup-01", "x") / "clusters.csv")
    assert audit["row"].tolist() == [0, 1, 2, 3, 4] and audit["is_center"].tolist() == [True, False, False, True, True]
    assert audit["center_row"].tolist() == [0, 0, 0, 3, 4]


def test_end_to_end_is_deterministic(built):
    ded.build_deduplication("dedup-01", "x", DEDUP_SEC, "cache-01", CACHE_SEC)
    d = dedup.deduplication_dir("dedup-01", "x")
    first = {f: (d / f).read_bytes() for f in ("deduplicated.json", "clusters.csv")}
    ded.build_deduplication("dedup-01", "x", DEDUP_SEC, "cache-01", CACHE_SEC)
    assert first == {f: (d / f).read_bytes() for f in ("deduplicated.json", "clusters.csv")}


def test_stale_cache_raises(built):
    run, recs = built
    (run / "final.json").write_text(json.dumps(recs[::-1]))               # same rows, different order
    with pytest.raises(ValueError, match="sha256 mismatch"):
        ded.build_deduplication("dedup-01", "x", DEDUP_SEC, "cache-01", CACHE_SEC)


def test_changed_cache_section_raises(built):
    changed = copy.deepcopy(CACHE_SEC)
    changed["summary_thresholds"] = [0.9]
    with pytest.raises(ValueError, match="section changed"):
        ded.build_deduplication("dedup-01", "x", DEDUP_SEC, "cache-01", changed)


def test_shared_measurement_id_is_allowed_and_tie_breaks_on_row(built):
    # postprocessed.json list-expanded children share a parent's measurement_id by design
    run, recs = built
    for r in recs[:3]:
        r["measurement_id"] = 10
    (run / "final.json").write_text(json.dumps(recs))
    dc.build_deduplicate_cache("cache-01", "x", CACHE_SEC)            # rebuild so the sha check passes
    ded.build_deduplication("dedup-01", "x", DEDUP_SEC, "cache-01", CACHE_SEC)
    out = ded.load_deduplicated("dedup-01", "x")
    assert out[0]["page_number"] == [1, 2, 3]                          # rows 0 and 2 tie on mean and id -> row 0
    meta = json.loads((dedup.deduplication_dir("dedup-01", "x") / "meta.json").read_text())
    assert meta["n_clusters_with_shared_measurement_id"] == 1 and meta["n_records_sharing_a_measurement_id"] == 2


def test_non_int_measurement_id_raises(built):
    run, recs = built
    recs[1]["measurement_id"] = "11"
    (run / "final.json").write_text(json.dumps(recs))
    dc.build_deduplicate_cache("cache-01", "x", CACHE_SEC)
    with pytest.raises(ValueError, match="measurement_id"):
        ded.build_deduplication("dedup-01", "x", DEDUP_SEC, "cache-01", CACHE_SEC)


def test_load_deduplicated_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(dedup, "DEDUP_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        ded.load_deduplicated("dedup-01", "x")
