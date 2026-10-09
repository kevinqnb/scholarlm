"""Deduplicate an extraction by exact weighted correlation clustering, keeping each cluster's center.

Reads the duplicate-candidate graph from deduplicate_cache.py (checked against the
current extraction file), then:
1. Clusters rows. With tau = the dataset's fuzzy_threshold, separating a pair costs
   max(w - tau, 0) and joining costs max(tau - w, 0). Pairs that aren't strict-equal
   can never share a cluster (isclose is not transitive, so connected components
   could wrongly merge them). Each component of the w >= tau graph is solved to
   proven optimality with an integer program; otherwise it is a hard error.
2. Keeps each cluster's center: the member with the highest mean raw weight to the
   others. Ties go to the lowest (measurement_id, row) and are counted in meta.json.
Kept rows are the original records; provenance fields hold the center's entries
followed by the dropped members' entries.

Writes analysis/results/deduplication/<config>/<id>/{deduplicated.json, clusters.csv, meta.json}.

Usage
-----
    python analysis/deduplication.py --config analysis/analysis-configs/recovery-validity/<id>.yaml
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from analysis.common import dedup
from analysis.common.config import get_section
from analysis.common.dedup import (
    DEDUP_SECTION as SECTION, DEDUP_SECTION_KEYS as SECTION_KEYS, deduplication_dir, load_deduplication_config,
)
from analysis.common.provenance import repo_relative, sha256_file
from experiments.run_extraction import load_dataset_config
import utils as paths


_TIE_ATOL = 1e-12  # float-noise tolerance when comparing mean weights


def get_threshold(experiment_id: str) -> float:
    """The clustering threshold tau: the experiment's dataset's fuzzy_threshold.

    Args:
        experiment_id: Extraction experiment id.

    Returns:
        tau in [0, 1].

    Raises:
        ValueError: The threshold is unset or out of range.
    """
    dataset = paths.find_result_dir(experiment_id).relative_to(paths.RESULTS_ROOT).parts[0]
    tau = load_dataset_config(dataset).fuzzy_threshold
    if tau is None or isinstance(tau, bool) or not isinstance(tau, (int, float)) or not 0.0 <= tau <= 1.0:
        raise ValueError(f"DatasetConfig for {dataset!r} has no usable fuzzy_threshold (got {tau!r})")
    return float(tau)


# ---------------------------------------------------------------------------
# Weighted correlation clustering
# ---------------------------------------------------------------------------

def connected_components(n: int, pairs) -> list[list[int]]:
    """Connected components of a graph on nodes 0..n-1 (union-find).

    Args:
        n: Number of nodes.
        pairs: Edges as (i, j).

    Returns:
        List of components, each a list of node ids.
    """
    parent = list(range(n))

    def find(x):
        """Union-find root of ``x``, with path halving."""
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in pairs:
        parent[find(i)] = find(j)
    comp: dict = collections.defaultdict(list)
    for r in range(n):
        comp[find(r)].append(r)
    return list(comp.values())


def pair_costs(w: float, tau: float) -> tuple[float, float]:
    """Correlation-clustering costs for a strict-equal pair.

    Args:
        w: Pair weight.
        tau: Threshold.

    Returns:
        ``(cost_to_separate, cost_to_join)`` = ``(max(w - tau, 0), max(tau - w, 0))``.
    """
    return max(w - tau, 0.0), max(tau - w, 0.0)


def partition_cost(clusters, W: dict, tau: float) -> float:
    """Total correlation-clustering cost of a partition.

    Args:
        clusters: Partition of the rows, as lists.
        W: (i, j) -> weight for strict-equal pairs, i < j.
        tau: Threshold.

    Returns:
        Sum of separate/join costs over all pairs in W.

    Raises:
        AssertionError: A cluster joins a pair absent from W.
    """
    cl = {v: k for k, g in enumerate(clusters) for v in g}
    cost = 0.0
    for a, b in itertools.combinations(sorted(cl), 2):
        w = W.get((a, b))
        if w is None:
            if cl[a] == cl[b]:
                raise AssertionError(f"rows {a} and {b} are not strict-equal but share a cluster")
            continue
        c_sep, c_join = pair_costs(w, tau)
        cost += c_sep if cl[a] != cl[b] else c_join
    return cost


def solve_component(members: list[int], W: dict, tau: float, time_limit_s: float) -> tuple[list[list[int]], float, float]:
    """Solve one component exactly as a MILP with transitivity constraints.

    Args:
        members: Row ids in the component.
        W: (i, j) -> weight for strict-equal pairs within ``members``; absent pairs
            are forced apart.
        tau: Threshold.
        time_limit_s: Solver time limit.

    Returns:
        ``(clusters, cost, seconds)``.

    Raises:
        RuntimeError: The solver did not prove optimality in time.
    """
    members = sorted(members)
    if len(members) == 1:
        return [members], 0.0, 0.0
    t0 = time.time()
    var = {}
    for a, b in itertools.combinations(members, 2):
        if (a, b) in W:
            var[(a, b)] = len(var)
    if not var:
        return [[v] for v in members], 0.0, 0.0
    const = 0.0
    coef = np.zeros(len(var))
    for p, idx in var.items():
        c_sep, c_join = pair_costs(W[p], tau)
        const += c_sep
        coef[idx] = c_join - c_sep

    def key(a, b):
        """Pair as (min, max)."""
        return (a, b) if a < b else (b, a)

    rows, cols, vals, r = [], [], [], 0
    for triple in itertools.combinations(members, 3):
        for j in triple:                                   # x_ij + x_jk - x_ik <= 1
            i, k = [v for v in triple if v != j]
            ij, jk, ik = key(i, j), key(j, k), key(i, k)
            if ij not in var or jk not in var:             # a forced-zero term: trivially true
                continue
            rows += [r, r]
            cols += [var[ij], var[jk]]
            vals += [1, 1]
            if ik in var:
                rows.append(r)
                cols.append(var[ik])
                vals.append(-1)
            r += 1
    constraints = []
    if r:
        A = coo_matrix((vals, (rows, cols)), shape=(r, len(var))).tocsr()
        constraints = [LinearConstraint(A, -np.inf, 1.0)]
    res = milp(coef, constraints=constraints, integrality=np.ones(len(var)), bounds=Bounds(0, 1),
               options={"time_limit": float(time_limit_s), "mip_rel_gap": 0.0})
    if res.status != 0:
        raise RuntimeError(
            f"component of {len(members)} rows (first row {members[0]}) not solved to proven optimality "
            f"within {time_limit_s}s: status {res.status}: {res.message}")
    x = np.round(res.x).astype(int)
    parent = {v: v for v in members}

    def find(v):
        """Union-find root of ``v``, with path halving."""
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    for p, idx in var.items():
        if x[idx]:
            parent[find(p[0])] = find(p[1])
    groups: dict = collections.defaultdict(list)
    for v in members:
        groups[find(v)].append(v)
    clusters = sorted(groups.values(), key=lambda g: g[0])
    cost = partition_cost(clusters, W, tau)         # also asserts no non-strict-equal pair is joined
    assert abs(cost - (const + res.fun)) < 1e-6, (cost, const + res.fun)
    return clusters, cost, time.time() - t0


def cluster_rows(n: int, edges: list, weights: list, tau: float, time_limit_s: float) -> dict:
    """Cluster all rows, solving each component of the w >= tau graph separately.

    Args:
        n: Number of rows.
        edges: Cached (i, j) pairs, i < j.
        weights: Weight per edge.
        tau: Threshold.
        time_limit_s: Per-component solver time limit.

    Returns:
        Dict with ``clusters`` (sorted by first row), ``cost``, ``n_components``,
        ``largest_component``, ``slowest_solve_s``.
    """
    W = {tuple(e): float(w) for e, w in zip(edges, weights)}
    assert len(W) == len(edges), "duplicate edges in cache"
    assert all(i < j for i, j in W) and all(0 <= i and j < n for i, j in W)
    adj = collections.defaultdict(set)
    for i, j in W:
        adj[i].add(j)
        adj[j].add(i)
    comps = connected_components(n, [e for e, w in W.items() if w >= tau])
    clusters, total_cost, slowest = [], 0.0, 0.0
    for comp in comps:
        S = set(comp)
        sub = {(a, b): W[(a, b)] for a in comp for b in adj[a] if b in S and a < b}
        cl, cost, secs = solve_component(comp, sub, tau, time_limit_s)
        clusters += cl
        total_cost += cost
        slowest = max(slowest, secs)
    clusters.sort(key=lambda g: g[0])
    assert sorted(v for g in clusters for v in g) == list(range(n))
    return {"clusters": clusters, "cost": total_cost, "n_components": len(comps),
            "largest_component": max(len(c) for c in comps), "slowest_solve_s": slowest}


def pick_center(cluster: list[int], W: dict, measurement_ids: list) -> tuple[int, dict, bool]:
    """Pick the member with the highest mean raw weight to the rest of its cluster.

    Ties (within _TIE_ATOL) go to the lowest (measurement_id, row).

    Args:
        cluster: Row ids.
        W: (i, j) -> weight, i < j.
        measurement_ids: measurement_id per row.

    Returns:
        ``(center_row, {row: mean weight}, decided_by_tiebreak)``; singletons get NaN.
    """
    if len(cluster) == 1:
        return cluster[0], {cluster[0]: float("nan")}, False
    mean = {}
    for v in cluster:
        ws = [W[(min(v, u), max(v, u))] for u in cluster if u != v]   # KeyError = non-strict-equal pair: a bug
        mean[v] = math.fsum(ws) / len(ws)
    best = max(mean.values())
    tied = [v for v in cluster if best - mean[v] <= _TIE_ATOL]
    center = min(tied, key=lambda v: (measurement_ids[v], v))
    return center, mean, len(tied) > 1


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def _check_cache_matches_extraction(cache_config_id, cache_section, experiment_id, extraction_file, records):
    """Check the cache sidecar matches this extraction file and config section.

    Args:
        cache_config_id: Config id the cache was built under.
        cache_section: Current ``deduplicate_cache`` section.
        experiment_id: Extraction experiment id.
        extraction_file: Extraction file in use.
        records: Its loaded records.

    Returns:
        The sidecar dict.

    Raises:
        FileNotFoundError: No sidecar.
        ValueError: Path, sha256, row count or section differs.
    """
    meta_path = dedup.deduplicate_cache_meta_path(cache_config_id, experiment_id)
    if not meta_path.exists():
        raise FileNotFoundError(f"{experiment_id}: no cache sidecar {meta_path}; build the deduplicate cache first")
    meta = json.loads(meta_path.read_text())
    if meta["extraction_file"] != repo_relative(extraction_file):
        raise ValueError(f"{experiment_id}: cache was built from {meta['extraction_file']}, "
                         f"config now points at {repo_relative(extraction_file)}")
    if meta["extraction_sha256"] != sha256_file(extraction_file):
        raise ValueError(f"{experiment_id}: {extraction_file.name} changed since the cache was built (sha256 mismatch) "
                         f"-- rebuild the deduplicate cache")
    if meta["n_rows"] != len(records):
        raise ValueError(f"{experiment_id}: cache n_rows {meta['n_rows']} != extraction rows {len(records)}")
    if meta["section"] != cache_section:
        raise ValueError(f"{experiment_id}: the cache config's section changed since the cache was built "
                         f"-- rebuild the deduplicate cache")
    return meta


def merge_provenance(records: list[dict], clusters: list[list[int]], centers: list[int], fields: list[str]) -> list[dict]:
    """Center records with their cluster's provenance entries merged in.

    Args:
        records: Original extraction records.
        clusters: Clusters, aligned with ``centers``.
        centers: Center row of each cluster.
        fields: Provenance fields to merge (list-valued, equal length per row).

    Returns:
        One record per center, in row order. Each provenance list is the center's
        entries, then other members' entries in row order.

    Raises:
        TypeError: A provenance value is not a list.
        ValueError: A row's provenance lists differ in length.
    """
    for f in fields:
        bad = [i for i, r in enumerate(records) if not isinstance(r.get(f), list)]
        if bad:
            raise TypeError(f"provenance field {f!r}: {len(bad)} rows are not lists (first rows {bad[:5]})")
    for i, r in enumerate(records):
        if len({len(r[f]) for f in fields}) != 1:
            raise ValueError(f"row {i}: provenance lists have unequal length across {fields}")
    by_center = {c: g for c, g in zip(centers, clusters)}
    out = []
    for c in sorted(centers):
        rec = dict(records[c])
        members = by_center[c]
        for f in fields:
            merged = list(records[c][f])
            for m in members:
                if m != c:
                    merged.extend(records[m][f])
            rec[f] = merged
        out.append(rec)
    for f in fields:
        assert sum(len(r[f]) for r in out) == sum(len(r[f]) for r in records), f"provenance entries not conserved: {f}"
    assert all(len({len(r[f]) for f in fields}) == 1 for r in out)
    return out


def build_deduplication(config_id: str, experiment_id: str, sec: dict, cache_config_id: str, cache_section: dict) -> Path:
    """Cluster one extraction and write deduplicated.json, clusters.csv and meta.json.

    Args:
        config_id: Config id the outputs are filed under.
        experiment_id: Extraction experiment id.
        sec: The ``deduplication`` config section.
        cache_config_id: Config id the deduplicate cache was built under.
        cache_section: The ``deduplicate_cache`` config section.

    Returns:
        Output directory.

    Raises:
        FileNotFoundError: Extraction file or cache missing.
        ValueError: Cache/extraction mismatch, empty extraction, or non-int measurement_id.
    """
    t0 = time.time()
    extraction_file = paths.find_result_dir(experiment_id) / cache_section["extraction_file"]
    if not extraction_file.exists():
        raise FileNotFoundError(f"{experiment_id}: {extraction_file} does not exist")
    records = json.loads(extraction_file.read_text())
    if not isinstance(records, list) or not records:
        raise ValueError(f"{experiment_id}: {extraction_file} is not a non-empty list of records")
    cache_meta = _check_cache_matches_extraction(cache_config_id, cache_section, experiment_id, extraction_file, records)
    cache = dedup.load_deduplicate_cache(cache_config_id, experiment_id)
    n = len(records)
    assert cache["n_rows"] == n

    mids = [r.get("measurement_id") for r in records]
    if any(isinstance(m, bool) or not isinstance(m, int) for m in mids):
        raise ValueError(f"{experiment_id}: measurement_id must be an int on every record")
    # NOT required unique: postprocessed.json's list-expanded child rows share their parent's id.

    tau = get_threshold(experiment_id)
    print(f"{experiment_id}: {n} rows, tau={tau}, {len(cache['edges'])} cached strict-equal edges", flush=True)
    res = cluster_rows(n, cache["edges"], cache["edge_weights"], tau, sec["solver_time_limit_s"])
    clusters = res["clusters"]
    W = {tuple(e): float(w) for e, w in zip(cache["edges"], cache["edge_weights"])}

    centers, audit, n_tiebreak, n_sub_tau, n_shared_mid = [], [], 0, 0, 0
    for g in clusters:
        center, mean, tied = pick_center(g, W, mids)
        centers.append(center)
        n_tiebreak += tied
        n_shared_mid += len({mids[v] for v in g}) < len(g)
        n_sub_tau += any(W[(a, b)] < tau for a, b in itertools.combinations(g, 2))
        for v in g:
            audit.append({"row": v, "measurement_id": mids[v], "cluster_id": g[0], "center_row": center,
                          "is_center": v == center, "cluster_size": len(g), "mean_w": mean[v]})
    kept = merge_provenance(records, clusters, centers, sec["provenance_fields"])
    assert len(kept) == len(clusters)
    audit_df = pd.DataFrame(audit).sort_values("row").reset_index(drop=True)
    assert audit_df["row"].tolist() == list(range(n)) and int(audit_df["is_center"].sum()) == len(clusters)

    out_dir = deduplication_dir(config_id, experiment_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("deduplicated.json", "clusters.csv", "meta.json"):
        (out_dir / stale).unlink(missing_ok=True)
    (out_dir / "deduplicated.json").write_text(json.dumps(kept))
    audit_df.to_csv(out_dir / "clusters.csv", index=False)
    meta = {
        "analysis_config_id": config_id, "experiment_id": experiment_id,
        "deduplicate_cache_config_id": cache_config_id,
        "extraction_file": repo_relative(extraction_file), "extraction_sha256": cache_meta["extraction_sha256"],
        "cache_sha256": sha256_file(dedup.deduplicate_cache_path(cache_config_id, experiment_id)),
        "tau": tau, "solver_time_limit_s": sec["solver_time_limit_s"], "provenance_fields": sec["provenance_fields"],
        "rows_in": n, "rows_out": len(kept), "n_clusters": len(clusters),
        "n_singletons": sum(len(g) == 1 for g in clusters), "largest_cluster": max(len(g) for g in clusters),
        "n_components": res["n_components"], "largest_component": res["largest_component"],
        "slowest_component_solve_s": res["slowest_solve_s"], "total_weighted_cost": res["cost"],
        "n_clusters_with_sub_tau_pair": int(n_sub_tau), "n_clusters_with_shared_measurement_id": int(n_shared_mid),
        "n_records_sharing_a_measurement_id": n - len(set(mids)), "n_centers_decided_by_tiebreak": int(n_tiebreak),
        "n_multi_row_clusters": sum(len(g) > 1 for g in clusters),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{experiment_id}: {n} -> {len(kept)} rows ({len(clusters)} clusters, {meta['n_multi_row_clusters']} multi-row; "
          f"{n_tiebreak} centers by tie-break; {n_sub_tau} clusters hold a sub-tau pair; cost {res['cost']:.3f}; "
          f"largest component {res['largest_component']}, slowest solve {res['slowest_solve_s']:.1f}s) "
          f"-> {out_dir} ({time.time() - t0:.0f}s)", flush=True)
    return out_dir


def load_deduplicated(config_id: str, experiment_id: str) -> list[dict]:
    """Read a built deduplicated.json (never builds one).

    Args:
        config_id: Config id.
        experiment_id: Extraction experiment id.

    Returns:
        The kept records.

    Raises:
        FileNotFoundError: Not built.
    """
    path = deduplication_dir(config_id, experiment_id) / "deduplicated.json"
    if not path.exists():
        raise FileNotFoundError(f"No deduplication for ({config_id!r}, {experiment_id!r}) at {path}")
    return json.loads(path.read_text())


def main() -> None:
    """CLI: deduplicate every experiment id in ``--config``."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="analysis-configs/recovery-validity/<id>.yaml with params.deduplicate_cache and params.deduplication")
    args = parser.parse_args()
    cfg = load_deduplication_config(args.config)
    sec = get_section(cfg, SECTION, SECTION_KEYS)
    # The caches were built by deduplicate_cache.py on this same config.
    cache_section = get_section(cfg, dedup.CACHE_SECTION, dedup.CACHE_SECTION_KEYS)
    for experiment_id in cfg["params"]["experiment_ids"]:
        build_deduplication(cfg["id"], experiment_id, sec, cfg["id"], cache_section)


if __name__ == "__main__":
    main()
