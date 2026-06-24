from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def match_datasets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    strict_matching: Dict[str, str],
    fuzzy_matching: Optional[Dict[str, str]] = None,
    fuzzy_threshold: float = 0.0,
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], List[float]]:
    """Match rows across two dataframes using strict and optional fuzzy criteria.

    Builds candidate edges between rows that satisfy strict criteria, optionally
    scores them with fuzzy similarity, then computes a maximum-weight bipartite
    matching (1-1 alignment) using NetworkX.

    Parameters
    ----------
    df_left, df_right:
        Dataframes to match. Call reset_index(drop=True) beforehand for stable positional indices.
    strict_matching:
        Mapping from column name in df_left -> column name in df_right that must be
        strictly equal. Numeric values are compared with np.isclose.
    fuzzy_matching:
        Mapping from column name in df_left -> column name in df_right compared with
        fuzzy ratios, averaged to produce an edge weight in [0, 1].
    fuzzy_threshold:
        Minimum average fuzzy score in [0, 1] required for candidate edges to be included
        in the graph and considered for matching. Defaults to 0.0 to include all
        edges that pass strict criteria.

    Returns
    -------
    (matching, edges, edge_weights)
        matching: list of (left_index, right_index) pairs.
        edges: list of (left_index, right_index) candidate edges.
        edge_weights: list of edge weights aligned with edges.
    """
    float_atol = 1e-3
    float_rtol = 0.0

    if fuzzy_matching is None:
        fuzzy_matching = {}

    if not isinstance(strict_matching, dict) or len(strict_matching) == 0:
        raise ValueError("strict_matching must be a non-empty dict mapping left_col -> right_col")

    missing_left = [c for c in strict_matching if c not in df_left.columns]
    missing_right = [c for c in strict_matching.values() if c not in df_right.columns]
    if missing_left:
        raise KeyError(f"Columns missing from df_left: {missing_left}")
    if missing_right:
        raise KeyError(f"Columns missing from df_right: {missing_right}")

    missing_left_f = [c for c in fuzzy_matching if c not in df_left.columns]
    missing_right_f = [c for c in fuzzy_matching.values() if c not in df_right.columns]
    if missing_left_f:
        raise KeyError(f"Columns missing from df_left (fuzzy): {missing_left_f}")
    if missing_right_f:
        raise KeyError(f"Columns missing from df_right (fuzzy): {missing_right_f}")

    def _is_null(x) -> bool:
        return bool(pd.isna(x))

    def _normalize_obj(x):
        if _is_null(x):
            return None
        if isinstance(x, str):
            return x.lower().strip()
        return x

    def _is_numeric_scalar(x) -> bool:
        if isinstance(x, (bool, np.bool_)):
            return False
        return isinstance(x, (int, float, np.integer, np.floating)) and not _is_null(x)

    def _strict_equal(v_left, v_right) -> bool:
        if _is_null(v_left) and _is_null(v_right):
            return True
        if _is_null(v_left) or _is_null(v_right):
            return False
        if _is_numeric_scalar(v_left) and _is_numeric_scalar(v_right):
            return bool(np.isclose(float(v_left), float(v_right), atol=float_atol, rtol=float_rtol))
        return _normalize_obj(v_left) == _normalize_obj(v_right)

    def _fuzzy_score(v_left, v_right) -> Optional[float]:
        if _is_null(v_left) or _is_null(v_right):
            return None
        s_left = _normalize_obj(v_left)
        s_right = _normalize_obj(v_right)
        if not isinstance(s_left, str) or not isinstance(s_right, str):
            return 1.0 if s_left == s_right else 0.0
        return float(fuzz.ratio(s_left, s_right)) / 100.0

    edges: List[Tuple[int, int]] = []
    edge_weights: List[float] = []

    strict_items = list(strict_matching.items())
    fuzzy_items = list(fuzzy_matching.items())

    for i, row_l in df_left.iterrows():
        for j, row_r in df_right.iterrows():
            if not all(_strict_equal(row_l[c_l], row_r[c_r]) for c_l, c_r in strict_items):
                continue

            if not fuzzy_items:
                score = 1.0
            else:
                scores = [
                    s for c_l, c_r in fuzzy_items
                    if (s := _fuzzy_score(row_l[c_l], row_r[c_r])) is not None
                ]
                if not scores:
                    continue
                score = float(np.mean(scores))

            if score < fuzzy_threshold:
                continue

            edges.append((int(i), int(j)))
            edge_weights.append(float(score))

    if not edges:
        return [], edges, edge_weights

    G = nx.Graph()
    G.add_edges_from(
        [(f"L_{i}", f"R_{j}", {"weight": w}) for (i, j), w in zip(edges, edge_weights)]
    )

    matching_nodes = nx.algorithms.matching.max_weight_matching(G, maxcardinality=False)

    matching: List[Tuple[int, int]] = []
    for u, v in matching_nodes:
        if u.startswith("L_"):
            matching.append((int(u[2:]), int(v[2:])))
        else:
            matching.append((int(v[2:]), int(u[2:])))

    matching.sort()
    return matching, edges, edge_weights
