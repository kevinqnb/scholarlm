from typing import Any, Callable, Dict, List, Optional, Tuple

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
    fuzzy_normalizers: Optional[Dict[str, Callable[[str], str]]] = None,
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
        strictly equal. Numeric values are compared with np.isclose. Nulls (None, NaN,
        pd.NA, and empty/whitespace-only strings): null on BOTH sides is equal; null on
        exactly one side is not equal, so the candidate pair is dropped. A NaN produced
        by upstream coercion (e.g. analysis/match_cache.py's numeric_coerce turning an
        unparseable string into NaN) is indistinguishable from a real null here, so it
        strict-matches a null on the other side.
    fuzzy_matching:
        Mapping from column name in df_left -> column name in df_right compared with
        fuzzy ratios, averaged to produce an edge weight in [0, 1]. Per field:
        null on BOTH sides -> the field abstains: it is not scored and does not enter
        the average at all; null on exactly one side -> the field scores 0.0 and IS
        included in the average (the edge is not rejected; the threshold decides).
        If every fuzzy field abstains (null on both sides for all of them) there is
        nothing to average and the edge weight is 1.0 (vacuous agreement; strict
        fields alone decided the pair). Nulls are judged after ``fuzzy_normalizers``
        runs, so a string a normalizer reduces to empty counts as null.
    fuzzy_threshold:
        Minimum average fuzzy score in [0, 1] for a candidate edge to be included in the
        graph: edges with score >= fuzzy_threshold are kept (inclusive). Callers that
        re-apply a threshold to the returned weights must also use >= (see
        analysis/match_cache.py's edges_above_threshold). Defaults to 0.0 to include
        all edges that pass strict criteria.

    fuzzy_normalizers:
        Optional mapping from a df_left fuzzy column -> str-to-str canonicaliser,
        applied to both sides' non-null values of that field before scoring
        (e.g. ``scholarlm.utils.normalization.canonical_formula_name``). Keys must
        be keys of ``fuzzy_matching``. Non-string values are not passed to it.

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
    fuzzy_normalizers = fuzzy_normalizers or {}
    unknown_norm = [c for c in fuzzy_normalizers if c not in fuzzy_matching]
    if unknown_norm:
        raise KeyError(f"fuzzy_normalizers keys not in fuzzy_matching: {unknown_norm}")

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
        if isinstance(x, str) and not x.strip():
            return True
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

    def _fuzzy_prep(v, normalizer):
        """Normalized comparison value for one side, or None if null (judged after
        the normalizer, so a string it empties out counts as null)."""
        if _is_null(v):
            return None
        if normalizer is not None and isinstance(v, str):
            v = normalizer(v)
        return _normalize_obj(v)

    def _fuzzy_score(v_left, v_right, normalizer=None) -> Optional[float]:
        """None means both sides are null: the field abstains and is left out of the
        average. Exactly one null side scores 0.0 (and is averaged in)."""
        s_left = _fuzzy_prep(v_left, normalizer)
        s_right = _fuzzy_prep(v_right, normalizer)
        if s_left is None and s_right is None:
            return None
        if s_left is None or s_right is None:
            return 0.0
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
                    _fuzzy_score(row_l[c_l], row_r[c_r], fuzzy_normalizers.get(c_l))
                    for c_l, c_r in fuzzy_items
                ]
                scored = [x for x in scores if x is not None]
                # All fields abstained (null on both sides everywhere): no evidence
                # either way, strict fields alone decided -> weight 1.0.
                score = float(np.mean(scored)) if scored else 1.0

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
