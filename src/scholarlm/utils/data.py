import json
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union

import networkx as nx
import numpy as np
import pandas as pd
from rapidfuzz import fuzz


def _try_coerce_value(value: Any, target_type: Type[Any]) -> Any:
    """Best-effort coercion of a single value to `target_type`.

    If coercion fails, returns the original `value`.
    Only performs very light cleaning intended for PDF-extracted numerics.
    """
    if value is None:
        return value

    if isinstance(value, target_type):
        return value

    try:
        if target_type is str:
            return str(value)

        if target_type in (int, float):
            if isinstance(value, str):
                s = value.strip().replace(",", "").replace("−", "-")
                if target_type is int:
                    return int(float(s))
                return float(s)
            if target_type is int:
                return int(value)
            return float(value)

        return target_type(value)
    except Exception:
        return value


def load_and_process_results(
    json_path: str,
    *,
    unit_conversion_table: Mapping[str, Mapping[str, Any]],
    attribute_types: Optional[Mapping[Any, Type[Any]]] = None,
    drop_keys: Optional[Iterable[str]] = None,
    drop_attrs: Optional[Iterable[Any]] = None,
    attribute_col: str = "attribute",
    value_col: str = "value",
    unit_col: str = "units",
    out_col: str = "processed_value",
) -> pd.DataFrame:
    """Load experiment results JSON, deduplicate, optionally coerce value types, and apply unit conversions.

    Parameters
    ----------
    json_path:
        Path to a JSON file containing a list of dict records.
    unit_conversion_table:
        Mapping like: unit_conversion_table[attribute][unit] -> multiplicative factor.
    attribute_types:
        Optional mapping: attribute_types[attribute] -> Python type (e.g., int/float/str).
        Coercion is attempted but not enforced.
    drop_keys:
        Optional iterable of keys to drop from each record before creating the DataFrame.
    drop_attrs:
        Optional iterable of attribute values to drop.
    attribute_col:
        Name of the column containing attribute names.
    value_col:
        Name of the column containing the values to be processed.
    unit_col:
        Name of the column containing unit names.
    out_col:
        Name of the new column for processed values after coercion and unit conversion.

    Returns
    -------
    A processed DataFrame with an additional `out_col` column.
    """
    with open(json_path, "r") as f:
        records: List[Dict[str, Any]] = json.load(f)

    if drop_keys:
        drop_set = set(drop_keys)
        records = [{k: v for k, v in r.items() if k not in drop_set} for r in records]
    if drop_attrs:
        drop_attr_set = set(drop_attrs)
        records = [r for r in records if r.get(attribute_col) not in drop_attr_set]

    df = pd.DataFrame(records)
    df = df.dropna(subset=[value_col])

    if attribute_types and attribute_col in df.columns and value_col in df.columns:
        coerced_values: List[Any] = []
        for _, row in df.iterrows():
            attr = row.get(attribute_col)
            val = row.get(value_col)
            target_type = attribute_types.get(attr)
            coerced_values.append(val if target_type is None else _try_coerce_value(val, target_type))
        df[value_col] = coerced_values

    processed: List[Any] = []
    for _, row in df.iterrows():
        attr = row.get(attribute_col)
        unit = row.get(unit_col)
        val = row.get(value_col)

        attr_table = unit_conversion_table.get(attr, {}) if attr is not None else {}
        factor = attr_table.get(unit) if isinstance(attr_table, Mapping) else None

        if factor is None:
            processed.append(val)
        else:
            try:
                processed.append(val * factor)
            except Exception:
                processed.append(val)

    df[out_col] = processed
    df = df.dropna(subset=[out_col])
    df = df.reset_index(drop=True)
    return df


def match_datasets(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    strict_matching: Dict[str, str],
    fuzzy_matching: Optional[Dict[str, str]] = None,
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


def matching_precision_recall(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    *,
    strict_matching: Dict[str, str],
    fuzzy_matching: Optional[Dict[str, str]] = None,
    **match_kwargs,
) -> Tuple[float, float]:
    """Estimate recall/precision under the matching rules.

    Recall is computed relative to df_left, precision relative to df_right.
    """
    total_left = int(df_left.shape[0])
    total_right = int(df_right.shape[0])

    matching, _edges, _weights = match_datasets(
        df_left,
        df_right,
        strict_matching=strict_matching,
        fuzzy_matching=fuzzy_matching,
        **match_kwargs,
    )

    tp = len(matching)
    precision = tp / total_right if total_right > 0 else 0.0
    recall = tp / total_left if total_left > 0 else 0.0

    return recall, precision
