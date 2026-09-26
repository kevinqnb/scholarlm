"""Strict + fuzzy deduplication of extracted measurement records.

Pairwise duplicate semantics deliberately mirror ``scholarlm.utils.data.match_datasets``
(the ground-truth matcher): two records are duplicates under the same pairwise rule
that makes a ``match_datasets`` candidate edge. (That does not imply both would match
the same ground-truth row -- A~B and B~GT does not give A~GT.) The helpers below are reimplemented rather
than imported because ``match_datasets`` keeps them as closures, and refactoring it
would be an eval-logic change. ``tests/test_deduplication.py`` checks parity against
``match_datasets`` directly.

When every fuzzy field is null on at least one side of a pair, both here and in
``match_datasets`` the pair counts as a match if the strict fields agree (no fuzzy
evidence either way) -- ``match_datasets`` represents that case with an edge weight
of ``fuzzy_threshold`` (the minimum passing score) rather than dropping the edge, as
it used to before 2026-09-26.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

# Same tolerances as match_datasets.
_FLOAT_ATOL = 1e-3
_FLOAT_RTOL = 0.0


def _is_null(x) -> bool:
    if isinstance(x, str) and not x.strip():
        return True
    return x is None or (isinstance(x, (float, np.floating)) and np.isnan(x))


def _is_numeric_scalar(x) -> bool:
    if isinstance(x, (bool, np.bool_)):
        return False
    return isinstance(x, (int, float, np.integer, np.floating)) and not _is_null(x)


def _normalize_obj(x):
    if _is_null(x):
        return None
    if isinstance(x, str):
        return x.lower().strip()
    return x


def _strict_equal(a, b) -> bool:
    if _is_null(a) and _is_null(b):
        return True
    if _is_null(a) or _is_null(b):
        return False
    if _is_numeric_scalar(a) and _is_numeric_scalar(b):
        return bool(np.isclose(float(a), float(b), atol=_FLOAT_ATOL, rtol=_FLOAT_RTOL))
    return _normalize_obj(a) == _normalize_obj(b)


def _fuzzy_score(a, b) -> Optional[float]:
    if _is_null(a) or _is_null(b):
        return None
    s_a = _normalize_obj(a)
    s_b = _normalize_obj(b)
    if not isinstance(s_a, str) or not isinstance(s_b, str):
        return 1.0 if s_a == s_b else 0.0
    return float(fuzz.ratio(s_a, s_b)) / 100.0


def pair_score(
    row_a: pd.Series,
    row_b: pd.Series,
    *,
    strict_fields: Sequence[str],
    fuzzy_fields: Sequence[str],
) -> Tuple[bool, Optional[float]]:
    """Score one pair of records.

    Returns ``(strict_ok, fuzzy_score)``. ``fuzzy_score`` is the mean fuzzy ratio in
    [0, 1] over the fuzzy fields that are non-null on both sides, or ``None`` if there
    is no such field (or if ``strict_ok`` is False, in which case it is not computed).
    """
    if not all(_strict_equal(row_a[c], row_b[c]) for c in strict_fields):
        return False, None
    scores = [s for c in fuzzy_fields if (s := _fuzzy_score(row_a[c], row_b[c])) is not None]
    if not scores:
        return True, None
    return True, float(np.mean(scores))


def _is_duplicate(strict_ok: bool, score: Optional[float], fuzzy_threshold: float) -> bool:
    if not strict_ok:
        return False
    if score is None:  # all fuzzy fields null on one side: strict match alone decides
        return True
    return score >= fuzzy_threshold


def _block_key(row: pd.Series, strict_fields: Sequence[str]) -> tuple:
    """Hashable key such that rows in different blocks can never be strict-equal.

    Numerics all share one bucket per field (np.isclose isn't hashable), so the full
    pairwise strict check still runs inside each block.
    """
    key = []
    for c in strict_fields:
        v = row[c]
        if _is_null(v):
            key.append(("null",))
        elif _is_numeric_scalar(v):
            key.append(("num",))
        else:
            key.append(("str", _normalize_obj(v)))
    return tuple(key)


def _validate(
    df: pd.DataFrame,
    strict_fields: List[str],
    fuzzy_fields: List[str],
    fuzzy_threshold: float,
    provenance_fields: List[str],
) -> None:
    if len(strict_fields) == 0:
        raise ValueError("strict_fields must be non-empty")
    if len(fuzzy_fields) == 0:
        raise ValueError("fuzzy_fields must be non-empty (use DataFrame.drop_duplicates for strict-only)")
    for name, fields in (("strict", strict_fields), ("fuzzy", fuzzy_fields), ("provenance", provenance_fields)):
        if len(set(fields)) != len(fields):
            raise ValueError(f"Duplicate column names in {name}_fields: {fields}")
        missing = [c for c in fields if c not in df.columns]
        if missing:
            raise KeyError(f"Columns missing from df ({name}): {missing}")
    overlap = (set(strict_fields) | set(fuzzy_fields)) & set(provenance_fields)
    if overlap:
        raise ValueError(f"Columns cannot be both a match field and a provenance field: {sorted(overlap)}")
    if set(strict_fields) & set(fuzzy_fields):
        raise ValueError(f"Columns cannot be both strict and fuzzy: {sorted(set(strict_fields) & set(fuzzy_fields))}")
    if isinstance(fuzzy_threshold, bool) or not isinstance(fuzzy_threshold, (int, float)):
        raise TypeError(f"fuzzy_threshold must be a number, got {type(fuzzy_threshold).__name__}")
    if not 0.0 <= fuzzy_threshold <= 1.0:
        raise ValueError(f"fuzzy_threshold must be in [0, 1], got {fuzzy_threshold}")
    if not df.index.is_unique:
        raise ValueError("df.index must be unique (labels are used in the duplicate audit)")

    # match_datasets' comparison helpers are only well-defined on scalars: pd.isna on a
    # list is ambiguous, and bools compare equal to 1/0 through the object path.
    for c in list(strict_fields) + list(fuzzy_fields):
        bad = df[c].map(
            lambda v: not (_is_null(v) or isinstance(v, str) or _is_numeric_scalar(v))
        )
        if bad.any():
            examples = df.loc[bad, c].head(3).tolist()
            raise TypeError(
                f"Column {c!r} has {int(bad.sum())} non-scalar/bool values "
                f"(only str, int, float, None/NaN allowed); e.g. {examples}"
            )

    for c in provenance_fields:
        not_list = df[c].map(lambda v: not isinstance(v, list))
        if not_list.any():
            raise TypeError(
                f"Provenance column {c!r} has {int(not_list.sum())} non-list values; "
                f"e.g. {df.loc[not_list, c].head(3).tolist()}"
            )
    if provenance_fields:
        lengths = df[provenance_fields].map(len)
        misaligned = lengths.nunique(axis=1) != 1
        if misaligned.any():
            raise ValueError(
                f"{int(misaligned.sum())} rows have provenance lists of unequal length "
                f"across {provenance_fields}; first offending labels: "
                f"{df.index[misaligned][:5].tolist()}"
            )


def deduplicate_records(
    df: pd.DataFrame,
    *,
    strict_fields: List[str],
    fuzzy_fields: List[str],
    fuzzy_threshold: float,
    provenance_fields: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop duplicate records, merging the dropped rows' provenance into the kept row.

    Pair rule (mirrors ``match_datasets`` edge semantics): rows ``a`` and ``b`` are
    duplicates iff every strict field is equal (null == null; numerics via
    ``np.isclose(atol=1e-3, rtol=0)``; strings case-insensitive and stripped; a
    numeric never equals a string) AND the mean ``rapidfuzz.fuzz.ratio`` / 100 over
    fuzzy fields non-null on both sides is ``>= fuzzy_threshold``. If no fuzzy field
    is non-null on both sides, the strict match alone makes them duplicates (matches
    ``match_datasets``, which assigns that case an edge weight of ``fuzzy_threshold``).

    Grouping rule: the pair relation isn't transitive, so rows are walked in
    ``df`` order and each row is compared only against rows already *kept*. A row that
    is a duplicate of one or more kept rows is dropped and assigned to the kept row
    with the highest fuzzy score (a pair with no fuzzy evidence ranks below any scored
    pair), ties broken by earliest kept row. There is no chaining: with A~B, B~C and
    A!~C, A is kept, B is merged into A, and C is kept. Consequently the result
    depends on row order.

    Leaving ``document_id`` out of ``strict_fields`` allows merges across documents.
    Normalize values (e.g. ``""`` vs ``None`` units, unit conversion) *before* calling;
    this function compares exactly what it is given.

    Parameters
    ----------
    df:
        Records to deduplicate. Index must be unique; it is preserved.
    strict_fields:
        Non-empty list of columns that must be strictly equal.
    fuzzy_fields:
        Non-empty list of columns compared by fuzzy ratio.
    fuzzy_threshold:
        Minimum mean fuzzy score in [0, 1].
    provenance_fields:
        List-valued columns (aligned per row, e.g. page_number/.../context). For each
        dropped row, its lists are appended onto the kept row's lists, in ``df`` order.
        Pass ``[]`` to merge nothing.

    Returns
    -------
    (deduped, duplicates)
        deduped: the kept rows, in original order with original index labels, with
            merged provenance lists.
        duplicates: one row per dropped record, columns ``dropped_label``,
            ``kept_label``, ``score`` (NaN when no fuzzy field was comparable).
    """
    _validate(df, strict_fields, fuzzy_fields, fuzzy_threshold, provenance_fields)

    blocks: Dict[tuple, List] = {}  # block key -> kept labels, in order
    kept_labels: List = []
    duplicate_rows: List[Tuple] = []
    merged_into: Dict = {}  # kept label -> list of dropped labels, in order

    for label, row in df.iterrows():
        kept_in_block = blocks.setdefault(_block_key(row, strict_fields), [])
        best_label, best_rank, best_score = None, None, None
        for k_label in kept_in_block:
            strict_ok, score = pair_score(
                df.loc[k_label], row, strict_fields=strict_fields, fuzzy_fields=fuzzy_fields
            )
            if not _is_duplicate(strict_ok, score, fuzzy_threshold):
                continue
            # scored pairs outrank unscored ones; strict '>' keeps the earliest on ties
            rank = (1, score) if score is not None else (0, 0.0)
            if best_rank is None or rank > best_rank:
                best_label, best_rank, best_score = k_label, rank, score
        if best_label is None:
            kept_in_block.append(label)
            kept_labels.append(label)
        else:
            duplicate_rows.append((label, best_label, np.nan if best_score is None else best_score))
            merged_into.setdefault(best_label, []).append(label)

    deduped = df.loc[kept_labels].copy()
    for c in provenance_fields:
        merged_col = []
        for k_label in kept_labels:
            merged = list(df.at[k_label, c])
            for d_label in merged_into.get(k_label, []):
                merged.extend(df.at[d_label, c])
            merged_col.append(merged)
        deduped[c] = pd.Series(merged_col, index=deduped.index, dtype=object)

    duplicates = pd.DataFrame(duplicate_rows, columns=["dropped_label", "kept_label", "score"])

    # Every input row is accounted for exactly once.
    assert len(deduped) + len(duplicates) == len(df)
    assert set(kept_labels).isdisjoint(duplicates["dropped_label"])
    assert set(kept_labels) | set(duplicates["dropped_label"]) == set(df.index)
    assert set(duplicates["kept_label"]) <= set(kept_labels)
    # Provenance entries are conserved and stay aligned.
    for c in provenance_fields:
        assert deduped[c].map(len).sum() == df[c].map(len).sum(), c
    if provenance_fields:
        assert (deduped[provenance_fields].map(len).nunique(axis=1) == 1).all()

    return deduped, duplicates
