"""Unit tests for ``scholarlm.utils.data.match_datasets``.

Fixtures are hand-built so expected edges/weights can be verified by inspection.
Null semantics (strict AND fuzzy): null == null agrees (fuzzy score 1.0); null vs
non-null never matches (edge rejected outright, not averaged away). Empty/whitespace
strings are null, same as None/NaN. Supersedes the 2026-09-26 behavior (commit
150aba2) where one-sided fuzzy nulls were excluded from the mean and an all-null
pair got weight == fuzzy_threshold.
"""
import numpy as np
import pandas as pd

from scholarlm.utils.data import match_datasets


def test_strict_null_equals_null():
    left = pd.DataFrame([{"id": "a", "units": None}])
    right = pd.DataFrame([{"id": "a", "units": np.nan}])
    matching, edges, weights = match_datasets(
        left, right, strict_matching={"id": "id", "units": "units"}
    )
    assert matching == [(0, 0)]


def test_strict_one_sided_null_does_not_match():
    left = pd.DataFrame([{"id": "a", "units": "mg/L"}])
    right = pd.DataFrame([{"id": "a", "units": None}])
    matching, edges, weights = match_datasets(
        left, right, strict_matching={"id": "id", "units": "units"}
    )
    assert matching == []
    assert edges == []


def test_fuzzy_null_equals_null_scores_one():
    # Both sides null on every fuzzy field -> agreement (1.0), at any threshold.
    left = pd.DataFrame([{"id": "a", "name": None, "prop": None}])
    right = pd.DataFrame([{"id": "a", "name": np.nan, "prop": ""}])
    for threshold in (0.0, 0.5, 1.0):
        matching, edges, weights = match_datasets(
            left, right,
            strict_matching={"id": "id"},
            fuzzy_matching={"name": "name", "prop": "prop"},
            fuzzy_threshold=threshold,
        )
        assert edges == [(0, 0)]
        assert weights == [1.0]
        assert matching == [(0, 0)]


def test_fuzzy_one_sided_null_never_matches():
    # null vs non-null on a fuzzy field rejects the edge, in either direction,
    # even at threshold 0.0.
    named = pd.DataFrame([{"id": "a", "name": "widget"}])
    unnamed = pd.DataFrame([{"id": "a", "name": None}])
    for left, right in ((named, unnamed), (unnamed, named)):
        matching, edges, weights = match_datasets(
            left, right,
            strict_matching={"id": "id"},
            fuzzy_matching={"name": "name"},
            fuzzy_threshold=0.0,
        )
        assert matching == [] and edges == [] and weights == []


def test_fuzzy_one_sided_null_not_rescued_by_other_fuzzy_field():
    # name agrees perfectly but prop is null on one side only: edge rejected,
    # not averaged down to 0.5 and kept.
    left = pd.DataFrame([{"id": "a", "name": "widget", "prop": None}])
    right = pd.DataFrame([{"id": "a", "name": "widget", "prop": "gadget"}])
    matching, edges, _ = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.0,
    )
    assert edges == []


def test_fuzzy_null_null_field_counts_as_one_in_the_mean():
    # prop null on both sides scores 1.0; name "abcd" vs "abxd" = ratio 0.75.
    left = pd.DataFrame([{"id": "a", "name": "abcd", "prop": None}])
    right = pd.DataFrame([{"id": "a", "name": "abxd", "prop": np.nan}])
    _, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.0,
    )
    assert edges == [(0, 0)]
    assert weights == [0.875]


def test_null_null_edge_ties_with_exact_named_edge_but_never_beats_it():
    # L0 (named) matches R0 only if R0 is also named; null R1 is rejected.
    left = pd.DataFrame([{"id": "a", "name": "widget"}])
    right = pd.DataFrame([
        {"id": "a", "name": None},       # one-sided null -> no edge
        {"id": "a", "name": "widget"},
    ])
    matching, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name"},
        fuzzy_threshold=0.3,
    )
    assert edges == [(0, 1)]
    assert weights == [1.0]
    assert matching == [(0, 1)]


def test_empty_string_treated_as_null_in_strict_field():
    left = pd.DataFrame([{"id": "a", "units": ""}])
    right = pd.DataFrame([{"id": "a", "units": "  "}])
    matching, _, _ = match_datasets(left, right, strict_matching={"id": "id", "units": "units"})
    assert matching == [(0, 0)]


def test_empty_string_vs_real_value_does_not_strict_match():
    left = pd.DataFrame([{"id": "a", "units": ""}])
    right = pd.DataFrame([{"id": "a", "units": "mg/L"}])
    matching, edges, _ = match_datasets(left, right, strict_matching={"id": "id", "units": "units"})
    assert matching == []
    assert edges == []


def test_empty_string_fuzzy_field_is_null_not_scored_as_string():
    # "" is null; "" vs "gadget" is one-sided null -> edge rejected.
    left = pd.DataFrame([{"id": "a", "name": "widget", "prop": ""}])
    right = pd.DataFrame([{"id": "a", "name": "widget", "prop": "gadget"}])
    _, edges, _ = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.0,
    )
    assert edges == []
