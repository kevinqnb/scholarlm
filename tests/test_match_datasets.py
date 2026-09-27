"""Unit tests for ``scholarlm.utils.data.match_datasets``.

Fixtures are hand-built so expected edges/weights can be verified by inspection.
Covers the null-handling behavior added 2026-09-26: strict null == null already
matched before this change; what's new here is (1) an all-null-fuzzy pair no longer
drops its candidate edge but gets weight == fuzzy_threshold, and (2) empty/whitespace
strings are treated as null, same as None/NaN.
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


def test_all_null_fuzzy_defaults_to_threshold_instead_of_dropping_edge():
    left = pd.DataFrame([{"id": "a", "name": None, "prop": None}])
    right = pd.DataFrame([{"id": "a", "name": np.nan, "prop": ""}])
    matching, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.3,
    )
    assert edges == [(0, 0)]
    assert weights == [0.3]
    assert matching == [(0, 0)]


def test_all_null_fuzzy_edge_included_at_any_threshold():
    left = pd.DataFrame([{"id": "a", "name": None}])
    right = pd.DataFrame([{"id": "a", "name": None}])
    for threshold in (0.0, 0.5, 1.0):
        matching, edges, weights = match_datasets(
            left, right,
            strict_matching={"id": "id"},
            fuzzy_matching={"name": "name"},
            fuzzy_threshold=threshold,
        )
        assert edges == [(0, 0)]
        assert weights == [threshold]


def test_no_evidence_edge_never_outranks_a_real_scored_edge():
    # L0 strict-matches both R0 (no fuzzy evidence) and R1 (real fuzzy score 0.9+).
    # Only one can be picked (bipartite 1-1); it must be R1.
    left = pd.DataFrame([{"id": "a", "name": None}])
    right = pd.DataFrame([
        {"id": "a", "name": None},       # no fuzzy evidence -> weight == threshold
        {"id": "a", "name": "widget"},   # widget vs widget -> ratio 1.0
    ])
    left2 = pd.DataFrame([{"id": "a", "name": "widget"}])
    matching, edges, weights = match_datasets(
        left2, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name"},
        fuzzy_threshold=0.3,
    )
    assert set(edges) == {(0, 0), (0, 1)}
    weight_by_edge = dict(zip(edges, weights))
    assert weight_by_edge[(0, 0)] == 0.3  # left2's name is non-null, right[0]'s is null -> no evidence
    assert weight_by_edge[(0, 1)] == 1.0
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


def test_empty_string_fuzzy_field_excluded_from_mean_not_scored_as_string():
    # prop is "" on the left (== null) so it's excluded; name carries the whole score.
    left = pd.DataFrame([{"id": "a", "name": "widget", "prop": ""}])
    right = pd.DataFrame([{"id": "a", "name": "widget", "prop": "gadget"}])
    matching, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.0,
    )
    assert edges == [(0, 0)]
    assert weights == [1.0]  # only `name` (exact match) counted, `prop` excluded
