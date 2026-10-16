"""Unit tests for ``scholarlm.utils.data.match_datasets``.

Fixtures are hand-built so expected edges/weights can be verified by inspection.
Null semantics (2026-10-03). Strict fields: null == null agrees, null vs non-null never
matches. Fuzzy fields: null on both sides abstains (the field is left out of the mean);
null on exactly one side scores 0.0 and stays in the mean (the threshold, not an outright
rejection, decides); if every fuzzy field abstains the edge weight is 1.0. Empty/whitespace
strings are null, same as None/NaN. Threshold compare is inclusive (score >= threshold
keeps the edge). Supersedes the 2026-09-30 behavior (both-null scored 1.0 in the mean,
one-sided null rejected the edge outright).
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


def test_fuzzy_all_fields_null_on_both_sides_weight_one():
    # Every fuzzy field abstains -> nothing to average -> weight 1.0 at any threshold.
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


def test_fuzzy_one_sided_null_scores_zero_not_rejected():
    # null vs non-null on the only fuzzy field scores 0.0, either direction: the edge
    # exists at threshold 0.0 (weight 0.0) and is dropped by any threshold > 0.
    named = pd.DataFrame([{"id": "a", "name": "widget"}])
    unnamed = pd.DataFrame([{"id": "a", "name": None}])
    for left, right in ((named, unnamed), (unnamed, named)):
        _, edges, weights = match_datasets(
            left, right,
            strict_matching={"id": "id"},
            fuzzy_matching={"name": "name"},
            fuzzy_threshold=0.0,
        )
        assert edges == [(0, 0)] and weights == [0.0]
        _, edges, weights = match_datasets(
            left, right,
            strict_matching={"id": "id"},
            fuzzy_matching={"name": "name"},
            fuzzy_threshold=0.1,
        )
        assert edges == [] and weights == []


def test_fuzzy_one_sided_null_is_averaged_in_as_zero():
    # name agrees perfectly (1.0), prop null on one side only (0.0): mean 0.5.
    left = pd.DataFrame([{"id": "a", "name": "widget", "prop": None}])
    right = pd.DataFrame([{"id": "a", "name": "widget", "prop": "gadget"}])
    _, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.0,
    )
    assert edges == [(0, 0)]
    assert weights == [0.5]


def test_fuzzy_both_null_field_is_left_out_of_the_mean():
    # prop null on both sides abstains; name "abcd" vs "abxd" = ratio 0.75 alone.
    # (Under the pre-2026-10-03 rule this was (0.75 + 1.0) / 2 = 0.875.)
    left = pd.DataFrame([{"id": "a", "name": "abcd", "prop": None}])
    right = pd.DataFrame([{"id": "a", "name": "abxd", "prop": np.nan}])
    _, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.0,
    )
    assert edges == [(0, 0)]
    assert weights == [0.75]


def test_fuzzy_mixed_abstain_zero_and_scored_fields():
    # a: both null (abstains), b: one-sided null (0.0), c: "abcd" vs "abxd" (0.75).
    # mean over the two scored fields = 0.375.
    left = pd.DataFrame([{"id": "a", "fa": None, "fb": None, "fc": "abcd"}])
    right = pd.DataFrame([{"id": "a", "fa": "", "fb": "x", "fc": "abxd"}])
    _, _, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"fa": "fa", "fb": "fb", "fc": "fc"},
        fuzzy_threshold=0.0,
    )
    assert weights == [0.375]


def test_fuzzy_threshold_is_inclusive():
    # weight is exactly 0.75; threshold 0.75 keeps the edge, 0.76 drops it.
    left = pd.DataFrame([{"id": "a", "name": "abcd"}])
    right = pd.DataFrame([{"id": "a", "name": "abxd"}])
    kw = dict(strict_matching={"id": "id"}, fuzzy_matching={"name": "name"})
    _, edges, weights = match_datasets(left, right, fuzzy_threshold=0.75, **kw)
    assert edges == [(0, 0)] and weights == [0.75]
    _, edges, _ = match_datasets(left, right, fuzzy_threshold=0.76, **kw)
    assert edges == []


def test_fuzzy_null_named_edge_loses_to_exact_named_edge():
    # L0 (named) vs R0 (null name): weight 0.0; vs R1 (same name): weight 1.0.
    left = pd.DataFrame([{"id": "a", "name": "widget"}])
    right = pd.DataFrame([
        {"id": "a", "name": None},
        {"id": "a", "name": "widget"},
    ])
    matching, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name"},
        fuzzy_threshold=0.0,
    )
    assert edges == [(0, 0), (0, 1)]
    assert weights == [0.0, 1.0]
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
    # "" is null; "" vs "gadget" is one-sided null -> 0.0, averaged with name's 1.0.
    left = pd.DataFrame([{"id": "a", "name": "widget", "prop": ""}])
    right = pd.DataFrame([{"id": "a", "name": "widget", "prop": "gadget"}])
    _, edges, weights = match_datasets(
        left, right,
        strict_matching={"id": "id"},
        fuzzy_matching={"name": "name", "prop": "prop"},
        fuzzy_threshold=0.0,
    )
    assert edges == [(0, 0)] and weights == [0.5]


def test_normalizer_that_empties_a_string_makes_it_null():
    # Nulls are judged after the normalizer. "!!!" -> "" is null: empty on both sides
    # abstains (weight 1.0 with a single field); empty on one side only scores 0.0.
    strip_punct = lambda s: "".join(ch for ch in s if ch.isalnum())
    kw = dict(strict_matching={"id": "id"}, fuzzy_matching={"name": "name"},
              fuzzy_normalizers={"name": strip_punct}, fuzzy_threshold=0.0)
    _, _, weights = match_datasets(
        pd.DataFrame([{"id": "a", "name": "!!!"}]), pd.DataFrame([{"id": "a", "name": "??"}]), **kw)
    assert weights == [1.0]
    _, _, weights = match_datasets(
        pd.DataFrame([{"id": "a", "name": "!!!"}]), pd.DataFrame([{"id": "a", "name": "abc"}]), **kw)
    assert weights == [0.0]


def test_strict_nan_from_coercion_matches_null_on_other_side():
    # Documents match_cache.py's numeric_coerce caveat: an unparseable value coerced to
    # NaN strict-matches a null (but never a real value) on the other side.
    kw = dict(strict_matching={"id": "id", "v": "v"})
    m, _, _ = match_datasets(pd.DataFrame([{"id": "a", "v": np.nan}]),
                             pd.DataFrame([{"id": "a", "v": np.nan}]), **kw)
    assert m == [(0, 0)]
    m, e, _ = match_datasets(pd.DataFrame([{"id": "a", "v": 1.0}]),
                             pd.DataFrame([{"id": "a", "v": np.nan}]), **kw)
    assert m == [] and e == []
