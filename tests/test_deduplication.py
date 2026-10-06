"""Unit tests for ``scholarlm.utils.deduplication.deduplicate_records``.

Fixtures are hand-built so the expected kept/dropped sets can be read off by
inspection; fuzzy ratios the expectations depend on are asserted as preconditions.
Pairwise semantics are also checked for parity against ``match_datasets`` (the
ground-truth matcher), including its null rules: a fuzzy field null on both sides
abstains from the mean, one null on exactly one side scores 0.0 and is averaged in, and
an all-abstaining pair scores 1.0.
"""
import numpy as np
import pandas as pd
import pytest
from rapidfuzz import fuzz

from scholarlm.utils.data import match_datasets
from scholarlm.utils.deduplication import deduplicate_records, pair_score

STRICT = ["document_id", "attribute", "value", "units"]
FUZZY = ["name", "ecosystem"]
PROV = ["page_number", "context"]


def _ratio(a, b):
    return fuzz.ratio(a, b) / 100.0


def _rec(name, *, doc="d1", attr="ph", value=7.0, units=None, eco="pond", page=1, ctx=None):
    return {
        "document_id": doc, "attribute": attr, "value": value, "units": units,
        "name": name, "ecosystem": eco,
        "page_number": [page], "context": [ctx if ctx is not None else f"ctx-{name}-{page}"],
    }


def _dedup(df, threshold, fuzzy=("name",), prov=PROV):
    return deduplicate_records(
        df, strict_fields=STRICT, fuzzy_fields=list(fuzzy),
        fuzzy_threshold=threshold, provenance_fields=list(prov),
    )


# ── Grouping rule ──────────────────────────────────────────────────────────────

def test_no_chaining_compares_only_against_kept_rows():
    # A~B (0.75), B~C (0.75), A!~C (0.5) at threshold 0.7.
    assert _ratio("aaaa", "aaab") == 0.75
    assert _ratio("aaab", "aabb") == 0.75
    assert _ratio("aaaa", "aabb") == 0.5
    df = pd.DataFrame([_rec("aaaa"), _rec("aaab"), _rec("aabb")])
    kept, dups = _dedup(df, 0.7)
    assert kept.index.tolist() == [0, 2]  # C survives: B was dropped, so no A-B-C chain
    assert dups[["dropped_label", "kept_label"]].values.tolist() == [[1, 0]]
    assert dups.score.tolist() == [0.75]


def test_threshold_is_inclusive():
    df = pd.DataFrame([_rec("aaaa"), _rec("aaab")])
    assert len(_dedup(df, 0.75)[0]) == 1
    assert len(_dedup(df, 0.7501)[0]) == 2


def test_assigns_to_highest_scoring_kept_row():
    # X="aaaaaa", Y="aaabbb" not dups of each other (0.5); D closer to Y (0.83) than X (0.67).
    assert _ratio("aaaaaa", "aaabbb") == 0.5
    assert _ratio("aaaaaa", "aaaabb") == pytest.approx(2 / 3)
    assert _ratio("aaabbb", "aaaabb") == pytest.approx(5 / 6)
    df = pd.DataFrame([_rec("aaaaaa"), _rec("aaabbb"), _rec("aaaabb")])
    kept, dups = _dedup(df, 0.6)
    assert kept.index.tolist() == [0, 1]
    assert dups[["dropped_label", "kept_label"]].values.tolist() == [[2, 1]]


def test_tie_goes_to_earliest_kept_row():
    assert _ratio("aaaaaa", "aaabbb") == 0.5
    assert _ratio("aaaaaa", "aabaab") == _ratio("aaabbb", "aabaab") == pytest.approx(2 / 3)
    df = pd.DataFrame([_rec("aaaaaa"), _rec("aaabbb"), _rec("aabaab")])
    kept, dups = _dedup(df, 0.6)
    assert kept.index.tolist() == [0, 1]
    assert dups[["dropped_label", "kept_label"]].values.tolist() == [[2, 0]]


# ── Strict semantics ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("override,expect_dup", [
    ({"doc": "d2"}, False),
    ({"attr": "do"}, False),
    ({"value": 7.0005}, True),       # within atol=1e-3
    ({"value": 7.01}, False),
    ({"value": 7}, True),            # int vs float
    ({"value": "7.0"}, False),       # string never equals a number
    ({"units": "mg/L"}, False),      # None vs str
])
def test_strict_fields(override, expect_dup):
    df = pd.DataFrame([_rec("pond a"), _rec("pond a", **override)])
    kept, _ = _dedup(df, 0.9)
    assert len(kept) == (1 if expect_dup else 2)


def test_strict_strings_case_and_whitespace_insensitive():
    df = pd.DataFrame([_rec("pond a", units="mg/L"), _rec("pond a", units=" MG/l ")])
    assert len(_dedup(df, 0.9)[0]) == 1


def test_null_equals_null_on_strict_fields():
    df = pd.DataFrame([_rec("pond a", units=None), _rec("pond a", units=np.nan)])
    assert len(_dedup(df, 0.9)[0]) == 1


def test_empty_string_equals_null_on_strict_fields():
    df = pd.DataFrame([_rec("pond a", units=None), _rec("pond a", units="  ")])
    assert len(_dedup(df, 0.9)[0]) == 1


def test_empty_string_does_not_equal_a_real_value_on_strict_fields():
    df = pd.DataFrame([_rec("pond a", units=""), _rec("pond a", units="mg/L")])
    assert len(_dedup(df, 0.9)[0]) == 2


# ── Fuzzy semantics ────────────────────────────────────────────────────────────

def test_fuzzy_all_null_both_sides_abstains_score_one():
    df = pd.DataFrame([_rec(None, eco=None), _rec(None, eco=None)])
    kept, dups = _dedup(df, 1.0, fuzzy=FUZZY)
    assert kept.index.tolist() == [0]
    assert dups.score.tolist() == [1.0]


def test_fuzzy_one_sided_null_scores_zero_and_threshold_decides():
    # name null on row 1 only; ecosystem agrees (1.0) -> mean (0.0 + 1.0) / 2 = 0.5.
    df = pd.DataFrame([_rec("pond a", eco="pond"), _rec(None, eco="pond")])
    kept, dups = _dedup(df, 0.5, fuzzy=FUZZY)   # inclusive threshold
    assert kept.index.tolist() == [0] and dups.score.tolist() == [0.5]
    assert len(_dedup(df, 0.51, fuzzy=FUZZY)[0]) == 2
    # and in the other direction / on the second field
    df = pd.DataFrame([_rec("pond a", eco=None), _rec("pond a", eco="pond")])
    assert _dedup(df, 0.5, fuzzy=FUZZY)[1].score.tolist() == [0.5]
    assert len(_dedup(df, 0.51, fuzzy=FUZZY)[0]) == 2


def test_fuzzy_null_still_needs_strict_match():
    df = pd.DataFrame([_rec(None), _rec(None, value=8.0)])
    assert len(_dedup(df, 0.9)[0]) == 2


def test_merge_goes_to_highest_scoring_eligible_kept_row():
    # X=(aaaa, pond), Y=(zzzz, pond) kept. D=(aaab, pond): X scores (0.75+1)/2,
    # Y scores (0+1)/2 -> D merges into X. Threshold 0.6 keeps Y apart from X (0.5).
    df = pd.DataFrame([_rec("aaaa", eco="pond"), _rec("zzzz", eco="pond"), _rec("aaab", eco="pond")])
    kept, dups = _dedup(df, 0.6, fuzzy=FUZZY)
    assert kept.index.tolist() == [0, 1]
    assert dups[["dropped_label", "kept_label"]].values.tolist() == [[2, 0]]


def test_fuzzy_both_null_field_abstains_from_the_mean():
    a, b = _rec("aaaa", eco=None), _rec("aaab", eco=None)
    eligible, score = pair_score(pd.Series(a), pd.Series(b), strict_fields=STRICT, fuzzy_fields=FUZZY)
    assert eligible and score == 0.75  # ecosystem abstains; mean over name alone


def test_one_sided_null_field_scores_zero_in_the_mean():
    a, b = _rec("aaaa", eco="pond"), _rec("aaab", eco=None)
    eligible, score = pair_score(pd.Series(a), pd.Series(b), strict_fields=STRICT, fuzzy_fields=FUZZY)
    assert eligible and score == 0.375  # (0.75 + 0.0) / 2


def test_all_fuzzy_fields_abstain_scores_one():
    a, b = _rec(None, eco=None), _rec(None, eco=None)
    assert pair_score(pd.Series(a), pd.Series(b), strict_fields=STRICT, fuzzy_fields=FUZZY) == (True, 1.0)


def test_strict_mismatch_is_ineligible_with_no_score():
    a, b = _rec("aaaa", value=7.0), _rec("aaaa", value=8.0)
    assert pair_score(pd.Series(a), pd.Series(b), strict_fields=STRICT, fuzzy_fields=FUZZY) == (False, None)


def test_fuzzy_mean_over_fields():
    a, b = _rec("aaaa", eco="pond"), _rec("aaab", eco="lake")
    _, score = pair_score(pd.Series(a), pd.Series(b), strict_fields=STRICT, fuzzy_fields=FUZZY)
    assert score == pytest.approx((0.75 + _ratio("pond", "lake")) / 2)


# ── Provenance merging ─────────────────────────────────────────────────────────

def test_provenance_merged_in_row_order_and_aligned():
    df = pd.DataFrame([
        _rec("aaaa", page=1, ctx="c1"),
        _rec("zzzz", page=9, ctx="c9"),   # not a dup
        _rec("aaab", page=2, ctx="c2"),
        {**_rec("aaaa", page=3, ctx="c3"), "page_number": [3, 4], "context": ["c3", "c4"]},
    ])
    kept, dups = _dedup(df, 0.7)
    assert kept.index.tolist() == [0, 1]
    assert kept.at[0, "page_number"] == [1, 2, 3, 4]
    assert kept.at[0, "context"] == ["c1", "c2", "c3", "c4"]
    assert kept.at[1, "page_number"] == [9]
    # input not mutated
    assert df.at[0, "page_number"] == [1]


def test_preserves_non_range_index_labels():
    df = pd.DataFrame([_rec("aaaa"), _rec("aaab")], index=[10, 5])
    kept, dups = _dedup(df, 0.7)
    assert kept.index.tolist() == [10]
    assert dups[["dropped_label", "kept_label"]].values.tolist() == [[5, 10]]


# ── Invariants ─────────────────────────────────────────────────────────────────

def _random_fixture(seed, n=60):
    rng = np.random.default_rng(seed)
    names = ["pond 1", "pond 2", "Pond 1 ", "lake a", "lake b", "site x", None]
    ecos = ["pond", "lake", "Pond", None]
    values = [7.0, 7.0004, 7.5, 8, "7.0", "ca. 7", None]
    units = ["mg/L", "MG/L", None, "", "  "]
    rows = []
    for i in range(n):
        rows.append({
            "document_id": rng.choice(["d1", "d2"]), "attribute": rng.choice(["ph", "do"]),
            "value": values[rng.integers(len(values))], "units": units[rng.integers(len(units))],
            "name": names[rng.integers(len(names))], "ecosystem": ecos[rng.integers(len(ecos))],
            "page_number": [i], "context": [f"c{i}"],
        })
    return pd.DataFrame(rows)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("threshold", [0.0, 1 / 3, 0.8, 1.0])
def test_pairwise_parity_with_match_datasets(seed, threshold):
    df = _random_fixture(seed)
    _, edges, _ = match_datasets(
        df, df, strict_matching={c: c for c in STRICT},
        fuzzy_matching={c: c for c in FUZZY}, fuzzy_threshold=threshold,
    )
    edge_set = set(edges)
    n_edge = n_non_edge = n_one_sided_null = 0
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            eligible, score = pair_score(df.iloc[i], df.iloc[j], strict_fields=STRICT, fuzzy_fields=FUZZY)
            ours = eligible and score >= threshold
            assert ours == ((i, j) in edge_set), (i, j, df.iloc[i].to_dict(), df.iloc[j].to_dict())
            n_edge += ours
            n_non_edge += not ours
            n_one_sided_null += any(
                pd.isna(df.iloc[i][c]) != pd.isna(df.iloc[j][c]) for c in FUZZY
            ) and all(
                (df.iloc[i][c] == df.iloc[j][c]) or (pd.isna(df.iloc[i][c]) and pd.isna(df.iloc[j][c]))
                for c in STRICT
            )
    assert (n_edge > 0 or threshold == 1.0) and n_non_edge > 100 and n_one_sided_null > 0  # fixture exercises every path


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_idempotent(seed):
    df = _random_fixture(seed)
    kept, _ = _dedup(df, 1 / 3, fuzzy=FUZZY)
    kept2, dups2 = _dedup(kept, 1 / 3, fuzzy=FUZZY)
    assert kept2.index.tolist() == kept.index.tolist()
    assert len(dups2) == 0


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_doubled_input_keeps_same_rows(seed):
    df = _random_fixture(seed)
    kept, _ = _dedup(df, 1 / 3, fuzzy=FUZZY)
    doubled = pd.concat([df, df.set_index(df.index + len(df))])
    kept_d, _ = _dedup(doubled, 1 / 3, fuzzy=FUZZY)
    assert kept_d.index.tolist() == kept.index.tolist()
    # every provenance entry survives, twice
    assert kept_d.page_number.map(len).sum() == 2 * len(df)


def test_row_accounting_on_random_fixture():
    df = _random_fixture(3)
    kept, dups = _dedup(df, 0.5, fuzzy=FUZZY)
    assert len(kept) + len(dups) == len(df)
    assert sorted(kept.index.tolist() + dups.dropped_label.tolist()) == list(range(len(df)))
    assert len(dups) > 0


# ── Fail loud ──────────────────────────────────────────────────────────────────

def _ok_df():
    return pd.DataFrame([_rec("aaaa"), _rec("aaab")])


@pytest.mark.parametrize("kwargs,exc", [
    ({"strict_fields": []}, ValueError),
    ({"fuzzy_fields": []}, ValueError),
    ({"strict_fields": STRICT + ["missing"]}, KeyError),
    ({"fuzzy_fields": ["missing"]}, KeyError),
    ({"provenance_fields": ["missing"]}, KeyError),
    ({"fuzzy_threshold": 1.5}, ValueError),
    ({"fuzzy_threshold": -0.1}, ValueError),
    ({"fuzzy_threshold": True}, TypeError),
    ({"fuzzy_fields": ["name", "document_id"]}, ValueError),     # strict & fuzzy overlap
    ({"strict_fields": STRICT + ["context"]}, ValueError),      # match & provenance overlap
])
def test_bad_arguments_raise(kwargs, exc):
    args = dict(strict_fields=STRICT, fuzzy_fields=["name"], fuzzy_threshold=0.5, provenance_fields=PROV)
    args.update(kwargs)
    with pytest.raises(exc):
        deduplicate_records(_ok_df(), **args)


@pytest.mark.parametrize("col,bad", [("value", [7.0]), ("value", True), ("name", ["a"]), ("units", pd.NA)])
def test_non_scalar_match_values_raise(col, bad):
    df = _ok_df()
    df[col] = df[col].astype(object)
    df.at[1, col] = bad
    with pytest.raises(TypeError):
        _dedup(df, 0.5)


def test_non_unique_index_raises():
    df = pd.DataFrame([_rec("aaaa"), _rec("aaab")], index=[0, 0])
    with pytest.raises(ValueError):
        _dedup(df, 0.5)


def test_non_list_provenance_raises():
    df = _ok_df()
    df.at[1, "context"] = "not a list"
    with pytest.raises(TypeError):
        _dedup(df, 0.5)


def test_misaligned_provenance_raises():
    df = _ok_df()
    df.at[1, "context"] = ["a", "b"]
    with pytest.raises(ValueError):
        _dedup(df, 0.5)
