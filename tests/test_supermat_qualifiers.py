"""Rung-1 unit tests for `_parse_qualifiers` in data/supermat/preprocessing.py
(rung 1 of CLAUDE.md's staged-gate ladder): a hand-built fixture of raw
qualifiers-GT `value` strings, each with an expected structured result I can
verify by inspection. Covers every branch of the parser -- plain values (incl.
the colon-decimal OCR fix and orphan-punctuation artifacts), ranges
(dash/tilde/word/inequality/one-sided), approximations, tolerance (± and
compact parenthetical notation), lists, the ascending-order-only range policy,
the tolerance-implausibility reinterpretation, and the "leave unparsed"
fallback for free text / too-garbled/too-ambiguous input.

point_value/lower/upper are asserted as float (matching pond/nfix's
point_value convention); tolerance/list_values stay strings (matching
PARSE_QUANTITY_INSTRUCTIONS and the supermat NuExtract few-shot examples).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "data" / "supermat" / "preprocessing.py"

_spec = importlib.util.spec_from_file_location("supermat_preprocessing", _SCRIPT)
pp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pp)


def _null_fields(**overrides):
    """All 7 fields null, `qualifiers` included -- the "shape unannotated"
    state. Every test that expects a successful parse must pass its own
    `qualifiers=` (`[]` for a confirmed-plain value, or the matched tag
    list); only the "left unparsed" tests rely on this default.
    """
    out = {f: None for f in pp.QUALIFIER_FIELDS}
    out.update(overrides)
    return out


def test_plain_float():
    assert pp._parse_qualifiers("30") == _null_fields(qualifiers=[], point_value=30.0)
    assert pp._parse_qualifiers("37.7") == _null_fields(qualifiers=[], point_value=37.7)


def test_plain_float_with_ocr_colon_decimal():
    assert pp._parse_qualifiers("46:5") == _null_fields(qualifiers=[], point_value=46.5)


def test_orphan_trailing_dash_is_plain_value():
    # "18-K" -> "18-" after unit-stripping (a pre-existing raw-data artifact,
    # see the 2026-09-24 qualifiers-unit-strip build note; confirmed against
    # the OCR source, "The 18-K transition temperature...", a plain value).
    assert pp._parse_qualifiers("18-") == _null_fields(qualifiers=[], point_value=18.0)


def test_orphan_trailing_paren_is_plain_value():
    # "13 K)" -> "13)" after unit-stripping; confirmed against the OCR
    # source (a stray closing paren unrelated to the number).
    assert pp._parse_qualifiers("13)") == _null_fields(qualifiers=[], point_value=13.0)


def test_dash_range():
    assert pp._parse_qualifiers("7-8") == _null_fields(
        qualifiers=["IsRange"], lower=7.0, upper=8.0
    )


def test_dash_range_descending_order_left_unparsed():
    # "60-58" is written high-to-low; hand-checking the OCR source for every
    # such reversed pair in this corpus found some are genuine ranges and
    # some are trend fragments (e.g. before/after values), not reliably
    # distinguishable from the bare cell text -- so descending order is
    # deliberately left unparsed rather than silently sorted into an
    # interval. See `_ascending_range_fields`'s docstring.
    assert pp._parse_qualifiers("60-58") == _null_fields()


def test_unicode_minus_range():
    assert pp._parse_qualifiers("37 − 38") == _null_fields(
        qualifiers=["IsRange"], lower=37.0, upper=38.0
    )


def test_ocr_dash_artifact_range():
    assert pp._parse_qualifiers("26À28") == _null_fields(
        qualifiers=["IsRange"], lower=26.0, upper=28.0
    )


def test_tilde_range():
    assert pp._parse_qualifiers("18 ∼ 30") == _null_fields(
        qualifiers=["IsRange"], lower=18.0, upper=30.0
    )


def test_approx_dash_range():
    # "∼28-30": an approximation mark in front of a whole dash range, not a
    # single approximate value -- confirmed against the OCR source ("bulk SC
    # ... with T_C ∼ 28-30 K").
    assert pp._parse_qualifiers("∼28-30") == _null_fields(
        qualifiers=["IsRange", "IsApproximate"], lower=28.0, upper=30.0
    )


def test_from_to_range():
    assert pp._parse_qualifiers("from 1.4 to 3.5") == _null_fields(
        qualifiers=["IsRange"], lower=1.4, upper=3.5
    )


def test_bare_to_range():
    assert pp._parse_qualifiers("16 to 26") == _null_fields(
        qualifiers=["IsRange"], lower=16.0, upper=26.0
    )


def test_from_to_range_descending_order_left_unparsed():
    # "from 4.5 to 2" -- confirmed against the OCR source this is a trend
    # ("the drop in Tc from 4.5 K to 2 K upon decreasing the Na
    # concentration"), not a reported interval.
    assert pp._parse_qualifiers("from 4.5 to 2") == _null_fields()


def test_from_only_left_unparsed():
    # "from 26" alone (no "to") -- confirmed against the OCR source this is
    # a truncated pressure-dependence trend fragment ("Tc increase from 26 K
    # at P=0 to ... 43 K at P=4 GPa"), not a lower-bound statement. There is
    # no pattern for this shape at all now (removed after verification).
    assert pp._parse_qualifiers("from 26") == _null_fields()


def test_up_to_is_upper_bound():
    assert pp._parse_qualifiers("up to 40") == _null_fields(
        qualifiers=["IsRange"], upper=40.0
    )


def test_up_to_approx_tags_both():
    assert pp._parse_qualifiers("up to ~3.8") == _null_fields(
        qualifiers=["IsRange", "IsApproximate"], upper=3.8
    )


def test_below_is_upper_bound():
    assert pp._parse_qualifiers("below 10") == _null_fields(
        qualifiers=["IsRange"], upper=10.0
    )


def test_as_high_as_is_upper_bound():
    assert pp._parse_qualifiers("as high as 38") == _null_fields(
        qualifiers=["IsRange"], upper=38.0
    )


def test_above_is_lower_bound():
    assert pp._parse_qualifiers("above 30") == _null_fields(
        qualifiers=["IsRange"], lower=30.0
    )


def test_leading_less_than():
    assert pp._parse_qualifiers("< 10") == _null_fields(
        qualifiers=["IsRange"], upper=10.0
    )


def test_leading_greater_than_no_space():
    assert pp._parse_qualifiers(">2") == _null_fields(
        qualifiers=["IsRange"], lower=2.0
    )


def test_trailing_le_is_lower_bound():
    assert pp._parse_qualifiers("30 ≤") == _null_fields(
        qualifiers=["IsRange"], lower=30.0
    )


def test_leading_tilde_approximate():
    assert pp._parse_qualifiers("~1.3") == _null_fields(
        qualifiers=["IsApproximate"], point_value=1.3
    )


def test_word_approximate():
    assert pp._parse_qualifiers("about 4") == _null_fields(
        qualifiers=["IsApproximate"], point_value=4.0
    )


def test_tolerance_symbol():
    assert pp._parse_qualifiers("23.7 ± 0.5") == _null_fields(
        qualifiers=["HasTolerance"], point_value=23.7, tolerance="± 0.5"
    )


def test_implausible_tolerance_reinterpreted_as_range():
    # "37 ± 38": a tolerance >= its own point value is physically
    # implausible for a (positive) Tc. Confirmed against the OCR source
    # ("...close values Tc ∼ 37-38 K") this is an OCR-corrupted range dash,
    # not a real tolerance -- '±' stood in for '-' in this document (which
    # also uses 'À' for '-' elsewhere, e.g. "Ba1Àx" for "Ba1-x").
    assert pp._parse_qualifiers("37 ± 38") == _null_fields(
        qualifiers=["IsRange"], lower=37.0, upper=38.0
    )


def test_implausible_tolerance_ocr_colon_decimal_reinterpreted_as_range():
    # "3:0 ± 4:2": same document/corruption as above; confirmed against the
    # OCR source ("...low-temperature (T_f ~ 3.0-4.2 K) superconductor").
    assert pp._parse_qualifiers("3:0 ± 4:2") == _null_fields(
        qualifiers=["IsRange"], lower=3.0, upper=4.2
    )


def test_compact_uncertainty_full_precision():
    assert pp._parse_qualifiers("6.32(19)") == _null_fields(
        qualifiers=["HasTolerance"], point_value=6.32, tolerance="± 0.19"
    )


def test_compact_uncertainty_padded():
    assert pp._parse_qualifiers("2.05(5)") == _null_fields(
        qualifiers=["HasTolerance"], point_value=2.05, tolerance="± 0.05"
    )


def test_compact_uncertainty_no_decimal_point():
    assert pp._parse_qualifiers("203(1)") == _null_fields(
        qualifiers=["HasTolerance"], point_value=203.0, tolerance="± 1"
    )


def test_list_and():
    assert pp._parse_qualifiers("4.2 and 4.6") == _null_fields(
        qualifiers=["IsList"], list_values=["4.2", "4.6"]
    )


def test_list_comma_and():
    assert pp._parse_qualifiers("86, 88 and 48") == _null_fields(
        qualifiers=["IsList"], list_values=["86", "88", "48"]
    )


def test_list_with_compact_uncertainty_last_item():
    assert pp._parse_qualifiers("39.21, 39.15, 39.23 and 39.08(5)") == _null_fields(
        qualifiers=["IsList"], list_values=["39.21", "39.15", "39.23", "39.08(5)"]
    )


def test_free_text_left_unparsed():
    assert pp._parse_qualifiers("temperature of liquid helium") == _null_fields()


def test_unparsed_qualifiers_is_none_not_confirmed_plain():
    # Per the 2026-09-22 qualifier-ground-truth build note's convention:
    # `qualifiers: null` ("shape unannotated") and `qualifiers: []`
    # ("confirmed plain") are not interchangeable -- an unparsed row must
    # get null, not the empty list a successful plain-value parse returns.
    result = pp._parse_qualifiers("temperature of liquid helium")
    assert result["qualifiers"] is None


def test_garbled_ocr_left_unparsed():
    assert pp._parse_qualifiers("92.5 6 0.5") == _null_fields()


def test_ambiguous_space_separated_left_unparsed():
    assert pp._parse_qualifiers("9 0") == _null_fields()
