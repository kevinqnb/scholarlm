"""Unit tests for src/scholarlm/utils/parsing.py: parse_quantity_shape (and
its numeric-converting wrapper), normalize_unit_notation, standardize_units,
and split_value_and_unit_suffix. Every case here is one a human can verify by
inspection -- no real repo data, no fixture tree.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / "src"))

from scholarlm.utils import parsing  # noqa: E402


# ---------------------------------------------------------------------------
# parse_quantity_shape
# ---------------------------------------------------------------------------


def test_plain_float_is_confirmed_plain():
    out = parsing.parse_quantity_shape("48")
    assert out["qualifiers"] == []
    assert out["point_value"] == "48"
    assert out["lower"] is None and out["upper"] is None


def test_ca_prefix_is_approximate():
    out = parsing.parse_quantity_shape("ca. 0.26")
    assert out["qualifiers"] == ["IsApproximate"]
    assert out["point_value"] == "0.26"


def test_scientific_notation_is_confirmed_plain():
    out = parsing.parse_quantity_shape("1.5 × 10^8")
    assert out["qualifiers"] == []
    assert out["point_value"] == "1.5 × 10^8"


def test_unicode_dash_and_approx_marks_normalized_first():
    # − (U+2212) and ∼ (U+223C) both appear in real OCR/LLM text standing in
    # for ASCII '-'/'~'.
    out = parsing.parse_quantity_shape("36−40")
    assert out["qualifiers"] == ["IsRange"]
    assert out["lower"] == "36" and out["upper"] == "40"


def test_tolerance_default_accepts_any_magnitude():
    # reinterpret_implausible_tolerance defaults to False: a tolerance at
    # least as large as its own point value is taken at face value, not
    # silently reinterpreted as a range (real for nfix's near-zero rates).
    out = parsing.parse_quantity_shape("-1.3 ± 5.0")
    assert out["qualifiers"] == ["HasTolerance"]
    assert out["point_value"] == "-1.3"
    assert out["tolerance"] == "± 5.0"


def test_tolerance_reinterpreted_as_range_when_flag_set_and_ascending():
    out = parsing.parse_quantity_shape("37 ± 38", reinterpret_implausible_tolerance=True)
    assert out["qualifiers"] == ["IsRange"]
    assert out["lower"] == "37" and out["upper"] == "38"


def test_tolerance_reinterpretation_refuses_descending_order():
    # Implausible (|-10| >= |5|), but reinterpreting as a range would need
    # 5 <= -10, which is false -- _ascending_range_fields refuses, so this
    # falls through to unparsed rather than silently sorting the pair.
    out = parsing.parse_quantity_shape("5 ± -10", reinterpret_implausible_tolerance=True)
    assert out["qualifiers"] is None
    assert out["point_value"] is None


def test_compact_uncertainty_notation():
    out = parsing.parse_quantity_shape("2.05(5)")
    assert out["qualifiers"] == ["HasTolerance"]
    assert out["point_value"] == "2.05"
    assert out["tolerance"] == "± 0.05"


def test_compact_uncertainty_digit_mismatch_left_unparsed_by_default():
    # 3 uncertainty digits, only 1 decimal place -- strict_compact_uncertainty
    # defaults to False, so this is left unparsed rather than raising.
    out = parsing.parse_quantity_shape("2.0(345)")
    assert out["qualifiers"] is None


def test_compact_uncertainty_digit_mismatch_raises_when_strict():
    with pytest.raises(AssertionError, match="more uncertainty digits"):
        parsing.parse_quantity_shape("2.0(345)", strict_compact_uncertainty=True)


def test_list_values():
    out = parsing.parse_quantity_shape("1, 2, 3")
    assert out["qualifiers"] == ["IsList"]
    assert out["list_values"] == ["1", "2", "3"]


def test_dash_range_ascending():
    out = parsing.parse_quantity_shape("7-8")
    assert out["qualifiers"] == ["IsRange"]
    assert out["lower"] == "7" and out["upper"] == "8"


def test_dash_range_descending_left_unparsed():
    out = parsing.parse_quantity_shape("8-7")
    assert out["qualifiers"] is None


def test_up_to_word():
    out = parsing.parse_quantity_shape("up to 33")
    assert out["qualifiers"] == ["IsRange"]
    assert out["upper"] == "33"


def test_le_leading_symbol():
    out = parsing.parse_quantity_shape("< 10")
    assert out["qualifiers"] == ["IsRange"]
    assert out["upper"] == "10"


def test_tilde_single_is_approximate():
    out = parsing.parse_quantity_shape("~2.3")
    assert out["qualifiers"] == ["IsApproximate"]
    assert out["point_value"] == "2.3"


def test_nan_and_inf_text_not_treated_as_a_plain_value():
    for text in ("nan", "NaN", "inf", "-inf", "Infinity"):
        out = parsing.parse_quantity_shape(text)
        assert out["point_value"] is None, text


def test_unparseable_text_left_all_null():
    out = parsing.parse_quantity_shape("temperature of liquid helium")
    assert all(v is None for v in out.values())


def test_parse_quantity_shape_numeric_converts_point_value_lower_upper():
    out = parsing.parse_quantity_shape_numeric("7-8")
    assert out["lower"] == 7.0 and out["upper"] == 8.0
    assert isinstance(out["lower"], float)


def test_parse_quantity_shape_numeric_converts_sci_notation_point_value():
    # Real nfix extraction output: "1.3 × 10-3" (no ^, no space before the
    # exponent's sign) -- bare float() can't parse this, only SCI_NOTATION_RE.
    out = parsing.parse_quantity_shape_numeric("1.3 × 10-3")
    assert out["point_value"] == pytest.approx(1.3e-3)


def test_parse_quantity_shape_numeric_leaves_tolerance_as_string():
    out = parsing.parse_quantity_shape_numeric("2.05(5)")
    assert out["point_value"] == 2.05
    assert out["tolerance"] == "± 0.05"


def test_to_float_raises_on_neither_plain_nor_sci_notation():
    with pytest.raises(ValueError, match="not a plain float"):
        parsing.to_float("not a number")


# ---------------------------------------------------------------------------
# normalize_unit_notation
# ---------------------------------------------------------------------------


def test_greek_mu_unified_to_micro_sign():
    assert parsing.normalize_unit_notation("μmol N m⁻² d⁻¹") == "µmol N m⁻² d⁻¹"


def test_negative_exponent_ascii_forms_converted_to_superscript():
    assert parsing.normalize_unit_notation("m-2") == "m⁻²"
    assert parsing.normalize_unit_notation("m^-2") == "m⁻²"
    assert parsing.normalize_unit_notation("m^{-2}") == "m⁻²"


def test_exponent_conversion_does_not_touch_chemical_subscripts():
    # No preceding '-' in "N2"/"C2H4" -- these must survive untouched.
    assert parsing.normalize_unit_notation("N2") == "N2"
    assert parsing.normalize_unit_notation("nmol C2H4 g⁻¹ h⁻¹") == "nmol C2H4 g⁻¹ h⁻¹"


def test_day_hour_year_words_normalized():
    assert parsing.normalize_unit_notation("mg N m-2 day-1") == "mg N m⁻² d⁻¹"
    assert parsing.normalize_unit_notation("g N m-2 hour-1") == "g N m⁻² h⁻¹"


def test_ascii_micro_prefix_normalized():
    assert parsing.normalize_unit_notation("ug/L") == "µg/L"
    assert parsing.normalize_unit_notation("umol/L") == "µmol/L"


# ---------------------------------------------------------------------------
# standardize_units
# ---------------------------------------------------------------------------


def test_standardize_units_literal_member_unchanged():
    assert parsing.standardize_units("K", frozenset({"K", "mK"})) == ("K", False)


def test_standardize_units_none_passes_through():
    assert parsing.standardize_units(None, frozenset({"K"})) == (None, False)


def test_standardize_units_dimensionless_alias_maps_to_none():
    std, changed = parsing.standardize_units("dimensionless", frozenset())
    assert std is None and changed is True


def test_standardize_units_dimensionless_unknown_phrase_unchanged():
    std, changed = parsing.standardize_units("GPa", frozenset())
    assert std == "GPa" and changed is False


def test_standardize_units_known_variant_maps_to_canonical():
    assert parsing.standardize_units("kelvin", frozenset({"K", "mK"})) == ("K", True)
    assert parsing.standardize_units("%", frozenset({"percent", "fraction"})) == ("percent", True)


def test_standardize_units_generic_normalization_hits_canonical():
    canonical = frozenset({"mg N m⁻² d⁻¹"})
    std, changed = parsing.standardize_units("mg N m-2 day-1", canonical)
    assert std == "mg N m⁻² d⁻¹" and changed is True


def test_standardize_units_species_ambiguous_unit_left_unchanged():
    # Missing the N/N2/C2H4 token -- genuinely ambiguous, never guessed.
    canonical = frozenset({"µmol N m⁻² d⁻¹", "µmol N2 m⁻² d⁻¹"})
    std, changed = parsing.standardize_units("µmol m⁻² d⁻¹", canonical)
    assert std == "µmol m⁻² d⁻¹" and changed is False


# ---------------------------------------------------------------------------
# split_value_and_unit_suffix
# ---------------------------------------------------------------------------


def test_split_strips_given_units_suffix():
    text, units = parsing.split_value_and_unit_suffix("3.0 GPa", "GPa", frozenset({"K", "mK"}))
    assert text == "3.0" and units == "GPa"


def test_split_given_units_not_a_suffix_leaves_value_unchanged():
    text, units = parsing.split_value_and_unit_suffix("0.05", "GPa", frozenset({"K", "mK"}))
    assert text == "0.05" and units == "GPa"


def test_split_recovers_percent_from_embedded_suffix():
    canonical = frozenset({"percent", "fraction"})
    text, units = parsing.split_value_and_unit_suffix("34%", None, canonical)
    assert text == "34" and units == "percent"


def test_split_strips_glued_suffix_with_no_separating_space():
    text, units = parsing.split_value_and_unit_suffix("48K", "K", frozenset({"K", "mK"}))
    assert text == "48" and units == "K"


def test_split_given_units_refuses_to_strip_down_to_nothing():
    text, units = parsing.split_value_and_unit_suffix("K", "K", frozenset({"K", "mK"}))
    assert text == "K" and units == "K"


def test_split_no_canonical_units_is_a_no_op():
    text, units = parsing.split_value_and_unit_suffix("34%", None, frozenset())
    assert text == "34%" and units is None


def test_split_unrecognized_suffix_leaves_value_and_units_unchanged():
    canonical = frozenset({"K", "mK"})
    text, units = parsing.split_value_and_unit_suffix("48 furlongs", None, canonical)
    assert text == "48 furlongs" and units is None
