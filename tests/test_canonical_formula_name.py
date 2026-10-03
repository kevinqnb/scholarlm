"""Unit tests for ``canonical_formula_name`` and its wiring into ``match_datasets``.

Pair fixtures are hand-built from real supermat names (GT and extractions) so the
expected canonical form can be checked by eye. Abbreviations/aliases are out of scope
and must NOT be folded.
"""
import numpy as np
import pandas as pd
import pytest

from scholarlm.utils.data import match_datasets
from scholarlm.utils.normalization import canonical_formula_name as cfn

# (a, b): same compound, different notation -> must canonicalize identically.
SAME = [
    ("Rb2 Cr3 As3", "Rb₂Cr₃As₃"),                          # spacing vs unicode subscripts
    ("Pr_2CuO_4", "Pr2CuO4"),                              # LaTeX underscore subscript
    ("La2−x Ba x CuO4", "La$_{2-x}$Ba$_x$CuO$_4$"),        # OCR spacing + minus glyph vs LaTeX
    ("La_{2-x}Sr_xCuO_4", "La2-xSrxCuO4"),                 # braces
    ("Bi$_2$Sr$_2$CaCu$_2$O$_{8+x}$", "Bi2Sr2CaCu2O8+x"),
    ("HgBa_2CuO_{4+x}", "HgBa₂CuO₄₊ₓ"),                    # subscript plus / subscript x
    ("YBa$_2$Cu$_3$O$_{7-\\delta}$", "YBa2Cu3O7−δ"),       # \delta vs unicode delta, minus glyph
    ("Pr0.5 Ca0.5 Ba2 Cu3 O7−δ", "Pr0.5Ca0.5Ba2Cu3O7-δ"),
    ("\\(Fe_xCu_{1-x}\\)BaSrYCu_2O_{7+\\delta}", "FexCu1-xBaSrYCu2O7+δ"),  # \\( \\) are math delimiters
    ("Na0.35CoO2\t1.3H2O", "Na0.35CoO2 1.3H2O"),            # tab vs space
    ("CuO·2H2O", "CuO•2H2O"),                               # dot-glyph variants
    ("CuO\\cdot 2H2O", "CuO.2H2O"),
    ("Sr2‑RuO4", "Sr2-RuO4"),                              # non-breaking hyphen
    ("Bi–2212", "Bi-2212"),                                 # en dash
]

# (a, b): different compounds / out-of-scope aliases -> must stay different.
DIFFERENT = [
    ("La_{2-x}Sr_xCuO_4", "La_{2-x}Ba_xCuO_4"),            # different dopant element
    ("Bi2Sr2CuO6", "Bi2Sr2CaCu2O8"),
    ("LSCO", "La_{2-x}Sr_xCuO_4"),                          # abbreviation: out of scope
    ("Hg-1201", "HgBa_2CuO_{4+x}"),                         # abbreviation: out of scope
    ("La1.75 Sr0.25 CuO4", "La_{2-x}Sr_xCuO_4"),            # number vs doping variable
    ("O7-\\delta", "O7+\\delta"),                           # sign matters
    ("MgB2", "MgB4"),
]


@pytest.mark.parametrize("a,b", SAME)
def test_same_compound_different_notation_canonicalizes_equal(a, b):
    assert cfn(a) == cfn(b)


@pytest.mark.parametrize("a,b", DIFFERENT)
def test_different_compounds_and_aliases_stay_different(a, b):
    assert cfn(a) != cfn(b)


def test_exact_canonical_forms_by_inspection():
    assert cfn("Rb2 Cr3 As3") == "Rb2Cr3As3"
    assert cfn("La$_{2-x}$Ba$_x$CuO$_4$") == "La2-xBaxCuO4"
    assert cfn("YBa$_2$Cu$_3$O$_{7-\\delta}$") == "YBa2Cu3O7-δ"
    assert cfn("\\(Fe_xCu_{1-x}\\)BaSrYCu_2O_{7+\\delta}") == "FexCu1-xBaSrYCu2O7+δ"
    assert cfn("(Fe_xCu_{1-x})BaSrY") == "(FexCu1-x)BaSrY"      # real parentheses are kept


def test_unknown_latex_command_left_as_literal_text():
    # Stricter, never looser: an unhandled command must not vanish and fake a match.
    assert cfn("Tc \\approx 30") == "Tc\\approx30"
    assert cfn("Tc \\approx 30") != cfn("Tc 30")


def test_case_is_not_folded():
    # The matcher lowercases itself; the canonicalizer must not decide case policy.
    assert cfn("CoO") == "CoO"


def test_non_string_raises():
    with pytest.raises(TypeError):
        cfn(None)
    with pytest.raises(TypeError):
        cfn(3.0)


# --- wiring into match_datasets -------------------------------------------------

def _one_pair(gt_name, ex_name, normalizers, threshold=0.5):
    left = pd.DataFrame([{"id": "a", "name": gt_name}])
    right = pd.DataFrame([{"id": "a", "name": ex_name}])
    return match_datasets(
        left, right, strict_matching={"id": "id"}, fuzzy_matching={"name": "name"},
        fuzzy_threshold=threshold, fuzzy_normalizers=normalizers,
    )


def test_normalizer_turns_a_sub_threshold_pair_into_an_exact_match():
    gt, ex = "YBa2Cu3O7−δ", "YBa$_2$Cu$_3$O$_{7-\\delta}$"   # raw fuzz.ratio = 0.47
    _, edges, _ = _one_pair(gt, ex, None)
    assert edges == []                      # raw fuzz.ratio is below 0.5
    _, edges, weights = _one_pair(gt, ex, {"name": cfn})
    assert edges == [(0, 0)] and weights == [1.0]


def test_normalizer_does_not_rescue_a_genuinely_different_name():
    _, edges, _ = _one_pair("MgB2", "YBa$_2$Cu$_3$O$_7$", {"name": cfn})
    assert edges == []


def test_normalizer_never_sees_a_null_and_null_rules_follow_the_matcher():
    seen = []
    def spy(x):
        assert isinstance(x, str) and x.strip(), f"normalizer must not see a null: {x!r}"
        seen.append(x)
        return x
    _, edges, weights = _one_pair(None, np.nan, {"name": spy})
    assert edges == [(0, 0)] and weights == [1.0]       # null on both sides: all fields abstain
    assert seen == []
    _, edges, weights = _one_pair("MgB2", None, {"name": spy}, threshold=0.0)
    assert edges == [(0, 0)] and weights == [0.0]       # one-sided null scores 0.0
    assert seen == ["MgB2"]                             # only the non-null side is normalized
    _, edges, _ = _one_pair("MgB2", None, {"name": spy})  # default threshold 0.5
    assert edges == []


def test_normalizer_key_must_be_a_fuzzy_column():
    with pytest.raises(KeyError):
        _one_pair("a", "a", {"nope": cfn})


def test_supermat_config_wires_the_normalizer_and_coercion():
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src"), str(root / "experiments")]
    from experiments.run_extraction import load_dataset_config
    cfg = load_dataset_config("supermat")
    assert cfg.fuzzy_normalizers == {"name": cfn}
    assert cfg.numeric_coerce == ["point_value"]
    assert set(cfg.fuzzy_normalizers) <= set(cfg.fuzzy_matching)
