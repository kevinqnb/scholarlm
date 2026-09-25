"""Shared parsing utilities for turning raw extracted/reported text into the
pipeline's standardized shapes: a measurement's qualifier/shape fields
(point_value/lower/upper/list_values/tolerance/standard_deviation/qualifiers,
matching ``ParseQuantityResponse`` in ``src/scholarlm/measurementlm.py``), and
a `units` string normalized against a dataset's own vocabulary.

Used by:
- ``data/supermat/preprocessing.py`` -- ground-truth qualifier-field parsing
  from ``raw_data.csv``'s ``tcValue`` text (via ``parse_quantity_shape``, with
  its two supermat-OCR-specific flags set -- see that module for why).
- ``analysis/postprocessing.py`` -- post-hoc qualifier-field fill and unit
  standardization for extraction/baseline `final.json` output, across pond,
  nfix, and supermat.

Both halves below are deliberately conservative: a value/units string that
doesn't confidently match a known pattern is left as-is, never guessed.
"""
from __future__ import annotations

import math
import re

# ---------------------------------------------------------------------------
# Quantity-shape parsing
# ---------------------------------------------------------------------------

# Must match ParseQuantityResponse in src/scholarlm/measurementlm.py exactly,
# minus `explanation` (a generation-only field, not part of a parsed shape).
QUALIFIER_FIELDS = [
    "qualifiers", "point_value", "lower", "upper",
    "list_values", "tolerance", "standard_deviation",
]

# Unicode minus/dash marks standing in for '-', and unicode approx marks
# standing in for '~' -- seen in both OCR'd ground-truth text and LLM output.
_DASH_APPROX_TRANSLATION = str.maketrans({
    "−": "-", "–": "-", "—": "-",
    "∼": "~", "≃": "~", "≈": "~",
})


def normalize_dash_and_approx_marks(s: str) -> str:
    """Unify unicode dash/approx-mark variants to plain ASCII '-'/'~', then
    collapse whitespace. Dataset-agnostic -- no OCR-corpus-specific fixups
    (those stay local to data/supermat/preprocessing.py, which knows exactly
    which artifacts its own raw_data.csv has)."""
    s = s.translate(_DASH_APPROX_TRANSLATION)
    return re.sub(r'\s+', ' ', s).strip()


# Matches "<mantissa> × 10^<exponent>" (also accepts x/X for ×, and a
# missing ^) -- the one non-plain-float numeric form observed in real
# point_value/value output so far (also used by analysis/match_cache.py's
# numeric-coercion at match time).
SCI_NOTATION_RE = re.compile(r"^\s*([-+]?\d*\.?\d+)\s*[×xX]\s*10\s*\^?\s*([-+]?\d+)\s*$")

# Ordered, mutually-exclusive-by-anchor patterns tried in `parse_quantity_shape`;
# _NUM matches a bare (optionally signed) int/float token.
_NUM = r'-?\d+\.?\d*'
_LIST_SPLIT_RE = re.compile(r',\s*|\s+and\s+', re.I)
_LIST_TOKEN_RE = re.compile(r'^' + _NUM + r'(?:\(\d+\))?$')
_TOLERANCE_RE = re.compile(r'^(' + _NUM + r')\s*±\s*(' + _NUM + r')$')
# Compact "value(uncertainty)" notation, e.g. "2.05(5)" == 2.05 ± 0.05,
# "203(1)" == 203 ± 1 -- the parenthesized digits replace the value's own
# last len(digits) digits (or, with no decimal point, are a plain integer
# uncertainty on the ones place).
_COMPACT_UNCERTAINTY_RE = re.compile(r'^(-?\d+(?:\.(\d+))?)\((\d+)\)$')
_TILDE_RANGE_RE = re.compile(r'^(' + _NUM + r')\s*~\s*(' + _NUM + r')$')
_DASH_RANGE_RE = re.compile(r'^(' + _NUM + r')\s*-\s*(' + _NUM + r')$')
_APPROX_DASH_RANGE_RE = re.compile(r'^~\s*(' + _NUM + r')\s*-\s*(' + _NUM + r')$')
_FROM_TO_RE = re.compile(r'^(?:from\s+)?(~)?\s*(' + _NUM + r')\s*to\s*(~)?\s*(' + _NUM + r')$', re.I)
_UPPER_WORD_RE = re.compile(r'^(?:up to|below|less than|as high as)\s*(~)?\s*(' + _NUM + r')$', re.I)
_LOWER_WORD_RE = re.compile(r'^(?:above|over|exceeds)\s*(' + _NUM + r')$', re.I)
# "ca." (circa) added alongside the original near/close to/around/about --
# a common abbreviation in real extraction/baseline `value` output that the
# supermat OCR corpus (where this pattern was first written) never needed.
_APPROX_WORD_RE = re.compile(r'^(?:near|close to|around|about|ca\.?)\s+(' + _NUM + r')$', re.I)
_LE_LEADING_RE = re.compile(r'^[<≤]\s*(' + _NUM + r')$')
_GE_LEADING_RE = re.compile(r'^[>≥]\s*(' + _NUM + r')$')
_LE_TRAILING_RE = re.compile(r'^(' + _NUM + r')\s*≤$')
_TILDE_SINGLE_RE = re.compile(r'^~\s*(' + _NUM + r')$')


def _ascending_range_fields(num1: str, num2: str, *, approximate: bool = False) -> dict | None:
    """IsRange qualifier fields for two number tokens, only when reported in
    ascending order (num1 <= num2) -- returns None otherwise (a descending
    "A-B"/"A to B" pair is a real, unresolvable ambiguity: sometimes a range
    written high-to-low, sometimes a trend fragment describing two different
    measurements -- see data/supermat/preprocessing.py's fuller discussion).
    Per CLAUDE.md's fail-loud policy, never guesses which reading is meant."""
    a, b = float(num1), float(num2)
    if a > b:
        return None
    fields = {"qualifiers": ["IsRange"], "lower": num1, "upper": num2}
    if approximate:
        fields["qualifiers"].append("IsApproximate")
    return fields


def parse_quantity_shape(
    raw_value: str,
    *,
    reinterpret_implausible_tolerance: bool = False,
    strict_compact_uncertainty: bool = False,
) -> dict:
    """Parse a reported-value string into the 7 QUALIFIER_FIELDS, with
    point_value/lower/upper left as the raw number strings matched (see
    `parse_quantity_shape_numeric` for a float-converting wrapper).

    Most inputs are a plain float (point_value, no tags); the rest are
    ranges/inequalities/approximations/tolerances/lists/compact
    parenthetical-uncertainty notation, matched by the ordered patterns
    above -- first match wins, and each pattern is `^...$`-anchored so
    there's no cross-pattern ambiguity to arbitrate.

    An input that matches nothing (a bare textual description, or text too
    garbled to disambiguate) is left all-null, `qualifiers` included --
    `qualifiers: None` means "shape unannotated"; only a successful parse
    sets it (to `[]` for a confirmed-plain value, or a tag list otherwise).
    Per CLAUDE.md's fail-loud policy, this function never invents a value
    it isn't confident the input actually states.

    Args:
        reinterpret_implausible_tolerance: When a "X ± Y" match has
            abs(Y) >= abs(X) (a tolerance at least as large as its own point
            value -- physically implausible for most of this repo's
            attributes), reinterpret it as an ascending range instead of
            accepting it as a real tolerance. Justified for supermat's raw
            OCR corpus (an OCR-corrupted range dash, e.g. "37 ± 38" for
            "37-38 K" -- see data/supermat/preprocessing.py), but WRONG in
            general: nfix's near-zero dinitrogen-fixation rates routinely
            have a real standard deviation/tolerance larger than the point
            value itself, negative values included. Off by default; only
            data/supermat/preprocessing.py's ground-truth builder sets it.
        strict_compact_uncertainty: When a compact "value(uncertainty)"
            match's parenthesized digit count exceeds the value's own
            decimal-place count (an invariant that always held for
            supermat's raw CSV text), raise AssertionError instead of
            leaving the input unparsed. Off by default -- arbitrary LLM
            output has no such guarantee, and one malformed row shouldn't
            abort postprocessing an entire experiment's worth of rows.

    Returns:
        {qualifiers, point_value, lower, upper, list_values, tolerance,
        standard_deviation}.
    """
    out = {field: None for field in QUALIFIER_FIELDS}
    s = normalize_dash_and_approx_marks(raw_value)

    try:
        # float() accepts "nan"/"inf"/"Infinity" -- text a model would only
        # ever emit as a placeholder/error string, never a real measurement.
        # isfinite() rejects those without rejecting a genuine "1e-10".
        if math.isfinite(float(s)):
            out["qualifiers"] = []
            out["point_value"] = s
            return out
    except ValueError:
        if SCI_NOTATION_RE.match(s):
            out["qualifiers"] = []
            out["point_value"] = s
            return out

    tokens = _LIST_SPLIT_RE.split(s)
    if len(tokens) >= 2 and all(_LIST_TOKEN_RE.match(tok) for tok in tokens):
        out["qualifiers"] = ["IsList"]
        out["list_values"] = tokens
        return out

    m = _TOLERANCE_RE.match(s)
    if m:
        point, tol = m.group(1), m.group(2)
        if not reinterpret_implausible_tolerance or abs(float(tol)) < abs(float(point)):
            out["qualifiers"] = ["HasTolerance"]
            out["point_value"] = point
            out["tolerance"] = f"± {tol}"
            return out
        # reinterpret_implausible_tolerance=True and the magnitude check failed
        # -- see docstring; only data/supermat/preprocessing.py hits this branch.
        fields = _ascending_range_fields(point, tol)
        if fields:
            out.update(fields)
            return out

    m = _COMPACT_UNCERTAINTY_RE.match(s)
    if m:
        value_text, frac, unc = m.group(1), m.group(2), m.group(3)
        frac_digits = len(frac) if frac else 0
        if len(unc) <= frac_digits or frac_digits == 0:
            tolerance_digits = unc if frac_digits == 0 else "0." + unc.rjust(frac_digits, "0")
            out["qualifiers"] = ["HasTolerance"]
            out["point_value"] = value_text
            out["tolerance"] = f"± {tolerance_digits}"
            return out
        if strict_compact_uncertainty:
            raise AssertionError(
                f"compact uncertainty {s!r} has more uncertainty digits than "
                "decimal places -- parsing assumption doesn't hold, check by hand"
            )
        # else: doesn't confidently apply -- fall through (no later pattern
        # matches a string containing '(', so this ends up unparsed).

    m = _TILDE_RANGE_RE.match(s) or _DASH_RANGE_RE.match(s)
    if m:
        fields = _ascending_range_fields(m.group(1), m.group(2))
        if fields:
            out.update(fields)
            return out

    m = _APPROX_DASH_RANGE_RE.match(s)
    if m:
        fields = _ascending_range_fields(m.group(1), m.group(2), approximate=True)
        if fields:
            out.update(fields)
            return out

    m = _FROM_TO_RE.match(s)
    if m:
        approx1, num1, approx2, num2 = m.groups()
        fields = _ascending_range_fields(num1, num2, approximate=bool(approx1 or approx2))
        if fields:
            out.update(fields)
            return out

    m = _UPPER_WORD_RE.match(s)
    if m:
        approx, num = m.groups()
        out["qualifiers"] = ["IsRange", "IsApproximate"] if approx else ["IsRange"]
        out["upper"] = num
        return out

    m = _LOWER_WORD_RE.match(s)
    if m:
        out["qualifiers"] = ["IsRange"]
        out["lower"] = m.group(1)
        return out

    m = _APPROX_WORD_RE.match(s)
    if m:
        out["qualifiers"] = ["IsApproximate"]
        out["point_value"] = m.group(1)
        return out

    m = _LE_LEADING_RE.match(s)
    if m:
        out["qualifiers"] = ["IsRange"]
        out["upper"] = m.group(1)
        return out

    m = _GE_LEADING_RE.match(s)
    if m:
        out["qualifiers"] = ["IsRange"]
        out["lower"] = m.group(1)
        return out

    m = _LE_TRAILING_RE.match(s)
    if m:
        out["qualifiers"] = ["IsRange"]
        out["lower"] = m.group(1)
        return out

    m = _TILDE_SINGLE_RE.match(s)
    if m:
        out["qualifiers"] = ["IsApproximate"]
        out["point_value"] = m.group(1)
        return out

    return out


def to_float(s: str) -> float:
    """float(s), falling back to SCI_NOTATION_RE's "<mantissa> × 10^<exponent>"
    form -- the one shape parse_quantity_shape's plain-value branch can put
    in point_value/lower/upper that bare float() can't parse on its own.
    Raises ValueError for anything else -- callers that need a best-effort,
    never-raise numeric coercion (e.g. analysis/postprocessing.py's
    expand_list_values, on an individual list_values entry it isn't the
    one that generated) should catch this themselves rather than assume
    every string reaching this function is one of these two forms."""
    try:
        return float(s)
    except ValueError:
        pass
    m = SCI_NOTATION_RE.match(s)
    if not m:
        raise ValueError(
            f"{s!r} is not a plain float or SCI_NOTATION_RE match"
        )
    mantissa, exponent = m.groups()
    return float(mantissa) * (10.0 ** int(exponent))


def parse_quantity_shape_numeric(raw_value: str, **kwargs) -> dict:
    """`parse_quantity_shape`, with point_value/lower/upper converted to
    float (via `to_float`, so a sci-notation point_value like
    "1.5 × 10^8" converts too) -- matching pond/nfix ground truth's
    `point_value` convention (copied from their already-numeric `value`).
    tolerance/list_values stay strings ("± 2", not 2.0).

    to_float is never expected to raise here: by construction, every string
    parse_quantity_shape assigns to point_value/lower/upper is already a
    plain float or a SCI_NOTATION_RE match -- letting a ValueError propagate
    uncaught is deliberate, since a failure would mean that invariant broke,
    not that the input text was merely unparseable.
    """
    out = parse_quantity_shape(raw_value, **kwargs)
    for field in ("point_value", "lower", "upper"):
        if out[field] is not None:
            out[field] = to_float(out[field])
    return out


# ---------------------------------------------------------------------------
# Units standardization
# ---------------------------------------------------------------------------

_GREEK_MU = "μ"       # μ (GREEK SMALL LETTER MU)
_MICRO_SIGN = "µ"     # µ (MICRO SIGN) -- the codepoint this repo's ground
                           # truth actually uses (checked against data/pond and
                           # data/nfix's committed ground_truth_review.json).
_MIDDLE_DOT = "·"     # ·
_SUPERSCRIPT_MINUS_DIGIT = {
    "1": "⁻¹",   # ⁻¹
    "2": "⁻²",   # ⁻²
    "3": "⁻³",   # ⁻³
}
# A per-unit negative exponent written as "-2", "^-2", or "^{-2}" right after
# a unit letter -- nfix's ground truth always uses the Unicode superscript
# form. Requires an explicit '-' (never bare "N2", "C2H4"), so it never
# touches a chemical-formula subscript.
_EXPONENT_RE = re.compile(r'(?<=[A-Za-z])\^?\{?-([123])(?!\d)\}?')
_TIME_WORD_MAP = {
    "day": "d", "days": "d",
    "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "year": "yr", "years": "yr",
}
_TIME_WORD_RE = re.compile(r'\b(' + '|'.join(_TIME_WORD_MAP) + r')\b', re.IGNORECASE)
# A bare ASCII 'u' standing in for the micro prefix (ug/L, umol/L) -- common
# in plain-text LLM output. Lowercase only ('U' is ambiguous elsewhere) and
# only before a unit token this repo actually uses a micro-prefixed form of.
_ASCII_MICRO_RE = re.compile(r'(?<![A-Za-z])u(g|mol|M)(?=[/\s]|$)')


def normalize_unit_notation(s: str) -> str:
    """Generic, reversible unit-notation cleanup: unifies the micro sign,
    collapses a middle dot to a space, converts a per-unit negative exponent
    to Unicode superscript form, normalizes day/hour/year spellings to
    d/h/yr, and normalizes a bare ASCII 'u' micro-prefix to the micro sign.

    Knows nothing about any dataset's canonical unit set -- `standardize_units`
    checks the result against one. Never applies a scale/factor conversion
    (mm -> m, %  -> fraction) or drops information (a missing species token) --
    only reversible spelling/formatting changes.
    """
    if not s:
        return s
    s = s.replace(_GREEK_MU, _MICRO_SIGN)
    s = s.replace(_MIDDLE_DOT, " ")
    s = _EXPONENT_RE.sub(lambda m: _SUPERSCRIPT_MINUS_DIGIT[m.group(1)], s)
    s = _TIME_WORD_RE.sub(lambda m: _TIME_WORD_MAP[m.group(1).lower()], s)
    s = _ASCII_MICRO_RE.sub(lambda m: _MICRO_SIGN + m.group(1), s)
    return re.sub(r'\s+', ' ', s).strip()


# Explicit per-canonical-unit notational variants that normalize_unit_notation's
# mechanical rules can't derive on their own: case differences, HTML/LaTeX
# artifacts, spelled-out unit names, slash-vs-inverse-notation. Each entry
# lists variants of exactly ONE physical unit -- never a different unit, a
# scale change, or a species-dropping generalization. For example nfix's
# "µmol m⁻² d⁻¹" (missing its N/N2/C2H4 token) is genuinely ambiguous between
# three different canonical units and is deliberately NOT listed anywhere
# here -- standardize_units leaves it untouched rather than guess.
_UNIT_VARIANTS: dict[str, list[str]] = {
    "m^2": ["m²", "m2", "m<sup>2</sup>", "m 2"],
    "km^2": ["km²", "km2", "km 2"],
    "percent": ["%", "percentage", "Percent", "PERCENT"],
    "K": ["kelvin", "Kelvin", "KELVIN", "°K", "$K$"],
    "mK": ["millikelvin", "milliKelvin"],
    "mg/L": ["mg/l", "mg L⁻¹", "mg L −1", "mg l −1", "mg L", "mg L-1", "mg l-1"],
    "µg/L": ["µg/l", "μg/l", "ug/L", "ug/l", "ug_l", "µg L⁻¹", "µg l⁻¹", "μg L⁻¹"],
    "mg/m^3": ["mg/m³", "mg m⁻³"],
    "µg/cm^2": ["µg/cm²", "µg cm⁻²"],
}


def _build_variant_to_canonical(table: dict[str, list[str]]) -> dict[str, str]:
    reverse: dict[str, str] = {}
    for canonical, variants in table.items():
        for variant in variants:
            if variant in reverse:
                raise ValueError(
                    f"unit variant {variant!r} is listed under both "
                    f"{reverse[variant]!r} and {canonical!r} -- _UNIT_VARIANTS "
                    "entries must be disjoint"
                )
            reverse[variant] = canonical
    return reverse


_UNIT_VARIANT_TO_CANONICAL = _build_variant_to_canonical(_UNIT_VARIANTS)

# Phrasings meaning "this attribute has no unit". Only ever consulted for an
# attribute whose ground truth is entirely null-units (e.g. pond's `ph`) --
# analysis.utils.data.match_datasets' strict equality treats null==null as a
# match, so standardizing one of these to real `None` (not any of these
# strings) is what actually lets such a row match.
_DIMENSIONLESS_ALIASES = frozenset({
    "dimensionless", "dimensionless quantity", "unitless", "n/a", "not applicable",
    "none", "null", "ph", "ph units", "ph unit", "su",
    "dimensionless (ph)", "dimensionless quantity (ph)",
    "dimensionless (ph unit)", "dimensionless (ph units)",
    "ph (nbs scale)",
})
# "score" deliberately excluded -- unlike the above (all genuine notational
# stand-ins for "no unit"), a bare "score" was observed alongside
# "abundance score" in real pond ph extractions, which looks like the model
# mislabeling a different measurement's value/units as ph rather than
# writing "ph has no unit" in yet another way. Not confidently a notational
# variant -- left unmapped.


def standardize_units(raw_units: str | None, canonical_units: frozenset) -> tuple:
    """Best-effort notational standardization of a `units` string against
    canonical_units (this attribute's own set of ground-truth unit strings).

    Never invents a unit: only ever returns something already in
    canonical_units, or None for a dimensionless attribute (canonical_units
    empty and raw_units is a known "no unit" phrasing -- see
    _DIMENSIONLESS_ALIASES). Anything it doesn't recognize as a notational
    variant of something already in canonical_units is returned unchanged.

    Args:
        raw_units: The extracted `units` string, or None.
        canonical_units: Every distinct non-null `units` value this
            attribute actually has in the ground truth (see
            analysis/postprocessing.py's canonical_units_by_attribute) --
            deliberately not attribute_info_dict's own unit list, which can
            name a unit the ground truth itself never uses, or omit one it
            does (see this build's own devlog/build note for an example).

    Returns:
        (standardized_units, changed).
    """
    if raw_units is None:
        return None, False
    if not canonical_units:
        if raw_units.strip().lower() in _DIMENSIONLESS_ALIASES:
            return None, True
        return raw_units, False
    if raw_units in canonical_units:
        return raw_units, False
    for candidate in (raw_units, normalize_unit_notation(raw_units)):
        mapped = _UNIT_VARIANT_TO_CANONICAL.get(candidate)
        if mapped is not None and mapped in canonical_units:
            return mapped, True
        if candidate in canonical_units:
            return candidate, True
    return raw_units, False


def split_value_and_unit_suffix(value: str, units: str | None, canonical_units: frozenset) -> tuple:
    """Peel a trailing unit token off `value` before shape-parsing, and
    recover `units` from it when the row's own `units` field is blank.

    Case 1 -- `units` already set: strips it from the end of `value` if
    present as a literal trailing suffix, whether space-separated ("3.0 GPa",
    units="GPa" -> "3.0") or glued directly onto the number ("34%",
    units="%" -> "34"; "48K", units="K" -> "48"). `units` is already this
    row's own reported unit, so any trailing occurrence of it is presumed
    intentional -- the only refusal is stripping it down to nothing. Never
    changes `units` itself here.

    Case 2 -- `units` is None/blank: some baselines drop the unit into
    `value` and leave `units` empty entirely (observed: GLiNER's
    value="34%", units=None). Tries a trailing '%', then progressively
    shorter whitespace-delimited trailing token-runs of `value` (longest
    first), against canonical_units via standardize_units -- so "34%" still
    resolves to "percent" when that's this attribute's canonical spelling,
    the same variant lookup standardize_units itself uses. Returns the
    first hit that actually lands in canonical_units; if nothing does,
    returns `value` and `units` completely unchanged -- never guesses at a
    unit that isn't already a real member of canonical_units.

    Returns:
        (value_text_for_shape_parsing, units_to_use).
    """
    s = value.strip()

    if units:
        u = units.strip()
        if u and s.endswith(u) and s != u:
            return s[: -len(u)].strip(), units
        return s, units

    if not canonical_units:
        return s, units

    if s.endswith("%"):
        standardized, _ = standardize_units("%", canonical_units)
        if standardized in canonical_units:
            return s[:-1].strip(), standardized

    tokens = s.split()
    for split_point in range(len(tokens) - 1, 0, -1):
        candidate = " ".join(tokens[split_point:])
        standardized, _ = standardize_units(candidate, canonical_units)
        if standardized in canonical_units:
            return " ".join(tokens[:split_point]), standardized

    return s, units
