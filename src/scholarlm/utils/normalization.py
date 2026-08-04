"""Canonicalisation of extracted values and unit strings prior to ground-truth matching.

Matching in :func:`scholarlm.utils.data.match_datasets` compares ``value`` and
``units`` with strict equality, so a system that reports the right measurement in
a different *encoding* -- ``nmol N L−1 h−1`` instead of ``nmol N L⁻¹ h⁻¹``, or
``19.9 ± 2.3`` instead of ``19.9`` -- is scored as a miss. That penalises systems
which were never shown the ground truth's preferred spelling, so the raw numbers
measure notation compliance as much as extraction accuracy.

This module removes that confound. The guiding rule is that normalisation may
change **how** a measurement is written but never **what** it says:

Permitted
    * Unicode form (NFKC), superscript/subscript digits, minus-sign glyphs,
      micro-sign vs Greek mu, HTML ``<sup>``/``<sub>``, LaTeX ``$...$``.
    * Notation for the same quantity: ``a/b`` -> ``a b⁻¹``, ``m^-2`` -> ``m⁻²``,
      ``mean ± sd`` -> mean, ``0·1`` -> ``0.1``, ``18,526`` -> ``18526``.
    * Synonyms of one unit (``day``/``d``, ``litre``/``l``, ``ethylene``/``c2h4``).
    * Token order, since unit multiplication commutes.
    * Descriptive qualifiers that carry no dimension (``dry weight``, ``DW``).

Refused, because each would invent information the extractor did not supply
    * A missing exponent sign. ``m2`` stays ``m²`` and will not match ``m⁻²``.
    * A missing analyte. ``nmol L⁻¹`` stays analyte-free and will not match
      ``nmol N L⁻¹``; ``N``, ``N2`` and ``C2H4`` differ by real stoichiometric
      factors and are never conflated.
    * A point value from a range or a bound. ``2–3``, ``<1`` and ``nd`` become
      NaN rather than 2, 1 and 0.
    * Any unit conversion. Rescaling values between units is a separate concern
      (see :mod:`scholarlm.utils.unit_conversion`); nothing here multiplies a value.

The functions are pure and system-agnostic, and are applied identically to the
ground truth and to every extraction arm.
"""
from __future__ import annotations

import re
import unicodedata

import numpy as np

# Tokens naming the measured species. Held out of the exponent parser because a
# trailing digit here is part of the name (``n2``) rather than an exponent (``m2``).
_ANALYTES = {"n", "n2", "c", "c2h4", "c2h2", "ch4", "p", "o2", "co2"}

# Splits an analyte off the unit it is written flush against: ``nmolN`` -> ``nmol N``.
# Deliberately narrow. A blanket lower-to-upper split also fires inside ``mL``, ``mM``
# and ``mK``, tearing the SI prefix off as a token of its own, so only a name that is
# actually an analyte is split out. Element symbols are guarded by a negative lookahead
# so that the ``C`` of ``Celsius`` is not mistaken for carbon.
_ANALYTE_SPLIT = re.compile(
    r"(?<=[a-z])(?="
    r"(?:C2H4|C2H2|CO2|CH4|N2|O2|[NCP])(?![a-z])"
    r"|(?:Nitrogen|Dinitrogen|Carbon|Ethylene|Acetylene)\b"
    r")"
)

# Molar concentration, whose symbol is an upper-case ``M`` and so cannot survive the
# lowercasing below: without this, ``μM`` (micromolar) and ``μm`` (micrometre) would
# both fold to ``um`` and compare equal. Rewriting the prefix as a word keeps them
# apart and, as a side effect, unifies the ASCII and Greek spellings of the prefix.
# The lookahead rejects ``MPa``, ``Mg`` and other units that merely begin with M.
_MOLAR = re.compile(r"(?<![A-Za-z0-9µμ])([mµμun]?)M(?![A-Za-z])")

# Dimensionless qualifiers describing the substrate rather than the unit.
_DESCRIPTORS = {
    "dry", "wet", "dw", "ww", "fdw", "weight", "mass", "biomass",
    "sediment", "soil", "tissue", "leaf", "root", "plant", "fixed",
    "of", "per",
}

# Distinct spellings of one unit.
_SYNONYMS = {
    "hour": "h", "hours": "h", "hr": "h", "hrs": "h",
    "day": "d", "days": "d",
    "year": "y", "years": "y", "yr": "y", "yrs": "y",
    "month": "mo", "months": "mo", "mon": "mo",
    "minute": "min", "minutes": "min",
    "second": "s", "seconds": "s", "sec": "s",
    "litre": "l", "litres": "l", "liter": "l", "liters": "l",
    "ethylene": "c2h4", "acetylene": "c2h2", "dinitrogen": "n2",
    "nitrogen": "n", "carbon": "c",
    "colony": "col", "colonies": "col",
    "percent": "%",
}

_MINUS = {"−": "-", "–": "-", "—": "-", "‒": "-", "⁻": "-"}


def _strip_markup(s: str) -> str:
    """Reduce HTML/LaTeX exponent markup to plain ``^n`` notation."""
    s = re.sub(r"<\s*sup\s*>(.*?)<\s*/\s*sup\s*>", r"^\1", s, flags=re.I)
    s = re.sub(r"<\s*sub\s*>(.*?)<\s*/\s*sub\s*>", r"\1", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)                 # any surviving tag
    s = s.replace(r"\mu", "μ").replace(r"\,", " ").replace(r"\;", " ")
    s = re.sub(r"\^\{\s*(-?\d+)\s*\}", r"^\1", s)  # ^{-2} -> ^-2
    s = s.replace("$", "")                         # LaTeX math delimiters
    return s


def _unify_glyphs(s: str) -> str:
    """Fold Unicode variants onto a plain-ASCII skeleton.

    NFKC already maps superscript/subscript digits to plain digits and the micro
    sign to Greek mu; this adds the minus-sign and mu foldings it leaves alone.
    """
    s = unicodedata.normalize("NFKC", s)
    for glyph, plain in _MINUS.items():
        s = s.replace(glyph, plain)
    s = s.replace("μ", "u").replace("µ", "u")  # mu / micro -> u
    return s


def parse_value(v) -> float:
    """Extract the scalar measurement from a raw ``value`` field.

    Recognises the notations extractors actually emit for a *single* number --
    ``19.9 ± 2.3``, ``1.1(0.1)``, ``~0.147``, ``0·1``, ``18,526``,
    ``0.95 × 10 10`` -- and returns NaN for anything that is not one, including
    ranges (``2–3``), bounds (``<1``) and sentinels (``nd``, ``NA``). The final
    ``fullmatch`` is what enforces that: a string only parses if, after cosmetic
    cleanup, nothing but a number remains.
    """
    if v is None:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)

    s = _unify_glyphs(_strip_markup(str(v))).strip()

    s = re.sub(r"(?<=\d)·(?=\d)", ".", s)        # 0·1 -> 0.1 (middle-dot decimal)
    s = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", s)     # 18,526 -> 18526
    s = s.split("±")[0].split("+/-")[0]          # mean ± sd -> mean
    s = re.sub(r"\(\s*[\d.]+\s*\)\s*$", "", s.strip())  # 1.1(0.1) -> 1.1
    s = re.sub(r"^(~|≈|ca\.?|approx\.?)\s*", "", s.strip(), flags=re.I)

    # 0.95 × 10 10  /  0.95 x 10^10  ->  9.5e9  (superscript lost in PDF extraction)
    m = re.fullmatch(r"\s*(-?\d*\.?\d+)\s*[×x*]\s*10\s*\^?\s*(-?\d+)\s*", s)
    if m:
        return float(m.group(1)) * 10 ** int(m.group(2))

    m = re.fullmatch(r"\s*(-?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*", s)
    return float(m.group(1)) if m else np.nan


def _parse_token(tok: str) -> tuple[str, int] | None:
    """Split one unit token into ``(name, exponent)``."""
    tok = _SYNONYMS.get(tok, tok)
    if tok in _DESCRIPTORS:
        return None
    if tok in _ANALYTES:
        return tok, 1
    m = re.fullmatch(r"([a-z%°]+)\^?(-?\d+)?", tok)
    if not m:
        return (tok, 1) if tok else None
    name = _SYNONYMS.get(m.group(1), m.group(1))
    if name in _DESCRIPTORS:
        return None
    return name, int(m.group(2)) if m.group(2) else 1


def canonical_units(u) -> str | None:
    """Canonicalise a unit string so that equal units compare equal.

    Returns a whitespace-joined, alphabetically sorted list of ``name`` /
    ``name<exp>`` tokens -- sorted because unit multiplication commutes, so
    ``nmol N₂ h⁻¹ g⁻¹`` and ``nmol N2 g⁻¹ h⁻¹`` are the same unit written two ways.

    Missing exponent signs and missing analytes are preserved as-is, so an
    incompletely reported unit stays a mismatch rather than being repaired.
    """
    if u is None or (isinstance(u, float) and np.isnan(u)):
        return None

    s = _strip_markup(str(u))
    s = _MOLAR.sub(lambda m: (m.group(1) or "") + "molar", s)  # μM -> μmolar, mM -> mmolar
    s = _ANALYTE_SPLIT.sub(" ", s)                             # nmolN -> nmol N
    s = _unify_glyphs(s).lower()
    s = re.sub(r"[()\[\]{}]", " ", s)
    s = re.sub(r"[·*,]", " ", s)           # nmol·h⁻¹ , mmol.liter⁻¹
    s = re.sub(r"(?<=[a-z])\.(?=[a-z])", " ", s)

    if "/" in s:                                # nmol N/mL/h -> nmol N mL-1 h-1
        head, *tail = s.split("/")
        s = " ".join([head] + [
            p.strip() if re.search(r"-\d", p) else f"{p.strip()}-1"
            for p in tail if p.strip()
        ])

    parts: list[str] = []
    for tok in s.split():
        # a bare exponent belongs to the token before it:  "L -1" -> "L-1"
        if re.fullmatch(r"\^?-?\d+", tok) and parts:
            parts[-1] += tok.lstrip("^")
        else:
            parts.append(tok)

    tokens = [t for t in (_parse_token(p) for p in parts) if t is not None]
    if not tokens:
        return None
    return " ".join(sorted(
        name if exp == 1 else f"{name}{exp}" for name, exp in tokens
    ))
