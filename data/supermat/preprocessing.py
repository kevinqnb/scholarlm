"""
Ground truth preprocessing for the supermat (superconductivity) dataset.

Pipeline
--------
    raw_data.csv
        ↓  remap duplicate-DOI codes, drop rows with no local PDF
        ↓  clean material -> name / identifiers / sample_details (with per-document
           forward-fill for bare doping values and pure sample-detail rows)
        ↓  normalize me_method, pressure
        ↓  parse tcValue -> value / units; drop rows reported as a range,
           approximation (~), or bound (< / > / "up to")
    ground_truth.json                  (all registered papers)
    ground_truth_ten.json              (top-10 paper development subset)

Note on document_id reconciliation
-----------------------------------
Of the 147 filename codes referenced in raw_data.csv, 5 have no local PDF under
their own name. Cross-referencing each code's bibliographic record (DOI) against
every other code revealed that 3 of those are duplicate registrations of a paper
we already have under a different code (see _DOCUMENT_ID_REMAP below); only 2
are genuinely missing PDFs, which we were unable to find or access.
Those 2 codes are dropped from the ground truth.

Usage
-----
Run from the repo root:

    python data/supermat/preprocessing.py

Or from data/supermat/:

    python preprocessing.py
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from scholarlm.utils.page_attribution import SUPERMAT_WEIGHTS, attribute_page, parse_ocr
from scholarlm.utils.parsing import (
    QUALIFIER_FIELDS,
    normalize_dash_and_approx_marks,
    parse_quantity_shape,
)

logger = logging.getLogger(__name__)

BASE = Path(__file__).parent  # data/supermat/

# Duplicate-DOI codes discovered during reconciliation: raw_data.csv rows filed
# under these codes describe the same paper as their real-PDF twin, just
# annotated in a separate pass. Merge their rows into the real-PDF document_id
# rather than dropping them.
_DOCUMENT_ID_REMAP: dict[str, str] = {
    "JPS0731655-CC": "JPS0730819-CC",
    "L095167004-CC": "JPS0732912-CC",
    "SSC1310125-CC": "MAT0305503-CC",
}

# Codes with no local PDF and no duplicate-DOI twin -- genuinely missing.
_NO_PDF_CODES = frozenset({"PHC2640145-CC", "yamaguchi2014ac"})

_TOP_PAPERS_N = 10

# ---------------------------------------------------------------------------
# material -> name / identifiers / sample_details
# ---------------------------------------------------------------------------

_QUALIFIER_WORDS = {
    "slightly", "lightly", "heavily", "nearly", "fully", "highly", "moderately",
    "optimally", "under-doped", "underdoped", "over-doped", "overdoped", "doped",
    "oxygen-deficient", "oxygen-deficient,", "fluorine-free", "as-grown", "as-grown,",
    "single-layer", "untwinned", "polycrystalline", "single", "crystal", "crystals",
    "pure", "third", "second", "first", "nominal", "stoichiometric",
    "with", "containing", "samples", "sample", "formula", "chemical",
    "the", "a", "an", "of", "for", "were", "was", "is", "are",
}

_ENGLISH_STOPWORDS = {
    "and", "or", "with", "in", "on", "at", "to", "of", "the", "a", "an",
    "is", "are", "were", "was", "for", "from", "grown",
}

_BARE_DOPING_RE = re.compile(r'^[0-9xXyYzZ.=,\s∼~<>%()-]+$')
_TRAILING_PAREN_RE = re.compile(r'^(?P<base>.*\S)\s*\((?P<abbr>[^()]{1,24})\)\s*$')


def _collapse_formula_spacing(s: str) -> str:
    """Join spurious whitespace between an element/subscript token and a following
    numeric run, e.g. "Bi 2 Sr 2 CuO 6" -> "Bi2 Sr2 CuO6". Skips plain English
    words (checked against _ENGLISH_STOPWORDS) so "and 0.70" is left untouched."""
    tokens = s.split(' ')
    out: list[str] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        while i + 1 < len(tokens):
            nxt = tokens[i + 1]
            if not nxt:
                i += 1
                continue
            prev_mergeable = (
                bool(re.search(r'[A-Za-z0-9]$', tok))
                and tok.lower() not in _ENGLISH_STOPWORDS
                and len(tok) <= 6
            )
            next_mergeable = bool(re.match(r'^\d', nxt))
            if prev_mergeable and next_mergeable:
                tok = tok + nxt
                i += 1
            else:
                break
        out.append(tok)
        i += 1
    return ' '.join(out)


def _extract_trailing_paren(s: str) -> tuple[str, str | None]:
    """Pull a short trailing parenthetical (<=3 words) into an identifier,
    e.g. "HgBa2CuO4+δ (Hg-1201)" -> ("HgBa2CuO4+δ", "Hg-1201")."""
    m = _TRAILING_PAREN_RE.match(s)
    if not m:
        return s, None
    abbr = m.group('abbr').strip()
    if len(abbr.split()) <= 3:
        return m.group('base').strip(), abbr
    return s, None


def _strip_leading_qualifiers(s: str) -> tuple[str, str]:
    """Split off a leading run of descriptive/filler words. Returns (qualifiers, remainder)."""
    tokens = s.split()
    i = 0
    while i < len(tokens) and tokens[i].lower().strip(',') in _QUALIFIER_WORDS:
        i += 1
    return " ".join(tokens[:i]), " ".join(tokens[i:])


# Elemental superconductors sometimes named in plain English rather than by symbol
# (e.g. historical-review papers): "mercury", "niobium", etc. Treated as formula-like
# so they aren't mistaken for descriptive prose.
_ELEMENT_NAMES = frozenset({
    "mercury", "lead", "tin", "aluminum", "aluminium", "niobium", "tantalum",
    "vanadium", "titanium", "zinc", "gallium", "indium", "thallium", "thorium",
    "uranium", "protactinium", "technetium", "rhenium", "ruthenium", "osmium",
    "iridium", "molybdenum", "tungsten", "zirconium", "hafnium", "lanthanum",
    "cadmium", "gadolinium", "carbon", "silicon", "boron", "diamond",
})


def _is_formula_like(tok: str) -> bool:
    core = tok.strip('(),.;:')
    if not core:
        return False
    if core.lower() in _ELEMENT_NAMES:
        return True
    if re.search(r'\d', core):
        return True
    if sum(1 for c in core if c.isupper()) >= 2:
        return True  # e.g. "YBCO", "NbTi", "PLCCO" -- multi-letter symbol/acronym
    if re.search(r'[δ•·+-]', core) and any(c.isupper() for c in core):
        return True
    return False


def _clean_material(raw: object) -> tuple[str | None, str | None, str | None]:
    """Clean one raw `material` cell.

    Returns (name, sample_details, identifiers). `name` is None when the row
    is a pure modifier (bare doping value or purely descriptive text) that
    should forward-fill from the most recent named material in the same
    document; the modifier text itself is preserved in `sample_details`.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None, None

    s = str(raw).strip()
    s = _collapse_formula_spacing(s)
    s, paren_abbrev = _extract_trailing_paren(s)

    if _BARE_DOPING_RE.match(s.strip()):
        return None, f"doping: {s.strip()}", paren_abbrev

    qualifiers, remainder = _strip_leading_qualifiers(s)
    has_formula_token = any(_is_formula_like(t) for t in remainder.split())
    if not has_formula_token:
        return None, s.strip(), paren_abbrev

    name = remainder.strip() or None
    sample_details = qualifiers.strip() or None
    return name, sample_details, paren_abbrev


# ---------------------------------------------------------------------------
# me_method normalization
# ---------------------------------------------------------------------------

_ME_METHOD_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'resist|ρ\(t\)|r-t curve', re.I), "resistivity"),
    (re.compile(r'suscep|magnetiz|magnetis|magnetic|m-t curve', re.I), "magnetic susceptibility"),
    (re.compile(r'specific[\s-]heat|heat capacity|c\s*\(\s*t\s*\)', re.I), "specific heat"),
    (re.compile(r'theoretic|calculat|predict|eliashberg', re.I), "theoretical calculation"),
    (re.compile(r'm\s*\(\s*t\s*\)', re.I), "magnetic susceptibility"),
]


def _normalize_me_method(raw: object) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    for pattern, canon in _ME_METHOD_PATTERNS:
        if pattern.search(s):
            return canon
    return s  # keep original text if unrecognized -- better than dropping information


# ---------------------------------------------------------------------------
# pressure normalization
# ---------------------------------------------------------------------------

_AMBIENT_PRESSURE_RE = re.compile(r'^(ambient(\s+p(ressure)?)?|ap|0(\.0*)?\s*gpa?|0)$', re.I)


def _normalize_pressure(raw: object) -> str | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    if _AMBIENT_PRESSURE_RE.match(s):
        return "ambient"
    s = s.replace("∼", "~").replace("−", "-")
    s = re.sub(r'\s+', ' ', s).strip()
    s = re.sub(r'(\d)(GPa)', r'\1 \2', s, flags=re.I)
    return s


# ---------------------------------------------------------------------------
# tcValue parsing
# ---------------------------------------------------------------------------

_JUNK_TCVALUES = frozenset({"exist", "nitrogen", "helium"})
_RANGE_RE = re.compile(r'(-?\d+\.?\d*)\s*[-–]\s*(-?\d+\.?\d*)')
_QUALIFIER_RE = re.compile(r'up to|[<>~∼≈]', re.I)
_NUMBER_RE = re.compile(r'-?\d+\.?\d*')

# Strips only the unit token itself (K / mK / kelvin(s)), not any inequality,
# range, or qualifier text around it. No left \b requirement -- "K" in "60K"
# has no word-boundary on its left (digit and letter are both \w) -- only a
# right \b, so we don't need a space or punctuation before the unit.
_VALUE_UNIT_RE = re.compile(r'\s*(?:m?K\b|(?i:kelvins?)\b)')


def _strip_value_units(s: str) -> str:
    """Remove Kelvin-unit tokens from a raw tcValue string, e.g. "36 K" ->
    "36", "up to 38 K" -> "up to 38", "16K to 26K" -> "16 to 26". Leaves
    inequality symbols (<, >, ~, ∼, ≈, ...), range dashes, and qualifier
    words (up to, from ... to, below, above, ...) untouched. The leading
    \\s* in _VALUE_UNIT_RE already consumes the unit's own preceding space,
    so no separate whitespace collapse is applied -- adding one would risk
    touching whitespace unrelated to a unit, beyond what was asked for."""
    return _VALUE_UNIT_RE.sub('', s).strip()


def _parse_tcvalue(raw: object) -> tuple[float | None, str | None]:
    """Returns (value, units). value=None signals the row should be dropped.

    Ground truth must be a single, unambiguous reported value -- matching the
    convention already used by pond/nfix (whose source data never contained
    ranges/bounds/approximations to begin with) and the judge's own criterion
    that a value merely inferred as a range endpoint should not be accepted.
    Rows reported as a range ("7-8 K"), an approximation ("~30 K", "∼2.3 K"),
    or a bound ("< 10 K", "> 30 K", "up to 33 K") are dropped rather than
    collapsed to a synthesized midpoint/endpoint.
    """
    s = str(raw).strip()
    if s.lower() in _JUNK_TCVALUES or not s:
        return None, None
    if _QUALIFIER_RE.search(s) or _RANGE_RE.search(s):
        return None, None

    num_match = _NUMBER_RE.search(s)
    numeric = float(num_match.group(0)) if num_match else None
    if numeric is None:
        return None, None

    units = "mK" if re.search(r'\bmk\b', s, re.I) else "K"
    return numeric, units


def _raw_tcvalue(raw: object) -> tuple[str | None, str | None]:
    """Returns (value, units) for the qualifiers ground truth: value is the
    raw reported tcValue text, verbatim -- no numeric parsing, and rows are
    no longer dropped for being a range/approximation/bound/junk-word/
    unparseable. The only drop reason is a genuinely absent tcValue
    (NaN/empty after stripping) -- there is no text to carry over at all.

    Unit tokens (K/mK/kelvin) are stripped from `value` later, in
    build_ground_truth, *after* page attribution runs -- page attribution's
    table pass float()-parses `value` and the unstripped text ("36 K")
    reliably fails that parse the same way a genuinely non-numeric qualifier
    ("up to 36 K") does, which is what keeps page attribution's behavior
    unaffected by this stripping. Stripping here instead would make plain
    values parse as floats and activate a code path that was never exercised
    for this GT before, silently changing page_number/page_score/
    page_confidence for hundreds of rows -- see the qualifiers-unit-strip
    build note for the numbers.

    units is still derived the same mK/K way as _parse_tcvalue: `tc` is
    always a temperature for this dataset regardless of the value's shape,
    so this is units bookkeeping for the attribute, not value parsing.
    """
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, None
    s = str(raw).strip()
    if not s:
        return None, None
    units = "mK" if re.search(r'\bmk\b', s, re.I) else "K"
    return s, units


# ---------------------------------------------------------------------------
# Qualifier-field parsing (fills point_value/lower/upper/tolerance/etc. from
# the raw tcValue text the qualifiers pipeline above keeps in `value`)
# ---------------------------------------------------------------------------

# OCR/typesetting variants seen in raw_data.csv's tcValue column, on top of
# the dash/approx-mark unification parse_quantity_shape already does: 'À'
# standing in for a dash between two digits (a font-mapping corruption
# specific to this OCR'd corpus -- e.g. "26À28" for "26-28"), and a colon
# directly between two digits standing in for a decimal point (e.g. "46:5"
# for "46.5"). Both are narrow enough (only ever between two digits) to be
# safe here but are NOT part of the shared parser -- they're an artifact of
# this specific OCR pass, not something real LLM output would produce.
_OCR_DASH_BETWEEN_DIGITS_RE = re.compile(r'(?<=\d)À(?=\d)')
_OCR_COLON_DECIMAL_RE = re.compile(r'(?<=\d):(?=\d)')
# A lone trailing ')' or '-' with nothing after it and no matching '(' --
# a pre-existing raw-data/unit-stripping artifact (see the 2026-09-24
# qualifiers-unit-strip build note's "orphan punctuation" rows), not range
# or bound syntax. Hand-verified against the OCR source text for all 4 rows
# this matches in the current corpus ("18-K"/"54.6-K" -> stray trailing
# dash; "6.3 K)"/"13 K)" -> stray trailing paren) that the number itself is
# a plain point value. Also corpus-specific -- not part of the shared parser.
_ORPHAN_PUNCTUATION_RE = re.compile(r'^(-?\d+\.?\d*)[)\-]$')


def _normalize_qualifier_text(s: str) -> str:
    s = normalize_dash_and_approx_marks(s)
    s = _OCR_DASH_BETWEEN_DIGITS_RE.sub('-', s)
    s = _OCR_COLON_DECIMAL_RE.sub('.', s)
    return re.sub(r'\s+', ' ', s).strip()


def _parse_qualifiers_text(raw_value: str) -> dict:
    """Parse one qualifiers-GT `value` string (already unit-stripped) into the
    7 QUALIFIER_FIELDS, with point_value/lower/upper left as the raw number
    strings matched (see `_parse_qualifiers`, which converts them to float).

    Applies this corpus's own OCR-artifact fixups (`_normalize_qualifier_text`)
    and orphan-punctuation check first, then delegates everything else to
    `scholarlm.utils.parsing.parse_quantity_shape` with
    `reinterpret_implausible_tolerance=True` (an OCR-corrupted range dash,
    e.g. "37 ± 38" for "37-38 K" -- hand-verified against the OCR source text,
    see notes/scholarlm/builds/2026-09-24-supermat-qualifiers-parsing-01.md)
    and `strict_compact_uncertainty=True` (a parenthesized-uncertainty digit
    count exceeding the value's own decimal places has always meant a bug
    worth catching by hand in this corpus, not a row to leave unparsed) --
    see that function's docstring for why neither is the shared default.
    """
    out = {field: None for field in QUALIFIER_FIELDS}
    s = _normalize_qualifier_text(raw_value)

    m = _ORPHAN_PUNCTUATION_RE.match(s)
    if m:
        out["qualifiers"] = []
        out["point_value"] = m.group(1)
        return out

    return parse_quantity_shape(
        s, reinterpret_implausible_tolerance=True, strict_compact_uncertainty=True,
    )


def _parse_qualifiers(raw_value: str) -> dict:
    """`_parse_qualifiers_text`, with point_value/lower/upper converted to
    float -- matching pond/nfix's `point_value` convention (copied from
    their already-numeric `value`). tolerance/list_values stay strings,
    matching `PARSE_QUANTITY_INSTRUCTIONS`/the supermat NuExtract few-shot
    examples in `experiments/dataset-configs/supermat.py` ("± 2", not 2.0).
    """
    out = _parse_qualifiers_text(raw_value)
    for field in ("point_value", "lower", "upper"):
        if out[field] is not None:
            out[field] = float(out[field])
    return out


# ---------------------------------------------------------------------------
# Page attribution
# ---------------------------------------------------------------------------


def _add_page_attribution(gt: pd.DataFrame, ocr_dir: Path) -> pd.DataFrame:
    """Append page_number, page_score, and page_confidence columns to *gt*."""
    gt = gt.copy()
    gt["page_number"] = pd.NA
    gt["page_score"] = pd.NA
    gt["page_confidence"] = pd.NA

    n_attributed = 0
    n_missing_ocr = 0
    confidence_counts: Counter[str] = Counter()

    for doc_id, group in gt.groupby("document_id"):
        ocr_path = ocr_dir / f"{doc_id}.txt"
        if not ocr_path.exists():
            n_missing_ocr += 1
            continue

        parsed = parse_ocr(ocr_path)
        for idx, row in group.iterrows():
            result = attribute_page(row.to_dict(), parsed, SUPERMAT_WEIGHTS)
            gt.at[idx, "page_number"] = result["candidates"]
            gt.at[idx, "page_score"] = result["score"]
            gt.at[idx, "page_confidence"] = result["confidence"]
            confidence_counts[result["confidence"]] += 1
            n_attributed += 1

    total = len(gt)
    print(f"  Page attribution: {n_attributed:,}/{total:,} rows attributed "
          f"({n_missing_ocr} docs with missing OCR)")
    print(f"  Confidence distribution: {dict(confidence_counts)}")
    return gt


# ---------------------------------------------------------------------------
# Ground truth builder
# ---------------------------------------------------------------------------

# QUALIFIER_FIELDS is now imported from scholarlm.utils.parsing (see the
# import block above) -- it must still match ParseQuantityResponse in
# src/scholarlm/measurementlm.py exactly, minus `explanation` (a
# generation-only field, not part of the GT record).


def build_ground_truth(raw_path: Path, out_dir: Path, *, build_qualifiers: bool = False) -> None:
    """Build ground_truth.json and ground_truth_ten.json from raw_data.csv.

    Output schema: document_id, name, identifiers, sample_details, pressure,
    me_method, additional_details, attribute, value, units[, page_number,
    page_score, page_confidence].

    build_qualifiers=True switches to the qualifiers ground truth instead:
    `value` becomes the raw, unparsed tcValue text (via `_raw_tcvalue`
    instead of `_parse_tcvalue`), with unit tokens (K/mK/kelvin) stripped out
    at the end via `_strip_value_units` -- inequality/range symbols and
    qualifier words (up to, from ... to, below, above, ...) are left as-is.
    Rows are no longer dropped for being a range/approximation/bound/
    unparseable -- only a genuinely absent tcValue still drops a row. The 7
    qualifier/shape fields (`QUALIFIER_FIELDS`) are then filled in by
    `_parse_qualifiers` from that same raw text: a plain point value becomes
    `point_value`, and ranges/inequalities/approximations/tolerances/
    parenthetical-uncertainty notation/lists are parsed into the matching
    fields and tagged in `qualifiers`. A row `_parse_qualifiers` can't
    confidently parse (free text, or too garbled to disambiguate) is left
    all-null, same as before this parsing step existed. Writes only
    `ground_truth_qualifiers.json` -- no ten-paper subset for this path.
    `ground_truth.json`/`ground_truth_ten.json` are untouched by this flag.
    """
    df = pd.read_csv(raw_path, encoding_errors="ignore")
    df = df.drop(columns=["id"])

    df["document_id"] = df["filename"].str.replace(
        r"\.superconductors\.tei\.xml$", "", regex=True
    )
    df["document_id"] = df["document_id"].replace(_DOCUMENT_ID_REMAP)
    n_before = len(df)
    df = df[~df["document_id"].isin(_NO_PDF_CODES)].reset_index(drop=True)
    print(f"  Dropped {n_before - len(df):,} rows with no local PDF "
          f"({sorted(_NO_PDF_CODES)})")

    # Clean material -> name / sample_details / identifiers, with per-document
    # forward-fill for bare-modifier rows.
    names: list[str | None] = []
    sample_details: list[str | None] = []
    identifiers: list[str | None] = []
    last_named: dict[str, str] = {}
    for doc_id, material in zip(df["document_id"], df["material"]):
        name, sd, ident = _clean_material(material)
        if name is not None:
            last_named[doc_id] = name
        else:
            name = last_named.get(doc_id)
        names.append(name)
        sample_details.append(sd)
        identifiers.append(ident)
    df["name"] = names
    df["sample_details"] = sample_details
    df["identifiers"] = identifiers

    df["me_method"] = df["me_method"].apply(_normalize_me_method)
    df["pressure"] = df["pressure"].apply(_normalize_pressure)

    if build_qualifiers:
        parsed_tc = df["tcValue"].apply(_raw_tcvalue)
    else:
        parsed_tc = df["tcValue"].apply(_parse_tcvalue)
    df["value"] = [t[0] for t in parsed_tc]
    df["units"] = [t[1] for t in parsed_tc]
    # raw_data.csv has no separate Tc-criterion column; additional_details is an
    # extraction-only field (see configs/supermat.py) with nothing to populate here.
    df["additional_details"] = None

    n_before = len(df)
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    if build_qualifiers:
        print(f"  Dropped {n_before - len(df):,} rows with a genuinely absent tcValue "
              f"(ranges/approximates/bounds/junk kept verbatim)")
    else:
        print(f"  Dropped {n_before - len(df):,} rows with unparseable, junk, or "
              f"qualified (range/approximate/bounded) tcValue")

    df["attribute"] = "tc"

    final_cols = [
        "document_id", "name", "identifiers", "sample_details", "pressure",
        "me_method", "additional_details", "attribute", "value", "units",
    ]
    if build_qualifiers:
        # Real values are filled in below, after page attribution and unit-
        # stripping -- these null placeholders only fix column order for now.
        for field in QUALIFIER_FIELDS:
            df[field] = None
        final_cols = final_cols + QUALIFIER_FIELDS
    df_final = df[final_cols].reset_index(drop=True)

    ocr_dir = BASE / "ocr_output_raw"
    if ocr_dir.exists():
        df_final = _add_page_attribution(df_final, ocr_dir)
    else:
        print(f"  Skipping page attribution: {ocr_dir} not found "
              f"(run experiments/run_ocr.py --dataset supermat first)")

    if build_qualifiers:
        # Strip unit tokens (K/mK/kelvin) from `value` only after page
        # attribution has run against the raw text -- see _raw_tcvalue's
        # docstring for why the order matters.
        df_final["value"] = df_final["value"].apply(_strip_value_units)
        assert (df_final["value"] != "").all(), "unit-stripping left an empty value"

        # Parse the (now unit-stripped) raw text into the qualifier fields.
        parsed = df_final["value"].apply(_parse_qualifiers)
        for field in QUALIFIER_FIELDS:
            df_final[field] = [p[field] for p in parsed]
        n_plain = sum(1 for p in parsed if not p["qualifiers"] and p["point_value"] is not None)
        n_tagged = sum(1 for p in parsed if p["qualifiers"])
        n_unparsed = sum(1 for p in parsed if not p["qualifiers"] and p["point_value"] is None)
        print(f"  Parsed qualifier fields: {n_plain:,} plain, {n_tagged:,} tagged "
              f"(range/approximate/tolerance/list), {n_unparsed:,} left unparsed")

        df_final.to_json(out_dir / "ground_truth_qualifiers.json", orient="records", indent=2)
        print(f"  Saved {len(df_final):,} rows -> ground_truth_qualifiers.json")
        return

    df_final.to_json(out_dir / "ground_truth.json", orient="records", indent=2)
    print(f"  Saved {len(df_final):,} rows -> ground_truth.json")

    top_papers = (
        df_final["document_id"].value_counts().head(_TOP_PAPERS_N).index.tolist()
    )
    gt_ten = df_final[df_final["document_id"].isin(top_papers)].reset_index(drop=True)
    gt_ten.to_json(out_dir / "ground_truth_ten.json", orient="records", indent=2)
    print(f"  Saved {len(gt_ten):,} rows -> ground_truth_ten.json "
          f"({len(top_papers)} papers: {top_papers})")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--qualifiers", action="store_true",
        help="Build only ground_truth_qualifiers.json (raw, undropped tcValue text, "
             "parsed into the qualifier/shape fields where confidently parseable); "
             "leaves ground_truth.json/ground_truth_ten.json untouched.",
    )
    args = parser.parse_args(argv)

    raw_path = BASE / "raw_data.csv"
    if args.qualifiers:
        print("Building qualifiers ground truth JSON ...")
        build_ground_truth(raw_path, BASE, build_qualifiers=True)
    else:
        print("Building ground truth JSONs ...")
        build_ground_truth(raw_path, BASE)


if __name__ == "__main__":
    main()
