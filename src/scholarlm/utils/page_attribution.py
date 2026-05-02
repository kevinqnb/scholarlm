"""Page attribution for scientific PDF OCR text.

Scores each page of an OCR document against a ground-truth measurement row
and returns the most likely page number along with a confidence tier.

OCR file conventions (both datasets)
--------------------------------------
- Pages:  ``<page number="N">`` where N is **0-indexed**.
  Output page numbers are 0-indexed (N), matching the OCR tags directly.
- Tables: ``<table number="N">`` where N is **1-indexed** and matches the
  table numbers cited in ``extraction_location_details`` strings.

Attribution strategy
--------------------
Two-pass approach inside :func:`attribute_page`:

1. **Table pass** — search for the numeric value in table cells (exact match
   within ``rtol``).  If the value is found in one or more tables, scoring is
   restricted to only those pages.  This typically narrows candidates to 1–2
   pages.
2. **Text pass** — if no table match is found, fall back to weighted fuzzy
   scoring across the full candidate page set (or all pages if no candidates
   were pre-specified).

Exported API
------------
parse_ocr                       -- parse an OCR .txt file into pages + table index
parse_numeric                   -- parse a string to float with OCR-correction
find_numeric_in_text            -- score how well a numeric value appears in text
find_value_in_tables            -- find pages where a value appears in a table cell
score_field                     -- fuzzy text score for a field value vs. page text
score_page                      -- weighted aggregate score for a row vs. a page
attribute_page                  -- pick the best page for a measurement row (two-pass)
extract_table_numbers           -- parse extraction_location_details → table list
get_candidate_pages_from_tables -- map table numbers to page numbers via OCR index

Default weight sets
-------------------
NFIX_WEIGHTS, POND_WEIGHTS -- recommended starting weights for each dataset.
Adjust after empirical spot-checking.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default weight sets
# ---------------------------------------------------------------------------

NFIX_WEIGHTS: dict[str, float] = {
    "value":       10.0,
    "name":         4.0,
    "attribute":    2.0,
    "units":        1.5,
    "nfix_method":  1.5,
    "site_type":    1.0,
    "date":         1.0,
    "location":     1.0,
}

POND_WEIGHTS: dict[str, float] = {
    "value":       10.0,
    "name":         4.0,
    "attribute":    2.0,
    "units":        1.5,
    "location":     1.0,
    "ecosystem":    1.0,
}

# ---------------------------------------------------------------------------
# OCR parsing
# ---------------------------------------------------------------------------

_PAGE_RE = re.compile(r'<page number="(\d+)">(.*?)</page>', re.DOTALL)
# Captures full table content (for cell extraction); non-greedy so multiple
# tables per page are matched independently.
_TABLE_CONTENT_RE = re.compile(r'<table number="(\d+)">(.*?)</table>', re.DOTALL)
# Extracts text from <td> and <th> cells (handles optional attributes like rowspan).
_CELL_RE = re.compile(r'<t[dh][^>]*>(.*?)</t[dh]>', re.DOTALL | re.IGNORECASE)
# Strips all remaining HTML/LaTeX tags for clean cell text.
_STRIP_TAGS_RE = re.compile(r'<[^>]+>|\\\([^)]*\)|\\\[[^\]]*\]')


def _cell_texts(table_html: str) -> list[str]:
    """Extract stripped text from all <td>/<th> cells in a table HTML fragment."""
    texts = []
    for m in _CELL_RE.finditer(table_html):
        raw = _STRIP_TAGS_RE.sub("", m.group(1)).strip()
        if raw:
            texts.append(raw)
    return texts


def parse_ocr(filepath: str | Path) -> dict:
    """Parse an OCR ``.txt`` file into a structured page/table index.

    Returns
    -------
    dict with three keys:

    ``"pages"``
        ``{page_num: {"text": str, "tables": list[int]}}``
        where ``page_num`` is 0-indexed, matching the OCR ``<page number="N">`` tags.

    ``"table_to_page"``
        ``{table_num: page_num}`` reverse lookup.
        Table numbers are 1-indexed and match ``extraction_location_details``.

    ``"table_cells"``
        ``{table_num: [cell_text, ...]}`` — stripped text from every
        ``<td>`` / ``<th>`` cell in each table.  Used by
        :func:`find_value_in_tables` for the first-pass table search.
    """
    text = Path(filepath).read_text(encoding="utf-8", errors="replace")
    pages: dict[int, dict] = {}
    table_to_page: dict[int, int] = {}
    table_cells: dict[int, list[str]] = {}

    for m in _PAGE_RE.finditer(text):
        page_num = int(m.group(1))  # 0-indexed, matching OCR <page number="N"> tags
        content = m.group(2)

        table_nums: list[int] = []
        for tm in _TABLE_CONTENT_RE.finditer(content):
            tn = int(tm.group(1))
            table_nums.append(tn)
            table_to_page[tn] = page_num
            table_cells[tn] = _cell_texts(tm.group(2))

        pages[page_num] = {"text": content, "tables": table_nums}

    return {"pages": pages, "table_to_page": table_to_page, "table_cells": table_cells}


# ---------------------------------------------------------------------------
# Numeric parsing
# ---------------------------------------------------------------------------

_UNICODE_SUPERSCRIPT = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁺", "0123456789-+")

# "× 10^{N}" or "× 10N" → "eN" (after superscript translation)
_SCI_CROSS_RE = re.compile(
    r"[×x]\s*10"
    r"(?:\^\{?([+-]?\d+)\}?"   # explicit caret: × 10^{-2}
    r"|([+-]?\d+))",            # bare digits: × 10-2 (after superscript translation)
    re.UNICODE,
)

# Numeric tokens: optional minus (ASCII or Unicode), digits with optional
# decimal, optional e-notation exponent.
_TOKEN_RE = re.compile(
    r"[-−]?"              # optional minus (ASCII - or Unicode −)
    r"\d+(?:[.,]\d+)?"         # integer or decimal (comma or period)
    r"(?:[eE][+-]?\d+)?",      # optional e-notation exponent
)


def parse_numeric(text: str) -> float | None:
    """Parse a string to float with common OCR-correction heuristics.

    Applies in order: Unicode superscript normalisation, ``× 10^N`` scientific
    notation conversion, whitespace/apostrophe removal, European decimal-comma
    substitution, Unicode minus normalisation, and OCR character confusion
    fixes (``O`` → ``0``, ``l`` → ``1``).

    Returns ``None`` if the string still cannot be parsed after all substitutions.
    """
    s = text.strip()
    if not s:
        return None

    s = s.translate(_UNICODE_SUPERSCRIPT)

    def _cross_sub(m: re.Match) -> str:
        exp = m.group(1) or m.group(2)
        return f"e{exp}"

    s = _SCI_CROSS_RE.sub(_cross_sub, s)
    s = re.sub(r"['\s]+", "", s)                    # apostrophe thousands-sep and whitespace
    s = re.sub(r"(\d),(\d)", r"\1.\2", s)           # European decimal comma
    s = s.replace("−", "-")                     # Unicode minus sign → ASCII hyphen
    s = s.replace("O", "0").replace("l", "1")       # OCR character confusions

    try:
        return float(s)
    except ValueError:
        return None


def _preprocess_text(text: str) -> str:
    """Normalise text so _TOKEN_RE can capture scientific-notation numbers."""
    t = text.translate(_UNICODE_SUPERSCRIPT)
    t = _SCI_CROSS_RE.sub(lambda m: f"e{m.group(1) or m.group(2)}", t)
    t = t.replace("−", "-")   # Unicode minus → ASCII so _TOKEN_RE matches it
    return t


def find_numeric_in_text(value: float, text: str, rtol: float = 1e-4) -> float:
    """Score how well *value* appears as a numeric token in *text*.

    Returns
    -------
    ``1.0``
        if any token matches *value* within relative tolerance *rtol*.
    A decayed score in ``[0.1, 1.0)``
        based on the closest match: ``1 / (1 + relative_error)``.
        The floor of ``0.1`` ensures pages with no plausible number are
        neither rewarded nor fully penalised.
    """
    normalized = _preprocess_text(text)
    tokens = _TOKEN_RE.findall(normalized)

    best_score = 0.1
    for token in tokens:
        parsed = parse_numeric(token)
        if parsed is None:
            continue
        if value == 0.0:
            if abs(parsed) < 1e-12:
                return 1.0
            continue
        rel_err = abs(parsed - value) / abs(value)
        if rel_err <= rtol:
            return 1.0
        score = 1.0 / (1.0 + rel_err)
        if score > best_score:
            best_score = score

    return best_score


# ---------------------------------------------------------------------------
# Table-cell value search
# ---------------------------------------------------------------------------

def find_value_in_tables(
    value: float,
    parsed_ocr: dict,
    rtol: float = 1e-4,
    restrict_to_pages: set[int] | None = None,
) -> list[int]:
    """Find pages where *value* appears as a numeric token in a table cell.

    Iterates over all tables in the document (or only those on
    *restrict_to_pages* if provided) and checks every cell using
    :func:`find_numeric_in_text`.  A cell is a hit when its score is ``1.0``
    (i.e. an exact match within *rtol*).

    Returns a sorted list of 1-indexed page numbers where the value was found
    in at least one table cell.  Returns an empty list if no table match exists.
    """
    table_to_page = parsed_ocr["table_to_page"]
    table_cells = parsed_ocr["table_cells"]

    matching_pages: set[int] = set()
    for tn, cells in table_cells.items():
        page = table_to_page.get(tn)
        if page is None:
            continue
        if restrict_to_pages is not None and page not in restrict_to_pages:
            continue
        for cell_text in cells:
            if find_numeric_in_text(value, cell_text, rtol) == 1.0:
                matching_pages.add(page)
                break  # one matching cell is enough to flag this table's page

    return sorted(matching_pages)


# ---------------------------------------------------------------------------
# Fuzzy field scoring
# ---------------------------------------------------------------------------

def score_field(field_value: object, page_text: str) -> float:
    """Score how well *field_value* appears in *page_text* using fuzzy matching.

    Returns a score in ``[0, 1]``.  ``None`` or empty values return ``0.5``
    (neutral — neither rewarding nor penalising missing data).
    Strings longer than 50 characters use ``token_set_ratio``; shorter ones
    use ``partial_ratio``.
    """
    if field_value is None:
        return 0.5
    s = str(field_value).strip()
    if not s or s.lower() == "nan":
        return 0.5
    if len(s) > 50:
        return fuzz.token_set_ratio(s, page_text) / 100.0
    return fuzz.partial_ratio(s, page_text) / 100.0


# ---------------------------------------------------------------------------
# Page scoring
# ---------------------------------------------------------------------------

def score_page(row: dict, page_text: str, weights: dict) -> float:
    """Compute a weighted score for *row* against *page_text*.

    The ``"value"`` key is scored via :func:`find_numeric_in_text`; all other
    keys are scored via :func:`score_field`.  Returns a weighted average in
    ``[0, 1]``.
    """
    total_weight = sum(weights.values())
    if total_weight == 0.0:
        return 0.0

    weighted_sum = 0.0
    for field, weight in weights.items():
        if field == "value":
            raw = row.get("value")
            if raw is not None:
                try:
                    s = find_numeric_in_text(float(raw), page_text)
                except (ValueError, TypeError):
                    s = score_field(raw, page_text)
            else:
                s = 0.5
        else:
            s = score_field(row.get(field), page_text)
        weighted_sum += weight * s

    return weighted_sum / total_weight


# ---------------------------------------------------------------------------
# Document-level attribution (two-pass)
# ---------------------------------------------------------------------------

def attribute_page(
    row: dict,
    parsed_ocr: dict,
    weights: dict,
    candidate_pages: list[int] | None = None,
    delta: float = 0.05,
) -> dict:
    """Attribute a ground-truth row to its most likely page in the OCR.

    Uses a two-pass strategy:

    1. **Table pass** — search for the row's numeric value in table cells
       (restricted to *candidate_pages* if given).  If the value is found on
       one or more pages, scoring is limited to those pages.
    2. **Text pass** — if no table match is found, score all pages in
       *candidate_pages* (or all pages) with weighted fuzzy matching.

    Parameters
    ----------
    row:
        Ground-truth row as a dict.  Keys should match the *weights* dict.
    parsed_ocr:
        Output of :func:`parse_ocr`.
    weights:
        Field-name → weight mapping.
    candidate_pages:
        If provided, restrict both the table search and the text scoring to
        these pages; otherwise use all pages.  Pages not present in the OCR
        are silently skipped (with fallback to all pages if the filtered set
        is empty).
    delta:
        Margin threshold for confidence tiers (default 0.05).

    Returns
    -------
    dict with:

    ``"page"``
        Best page number (1-indexed), or ``None`` if the OCR has no pages.
    ``"score"``
        Score of the best page (rounded to 4 decimal places).
    ``"candidates"``
        Sorted list of all pages within *delta* of the top score.
    ``"confidence"``
        One of ``"high"``, ``"medium"``, or ``"ambiguous"``.
        (Callers may override to ``"table-anchored"`` when appropriate.)

    Confidence tiers
    ----------------
    - ``"high"``      : single candidate, score > 0.7, margin to runner-up > delta
    - ``"medium"``    : single candidate but score <= 0.7 or margin <= delta
    - ``"ambiguous"`` : multiple candidates within delta of top score
    """
    all_pages = parsed_ocr["pages"]

    # Resolve the initial candidate page set
    if candidate_pages is not None:
        page_set = [p for p in candidate_pages if p in all_pages]
        if not page_set:
            page_set = list(all_pages.keys())
    else:
        page_set = list(all_pages.keys())

    if not page_set:
        return {"page": None, "score": 0.0, "candidates": [], "confidence": "ambiguous"}

    # Pass 1: look for the numeric value in table cells
    score_pages = page_set  # default: full candidate set
    raw_value = row.get("value")
    if raw_value is not None:
        try:
            numeric_value = float(raw_value)
            table_hit_pages = find_value_in_tables(
                numeric_value, parsed_ocr, restrict_to_pages=set(page_set)
            )
            if table_hit_pages:
                score_pages = table_hit_pages
        except (ValueError, TypeError):
            pass

    # Pass 2: weighted fuzzy scoring on the selected page set
    page_scores = {
        pn: score_page(row, all_pages[pn]["text"], weights)
        for pn in score_pages
    }

    sorted_pages = sorted(page_scores.items(), key=lambda x: -x[1])
    top_page, top_score = sorted_pages[0]

    candidates = sorted(p for p, s in page_scores.items() if top_score - s <= delta)

    if len(candidates) == 1:
        margin = (top_score - sorted_pages[1][1]) if len(sorted_pages) > 1 else top_score
        confidence = "high" if (top_score > 0.7 and margin > delta) else "medium"
    else:
        confidence = "ambiguous"

    return {
        "page": top_page,
        "score": round(top_score, 4),
        "candidates": candidates,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# nfix table pre-filtering helpers
# ---------------------------------------------------------------------------

# Capture digits/spaces/semicolons/commas after "table(s)" — stops at any letter
# (so "supplemental tables s3; s5" captures nothing after "tables ")
_TABLE_SECTION_RE = re.compile(r"\btables?\s+([\d\s;,]+)")
_SECTION_BREAK_RE = re.compile(r"[-–—]|figure|text|supplement|abstract|page")


def extract_table_numbers(detail_string: str) -> list[int] | None:
    """Parse an ``extraction_location_details`` string into integer table numbers.

    Examples
    --------
    ``"table 1 - results"``               → ``[1]``
    ``"tables 2; 3; 4; 6; 8 - figures"`` → ``[2, 3, 4, 6, 8]``
    ``"table 6"``                         → ``[6]``
    ``"page 95; abstract"``               → ``None``
    ``"supplemental tables s3; s5"``      → ``None``  (no digits follow "tables")
    ``"figure 2 - text"``                 → ``None``

    Returns ``None`` when no table numbers can be found.
    """
    if not detail_string:
        return None
    s = str(detail_string).strip()
    if s.lower() in ("nan", ""):
        return None

    s_lower = s.lower()
    if "table" not in s_lower:
        return None

    table_nums: list[int] = []
    for m in _TABLE_SECTION_RE.finditer(s_lower):
        # Stop at section separators so "table 2 - page 461" → [2] not [2, 461]
        nums_str = _SECTION_BREAK_RE.split(m.group(1))[0]
        for n in re.findall(r"\d+", nums_str):
            table_nums.append(int(n))

    return table_nums if table_nums else None


def get_candidate_pages_from_tables(
    table_numbers: list[int],
    parsed_ocr: dict,
) -> list[int]:
    """Map table numbers to page numbers using the OCR's table index.

    Tables missing from the OCR (OCR likely missed them) are logged as warnings
    and omitted.  Returns a deduplicated, sorted list of page numbers.
    """
    table_to_page = parsed_ocr["table_to_page"]
    pages: set[int] = set()
    for tn in table_numbers:
        if tn in table_to_page:
            pages.add(table_to_page[tn])
        else:
            logger.warning("Table %d not found in OCR index (OCR may have missed it)", tn)
    return sorted(pages)
