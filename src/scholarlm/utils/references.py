"""Strip a references/bibliography section from page-wrapped OCR text.

Applied once, generically, by ``DocumentLM.fit()`` to the assembled
``<page number="N">...</page>`` document text *before* the chandra-ocr-2 /
olmOCR format branch -- not duplicated per-formatter. See
notes/scholarlm/builds/2026-08-20-per-document-isolation-01.md.

Detects a references/bibliography/literature-cited/works-cited heading
(case-insensitive, optionally numbered, e.g. "6. References") in any of the
shapes seen in practice:

* chandra-ocr-2: the heading is the entire text content between a tag-close and
  the next tag-open, e.g. ``<h2>6. References</h2>``.
* olmOCR: the heading is a bare, unmarked-up entire line, e.g. ``REFERENCES``.
* chandra-ocr-2, div-encoded: the reference list is a single
  ``<div data-label="Bibliography">`` (or ``"References"``) with no heading
  text node of its own; or a ``<div data-label="Section-Header">`` whose
  heading text isn't sitting directly between ``>`` and ``<``. Both are cut at
  the ``<div`` itself. See
  notes/scholarlm/builds/2026-09-01-chandra-unknown-label-fallback-01.md.

Everything from the heading to end-of-document is dropped -- *unless* a bare
resume heading (olmOCR shape: an unmarked-up entire line matching
``_RESUME_HEADING_RE`` -- appendix, figures/tables, or supplementary
material) is found after it, in which case only the
references-through-resume-heading span is dropped and everything from that
heading onward is kept -- these commonly land *after* the references section
in a paper's layout and aren't reference-list content themselves. Running
this before the chandra branch means a div inside the dropped region can
carry any data-label -- including one ``chandra_format.py`` doesn't
recognize -- without ever reaching its label validation.
"""
import re

_HEADING_WORDS = r"(?:references|bibliography|literature cited|works cited)"
_NUMBER_PREFIX = r"(?:\d+\.?\s*)?"

# chandra shape: heading text sitting between a tag-close and the next tag-open.
_TAG_HEADING_RE = re.compile(
    rf">\s*{_NUMBER_PREFIX}{_HEADING_WORDS}\s*<", re.IGNORECASE
)
# olmOCR shape: heading as a bare, entire line.
_LINE_HEADING_RE = re.compile(
    rf"^[ \t]*{_NUMBER_PREFIX}{_HEADING_WORDS}[ \t]*$", re.IGNORECASE | re.MULTILINE
)

_PAGE_OPEN_RE = re.compile(r'<page number="\d+">')

# chandra div-encoded shapes the two heading regexes above miss: a reference list
# emitted as a single <div data-label="Bibliography">...</div> carrying no
# heading text node, or a <div data-label="Section-Header"> whose heading text
# isn't sitting directly between a '>' and a '<' (e.g. "References Cited",
# "7 References and notes"). See
# notes/scholarlm/builds/2026-09-01-chandra-unknown-label-fallback-01.md.
_DIV_OPEN_RE = re.compile(r'<div\b[^>]*\bdata-label="([^"]*)"[^>]*>', re.IGNORECASE)
_BIBLIOGRAPHY_LABELS = {"bibliography", "references"}
_SECTION_HEADER_LABEL = "section-header"
_TAG_STRIP_RE = re.compile(r"<[^>]+>")
_HEADING_TEXT_RE = re.compile(
    rf"^\s*{_NUMBER_PREFIX}{_HEADING_WORDS}\b", re.IGNORECASE
)

# Resuming past the references section, olmOCR shape: a bare, otherwise-
# unmarked-up heading line for a section that commonly lands *after*
# references in a paper's layout and carries appendix-like content -- e.g.
# "APPENDIX A", "LIST OF FIGURES", "SUPPORTING INFORMATION", "ELECTRONIC
# SUPPLEMENTARY MATERIAL", "SUPPLEMENTARY DATA". Enumerated rather than a
# generic "any bare heading-like line" heuristic: a real reference list is
# full of short all-caps/title-case lines of its own (author names, journal
# titles, DOI lines) that a generic heading-shape match would false-trigger
# on, resuming too early and keeping reference-list content -- wrong in the
# opposite direction from the bug this fixes. Deliberately excludes
# metadata-only sections (data availability, additional files, online
# resource) that aren't appendix-like content and, worse, commonly appear as
# a bare "Data availability:" line *inside* the tail of a reference/citation
# block -- adding them measurably raised the false-resume rate without
# advancing the extraction goal.
#
# Multi-word phrases (case-insensitive, arbitrary trailing text via the
# grouped alternation below) are safe to match loosely, the same way
# "appendix" is: a reference-list line essentially never happens to start
# with "supporting information" or "electronic supplementary material".
_RESUME_PHRASES = (
    r"(?:appendix|appendices"
    r"|list of figures|list of tables"
    r"|electronic supplementary material"
    r"|supplementary material|supplementary information|supplementary data"
    r"|supplemental material|supplemental information|supplemental data"
    r"|supporting information)"
)
# "figure(s)"/"table(s)" bare are common English words that legitimately
# start a citation-continuation line inside a real reference list (an
# OCR-wrapped "Tables 2-4 in Smith et al. (2019) show that..." starts with
# title-case "Tables", not a heading). Restricted to whole-word, case-
# SENSITIVE all-caps -- matching this codebase's existing olmOCR bare-heading
# convention (see the REFERENCES/BIBLIOGRAPHY fixtures in test_references.py)
# -- since citation prose is essentially never rendered as "FIGURES"/"TABLES"
# in full caps. Scoped inline ``(?i:...)`` keeps _RESUME_PHRASES
# case-insensitive while this alternative stays case-sensitive in the same
# compiled pattern.
_RESUME_HEADING_RE = re.compile(
    rf"^[ \t]*(?:(?i:{_RESUME_PHRASES})|FIGURES?|TABLES?)\b.*$", re.MULTILINE
)

_PAGE_OPEN_NUM_RE = re.compile(r'<page number="(\d+)">')
_PAGE_TAG_TOKEN_RE = re.compile(r'<page number="(\d+)">|</page>')


def _bibliography_label_cut(doc_text: str) -> int | None:
    """Offset of the first ``<div data-label="Bibliography">`` / ``"References"``.

    Cut at the ``<div`` itself so the labelled open tag doesn't survive to reach
    chandra's label validation. Pooled with the two heading-text regexes (not a
    fallback): a bibliography-labelled div can legitimately sit *earlier* in the
    document than an unrelated later heading-text match.
    """
    for m in _DIV_OPEN_RE.finditer(doc_text):
        if m.group(1).strip().lower() in _BIBLIOGRAPHY_LABELS:
            return m.start()
    return None


def _section_header_text_cut(doc_text: str) -> int | None:
    """Offset of the first ``<div data-label="Section-Header">`` whose stripped
    text content starts with a references/bibliography heading word.

    Fallback only -- consulted when every other detector missed -- because a
    Section-Header div whose heading text *is* sitting between ``>`` and ``<``
    is already caught by ``_TAG_HEADING_RE`` at a well-defined offset, and
    pooling this looser match would shift that cut point.
    """
    for m in _DIV_OPEN_RE.finditer(doc_text):
        if m.group(1).strip().lower() != _SECTION_HEADER_LABEL:
            continue
        close = doc_text.find("</div>", m.end())
        inner = doc_text[m.end():close] if close != -1 else doc_text[m.end():]
        if _HEADING_TEXT_RE.match(_TAG_STRIP_RE.sub("", inner).strip()):
            return m.start()
    return None


def _close_dangling_page(text: str) -> str:
    """Append a closing ``</page>`` if text ends inside an unclosed page block."""
    n_open = len(_PAGE_OPEN_RE.findall(text))
    n_close = text.count("</page>")
    if n_open > n_close:
        return text + "\n</page>\n\n"
    return text


def _assert_pages_well_formed(text: str) -> None:
    """Every ``<page number="N">`` is immediately followed, before any other
    ``<page ...>``, by exactly one matching ``</page>`` -- no tag left open
    at end-of-string, no orphan close, no open-inside-open.

    A raw open/close *count* comparison (``n_open == n_close``) can pass
    while a splice has stitched a later page's close onto an earlier page's
    open -- same total count, wrong boundaries, and the kept content is
    silently misattributed to the wrong page. Downstream entity_prov /
    attribute_prov steps consume that page attribution, so this walks the
    tags in document order instead of trusting a count.
    """
    open_page = None
    for m in _PAGE_TAG_TOKEN_RE.finditer(text):
        if m.group(1) is not None:
            assert open_page is None, (
                f"page {m.group(1)} opened while page {open_page} is still open"
            )
            open_page = m.group(1)
        else:
            assert open_page is not None, "</page> with no matching open <page> tag"
            open_page = None
    assert open_page is None, f"page {open_page} opened but never closed"


def _drop_references(doc_text: str) -> tuple[str, dict]:
    """Shared implementation behind ``drop_references_section`` (the text)
    and ``describe_drop_references`` (the diagnostics) -- one detection pass,
    two views of its result, so the diagnostics can never drift from what
    actually got dropped.

    See ``drop_references_section`` for the behavior this implements. The
    ``info`` dict returned alongside the text has keys: ``reference_heading_found``,
    ``reference_cut`` (char offset or None), ``reference_cut_shape`` (one of
    "tag", "line", "bibliography_div", "section_header_div", or None --
    "line" is the only shape resume-splicing is eligible for),
    ``resume_heading_found``, ``resume_heading_text`` (the matched line,
    stripped, or None), ``resume_start`` (char offset or None), ``total_chars``,
    ``dropped_chars`` (input length minus output length), and
    ``rescued_chars`` (chars kept *because* of the resume match -- 0 when no
    resume fired; this is the number that answers "what did the old
    truncate-to-end-of-document behavior use to throw away that this keeps").
    """
    candidates: list[tuple[int, str]] = []
    tag_match = _TAG_HEADING_RE.search(doc_text)
    if tag_match is not None:
        candidates.append((tag_match.start() + 1, "tag"))  # keep through the tag-close '>'
    line_match = _LINE_HEADING_RE.search(doc_text)
    if line_match is not None:
        candidates.append((line_match.start(), "line"))
    biblio_cut = _bibliography_label_cut(doc_text)
    if biblio_cut is not None:
        candidates.append((biblio_cut, "bibliography_div"))

    if not candidates:
        # Nothing found via a heading text node or a bibliography-labelled div;
        # last resort is a Section-Header div whose heading text isn't sitting
        # directly between '>' and '<'.
        section_cut = _section_header_text_cut(doc_text)
        if section_cut is None:
            info = {
                "reference_heading_found": False,
                "reference_cut": None,
                "reference_cut_shape": None,
                "resume_heading_found": False,
                "resume_heading_text": None,
                "resume_start": None,
                "total_chars": len(doc_text),
                "dropped_chars": 0,
                "rescued_chars": 0,
            }
            return doc_text, info
        candidates.append((section_cut, "section_header_div"))

    ref_cut, ref_cut_shape = min(candidates, key=lambda c: c[0])

    # Resume-splicing only applies when the winning cut is the olmOCR bare-
    # line shape (_LINE_HEADING_RE) -- the shape this was written for.
    # Every other cut (_TAG_HEADING_RE, a Bibliography/Section-Header div)
    # lands at a chandra tag/div boundary; splicing text out from the middle
    # of a chandra-formatted document, rather than truncating to
    # end-of-document, risks keeping a div's closing </div> without its
    # opening tag (or vice versa) if a resume heading happens to fall inside
    # that div's content -- a corruption _assert_pages_well_formed can't see,
    # since it only tracks <page> tags, and chandra_format.py's own label
    # validation would only catch some shapes of it, not all.
    is_bare_line_cut = ref_cut_shape == "line"

    resume_match = _RESUME_HEADING_RE.search(doc_text, ref_cut) if is_bare_line_cut else None
    if resume_match is None:
        result = _close_dangling_page(doc_text[:ref_cut])
        resume_start = None
        resume_heading_text = None
    else:
        resume_start = resume_match.start()
        resume_heading_text = resume_match.group(0).strip()
        middle = doc_text[ref_cut:resume_start]
        if _PAGE_TAG_TOKEN_RE.search(middle) is None:
            # Resume heading is on the same page as the reference cut --
            # that page's own open tag (kept in the prefix) and close tag
            # (kept in the suffix) already span the splice untouched.
            result = doc_text[:ref_cut] + doc_text[resume_start:]
        else:
            prefix = _close_dangling_page(doc_text[:ref_cut])
            page_num_match = None
            for m in _PAGE_OPEN_NUM_RE.finditer(doc_text, 0, resume_start):
                page_num_match = m
            assert page_num_match is not None, (
                "resume heading follows a <page> boundary with no preceding "
                "<page number=...> open tag to resume from"
            )
            suffix = (
                f'<page number="{page_num_match.group(1)}">\n\n'
                + doc_text[resume_start:]
            )
            result = prefix + suffix

    _assert_pages_well_formed(result)

    info = {
        "reference_heading_found": True,
        "reference_cut": ref_cut,
        "reference_cut_shape": ref_cut_shape,
        "resume_heading_found": resume_match is not None,
        "resume_heading_text": resume_heading_text,
        "resume_start": resume_start,
        "total_chars": len(doc_text),
        "dropped_chars": len(doc_text) - len(result),
        "rescued_chars": (len(doc_text) - resume_start) if resume_start is not None else 0,
    }
    return result, info


def drop_references_section(doc_text: str) -> str:
    """Truncate doc_text at the first references/bibliography heading found,
    then, if that cut is the olmOCR bare-line shape, resume at the first
    bare appendix/figures/tables/supplementary-material heading found after
    it.

    If no reference/bibliography heading is found, doc_text is returned
    unchanged. If a reference heading is found but no resume heading follows
    it -- or the cut is a chandra tag/div shape rather than a bare line, see
    below -- everything from the heading to end-of-document is dropped (this
    is the whole of the behavior before resume-heading support was added).
    If a resume heading *does* follow a bare-line cut, the
    references-through-resume-heading span is dropped and everything from
    that heading onward is kept.

    Resume-splicing is deliberately restricted to a bare-line reference cut
    (``_LINE_HEADING_RE`` winning the pooled ``min(candidates)``) rather than
    a chandra tag/div cut: splicing content out of the middle of a
    chandra-formatted document risks keeping a div's closing ``</div>``
    without its opening tag (or vice versa) if the resume heading happens to
    land inside that div's content -- a corruption ``_assert_pages_well_formed``
    can't see, since it only tracks ``<page>`` tags.

    Page-tag bookkeeping matters here: if the reference cut lands inside an
    open ``<page number="N">`` block that does *not* also contain the resume
    heading, a closing ``</page>`` is inserted right at the cut -- not at the
    very end of the result, which would instead silently misattribute the
    kept content to the wrong page, since its own opening tag was dropped
    along with the references span. The resume heading's original page
    number is then re-opened synthetically at the resume point, preserving
    provenance rather than renumbering. If the resume heading sits on the
    *same* page as the reference cut, no synthetic tags are needed -- that
    page's own open/close pair already spans the splice. A final structural
    assert (``_assert_pages_well_formed``) catches any input shape that
    breaks these assumptions (e.g. a dropped page whose close tag went
    missing) rather than silently emitting a mis-attributed result --
    downstream entity_prov/attribute_prov steps consume this page
    attribution.

    See ``describe_drop_references`` for a diagnostics-returning variant
    built from the same detection pass.
    """
    result, _ = _drop_references(doc_text)
    return result


def describe_drop_references(doc_text: str) -> dict:
    """Like ``drop_references_section``, but returns the diagnostic ``info``
    dict instead of the transformed text (see ``_drop_references`` for its
    keys). For measuring what a real corpus's reference-dropping actually
    does -- e.g. the retroactive drop_references experiment-configs -- without
    re-deriving the detection logic in a second place.
    """
    _, info = _drop_references(doc_text)
    return info
