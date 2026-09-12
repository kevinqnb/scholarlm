"""Unit tests for the shared reference-dropping step in
``scholarlm.utils.references``.

Fixtures mirror the two OCR-model shapes documented in
notes/scholarlm/builds/2026-08-20-per-document-isolation-01.md: chandra-ocr-2's
references heading as an ``<h2>`` inside a ``Section-Header`` div, and olmOCR's
references heading as a bare, unmarked-up line. Both fixtures include content
past the heading that would otherwise break something downstream (an
unrecognized chandra data-label for the chandra fixture; nothing structural for
olmOCR, just content that must not survive), so a passing test demonstrates the
truncation actually happened, not just that no exception was raised.
"""
import pytest

from scholarlm.utils.chandra_format import format_chandra_output
from scholarlm.utils.references import drop_references_section


CHANDRA_PAGE_0 = (
    '<div data-bbox="0 0 100 20" data-label="Text"><p>Intro paragraph text.</p></div>'
)
CHANDRA_PAGE_1 = (
    '<div data-bbox="0 0 100 20" data-label="Section-Header"><h2>6. References</h2></div>'
    '<div data-bbox="0 20 100 40" data-label="List-Group">Smith, J. (2020). A citation.</div>'
    '<div data-bbox="0 40 100 60" data-label="Chemical-Block">Unrecognized label content.</div>'
)
CHANDRA_DOC = (
    f'<page number="0">\n\n{CHANDRA_PAGE_0}\n\n</page>\n\n'
    f'<page number="1">\n\n{CHANDRA_PAGE_1}\n\n</page>\n\n'
)


def test_chandra_style_unrecognized_label_raises_without_drop_references():
    with pytest.raises(ValueError):
        format_chandra_output(CHANDRA_DOC)


def test_chandra_style_drop_references_removes_heading_and_unrecognized_label():
    dropped = drop_references_section(CHANDRA_DOC)

    assert "References" not in dropped
    assert "Chemical-Block" not in dropped
    assert "Smith, J." not in dropped

    formatted = format_chandra_output(dropped)  # must not raise
    assert "Intro paragraph text." in formatted


def test_chandra_style_content_before_heading_is_byte_identical():
    dropped = drop_references_section(CHANDRA_DOC)
    prefix_len = CHANDRA_DOC.index("<h2>6. References</h2>")
    assert dropped[:prefix_len] == CHANDRA_DOC[:prefix_len]


def test_chandra_style_dangling_page_tag_is_closed():
    dropped = drop_references_section(CHANDRA_DOC)
    n_open = dropped.count('<page number="0">') + dropped.count('<page number="1">')
    assert n_open == dropped.count("</page>")


OLMOCR_DOC = (
    '<page number="0">\n\nIntro text on page zero.\n\n</page>\n\n'
    '<page number="1">\n\n'
    'Some body text before references.\n'
    'REFERENCES\n'
    'Smith, J. (2020). A citation.\n'
    'Jones, A. (2021). Another citation.\n'
    '\n\n</page>\n\n'
)


def test_olmocr_style_drop_references_truncates_at_bare_line():
    dropped = drop_references_section(OLMOCR_DOC)

    assert "Some body text before references." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Jones, A." not in dropped


def test_olmocr_style_content_before_heading_is_byte_identical():
    dropped = drop_references_section(OLMOCR_DOC)
    prefix_len = OLMOCR_DOC.index("REFERENCES")
    assert dropped[:prefix_len] == OLMOCR_DOC[:prefix_len]


def test_olmocr_style_dangling_page_tag_is_closed():
    dropped = drop_references_section(OLMOCR_DOC)
    n_open = dropped.count('<page number="0">') + dropped.count('<page number="1">')
    assert n_open == dropped.count("</page>")


def test_no_heading_present_returns_input_unchanged():
    doc = '<page number="0">\n\n<div data-label="Text">No references here.</div>\n\n</page>\n\n'
    assert drop_references_section(doc) == doc


def test_bare_uppercase_heading_without_number_prefix_matches():
    doc = 'Body text.\nBIBLIOGRAPHY\nCitation one.\n'
    dropped = drop_references_section(doc)
    assert dropped == 'Body text.\n'


# ---------------------------------------------------------------------------
# chandra div-encoded reference headings the two heading-text regexes miss
# (see 2026-09-01-chandra-unknown-label-fallback-01.md).
# ---------------------------------------------------------------------------

BIBLIO_DIV_DOC = (
    '<page number="0">\n\n'
    '<div data-bbox="0 0 100 20" data-label="Text"><p>Body of the paper.</p></div>\n\n'
    '</page>\n\n'
    '<page number="1">\n\n'
    '<div data-bbox="0 0 100 40" data-label="Bibliography">'
    'Smith, J. (2020). A citation. Jones, A. (2021). Another citation.</div>\n\n'
    '</page>\n\n'
)


def test_bibliography_div_truncates_the_document():
    dropped = drop_references_section(BIBLIO_DIV_DOC)

    assert "Body of the paper." in dropped
    assert 'data-label="Bibliography"' not in dropped  # open tag itself is gone
    assert "Smith, J." not in dropped
    assert "Another citation" not in dropped


def test_bibliography_div_content_before_cut_is_byte_identical():
    dropped = drop_references_section(BIBLIO_DIV_DOC)
    cut = BIBLIO_DIV_DOC.index('<div data-bbox="0 0 100 40" data-label="Bibliography"')
    assert dropped[:cut] == BIBLIO_DIV_DOC[:cut]


def test_bibliography_div_dangling_page_tag_is_closed():
    dropped = drop_references_section(BIBLIO_DIV_DOC)
    n_open = dropped.count('<page number="0">') + dropped.count('<page number="1">')
    assert n_open == dropped.count("</page>")


def test_bibliography_div_pooled_wins_over_a_later_heading_match():
    # A bibliography-labelled div sits *before* a later bare-line heading that
    # _LINE_HEADING_RE also matches. Pooling (min of all candidates) must cut at
    # the earlier div; a fallback-only design would skip it and cut later.
    doc = (
        '<page number="0">\n\nIntro.\n'
        '<div data-label="Bibliography">Ref list here.</div>\n'
        'trailing body\nREFERENCES\nmore\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)
    assert "Intro." in dropped
    assert "Ref list here." not in dropped
    assert "trailing body" not in dropped


def test_section_header_div_with_heading_text_the_regex_misses_truncates():
    # ">References Cited<" is not ">\s*references\s*<" -- existing _TAG_HEADING_RE
    # misses it; the Section-Header fallback catches it.
    doc = (
        '<page number="0">\n\nMain text.\n'
        '<div data-bbox="0 0 1 1" data-label="Section-Header"><h2>References Cited</h2></div>\n'
        '<div data-bbox="0 0 1 1" data-label="List-Group">A citation line.</div>\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Main text." in dropped
    assert "References Cited" not in dropped
    assert "A citation line." not in dropped


def test_section_header_div_heading_between_tags_still_handled_by_existing_regex():
    # The spec's literal shape: ">References<" -- already caught by _TAG_HEADING_RE,
    # asserted here so the behavior is pinned regardless of which detector fires.
    doc = (
        '<page number="0">\n\nMain text.\n'
        '<div data-bbox="0 0 1 1" data-label="Section-Header">References</div>\n'
        '<div data-bbox="0 0 1 1" data-label="List-Group">A citation line.</div>\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Main text." in dropped
    assert "References" not in dropped
    assert "A citation line." not in dropped


def test_non_reference_section_header_div_is_left_alone():
    doc = (
        '<page number="0">\n\n'
        '<div data-bbox="0 0 1 1" data-label="Section-Header"><h2>3. Methods</h2></div>\n'
        '<div data-bbox="0 0 1 1" data-label="Text">Methods body.</div>\n\n</page>\n\n'
    )
    assert drop_references_section(doc) == doc
