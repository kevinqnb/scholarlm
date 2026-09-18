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
from scholarlm.utils.references import (
    _assert_pages_well_formed as assert_pages_well_formed,
    describe_drop_references,
    drop_references_section,
)


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
    assert_pages_well_formed(dropped)


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
    assert_pages_well_formed(dropped)


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
    assert_pages_well_formed(dropped)


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


# ---------------------------------------------------------------------------
# Resume-heading support: drop references-through-resume-heading, keep from
# the resume heading onward (appendix/figures/tables/supplementary material),
# rather than dropping to end-of-document.
# ---------------------------------------------------------------------------


def test_appendix_on_a_later_page_resumes_with_reopened_page_tag():
    doc = (
        '<page number="0">\n\nIntro.\n\n</page>\n\n'
        '<page number="1">\n\n'
        'Body text before references.\n'
        'REFERENCES\n'
        'Smith, J. (2020). A citation.\n'
        '\n\n</page>\n\n'
        '<page number="2">\n\n'
        'Jones, A. (2021). Another citation.\n'
        'APPENDIX A\n'
        'Supplementary methods.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Intro." in dropped
    assert "Body text before references." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Jones, A." not in dropped  # still in the dropped span, before the heading
    assert "APPENDIX A" in dropped
    assert "Supplementary methods." in dropped

    # page 1 is closed at the cut, page 2 is reopened at the resume point --
    # not left as a fragment with no page 1 close / no page 2 open.
    assert dropped.count('<page number="0">') == 1
    assert dropped.count('<page number="1">') == 1
    assert dropped.count('<page number="2">') == 1
    assert dropped.count("</page>") == 3
    assert_pages_well_formed(dropped)


def test_appendix_spanning_multiple_fully_dropped_pages_reopens_correct_number():
    # Pages 1 and 2 sit entirely inside the dropped span (no cut and no resume
    # heading on either) and must vanish as whole open/close pairs, not leave
    # a stray tag; page 3 is reopened with ITS OWN number, not page 1's or 2's.
    doc = (
        '<page number="0">\n\nIntro.\nREFERENCES\ncite 0\n\n</page>\n\n'
        '<page number="1">\n\ncite 1\n\n</page>\n\n'
        '<page number="2">\n\ncite 2\n\n</page>\n\n'
        '<page number="3">\n\nAPPENDIX A\nSupplementary methods.\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Intro." in dropped
    assert "cite 0" not in dropped
    assert "cite 1" not in dropped
    assert "cite 2" not in dropped
    assert "APPENDIX A" in dropped
    assert "Supplementary methods." in dropped
    assert '<page number="1">' not in dropped
    assert '<page number="2">' not in dropped
    assert dropped.count('<page number="3">') == 1
    assert_pages_well_formed(dropped)


def test_appendix_on_the_same_page_as_the_cut_needs_no_synthetic_tags():
    doc = (
        '<page number="0">\n\n'
        'Body text before references.\n'
        'REFERENCES\n'
        'Smith, J. (2020). A citation.\n'
        'APPENDIX A\n'
        'Supplementary methods.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Body text before references." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "APPENDIX A" in dropped
    assert "Supplementary methods." in dropped
    assert dropped.count('<page number="0">') == 1
    assert dropped.count("</page>") == 1
    assert_pages_well_formed(dropped)


def test_references_with_no_appendix_is_unchanged_from_pre_appendix_behavior():
    dropped = drop_references_section(OLMOCR_DOC)
    assert "Some body text before references." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Jones, A." not in dropped
    assert_pages_well_formed(dropped)


def test_inline_appendix_mention_does_not_trigger_a_resume():
    doc = (
        '<page number="0">\n\n'
        'Body text.\n'
        'REFERENCES\n'
        'Smith, J. et al., see Appendix A for details.\n'
        'Jones, A. (2021). Another citation.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Body text." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Appendix A" not in dropped
    assert "Jones, A." not in dropped
    assert_pages_well_formed(dropped)


def test_first_of_two_appendix_headings_wins():
    doc = (
        '<page number="0">\n\n'
        'Body text.\n'
        'REFERENCES\n'
        'A citation.\n'
        'APPENDIX A\n'
        'First appendix body.\n'
        'APPENDIX B\n'
        'Second appendix body.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "A citation." not in dropped
    assert "APPENDIX A" in dropped
    assert "First appendix body." in dropped
    # Everything from the first appendix heading onward is kept wholesale,
    # including the second appendix's own section.
    assert "APPENDIX B" in dropped
    assert "Second appendix body." in dropped
    assert_pages_well_formed(dropped)


def test_chandra_bibliography_div_with_trailing_bare_appendix_is_not_spliced():
    # The reference cut here is a Bibliography div (a chandra tag/div shape,
    # not the olmOCR bare-line shape resume-splicing is scoped to), and a
    # bare "APPENDIX A" line follows it on the SAME page, inside a div that
    # closes after the heading. Resume-splicing would keep that div's
    # closing </div> without its opening tag; scoping to bare-line cuts only
    # means this falls back to plain truncation instead, same as before
    # resume-heading support existed.
    doc = (
        '<page number="0">\n\n'
        '<div data-bbox="0 0 100 20" data-label="Text"><p>Body of the paper.</p></div>\n\n'
        '</page>\n\n'
        '<page number="1">\n\n'
        '<div data-bbox="0 0 100 40" data-label="Bibliography">'
        'Smith, J. (2020). A citation.\n'
        'APPENDIX A\n'
        'Supplementary methods.</div>\n\n'
        '</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Body of the paper." in dropped
    assert 'data-label="Bibliography"' not in dropped
    assert "Smith, J." not in dropped
    # Falls back to plain truncation -- the appendix content is NOT kept,
    # since resume-splicing doesn't apply to this cut shape.
    assert "APPENDIX A" not in dropped
    assert "Supplementary methods." not in dropped
    assert_pages_well_formed(dropped)
    # No orphan </div> either -- the whole div (open tag through close) is
    # dropped as a unit, exactly like the pre-existing bibliography-div
    # tests. format_chandra_output raising ValueError on an unbalanced/
    # unknown-label div is what would catch that; it must not raise here.
    format_chandra_output(dropped)


def test_appendix_heading_before_the_reference_cut_is_not_searched():
    doc = (
        '<page number="0">\n\n'
        'See Appendix A of the companion report for methods.\n'
        'Body text.\n'
        'REFERENCES\n'
        'A citation.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "See Appendix A" in dropped
    assert "Body text." in dropped
    assert "REFERENCES" not in dropped
    assert "A citation." not in dropped
    assert_pages_well_formed(dropped)


# ---------------------------------------------------------------------------
# Broader resume-heading vocabulary: figures, tables, supplementary material.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "heading",
    [
        "FIGURES",
        "FIGURE CAPTIONS",
        "LIST OF FIGURES",
        "TABLES",
        "TABLE CAPTIONS",
        "LIST OF TABLES",
        "SUPPLEMENTARY MATERIAL",
        "SUPPLEMENTARY INFORMATION",
        "SUPPLEMENTARY DATA",
        "SUPPLEMENTAL INFORMATION",
        "ELECTRONIC SUPPLEMENTARY MATERIAL",
        "SUPPORTING INFORMATION",
    ],
)
def test_non_appendix_resume_headings_also_trigger_a_resume(heading):
    doc = (
        '<page number="0">\n\nIntro.\nREFERENCES\ncitation\n\n</page>\n\n'
        f'<page number="1">\n\n{heading}\nSupplementary content.\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Intro." in dropped
    assert "citation" not in dropped
    assert heading in dropped
    assert "Supplementary content." in dropped
    assert_pages_well_formed(dropped)


# ---------------------------------------------------------------------------
# Negative controls: words/phrases that legitimately show up *inside* a real
# reference list's own text and must NOT be mistaken for a resume heading.
# Metadata-only sections (data availability, additional files, online
# resource) are deliberately excluded from the vocabulary rather than
# guarded against here, since they aren't appendix-like content in the first
# place; "figure(s)"/"table(s)" bare are kept but restricted to
# case-sensitive ALL-CAPS specifically because of the last two cases below.
# ---------------------------------------------------------------------------


def test_data_availability_line_inside_reference_span_does_not_trigger_a_resume():
    doc = (
        '<page number="0">\n\n'
        'Body text.\n'
        'REFERENCES\n'
        'Smith, J. (2020). A citation.\n'
        'Data availability: sequences deposited at NCBI under accession XYZ.\n'
        'Jones, A. (2021). Another citation.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Body text." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Data availability" not in dropped
    assert "Jones, A." not in dropped
    assert_pages_well_formed(dropped)


def test_additional_file_and_online_resource_lines_do_not_trigger_a_resume():
    doc = (
        '<page number="0">\n\n'
        'Body text.\n'
        'REFERENCES\n'
        'Smith, J. (2020). A citation.\n'
        'Additional file 1: raw data tables.\n'
        'Online resource 2 contains the full protocol.\n'
        'Jones, A. (2021). Another citation.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Body text." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Additional file" not in dropped
    assert "Online resource" not in dropped
    assert "Jones, A." not in dropped
    assert_pages_well_formed(dropped)


def test_supplementary_table_citation_continuation_does_not_trigger_a_resume():
    # An OCR-wrapped line starting with "Supplementary Table..." is not the
    # two/three-word phrase "supplementary material/information/data" the
    # vocabulary matches -- a bare "supplementary" would have false-triggered
    # here, which is exactly why it isn't in the vocabulary.
    doc = (
        '<page number="0">\n\n'
        'Body text.\n'
        'REFERENCES\n'
        'Smith, J. (2020).\n'
        'Supplementary Table S3 shows the full citation list continued here.\n'
        'Jones, A. (2021). Another citation.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Body text." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Supplementary Table" not in dropped
    assert "Jones, A." not in dropped
    assert_pages_well_formed(dropped)


def test_titlecase_tables_citation_continuation_does_not_trigger_a_resume():
    # "Tables" here is title-case (an OCR-wrapped sentence, not a heading);
    # the case-sensitive ALL-CAPS restriction on bare figure(s)/table(s) is
    # exactly what keeps this from being mistaken for a "TABLES" heading.
    doc = (
        '<page number="0">\n\n'
        'Body text.\n'
        'REFERENCES\n'
        'Smith, J. (2020).\n'
        'Tables 2-4 in Smith et al. (2019) show that concentrations varied.\n'
        'Jones, A. (2021). Another citation.\n'
        '\n\n</page>\n\n'
    )
    dropped = drop_references_section(doc)

    assert "Body text." in dropped
    assert "REFERENCES" not in dropped
    assert "Smith, J." not in dropped
    assert "Tables 2-4" not in dropped
    assert "Jones, A." not in dropped
    assert_pages_well_formed(dropped)


# ---------------------------------------------------------------------------
# describe_drop_references: diagnostics built from the same detection pass
# drop_references_section uses, not a re-derivation of it.
# ---------------------------------------------------------------------------


def test_describe_no_heading_found():
    doc = '<page number="0">\n\n<div data-label="Text">No references here.</div>\n\n</page>\n\n'
    info = describe_drop_references(doc)

    assert info["reference_heading_found"] is False
    assert info["reference_cut"] is None
    assert info["reference_cut_shape"] is None
    assert info["resume_heading_found"] is False
    assert info["resume_heading_text"] is None
    assert info["dropped_chars"] == 0
    assert info["rescued_chars"] == 0
    assert info["total_chars"] == len(doc)


def test_describe_reference_found_no_resume_matches_plain_truncation():
    info = describe_drop_references(OLMOCR_DOC)
    dropped = drop_references_section(OLMOCR_DOC)

    assert info["reference_heading_found"] is True
    assert info["reference_cut_shape"] == "line"
    assert info["resume_heading_found"] is False
    assert info["resume_heading_text"] is None
    assert info["rescued_chars"] == 0
    assert info["dropped_chars"] == len(OLMOCR_DOC) - len(dropped)
    assert info["total_chars"] == len(OLMOCR_DOC)


def test_describe_resume_reports_shape_matched_text_and_rescued_chars():
    doc = (
        '<page number="0">\n\n'
        'Body text before references.\n'
        'REFERENCES\n'
        'Smith, J. (2020). A citation.\n'
        'APPENDIX A\n'
        'Supplementary methods.\n'
        '\n\n</page>\n\n'
    )
    info = describe_drop_references(doc)
    dropped = drop_references_section(doc)

    assert info["reference_heading_found"] is True
    assert info["reference_cut_shape"] == "line"
    assert info["resume_heading_found"] is True
    assert info["resume_heading_text"] == "APPENDIX A"
    assert info["resume_start"] == doc.index("APPENDIX A")
    assert info["rescued_chars"] == len(doc) - doc.index("APPENDIX A")
    assert info["dropped_chars"] == len(doc) - len(dropped)
    assert info["total_chars"] == len(doc)


def test_describe_chandra_cut_shape_is_not_line_and_never_resumes():
    # The Bibliography-div cut shape ("bibliography_div") is exactly the case
    # resume-splicing is scoped away from -- confirmed by shape, not just by
    # the absence of a resume match.
    info = describe_drop_references(BIBLIO_DIV_DOC)

    assert info["reference_heading_found"] is True
    assert info["reference_cut_shape"] == "bibliography_div"
    assert info["resume_heading_found"] is False
    assert info["resume_heading_text"] is None
    assert info["rescued_chars"] == 0
