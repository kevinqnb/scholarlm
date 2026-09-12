"""Unit tests for the chandra-ocr-2 output formatter in ``scholarlm.utils.chandra_format``.

Fixture mirrors the real shape found in data/pond/ocr_output_chandra_ocr_2/
(see notes/scholarlm/builds/2026-08-13-chandra-ocr-adapter-01.md for the audit
each case below traces back to): a caption preceding its table (as seen for
"TABLE 1"), a figure with an inline caption paragraph, a figure whose caption
is an external div *following* it and which also contains a nested <table>
that must survive as figure content without being separately numbered, a
decorative logo image that must vanish entirely, and an orphaned caption with
no adjacent table/figure.
"""
import pytest

from scholarlm.utils.chandra_format import format_chandra_output


PAGE_0 = (
    '<div data-bbox="0 0 100 20" data-label="Page-Header">HeaderJunk</div>'
    '<div data-bbox="0 20 100 40" data-label="Text"><p>Intro paragraph text.</p></div>'
    '<div data-bbox="0 40 100 50" data-label="Caption">'
    '<p><b>TABLE 1</b> | Pond water characteristics.</p></div>'
    '<div data-bbox="0 50 100 80" data-label="Table">\n'
    '<table border="1">\n<tbody>\n'
    '<tr><td>Ambient temp</td><td>17.2</td></tr>\n'
    '</tbody>\n</table>\n</div>'
    '<div data-bbox="0 80 100 120" data-label="Figure">\n'
    '<img alt="Figure 1: Bacterial diversity chart."/>\n'
    '<p><b>FIGURE 1 | Bacterial diversity chart caption.</b> More detail.</p>\n'
    '</div>'
    '<div data-bbox="0 120 100 140" data-label="Image"><img alt="Springer logo"/></div>'
    '<div data-bbox="0 140 100 160" data-label="Page-Footer">FooterJunk</div>'
)

PAGE_1 = (
    '<div data-bbox="0 0 100 20" data-label="Text"><p>Second page intro.</p></div>'
    '<div data-bbox="0 20 100 80" data-label="Figure">\n'
    '<img alt="Figure 2: Nested data figure."/>\n'
    '<p><b>A</b></p>\n'
    '<table border="1">\n<tbody>\n'
    '<tr><td>NestedCell</td></tr>\n'
    '</tbody>\n</table>\n'
    '</div>'
    '<div data-bbox="0 80 100 90" data-label="Caption">'
    '<p><b>FIGURE 2 |</b> Nested data figure caption.</p></div>'
    '<div data-bbox="0 90 100 100" data-label="Caption">'
    '<p>Orphaned caption with no adjacent figure or table.</p></div>'
    '<div data-bbox="0 100 100 110" data-label="Page-Footer">FooterJunk2</div>'
)

DOC = f'<page number="0">\n\n{PAGE_0}\n\n</page>\n\n<page number="1">\n\n{PAGE_1}\n\n</page>\n\n'


@pytest.fixture
def formatted():
    return format_chandra_output(DOC)


def test_no_bbox_coordinates_survive(formatted):
    assert "data-bbox" not in formatted


def test_page_headers_and_footers_dropped(formatted):
    assert "HeaderJunk" not in formatted
    assert "FooterJunk" not in formatted
    assert "FooterJunk2" not in formatted


def test_decorative_logo_image_dropped_entirely(formatted):
    assert "Springer logo" not in formatted


def test_page_tags_preserved(formatted):
    assert '<page number="0">' in formatted
    assert '<page number="1">' in formatted
    assert "</page>" in formatted


def test_table_numbered_and_caption_attached_before_it(formatted):
    assert '<table number="1">' in formatted
    assert "Ambient temp" in formatted
    assert "Pond water characteristics" in formatted
    # exactly one numbered table -- the nested table inside Figure 2 must not
    # get its own number
    assert formatted.count("<table number=") == 1


def test_figure_inline_caption_and_alt_both_present(formatted):
    assert '<figure number="1">' in formatted
    assert "Figure 1: Bacterial diversity chart." in formatted
    assert "Bacterial diversity chart caption." in formatted


def test_figure_external_caption_attached_after_it(formatted):
    assert '<figure number="2">' in formatted
    assert "Nested data figure caption." in formatted


def test_nested_table_inside_figure_survives_as_figure_content(formatted):
    assert "NestedCell" in formatted


def test_figure_numbering_sequential_across_pages(formatted):
    assert formatted.count("<figure number=") == 2
    assert '<figure number="1">' in formatted
    assert '<figure number="2">' in formatted


def test_orphaned_caption_kept_not_dropped(formatted):
    assert "Orphaned caption with no adjacent figure or table." in formatted


def test_raises_on_input_with_no_page_tags():
    with pytest.raises(ValueError):
        format_chandra_output('<div data-bbox="0 0 1 1" data-label="Text"><p>x</p></div>')


def test_raises_on_unrecognized_label():
    bad_doc = (
        '<page number="0">\n\n'
        '<div data-bbox="0 0 1 1" data-label="Not-A-Real-Label">x</div>'
        '\n\n</page>\n\n'
    )
    with pytest.raises(ValueError):
        format_chandra_output(bad_doc)


# ---------------------------------------------------------------------------
# unknown_label_policy="coerce": structure-based fallback for labels outside
# the audited set (see 2026-09-01-chandra-unknown-label-fallback-01.md).
# ---------------------------------------------------------------------------

def _wrap(page_body: str) -> str:
    return f'<page number="0">\n\n{page_body}\n\n</page>\n\n'


def test_coerce_unknown_label_with_table_rendered_as_numbered_table():
    doc = _wrap(
        '<div data-bbox="0 0 1 1" data-label="Chemical-Block">\n'
        '<table border="1"><tbody><tr><td>CoercedCell</td></tr></tbody></table>\n'
        '</div>'
    )
    log: list[tuple[str, str]] = []
    out = format_chandra_output(doc, unknown_label_policy="coerce", coercion_log=log)

    assert '<table number="1">' in out
    assert "CoercedCell" in out
    assert log == [("Chemical-Block", "table")]


def test_coerce_unknown_label_with_image_rendered_as_figure():
    doc = _wrap(
        '<div data-bbox="0 0 1 1" data-label="Chemical-Block">'
        '<img alt="Structure of compound 4."/>'
        '</div>'
    )
    log: list[tuple[str, str]] = []
    out = format_chandra_output(doc, unknown_label_policy="coerce", coercion_log=log)

    assert '<figure number="1">' in out
    assert "Structure of compound 4." in out
    assert log == [("Chemical-Block", "figure")]


def test_coerce_unknown_label_text_only_unwrapped_as_body_text():
    doc = _wrap(
        '<div data-bbox="0 0 1 1" data-label="Chemical-Block">'
        '<p>2 H2 + O2 -> 2 H2O</p>'
        '</div>'
    )
    log: list[tuple[str, str]] = []
    out = format_chandra_output(doc, unknown_label_policy="coerce", coercion_log=log)

    assert "2 H2 + O2 -&gt; 2 H2O" in out or "2 H2 + O2 -> 2 H2O" in out
    assert "<figure" not in out and "<table number" not in out
    assert log == [("Chemical-Block", "text")]


def test_coerce_unknown_label_empty_after_decorative_strip_emits_nothing():
    doc = _wrap(
        '<div data-bbox="0 0 1 1" data-label="Text"><p>Real body.</p></div>'
        '<div data-bbox="0 0 1 1" data-label="Weird"><img alt="Elsevier logo"/></div>'
    )
    log: list[tuple[str, str]] = []
    out = format_chandra_output(doc, unknown_label_policy="coerce", coercion_log=log)

    assert "Real body." in out
    assert "Elsevier logo" not in out
    assert log == []  # nothing to coerce -- the div held only a decorative image


def test_coerce_nested_table_in_figure_shaped_div_stays_a_figure():
    # A <table> nested below a panel <p> is figure content, not a table in its
    # own right -- classification keys on a *direct-child* <table> only.
    doc = _wrap(
        '<div data-bbox="0 0 1 1" data-label="Weird">'
        '<img alt="Figure 9: panel layout."/>'
        '<p>A</p>'
        '<div><table border="1"><tbody><tr><td>PanelCell</td></tr></tbody></table></div>'
        '</div>'
    )
    log: list[tuple[str, str]] = []
    out = format_chandra_output(doc, unknown_label_policy="coerce", coercion_log=log)

    assert '<figure number="1">' in out
    assert "PanelCell" in out          # survives as figure content
    assert "<table number=" not in out  # not separately numbered
    assert log == [("Weird", "figure")]


def test_coerce_records_counts_by_label_with_multiplicity():
    doc = _wrap(
        '<div data-bbox="0 0 1 1" data-label="Chemical-Block"><p>first block</p></div>'
        '<div data-bbox="0 0 1 1" data-label="Chemical-Block"><p>second block</p></div>'
        '<div data-bbox="0 0 1 1" data-label="Bibliography"><p>a citation</p></div>'
    )
    log: list[tuple[str, str]] = []
    format_chandra_output(doc, unknown_label_policy="coerce", coercion_log=log)

    counts: dict[str, int] = {}
    for lbl, _kind in log:
        counts[lbl] = counts.get(lbl, 0) + 1
    assert counts == {"Chemical-Block": 2, "Bibliography": 1}


def test_coerce_div_with_no_data_label_logged_under_sentinel():
    doc = _wrap('<div data-bbox="0 0 1 1"><p>label-less content</p></div>')
    log: list[tuple[str, str]] = []
    out = format_chandra_output(doc, unknown_label_policy="coerce", coercion_log=log)

    assert "label-less content" in out
    assert log == [("<no-data-label>", "text")]


def test_raise_policy_is_the_default_and_still_raises():
    doc = _wrap('<div data-bbox="0 0 1 1" data-label="Chemical-Block"><p>x</p></div>')
    with pytest.raises(ValueError, match="Unrecognized chandra-ocr-2 data-label"):
        format_chandra_output(doc)
    with pytest.raises(ValueError, match="Unrecognized chandra-ocr-2 data-label"):
        format_chandra_output(doc, unknown_label_policy="raise")


def test_invalid_unknown_label_policy_raises():
    with pytest.raises(ValueError, match="unknown_label_policy"):
        format_chandra_output(DOC, unknown_label_policy="skip")


# ---------------------------------------------------------------------------
# Regression: a document with no unrecognized label must format byte-identically
# regardless of policy, and identically to a checked-in golden.
# ---------------------------------------------------------------------------

GOLDEN = (
    '<page number="0">\n\n<p>Intro paragraph text.</p>\n\n'
    '<p><b>TABLE 1</b> | Pond water characteristics.</p>\n\n'
    '<table number="1"><tbody>\n<tr><td>Ambient temp</td><td>17.2</td></tr>\n</tbody></table>\n\n'
    '<figure number="1">\nFigure 1: Bacterial diversity chart.\n'
    '<p><b>FIGURE 1 | Bacterial diversity chart caption.</b> More detail.</p>\n</figure>\n\n'
    '</page>\n\n\n\n'
    '<page number="1">\n\n<p>Second page intro.</p>\n\n'
    '<figure number="2">\nFigure 2: Nested data figure.\n'
    '<p><b>FIGURE 2 |</b> Nested data figure caption.</p>\n'
    '<p><b>A</b></p>\n<table border="1">\n<tbody>\n<tr><td>NestedCell</td></tr>\n</tbody>\n</table>\n</figure>\n\n'
    '<p>Orphaned caption with no adjacent figure or table.</p>\n\n'
    '</page>\n\n\n\n'
)


def test_clean_document_matches_golden_and_is_policy_invariant():
    assert format_chandra_output(DOC) == GOLDEN
    assert format_chandra_output(DOC, unknown_label_policy="coerce") == GOLDEN

    log: list[tuple[str, str]] = []
    format_chandra_output(DOC, unknown_label_policy="coerce", coercion_log=log)
    assert log == []
