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

Everything from the heading to end-of-document is dropped. Running this before
the chandra branch means a div inside the dropped region can carry any
data-label -- including one ``chandra_format.py`` doesn't recognize -- without
ever reaching its label validation.
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


def drop_references_section(doc_text: str) -> str:
    """Truncate doc_text at the first references/bibliography heading found.

    If no heading is found, doc_text is returned unchanged. If the cut lands
    inside an open ``<page number="N">`` block, a closing ``</page>`` is
    appended so downstream page-regexes (chandra_format.py's ``_PAGE_RE``,
    which requires a matched ``</page>``) don't silently skip -- and therefore
    leave un-validated -- the boundary page's un-truncated prefix.
    """
    candidates = []
    tag_match = _TAG_HEADING_RE.search(doc_text)
    if tag_match is not None:
        candidates.append(tag_match.start() + 1)  # keep through the tag-close '>'
    line_match = _LINE_HEADING_RE.search(doc_text)
    if line_match is not None:
        candidates.append(line_match.start())
    biblio_cut = _bibliography_label_cut(doc_text)
    if biblio_cut is not None:
        candidates.append(biblio_cut)

    if not candidates:
        # Nothing found via a heading text node or a bibliography-labelled div;
        # last resort is a Section-Header div whose heading text isn't sitting
        # directly between '>' and '<'.
        section_cut = _section_header_text_cut(doc_text)
        if section_cut is None:
            return doc_text
        candidates.append(section_cut)

    truncated = doc_text[: min(candidates)]

    n_open = len(_PAGE_OPEN_RE.findall(truncated))
    n_close = truncated.count("</page>")
    if n_open > n_close:
        truncated += "\n</page>\n\n"

    return truncated
