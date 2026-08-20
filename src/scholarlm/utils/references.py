"""Strip a references/bibliography section from page-wrapped OCR text.

Applied once, generically, by ``DocumentLM.fit()`` to the assembled
``<page number="N">...</page>`` document text *before* the chandra-ocr-2 /
olmOCR format branch -- not duplicated per-formatter. See
notes/scholarlm/builds/2026-08-20-per-document-isolation-01.md.

Detects a references/bibliography/literature-cited/works-cited heading
(case-insensitive, optionally numbered, e.g. "6. References") in either of the
two shapes seen in practice:

* chandra-ocr-2: the heading is the entire text content between a tag-close and
  the next tag-open, e.g. ``<h2>6. References</h2>``.
* olmOCR: the heading is a bare, unmarked-up entire line, e.g. ``REFERENCES``.

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

    if not candidates:
        return doc_text

    truncated = doc_text[: min(candidates)]

    n_open = len(_PAGE_OPEN_RE.findall(truncated))
    n_close = truncated.count("</page>")
    if n_open > n_close:
        truncated += "\n</page>\n\n"

    return truncated
