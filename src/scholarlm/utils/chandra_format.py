"""
Reformat chandra-ocr-2's bbox-tagged HTML layout output into the same
``<page number="N">`` / ``<table number="N">`` convention olmOCR's output uses.

``format_chandra_output`` expects input that has already been through
``DocumentLM.fit()``'s generic, model-agnostic page-wrap step -- i.e. text
already split into ``<page number="N">...</page>`` blocks, each containing a
flat stream of chandra's ``<div data-bbox="x1 y1 x2 y2" data-label="...">``
region blocks. It does not render PDFs or call a model itself.

See notes/scholarlm/builds/2026-08-13-chandra-ocr-adapter-01.md for the
provenance of every rule below -- each one traces back to a concrete example
found by auditing the cached chandra-ocr-2 output for the pond dataset.
"""
import re
from itertools import count

from bs4 import BeautifulSoup

# Labels observed in chandra-ocr-2 output (see build note for full audit).
_DROP_LABELS = {"Page-Header", "Page-Footer"}
_FIGURE_LABELS = {"Figure", "Diagram", "Image"}
_TABLE_LABEL = "Table"
_CAPTION_LABEL = "Caption"
_PLAIN_LABELS = {"Text", "Section-Header", "List-Group", "Footnote", "Equation-Block"}

# Phrase-level, not bare "cover": pond/wetland papers measure vegetation/ice/percent
# cover, which must not be mistaken for a decorative journal-cover image.
_DECORATIVE_ALT_KEYWORDS = ("logo", "icon", "cover image", "journal cover", "cover of the")

# Distinguishes a real caption paragraph ("FIGURE 1 | Bacterial community...") from
# a chart-internal label paragraph ("A", "Phyla") that also happens to be the first
# <p> after a figure's <img>.
_CAPTION_PREFIX_RE = re.compile(r"^(FIGURE|TABLE|FIG\.?)\s*\d", re.IGNORECASE)

_PAGE_RE = re.compile(r'<page number="(\d+)">(.*?)</page>', re.DOTALL)


def _is_decorative(img_tag) -> bool:
    alt = (img_tag.get("alt") or "").lower()
    return any(keyword in alt for keyword in _DECORATIVE_ALT_KEYWORDS)


def _render_children(tag) -> str:
    return "".join(str(child) for child in tag.contents).strip()


def _render_entry(entry: dict) -> str:
    if entry["kind"] == "text":
        return entry["body"]

    if entry["kind"] == "table":
        parts = []
        if entry["caption"]:
            parts.append(entry["caption"])
        parts.append(f'<table number="{entry["number"]}">{entry["body"]}</table>')
        return "\n\n".join(parts)

    if entry["kind"] == "figure":
        parts = [f'<figure number="{entry["number"]}">']
        if entry["alt"]:
            parts.append(entry["alt"])
        if entry["caption"]:
            parts.append(entry["caption"])
        if entry["body"]:
            parts.append(entry["body"])
        parts.append("</figure>")
        return "\n".join(parts)

    raise AssertionError(f"Unhandled entry kind: {entry['kind']!r}")


def _format_page(page_html: str, table_counter: count, figure_counter: count) -> str:
    soup = BeautifulSoup(page_html, "html.parser")
    divs = [child for child in soup.contents if getattr(child, "name", None) == "div"]

    # Strip decorative images before classifying labels: done_when requires this
    # regardless of which data-label div encloses the <img> (see build note item 5).
    for div in divs:
        for img in div.find_all("img"):
            if _is_decorative(img):
                img.decompose()

    entries: list[dict] = []
    pending_caption: str | None = None
    open_entry: dict | None = None

    for div in divs:
        label = div.get("data-label")

        if label in _DROP_LABELS:
            continue

        if label == _CAPTION_LABEL:
            caption_text = _render_children(div)
            if open_entry is not None and open_entry["caption"] is None:
                open_entry["caption"] = caption_text
                open_entry = None
            else:
                pending_caption = caption_text
            continue

        if label == _TABLE_LABEL:
            table_tag = div.find("table")
            if table_tag is None:
                raise ValueError(
                    "Table div has no nested <table> element -- unexpected "
                    "chandra-ocr-2 output shape."
                )
            entry = {
                "kind": "table",
                "number": next(table_counter),
                "caption": None,
                "body": _render_children(table_tag),
            }
            if pending_caption is not None:
                entry["caption"] = pending_caption
                pending_caption = None
            entries.append(entry)
            open_entry = entry
            continue

        if label in _FIGURE_LABELS:
            img = div.find("img")
            alt = img.get("alt", "").strip() if img else None
            if img is not None:
                img.decompose()

            first_p = div.find("p")
            inline_caption = None
            if first_p is not None and _CAPTION_PREFIX_RE.match(first_p.get_text(" ", strip=True)):
                inline_caption = str(first_p)
                first_p.decompose()

            body = _render_children(div)

            # A logo/icon Image div reduces to nothing once its <img> is stripped
            # as decorative -- drop it transparently, like Page-Header/Page-Footer,
            # without disturbing pending_caption/open_entry adjacency.
            if alt is None and not inline_caption and not body:
                continue

            entry = {
                "kind": "figure",
                "number": next(figure_counter),
                "alt": alt,
                "caption": inline_caption,
                "body": body,
            }
            if entry["caption"] is None and pending_caption is not None:
                entry["caption"] = pending_caption
                pending_caption = None
            entries.append(entry)
            open_entry = entry
            continue

        if label in _PLAIN_LABELS:
            if pending_caption is not None:
                entries.append({"kind": "text", "body": pending_caption})
                pending_caption = None
            entries.append({"kind": "text", "body": _render_children(div)})
            open_entry = None
            continue

        raise ValueError(
            f"Unrecognized chandra-ocr-2 data-label {label!r}; not in the audited "
            "label set from notes/scholarlm/builds/2026-08-13-chandra-ocr-adapter-01.md."
        )

    # An orphaned caption with no adjacent table/figure is kept as body prose,
    # never silently dropped.
    if pending_caption is not None:
        entries.append({"kind": "text", "body": pending_caption})

    return "\n\n".join(_render_entry(entry) for entry in entries)


def format_chandra_output(doc_text: str) -> str:
    """
    Reformat a chandra-ocr-2 document's bbox-tagged HTML into readable prose.

    Args:
        doc_text: Full document text already page-wrapped by
            ``DocumentLM.fit()``'s generic step, i.e. a sequence of
            ``<page number="N">...</page>`` blocks each containing chandra's
            flat ``<div data-bbox="..." data-label="...">`` region stream.

    Returns:
        str: The same page structure with all ``data-bbox`` coordinates and
            decorative page furniture stripped, and tables/figures numbered
            sequentially with their captions inlined as prose.
    """
    table_counter = count(1)
    figure_counter = count(1)

    def _replace(match: re.Match) -> str:
        page_number = match.group(1)
        cleaned = _format_page(match.group(2), table_counter, figure_counter)
        return f'<page number="{page_number}">\n\n{cleaned}\n\n</page>\n\n'

    result, n_subs = _PAGE_RE.subn(_replace, doc_text)
    if n_subs == 0:
        raise ValueError(
            "No <page number=\"N\"> tags found in input. format_chandra_output "
            "expects DocumentLM.fit()'s already page-wrapped document text, not "
            "raw per-page model output."
        )
    return result
