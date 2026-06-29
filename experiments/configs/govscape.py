"""
Dataset configuration for the govscape dataset.

Government document metadata extraction. Each document is a government-generated
PDF; the extraction task is to recover basic bibliographic metadata from the OCR text.

Only Ablation 1 (direct extraction) is used for this dataset.
"""
from __future__ import annotations

from pydantic import BaseModel

from scholarlm.config import DatasetConfig


# ---------------------------------------------------------------------------
# Entity schema  (required by DatasetConfig; not used by Ablation 1)
# ---------------------------------------------------------------------------


class EntitySchema(BaseModel):
    """A single government document."""

    title: str | None


ENTITY_IDENTIFICATION_PROMPT = """Given the provided text, identify the government document and extract its title.

Response schema:
- title: the official title of the document.

Output format:
{
  "items": [
    { "title": "..." }
  ]
}
If no title is found, output exactly:
{ "items": [] }
"""


# ---------------------------------------------------------------------------
# Attribute catalogue
# ---------------------------------------------------------------------------


_ATTRIBUTE_INFO_DICT: dict[str, dict] = {
    "title": {
        "description": (
            "The official title of the document as it appears on the cover page or header."
        ),
        "units": [],
    },
    "authors": {
        "description": (
            "The individual authors or creators of the document. "
            "If multiple, list all names separated by semicolons. "
            "Set to None if the document is unsigned or no named authors are given."
        ),
        "units": [],
    },
    "publication_date": {
        "description": (
            "The date the document was published, issued, or finalized. "
            "Use YYYY-MM-DD if the full date is known, YYYY-MM for month/year only, "
            "or YYYY for year only. Set to None if no date is stated."
        ),
        "units": [],
    },
    "organization": {
        "description": (
            "The agency, office, department, bureau, or funding source responsible for "
            "the document's creation or publication (e.g. 'U.S. Army Corps of Engineers', "
            "'EPA Region 5', 'National Park Service, Southeast Regional Office'). "
            "If multiple organizations are listed, join them with semicolons."
        ),
        "units": [],
    },
    "document_type": {
        "description": (
            "The type of government document. One of: report, letter, form, notice, "
            "memorandum, guidance, plan, agreement, minutes, statement, assessment, "
            "press_release, other."
        ),
        "units": [],
    },
}


# ---------------------------------------------------------------------------
# Ablation 1: direct extraction schema and prompt
# ---------------------------------------------------------------------------


class DirectExtractionItemSchema(BaseModel):
    """One metadata attribute extracted from a government document."""

    attribute: str
    value: str | None
    units: str | None


_DIRECT_EXTRACTION_PROMPT = """Extract the following metadata attributes from this government document. Produce exactly one item per attribute (5 items total).

Attributes to extract:

1. title — The official title of the document as it appears on the cover page or header. If no explicit title is present, use the most prominent heading.
2. authors — The individual authors or creators of the document. If multiple, join names with semicolons. Set to None if the document is unsigned or no named authors are given.
3. publication_date — The date the document was published, issued, or finalized. Use YYYY-MM-DD for a full date, YYYY-MM for month/year, or YYYY for year only. Set to None if no date is stated.
4. organization — The agency, office, department, bureau, or funding source responsible for the document (e.g. "U.S. Army Corps of Engineers", "EPA Region 5", "Missouri Department of Natural Resources"). If multiple organizations, join with semicolons. Set to None if not stated.
5. document_type — The type of document. Choose exactly one from:
   - report: technical reports, annual reports, survey reports, feasibility studies, research reports
   - letter: formal correspondence, comment letters, decision letters, response letters, cover letters
   - form: regulatory forms, permit applications, grant applications, intake forms, worksheets
   - notice: Federal Register notices, public notices, determination notices, legal notices
   - memorandum: internal memos, decision memos, briefing memos, staff memos, routing slips
   - guidance: policy guidance documents, regulatory guidance, directives, standards, manuals
   - plan: management plans, strategic plans, work plans, conservation plans, action plans
   - agreement: contracts, cooperative agreements, grant agreements, memoranda of understanding
   - minutes: meeting minutes, public hearing transcripts, committee proceedings, agendas
   - statement: environmental impact statements, policy statements, finding of no significant impact
   - assessment: environmental assessments, risk assessments, cultural resource assessments
   - press_release: press releases, news releases, public announcements
   - other: any document type not listed above

Extraction rules:
- Output exactly one item per attribute (5 items total), in the order listed above.
- Use ONLY information explicitly stated in the document text. Do NOT infer or fabricate values.
- Set value to None if the attribute cannot be determined from the text.
- Always set units to null (these are not numerical measurements).
- Use the exact lowercase attribute name from the list above (e.g. "title", not "Title").

Output format:
{
  "items": [
    { "attribute": "title", "value": "...", "units": null },
    { "attribute": "authors", "value": "...", "units": null },
    { "attribute": "publication_date", "value": "...", "units": null },
    { "attribute": "organization", "value": "...", "units": null },
    { "attribute": "document_type", "value": "...", "units": null }
  ]
}
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


CONFIG = DatasetConfig(
    name="govscape",
    data_dir="data/govscape",
    metadata_file="data/govscape/directory.json",
    entity_schema=EntitySchema,
    entity_identification_prompt=ENTITY_IDENTIFICATION_PROMPT,
    entity_type_description=(
        "A government document — a PDF produced by a government agency, office, or program."
    ),
    attribute_info_dict=_ATTRIBUTE_INFO_DICT,
    direct_extraction_schema=DirectExtractionItemSchema,
    direct_extraction_prompt=_DIRECT_EXTRACTION_PROMPT,
)
