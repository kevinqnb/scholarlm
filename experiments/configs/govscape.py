"""
Dataset configuration for the govscape dataset.

Government document metadata extraction. Each document is a government-generated
PDF; the extraction task is to recover basic bibliographic metadata from the OCR text.

Uses the MetadataLM pipeline (run_metadata_extraction.py), which makes a single
LLM call per document and returns one metadata record per document.
"""
from __future__ import annotations

from pydantic import BaseModel

from scholarlm.config import DatasetConfig


# ---------------------------------------------------------------------------
# Metadata schema and prompt
# ---------------------------------------------------------------------------


class GovscapeMetadataSchema(BaseModel):
    """Metadata extracted from a single government document."""

    title: str | None
    authors: str | None
    publication_date: str | None
    organization: str | None
    document_type: str | None


_METADATA_EXTRACTION_PROMPT = """Extract the following metadata from this government document.

Fields:

- title: The official title of the document as it appears on the cover page or header. If no explicit title is present, use the most prominent heading.
- authors: The individual authors or creators of the document. If multiple, join names with semicolons. Set to null if the document is unsigned or no named authors are given.
- publication_date: The date the document was published, issued, or finalized. Use YYYY-MM-DD for a full date, YYYY-MM for month/year, or YYYY for year only. Set to null if no date is stated.
- organization: The agency, office, department, bureau, or funding source responsible for the document (e.g. "U.S. Army Corps of Engineers", "EPA Region 5", "Missouri Department of Natural Resources"). If multiple organizations, join with semicolons. Set to null if not stated.
- document_type: The type of document. Choose exactly one from:
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
- Use ONLY information explicitly stated in the document text. Do NOT infer or fabricate values.
- Set a field to null if its value cannot be determined from the text.
"""


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


CONFIG = DatasetConfig(
    name="govscape",
    data_dir="data/govscape",
    metadata_file="data/govscape/directory.json",
    metadata_extraction_schema=GovscapeMetadataSchema,
    metadata_extraction_prompt=_METADATA_EXTRACTION_PROMPT,
)
