"""GLiNER2 baseline: local encoder-model structured extraction.

Faithful adaptation of GLiNER2 (Fastino AI, EMNLP 2025 System Demo;
``fastino/gliner2-{base,large}-v1``) as a `MeasurementLM` subclass. GLiNER2 is a
small (205M/340M) DeBERTa-style encoder run **locally** via
``GLiNER2.from_pretrained(...)`` — it is NOT an LLM behind an OpenAI-compatible
server, so this baseline makes no `_acall` / `_call_batch` calls at all. It only
reuses `MeasurementLM`'s programmatic `_deduplicate` (and its stored config
attributes); the base ctor still constructs unused OpenAI clients, which is
harmless.

How it maps onto this library (decisions locked with the user):

* **Per-attribute structured passes.** GLiNER2's *structured data extraction*
  returns a list of objects, each holding several linked sub-fields (see its
  tutorial ``3-json_extraction.md``: a ``prescription`` structure yields records
  of ``{medication, dosage, frequency}``). We run **one structure per dataset
  attribute** — each structure's fields carry that attribute's
  ``attribute_info_dict[key]["description"]`` as guidance — so the small model
  gets the same semantic descriptions MeasurementLM uses, rather than having to
  classify an opaque attribute label. Each document is split into overlapping
  line-based word windows (GLiNER2's pip release operates on a bounded word
  window, so long papers must be chunked); work items
  ``(document × attribute × chunk)`` are dispatched together through
  `batch_extract` with a list of per-attribute schemas. Overlap-induced duplicate
  detections are collapsed downstream by `_deduplicate` (which merges records
  sharing entity/attribute/event with equal value+units), so no span-level
  chunk merging is needed.

* **Field scope = name + date + value + units.** Each structure extracts the
  entity ``name``, the measurement ``value`` (numeric), its ``units``, and — when
  the dataset's ``measurement_event_schema`` declares a ``date`` field — the
  measurement ``date``. All other entity/event fields are left ``None`` and are
  matched via the fuzzy ``name`` matcher, exactly as the ChatExtract baseline does.

Like the NuExtract and ChatExtract baselines, `fit()` deliberately skips
`_standardize` (an extra MeasurementLM-specific LLM pass) and applies only the
programmatic `_deduplicate`, for a fair, non-conflating cleanup of duplicate
mentions.
"""

from __future__ import annotations

import re

from .measurementlm import MeasurementLM

# ---------------------------------------------------------------------------
# OCR tag handling. The OCR text is `<page number="N">...</page>` blocks with
# `<table number="M">...<tr><td>...` HTML tables inside. GLiNER is a word-token
# encoder, so raw HTML markup is pure noise that also eats into the fixed word
# chunk budget. We flatten to plain text while preserving row structure: table
# rows become `|`-delimited lines so a row's entity and its cells stay adjacent
# (GLiNER groups a structure's fields by proximity within a chunk).
# ---------------------------------------------------------------------------
_ROW_END_RE = re.compile(r"</tr>|</page>|</p>|<br\s*/?>", re.IGNORECASE)
_CELL_END_RE = re.compile(r"</t[dh]>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]+")
_BLANKS_RE = re.compile(r"\n\s*\n+")
# Splits a document into its 0-indexed <page number="N"> blocks so each chunk
# can be attributed to a single source page (canonical copy in utils/page_attribution.py).
_PAGE_RE = re.compile(r'<page number="(\d+)">(.*?)</page>', re.DOTALL)


class MeasurementLMGliner(MeasurementLM):
    """Local structured-extraction baseline using GLiNER2."""

    def __init__(
        self,
        *args,
        gliner_property_names: dict[str, str] | None = None,
        entity_type_description: str | None = None,
        threshold: float = 0.5,
        batch_size: int = 8,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        device: str | None = None,
        **kwargs,
    ):
        # GLiNER reads OCR text directly; never run the image-based table
        # cleaning pass (mirrors the other baselines).
        kwargs.setdefault("clean_tables", False)
        super().__init__(*args, **kwargs)

        # Imported here (not at module top) so `analysis`/config imports of this
        # module don't require the optional `gliner2[local]` dependency.
        from gliner2 import GLiNER2

        from_pretrained_kwargs = {"map_location": device} if device else {}
        self.extractor = GLiNER2.from_pretrained(self.model_name, **from_pretrained_kwargs)

        self.gliner_property_names = gliner_property_names or {}
        self.entity_type_description = entity_type_description
        self.threshold = threshold
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # -----------------------------------------------------------------------
    # Schema vocabulary
    # -----------------------------------------------------------------------

    def _phrase(self, attr_key: str) -> str:
        """Short human-readable property phrase for one attribute.

        Falls back ``gliner_property_names`` → ``key`` with underscores replaced
        by spaces (there is no ChatExtract map on this class, but the runner may
        pass those phrases in via ``gliner_property_names``).
        """
        return self.gliner_property_names.get(attr_key) or attr_key.replace("_", " ")

    def _has_date_event(self) -> bool:
        """Whether the dataset's event schema declares a ``date`` field."""
        return (
            self.measurement_event_schema is not None
            and "date" in self.measurement_event_schema.model_fields
        )

    def _entity_name_field(self) -> str:
        """Primary entity string field (``name`` for pond/nfix; else the first)."""
        fields = list(self.entity_identification_schema.model_fields)
        return "name" if "name" in fields else fields[0]

    def _build_structure(self, attr_key: str):
        """Build a one-structure GLiNER2 `Schema` for a single attribute.

        Fields: entity ``name``, measurement ``value`` (numeric) + ``units``,
        and — when the dataset tracks measurement dates — ``date``. Each field's
        description is what steers this small model; the ``value`` field carries
        the attribute's full ``attribute_info_dict`` description.
        """
        phrase = self._phrase(attr_key)
        info = self.attribute_info_dict.get(attr_key, {})
        description = info.get("description", "")
        units = info.get("units") or []
        unit_hint = f" (e.g. {', '.join(units[:4])})" if units else ""

        entity_desc = (
            self.entity_type_description
            or "the entity that this measurement belongs to"
        )

        struct_name = attr_key  # unique per work item; used to read results back
        schema = self.extractor.create_schema()
        builder = schema.structure(struct_name)
        builder.field(
            self._entity_name_field(),
            dtype="str",
            description=f"The name or identifier of {entity_desc} for which the "
            f"{phrase} is reported.",
        )
        builder.field(
            "value",
            dtype="str",
            description=f"The numeric value of the {phrase} measurement. {description} "
            f"Give digits only (e.g. '2.3', '850').",
        )
        builder.field(
            "units",
            dtype="str",
            description=f"The unit of the {phrase} value{unit_hint}. "
            f"Leave empty if the quantity is dimensionless or no unit is given.",
        )
        if self._has_date_event():
            builder.field(
                "date",
                dtype="str",
                description="The date or time period when this measurement was taken "
                "(e.g. 'June 2019', 'Summer 2021', '2015'), if stated.",
            )
        schema.build()  # finalize the active builder on the Schema object
        return struct_name, schema

    # -----------------------------------------------------------------------
    # Text preparation
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Flatten `<page>`/`<table>` OCR markup to plain, row-structured text."""
        text = _CELL_END_RE.sub(" | ", text)
        text = _ROW_END_RE.sub("\n", text)
        text = _ANY_TAG_RE.sub(" ", text)
        text = _WS_RE.sub(" ", text)
        text = _BLANKS_RE.sub("\n", text)
        return text.strip()

    def _chunk_text(self, text: str) -> list[str]:
        """Split cleaned text into overlapping, line-aligned word windows.

        Packs whole lines (table rows / sentences kept intact) up to
        ``chunk_size`` words per chunk, carrying the trailing ``chunk_overlap``
        words into the next chunk so a measurement whose fields straddle a
        boundary is still seen whole in at least one chunk.
        """
        lines = [ln for ln in text.split("\n") if ln.strip()]
        chunks: list[str] = []
        cur: list[str] = []
        cur_words = 0
        for line in lines:
            w = len(line.split())
            if cur and cur_words + w > self.chunk_size:
                chunks.append("\n".join(cur))
                overlap: list[str] = []
                ov_words = 0
                for prev in reversed(cur):
                    pw = len(prev.split())
                    if ov_words + pw > self.chunk_overlap:
                        break
                    overlap.insert(0, prev)
                    ov_words += pw
                cur, cur_words = list(overlap), ov_words
            cur.append(line)
            cur_words += w
        if cur:
            chunks.append("\n".join(cur))
        return chunks or [text]

    # -----------------------------------------------------------------------
    # Record construction
    # -----------------------------------------------------------------------

    @staticmethod
    def _slug(name: str | None) -> str:
        if not name:
            return "none"
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        return slug or "none"

    @staticmethod
    def _clean_field(value) -> str | None:
        """Normalize a GLiNER field value to a stripped string or None."""
        if value is None:
            return None
        text = str(value).strip()
        if not text or text.lower() == "none":
            return None
        return text

    def _make_record(
        self, doc_idx: int, attribute: str, name: str | None, date: str | None,
        value: str, units: str | None, page_num: int | None,
    ) -> dict:
        """Build one extraction record in the standard flat schema.

        Entity/event fields are derived from the dataset's schemas and set to
        ``None`` except the entity name (and ``date`` when tracked), so the same
        record shape works across datasets without hardcoding their union.
        ``page_num`` is the 0-indexed OCR ``<page number="N">`` the source chunk
        came from; ``_deduplicate`` aggregates it into an aligned ``page_number``
        list so the judge can limit its context to the source page(s).
        """
        name_field = self._entity_name_field()
        record: dict = {f: None for f in self.entity_identification_schema.model_fields}
        record[name_field] = name
        if self.measurement_event_schema is not None:
            for f in self.measurement_event_schema.model_fields:
                record[f] = None
            if self._has_date_event():
                record["date"] = date

        record |= {"attribute": attribute, "value": value, "units": units}
        entity_id = f"doc_{doc_idx}_{attribute}_{self._slug(name)}"
        return {"document_id": doc_idx} | record | {
            "entity_id": entity_id, "attribute_terms": [], "page_number": page_num,
        }

    # -----------------------------------------------------------------------
    # Extraction driver
    # -----------------------------------------------------------------------

    def _page_chunks(self, doc: str) -> list[tuple[int | None, str]]:
        """Split a document into per-page ``(page_num, chunk)`` work units.

        The OCR ``<page number="N">`` block is a hard chunk boundary — a chunk
        never straddles two pages, so every chunk (and thus every record) maps to
        exactly one 0-indexed page.  Pages larger than ``chunk_size`` words are
        still sub-split by ``_chunk_text`` (most pages exceed GLiNER2's window, so
        one-chunk-per-page would truncate); all sub-chunks share the page number.
        Untagged documents fall back to a single ``None`` page.
        """
        page_bodies: list[tuple[int | None, str]] = [
            (int(m.group(1)), m.group(2)) for m in _PAGE_RE.finditer(doc)
        ] or [(None, doc)]
        chunks: list[tuple[int | None, str]] = []
        for page_num, body in page_bodies:
            for chunk in self._chunk_text(self._clean_ocr_text(body)):
                chunks.append((page_num, chunk))
        return chunks

    def _extract_records(self, documents: list[str]) -> list[dict]:
        """Run one structured GLiNER pass per (document, attribute, page-chunk)."""
        doc_chunks = [self._page_chunks(doc) for doc in documents]
        attr_keys = list(self.attribute_info_dict)

        texts: list[str] = []
        schemas: list = []
        struct_names: list[str] = []
        work: list[tuple[int, str, int | None]] = []  # (doc_idx, attr_key, page_num)
        for doc_idx, chunks in enumerate(doc_chunks):
            for attr_key in attr_keys:
                for page_num, chunk in chunks:
                    struct_name, schema = self._build_structure(attr_key)
                    texts.append(chunk)
                    schemas.append(schema)
                    struct_names.append(struct_name)
                    work.append((doc_idx, attr_key, page_num))

        total_chunks = sum(len(c) for c in doc_chunks)
        print(
            f"Running GLiNER on {len(documents)} documents "
            f"({total_chunks} chunks) × {len(attr_keys)} attributes "
            f"= {len(work)} structured passes (batch_size={self.batch_size}, "
            f"threshold={self.threshold})..."
        )

        results = self.extractor.batch_extract(
            texts,
            schemas,
            batch_size=self.batch_size,
            threshold=self.threshold,
            max_len=self.chunk_size,
        )

        records: list[dict] = []
        has_date = self._has_date_event()
        for (doc_idx, attr_key, page_num), struct_name, result in zip(work, struct_names, results):
            for item in result.get(struct_name, []):
                value = self._clean_field(item.get("value"))
                if value is None or not re.search(r"\d", value):
                    continue  # values are always numeric
                name = self._clean_field(item.get(self._entity_name_field()))
                units = self._clean_field(item.get("units"))
                date = self._clean_field(item.get("date")) if has_date else None
                records.append(self._make_record(doc_idx, attr_key, name, date, value, units, page_num))

        return records

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def fit(self, documents: list[str]) -> list[dict]:
        """Run the GLiNER baseline over the given documents' OCR text."""
        self.data = [{"document_id": i} for i in range(len(documents))]
        self.data = self._extract_records(documents)
        self.data = self._deduplicate(self.data)
        return self.data
