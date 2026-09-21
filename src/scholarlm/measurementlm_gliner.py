"""GLiNER2 baseline: local encoder-model structured extraction.

Faithful adaptation of GLiNER2 (Fastino AI, EMNLP 2025 System Demo;
``fastino/gliner2-{base,large}-v1``) as a `MeasurementLM` subclass. GLiNER2 is a
small (205M/340M) DeBERTa-style encoder run **locally** via
``GLiNER2.from_pretrained(...)`` — it is NOT an LLM behind an OpenAI-compatible
server, so this baseline makes no `_acall` / `_call_batch` calls at all. It only
reuses `MeasurementLM`'s stored config attributes; the base ctor still constructs
unused OpenAI clients, which is harmless.

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
  `batch_extract` with a list of per-attribute schemas.

  Unlike `measurementlm_nuextract.py` (which also chunks and, for exactly that
  reason, keeps `_deduplicate` to merge cross-chunk repeats of the same
  measurement), this baseline does **not** deduplicate: `_deduplicate` is
  `MeasurementLM`'s own pipeline machinery, not part of GLiNER2's method, and per
  `measurementlm_chatextract.py` / `measurementlm_nuextract3.py`'s precedent a
  reimplemented reference baseline shouldn't be credited with it. The practical
  consequence is that `chunk_overlap` (default 64 words) can cause the same
  reported value to surface as more than one record when it falls in the
  overlapping region between two adjacent chunks — a known, accepted artifact of
  adapting a bounded-window encoder to long documents, not a modeling gap.

* **Field scope = every entity/event field the dataset opts in, not just
  name.** GLiNER2's structures support multiple linked sub-fields (see above),
  so — unlike ChatExtract, whose method is fixed to a single Material/Value/Unit
  triple — restricting GLiNER2 to name+value+units would be leaving capability
  on the table, not matching the reference method. Each structure asks for
  every entity/event field named in ``DatasetConfig.gliner_field_descriptions``
  (in addition to the subject name, value, and units); a field with no entry
  there is never asked for. This is a GLiNER-only mapping, never
  ``entity_schema``/``measurement_event_schema`` themselves — those feed the
  real pipeline's (and other baselines') structured-decoding schemas directly,
  and GLiNER's field scope must not change what they send. A field with no
  entry (e.g. ``identifiers``, an alias-resolution aid for the real pipeline's
  entity matching, never reported content in its own right) is a deliberate
  exclusion, not an oversight. Description text should be copied verbatim from
  this dataset's own prompts (typically ``direct_extraction_prompt``), not
  freshly authored.

Like the ChatExtract and NuExtract3 baselines, `fit()` deliberately skips both
`_standardize` and `_deduplicate` — both are `MeasurementLM`'s own pipeline
contributions, not something a reimplemented reference method should be
credited with.
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
#
# The OCR HTML is *pretty-printed* (a newline after every cell), so we must
# collapse ALL source whitespace — newlines included — up front; otherwise those
# intra-row newlines survive and shatter each table row into one cell per line.
# After that, only the row/page tags reintroduce line breaks, so a `<tr>`'s cells
# all land on a single `|`-delimited line.
# ---------------------------------------------------------------------------
_ROW_END_RE = re.compile(r"</tr>|</page>|</p>|<br\s*/?>", re.IGNORECASE)
_CELL_END_RE = re.compile(r"</t[dh]>", re.IGNORECASE)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_ALL_WS_RE = re.compile(r"\s+")
_WS_RE = re.compile(r"[ \t]+")
_PIPE_RUN_RE = re.compile(r"(?:\|[ \t]*){2,}")  # collapse empty-cell `| | |` runs
# Splits a document into its 0-indexed <page number="N"> blocks so each chunk
# can be attributed to a single source page (canonical copy in utils/page_attribution.py).
_PAGE_RE = re.compile(r'<page number="(\d+)">(.*?)</page>', re.DOTALL)
# Isolates each `<table>...</table>` span (before flattening) so the chunker can
# treat a table as an atomic block rather than splitting it mid-rows.
_TABLE_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)

# The qualifier/shape fields every other extraction method carries as
# qualifiers/point_value/lower/upper/list_values/tolerance/standard_deviation
# (MeasurementLM.ParseQuantityResponse) -- see _build_structure's docstring
# for why GLiNER2 represents qualifiers/list_values as comma-joined strings
# rather than JSON lists.
QUANTITY_FIELD_NAMES = (
    "qualifiers", "point_value", "lower", "upper",
    "list_values", "tolerance", "standard_deviation",
)


class MeasurementLMGliner(MeasurementLM):
    """Local structured-extraction baseline using GLiNER2."""

    def __init__(
        self,
        *args,
        gliner_property_names: dict[str, str] | None = None,
        entity_type_description: str | None = None,
        gliner_entity_description: str | None = None,
        gliner_field_descriptions: dict[str, str] | None = None,
        threshold: float = 0.5,
        batch_size: int = 8,
        chunk_size: int = 384,
        chunk_overlap: int = 64,
        device: str | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Imported here (not at module top) so `analysis`/config imports of this
        # module don't require the optional `gliner2[local]` dependency.
        from gliner2 import GLiNER2

        from_pretrained_kwargs = {"map_location": device} if device else {}
        self.extractor = GLiNER2.from_pretrained(self.model_name, **from_pretrained_kwargs)

        self.gliner_property_names = gliner_property_names or {}
        self.entity_type_description = entity_type_description
        self.gliner_entity_description = gliner_entity_description
        self.gliner_field_descriptions = gliner_field_descriptions or {}
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

    def _extra_entity_fields(self) -> list[str]:
        """Entity-schema fields to ask for beyond the subject name field.

        Only fields named in ``gliner_field_descriptions`` are asked for. A
        field with no entry there is a deliberate exclusion, not an oversight
        -- e.g. ``identifiers`` (an alias-resolution aid for the real
        pipeline's entity matching, not reported content).
        """
        name_field = self._entity_name_field()
        return [
            f for f in self.entity_identification_schema.model_fields
            if f != name_field and f in self.gliner_field_descriptions
        ]

    def _extra_event_fields(self) -> list[str]:
        """Measurement-event-schema fields to ask for beyond the subject name field.

        ``name_field`` is excluded here too, in case a dataset's event schema
        happens to declare a same-named field. See `_extra_entity_fields` for
        the description-required filter.
        """
        if self.measurement_event_schema is None:
            return []
        name_field = self._entity_name_field()
        return [
            f for f in self.measurement_event_schema.model_fields
            if f != name_field and f in self.gliner_field_descriptions
        ]

    def _field_description(self, field_name: str) -> str:
        """Description for one extra entity/event field, from
        ``gliner_field_descriptions`` (see `_extra_entity_fields`)."""
        return self.gliner_field_descriptions[field_name]

    def _entity_name_field(self) -> str:
        """Field that holds the measurement subject's name.

        Every dataset config enumerates the measurement subject as ``name`` on
        its entity schema (pond/nfix/supermat/measeval). Fail loud rather than
        guessing a fallback field: writing the extracted subject into the
        wrong column would silently drop every candidate edge in
        ``match_datasets`` instead of raising here.
        """
        fields = list(self.entity_identification_schema.model_fields)
        if "name" not in fields:
            raise ValueError(
                f"entity_identification_schema {self.entity_identification_schema.__name__!r} "
                f"has no 'name' field (fields: {fields}) -- MeasurementLMGliner requires the "
                f"measurement subject to be named 'name' on the entity schema."
            )
        return "name"

    def _build_structure(self, attr_key: str):
        """Build a one-structure GLiNER2 `Schema` for a single attribute.

        Fields: subject ``name``, measurement ``value`` (full raw text) +
        ``units``, the qualifier/shape fields (see QUANTITY_FIELD_NAMES below),
        plus every other entity- and measurement-event-schema field (see
        `_extra_entity_fields` / `_extra_event_fields`) — module docstring's
        "Field scope" note explains why GLiNER2 isn't restricted to name-only the
        way ChatExtract is. Each field's description is what steers this small
        model; the ``value`` field carries the attribute's full
        ``attribute_info_dict`` description.

        GLiNER2 structure fields are scalar (``dtype="str"`` only, no array
        type) — unlike every other extraction method in this repo, ``qualifiers``
        and ``list_values`` are comma-joined strings here, not JSON lists. This
        is a real, documented divergence from the shared
        qualifiers/point_value/lower/upper/list_values/tolerance/
        standard_deviation shape (MeasurementLM.ParseQuantityResponse), forced
        by GLiNER2's own type system, not an oversight.
        """
        phrase = self._phrase(attr_key)
        info = self.attribute_info_dict.get(attr_key, {})
        description = info.get("description", "")
        units = info.get("units") or []
        unit_hint = f" (e.g. {', '.join(units)})" if units else ""

        entity_desc = (self.gliner_entity_description or self.entity_type_description).rstrip(". ")

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
            description=f"The value of the {phrase} measurement exactly as reported, in full -- "
            f"including any range, list, inequality, mean/median/count label, or uncertainty "
            f"measure (+/- value, confidence interval, standard deviation) reported alongside "
            f"it. {description} Do not convert, round, drop, or otherwise modify any part of it.",
        )
        builder.field(
            "units",
            dtype="str",
            description=f"The unit of the {phrase} value{unit_hint}. "
            f"Leave empty if the quantity is dimensionless or no unit is given.",
        )
        builder.field(
            "qualifiers",
            dtype="str",
            description="Comma-separated tags describing the shape of the value, drawn only "
            "from: IsCount (a count of discrete items), IsApproximate (explicitly hedged, e.g. "
            "'about 50'), IsList (an enumerated list of separate values), IsRange (a reported "
            "interval or one-sided bound, e.g. '3-7', '< 5'), IsMean (an explicitly stated "
            "mean/average), IsMedian (an explicitly stated median), HasTolerance (an explicit "
            "+/- value or confidence interval reported alongside the value), HasSD (an explicit "
            "standard deviation reported alongside the value). Combine tags freely when the "
            "text supports it (e.g. 'IsApproximate, IsMean'). Leave empty for a plain, unhedged "
            "single value.",
        )
        builder.field(
            "point_value",
            dtype="str",
            description="The single central value, when one is directly reported -- a plain "
            "point value, or the stated mean/median/count. Leave empty if no single central "
            "value is reported.",
        )
        builder.field(
            "lower",
            dtype="str",
            description="The lower bound of a reported range or one-sided inequality (e.g. "
            "'at least 10'). Leave empty if no range or lower bound is reported.",
        )
        builder.field(
            "upper",
            dtype="str",
            description="The upper bound of a reported range or one-sided inequality (e.g. "
            "'< 5'). Leave empty if no range or upper bound is reported.",
        )
        builder.field(
            "list_values",
            dtype="str",
            description="If the value is an enumerated list of separate values, the parsed "
            "items joined by commas, in the order reported. Leave empty otherwise.",
        )
        builder.field(
            "tolerance",
            dtype="str",
            description="The confidence interval or +/- value exactly as reported (e.g. "
            "'± 0.5', '95% CI: 5-9'), if one is given alongside the value. Leave empty otherwise.",
        )
        builder.field(
            "standard_deviation",
            dtype="str",
            description="The standard deviation exactly as reported, if one is given "
            "alongside the value. Leave empty otherwise.",
        )
        for field_name in self._extra_entity_fields() + self._extra_event_fields():
            builder.field(field_name, dtype="str", description=self._field_description(field_name))
        schema.build()  # finalize the active builder on the Schema object
        return struct_name, schema

    # -----------------------------------------------------------------------
    # Text preparation
    # -----------------------------------------------------------------------

    @staticmethod
    def _clean_ocr_text(text: str) -> str:
        """Flatten `<page>`/`<table>` OCR markup to plain, row-structured text.

        Collapses all source whitespace (newlines included) *first* so the
        pretty-printed HTML's per-cell newlines can't split a row; only the
        row/page tags below reintroduce line breaks, keeping each `<tr>`'s cells
        together on one `|`-delimited line (entity adjacent to its values). Empty
        cells (`<td></td>`) collapse away rather than leaving `| |` gaps.
        """
        text = _ALL_WS_RE.sub(" ", text)      # kill pretty-print newlines first
        text = _CELL_END_RE.sub(" | ", text)  # cell boundary -> " | "
        text = _ROW_END_RE.sub("\n", text)    # row/page/paragraph -> newline
        text = _ANY_TAG_RE.sub(" ", text)     # drop remaining (opening) tags
        lines: list[str] = []
        for line in text.split("\n"):
            line = _WS_RE.sub(" ", line)
            line = _PIPE_RUN_RE.sub("| ", line)   # squeeze empty-cell runs
            line = line.strip().strip("|").strip()
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _split_blocks(self, body: str) -> list[tuple[str, str]]:
        """Segment a raw page body into ordered ``(kind, cleaned_text)`` blocks.

        ``kind`` is ``"table"`` or ``"prose"``. Table spans are isolated from the
        raw HTML *before* flattening so the chunker can keep each one atomic;
        everything between/around tables is prose. Each segment is run through
        `_clean_ocr_text`, so a table block's lines are its ``|``-delimited rows
        (with the first line being the header row).
        """
        blocks: list[tuple[str, str]] = []
        pos = 0
        for m in _TABLE_RE.finditer(body):
            if m.start() > pos:
                prose = self._clean_ocr_text(body[pos:m.start()])
                if prose:
                    blocks.append(("prose", prose))
            table = self._clean_ocr_text(m.group(0))
            if table:
                blocks.append(("table", table))
            pos = m.end()
        if pos < len(body):
            prose = self._clean_ocr_text(body[pos:])
            if prose:
                blocks.append(("prose", prose))
        return blocks

    def _split_table_lines(self, lines: list[str]) -> list[str]:
        """Row-split an oversized table into ``<= chunk_size`` word sub-chunks.

        The header row (``lines[0]``) is repeated at the top of every sub-chunk so
        each fragment keeps its column semantics (which column a value sits under).
        A single row wider than ``chunk_size`` cannot be split further, so it is
        emitted whole (with its header) and left for GLiNER's ``max_len`` to bound.
        """
        header = lines[0]
        header_words = len(header.split())
        body_rows = lines[1:]
        if not body_rows:
            return ["\n".join(lines)]
        subs: list[str] = []
        cur = [header]
        cur_words = header_words
        for row in body_rows:
            rw = len(row.split())
            if len(cur) > 1 and cur_words + rw > self.chunk_size:
                subs.append("\n".join(cur))
                cur = [header]
                cur_words = header_words
            cur.append(row)
            cur_words += rw
        if len(cur) > 1:
            subs.append("\n".join(cur))
        return subs

    def _chunk_page(self, body: str) -> list[str]:
        """Table-aware chunking of one raw page body into GLiNER-sized windows.

        Prose is packed into overlapping ``chunk_size``-word windows, carrying the
        trailing ``chunk_overlap`` words across boundaries (as before). A table is
        kept intact when it fits within ``chunk_size`` — flushed into a fresh chunk
        rather than split across a boundary — and only row-split (with its header
        repeated, `_split_table_lines`) when it exceeds ``chunk_size`` on its own.
        """
        chunks: list[str] = []
        cur: list[str] = []
        cur_words = 0

        def flush() -> None:
            nonlocal cur, cur_words
            if cur:
                chunks.append("\n".join(cur))
            cur, cur_words = [], 0

        for kind, text in self._split_blocks(body):
            lines = [ln for ln in text.split("\n") if ln.strip()]
            if kind == "table":
                table_words = sum(len(ln.split()) for ln in lines)
                if table_words <= self.chunk_size:
                    # Keep atomic: start a fresh chunk if it won't fit whole.
                    if cur and cur_words + table_words > self.chunk_size:
                        flush()
                    cur.extend(lines)
                    cur_words += table_words
                else:
                    # Oversized: flush prose, emit header-repeated row splits.
                    flush()
                    chunks.extend(self._split_table_lines(lines))
                continue
            for line in lines:  # prose
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
        flush()
        return chunks or [self._clean_ocr_text(body)]

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
        self, doc_idx: int, attribute: str, name: str | None,
        value: str, units: str | None, quantity: dict[str, str | None],
        page_num: int | None, extra: dict[str, str | None],
    ) -> dict:
        """Build one extraction record in the standard flat schema.

        Entity/event fields are derived from the dataset's schemas: every field
        GLiNER extracted (``extra``, keyed by field name — see
        `_extra_entity_fields` / `_extra_event_fields`) is filled in, the rest
        default to ``None``, so the same record shape works across datasets
        without hardcoding their union. ``quantity`` carries QUANTITY_FIELD_NAMES
        (qualifiers/list_values as comma-joined strings, not lists -- see
        `_build_structure`'s docstring). ``page_num`` is the 0-indexed OCR
        ``<page number="N">`` the source chunk came from; with no `_deduplicate`
        step (see module docstring), it stays a plain scalar here, matching
        ChatExtract's ``_make_record``.
        """
        name_field = self._entity_name_field()
        record: dict = {f: None for f in self.entity_identification_schema.model_fields}
        if self.measurement_event_schema is not None:
            for f in self.measurement_event_schema.model_fields:
                record[f] = None
        record.update(extra)
        # After the extra-field fill, not before: ``name_field`` may itself be an
        # event field (see _entity_name_field), which the loop above would
        # otherwise overwrite with None.
        record[name_field] = name

        record |= {"attribute": attribute, "value": value, "units": units} | quantity
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
        still sub-split by ``_chunk_page`` (most pages exceed GLiNER2's window, so
        one-chunk-per-page would truncate); all sub-chunks share the page number.
        Untagged documents fall back to a single ``None`` page.
        """
        page_bodies: list[tuple[int | None, str]] = [
            (int(m.group(1)), m.group(2)) for m in _PAGE_RE.finditer(doc)
        ] or [(None, doc)]
        chunks: list[tuple[int | None, str]] = []
        for page_num, body in page_bodies:
            for chunk in self._chunk_page(body):
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

        extra_field_names = self._extra_entity_fields() + self._extra_event_fields()

        records: list[dict] = []
        for (doc_idx, attr_key, page_num), struct_name, result in zip(work, struct_names, results):
            for item in result.get(struct_name, []):
                value = self._clean_field(item.get("value"))
                if value is None or not re.search(r"\d", value):
                    continue  # values are always numeric
                name = self._clean_field(item.get(self._entity_name_field()))
                units = self._clean_field(item.get("units"))
                quantity = {f: self._clean_field(item.get(f)) for f in QUANTITY_FIELD_NAMES}
                extra = {f: self._clean_field(item.get(f)) for f in extra_field_names}
                records.append(
                    self._make_record(doc_idx, attr_key, name, value, units, quantity, page_num, extra)
                )

        return records

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def fit(self, documents: list[str]) -> list[dict]:
        """Run the GLiNER baseline over the given documents' OCR text.

        Like ChatExtract and NuExtract3, skips both `_standardize` and
        `_deduplicate` — see module docstring's "Field scope" and dedup notes.
        """
        self.data = self._extract_records(documents)
        return self.data
