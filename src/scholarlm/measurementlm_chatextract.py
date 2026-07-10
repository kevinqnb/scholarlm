"""ChatExtract baseline: conversational, sentence-by-sentence triplet extraction.

Faithful reimplementation of ChatExtract (Polak & Morgan, *Nature Communications*
2024, arXiv:2303.05352) as a `MeasurementLM` subclass. ChatExtract extracts
``Material, Value, Unit`` triplets for a *single target property (attribute)* from scientific
text using a multi-turn conversation with redundant yes/no verification questions
to suppress hallucination. The prompt strings below are copied verbatim 
from the author's original implementation.

How it maps onto this library:

* **Single-property → multi-attribute.** ChatExtract targets one property or attribute 
  at a time. Our datasets have several attributes, so the *entire* conversation is run once per
  attribute per text unit.

* **Material → entity.** ChatExtract's "material" is our entity; the extracted
  material string becomes the record's ``name``. ChatExtract does not extract the
  other entity/event fields (location, ecosystem, date, ...), so those stay
  ``None`` and matching leans on the fuzzy matcher's ``name`` threshold. Values
  are kept verbatim as the model returns them (e.g. ``"∼1000"``), exactly as the
  paper does.

* **Preprocessing.** OCR text is split into sentences (`utils.sentences`,
  pysbd). Sentences with no digit are dropped (values are always numeric).
  Classification runs on the bare sentence; every later turn runs on the
  ``passage = title + preceding sentence + target sentence`` (the paper's
  deliberately short context). Real document tables are handled by a separate
  classify-then-extract workflow (no redundant verification, per the paper);
  figures are ignored.
"""

from __future__ import annotations

import asyncio
import re

from .measurementlm import MeasurementLM
from .utils.sentences import split_sentences

# ---------------------------------------------------------------------------
# OCR tag regexes (canonical copies live in utils/page_attribution.py).
# ---------------------------------------------------------------------------
_PAGE_RE = re.compile(r'<page number="(\d+)">(.*?)</page>', re.DOTALL)
_TABLE_RE = re.compile(r'<table number="(\d+)">.*?</table>', re.DOTALL)
_CAPTION_RE = re.compile(r'<caption>(.*?)</caption>', re.DOTALL)
_STRIP_TAGS_RE = re.compile(r'<[^>]+>|\\\([^)]*\)|\\\[[^\]]*\]')

_ORDINALS = [
    "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth",
    "ninth", "tenth", "eleventh", "twelfth", "thirteenth", "fourteenth",
    "fifteenth", "sixteenth", "seventeenth", "eighteenth", "nineteenth", "twentieth",
]

# ---------------------------------------------------------------------------
# Prompt templates — copied verbatim from ChatExtract.py (lines 85-103), with
# the property name parameterized as ``{prop}``. Each is prepended to the text
# (sentence, passage, or table) with the trailing "\n\n" separator intact.
# ---------------------------------------------------------------------------

CLASSIF_Q = 'Answer "Yes" or "No" only. Does the following text contain a value of {prop}?\n\n'
IFMULTI_Q = 'Answer "Yes" or "No" only. Does the following text contain more than one value of {prop}?\n\n'

# value, unit, material — asked in this order.
SINGLE_Q = [
    'Give the number only without units, do not use a full sentence. If the value is not present in the text, type "None". What is the value of the {prop} in the following text?\n\n',
    'Give the unit only, do not use a full sentence. If the unit is not present in the text, type "None". What is the unit of the {prop} in the following text?\n\n',
    'Give the name of the material only, do not use a full sentence. If the name of the material is not present in the text, type "None". What is the material for which the {prop} is given in the following text?\n\n',
]
# Redundant follow-up verification (two-part; the extracted answer is inserted
# between the parts). The reference script defines these but only ever runs the
# multi-valued verifications (ChatExtract.py:184-203); the single-valued branch
# is gated solely by a literal "none" check with no verification call. We match
# that by default; ``include_single_verification`` optionally enables single-
# branch verification as an explicit, non-faithful ablation.
SINGLE_FOLLOWUP_Q = [
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the value of the {prop} for the compound in the following text?\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the unit of the value of {prop} in the following text?\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is "', '" the compound for which the value of {prop} is given in the following text? Make sure it is a real compound.\n\n'],
]

TAB_Q = 'Use only data present in the text. If data is not present in the text, type "None". Summarize the values of {prop} in the following text in a form of a table consisting of: Material, Value, Unit\n\n'
# Three-part templates: cell value, ordinal, and passage are spliced in.
TAB_FOLLOWUP_Q = [
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is "', '" the ', ' compound for which the value of {prop} is given in the following text? Make sure it is a real compound.\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the value of the {prop} for the ', ' compound in the following text?\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the unit of the ', ' value of {prop} in the following text?\n\n'],
]

# Real-document-table workflow (paper Sec. II.C; not in the reference script).
TABLE_CLASSIFY_Q = 'Answer "Yes" or "No" only. Does the following table contain values of {prop}?\n\n'
TABLE_EXTRACT_Q = (
    'Use only data given in the table and its caption. Extract the material names and '
    'values of {prop} from the following table. Present these values in a new table with '
    'columns only for: Material, Value, Unit\n\n'
)

# Token budgets by call type (ChatExtract.py: 6 for yes/no, 500 otherwise). We
# use a slightly larger yes/no budget to tolerate models that prefix a word.
_YN_TOKENS = 12
_EXTRACT_TOKENS = 512


class MeasurementLMChatExtract(MeasurementLM):
    """Conversational sentence-by-sentence extraction baseline (ChatExtract)."""

    def __init__(
        self,
        *args,
        attribute_property_names: dict[str, str] | None = None,
        include_single_verification: bool = False,
        extract_tables: bool = True,
        max_tables_per_document: int = 30,
        max_concurrent: int = 32,
        **kwargs,
    ):
        # ChatExtract reads OCR text directly; never run the image-based table
        # cleaning pass (mirrors the other baselines).
        kwargs.setdefault("clean_tables", False)
        super().__init__(*args, max_concurrent=max_concurrent, **kwargs)
        self.attribute_property_names = attribute_property_names or {}
        self.include_single_verification = include_single_verification
        self.extract_tables = extract_tables
        self.max_tables_per_document = max_tables_per_document

    # -----------------------------------------------------------------------
    # Property vocabulary
    # -----------------------------------------------------------------------

    def _property_items(self) -> list[tuple[str, str]]:
        """(attribute_key, <PROPERTY> phrase) pairs, one per dataset attribute."""
        items = []
        for attr_key in self.attribute_info_dict:
            phrase = self.attribute_property_names.get(attr_key, attr_key.replace("_", " "))
            items.append((attr_key, phrase))
        return items

    # -----------------------------------------------------------------------
    # Text preparation
    # -----------------------------------------------------------------------

    def _prepare_document(self, context: str, title: str) -> dict:
        """Split one document's OCR text into sentence and table work units.

        Returns ``{"sentences": [(sentence, passage), ...], "tables": [str, ...]}``.
        Prose is gathered per page (in page order), tables are pulled out and
        handled separately, and each candidate sentence's ``passage`` is built as
        ``title + preceding sentence + target sentence``. Sentences with no digit
        are dropped up front (values are always numeric).
        """
        title = (title or "").strip()

        prose_sentences: list[str] = []
        tables: list[str] = []

        page_matches = list(_PAGE_RE.finditer(context))
        # Fall back to treating the whole context as one page if untagged.
        page_bodies = [m.group(2) for m in page_matches] if page_matches else [context]

        for body in page_bodies:
            for tm in _TABLE_RE.finditer(body):
                table_block = tm.group(0)
                if _CAPTION_RE.search(table_block):
                    # Cleaned OCR: the caption is a <caption> element inside the
                    # table block, so it is already fed to the model as-is.
                    tables.append(table_block)
                else:
                    # Raw OCR: no caption element; attach the nearest preceding
                    # text line as the caption.
                    caption = self._table_caption(body, tm.start())
                    tables.append((caption + "\n" + table_block).strip() if caption else table_block)
            # Prose = page body with table blocks removed, then sentence-split.
            prose = _TABLE_RE.sub(" ", body)
            prose_sentences.extend(split_sentences(prose))

        sentences: list[tuple[str, str]] = []
        for i, sentence in enumerate(prose_sentences):
            if not re.search(r"\d", sentence):  # digit pre-filter
                continue
            prev = prose_sentences[i - 1] if i > 0 else ""
            passage = self._build_passage(title, prev, sentence)
            sentences.append((sentence, passage))

        if len(tables) > self.max_tables_per_document:
            tables = tables[: self.max_tables_per_document]

        return {"sentences": sentences, "tables": tables}

    @staticmethod
    def _build_passage(title: str, prev: str, sentence: str) -> str:
        parts = []
        if title:
            parts.append(title.rstrip(".") + ".")
        if prev:
            parts.append(prev)
        parts.append(sentence)
        return " ".join(parts)

    @staticmethod
    def _table_caption(body: str, table_start: int) -> str:
        """Caption fallback for *raw* OCR tables with no <caption> element:
        the last non-empty text line immediately preceding the table tag."""
        before = _STRIP_TAGS_RE.sub("", body[:table_start])
        lines = [ln.strip() for ln in before.splitlines() if ln.strip()]
        return lines[-1] if lines else ""

    # -----------------------------------------------------------------------
    # Conversation helpers (stateful; reuse _acall)
    # -----------------------------------------------------------------------

    async def _acall_retry(self, messages, max_tokens, max_retries: int = 3) -> str:
        """`_acall` at temperature 0 with a light retry on empty responses."""
        answer = ""
        for attempt in range(max_retries):
            answer = await self._acall(messages, temperature=0.0, max_tokens=max_tokens)
            if answer:
                return answer
            await asyncio.sleep(2 ** attempt)
        return answer

    async def _ask(self, messages: list[dict], question: str, yes_no: bool) -> str:
        """Append a user turn, call the model, append the assistant turn, return it."""
        messages.append({"role": "user", "content": question})
        answer = await self._acall_retry(
            messages, max_tokens=_YN_TOKENS if yes_no else _EXTRACT_TOKENS
        )
        messages.append({"role": "assistant", "content": answer})
        return answer

    # -----------------------------------------------------------------------
    # Per-sentence conversation (single/multi branching)
    # -----------------------------------------------------------------------

    async def _process_sentence(self, doc_idx, sentence, passage, attr_key, prop) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": ""}]

        # Stage A: classify the bare sentence.
        ans = await self._ask(messages, CLASSIF_Q.format(prop=prop) + sentence, yes_no=True)
        if "yes" not in ans.strip().lower():
            return []

        # Stage B gate: single vs. multiple values (on the passage).
        ans = await self._ask(messages, IFMULTI_Q.format(prop=prop) + passage, yes_no=True)
        low = ans.lower()
        if "no" in low:
            return await self._extract_single(messages, doc_idx, passage, attr_key, prop)
        if "yes" in low:
            return await self._extract_multi(messages, doc_idx, passage, attr_key, prop)
        return []  # ambiguous gate → extract nothing (matches reference behavior)

    async def _extract_single(self, messages, doc_idx, passage, attr_key, prop) -> list[dict]:
        """Ask value → unit → material, each with an optional strict verification."""
        fields = ["value", "unit", "material"]
        extracted: dict[str, str] = {}
        valid: dict[str, bool] = {}

        for idx, field in enumerate(fields):
            ans = (await self._ask(messages, SINGLE_Q[idx].format(prop=prop) + passage, yes_no=False)).strip()
            extracted[field] = ans
            ok = bool(ans) and "none" not in ans.lower()
            if ok and self.include_single_verification:
                pre, post = SINGLE_FOLLOWUP_Q[idx]
                followup = pre + ans + post.format(prop=prop) + passage
                verdict = await self._ask(messages, followup, yes_no=True)
                if "no" in verdict.lower():
                    ok = False
            valid[field] = ok

        if not valid["value"]:
            return []
        material = extracted["material"] if valid["material"] else None
        units = extracted["unit"] if valid["unit"] else None
        return [self._make_record(doc_idx, attr_key, material, extracted["value"], units)]

    async def _extract_multi(self, messages, doc_idx, passage, attr_key, prop) -> list[dict]:
        """Ask for a Material/Value/Unit table, then verify each cell strictly."""
        table_text = await self._ask(messages, TAB_Q.format(prop=prop) + passage, yes_no=False)
        rows = self._parse_table_rows(table_text)

        records: list[dict] = []
        for k, (material, value, unit) in enumerate(rows):
            ordinal = _ORDINALS[k] if k < len(_ORDINALS) else f"{k + 1}th"
            cells = {"material": material, "value": value, "unit": unit}
            valid: dict[str, bool] = {}
            row_ok = True
            for col_idx, col in enumerate(["material", "value", "unit"]):
                cell = str(cells[col]).strip()
                if not row_ok:
                    valid[col] = False  # short-circuited: not asked
                    continue
                if not cell or "none" in cell.lower():
                    valid[col] = False
                    row_ok = False
                    continue
                p0, p1, p2 = TAB_FOLLOWUP_Q[col_idx]
                followup = p0 + cell + p1 + ordinal + p2.format(prop=prop) + passage
                verdict = await self._ask(messages, followup, yes_no=True)
                if "no" in verdict.lower():
                    valid[col] = False
                    row_ok = False
                else:
                    valid[col] = True

            if valid.get("material") and valid.get("value"):
                units = cells["unit"].strip() if valid.get("unit") else None
                records.append(
                    self._make_record(doc_idx, attr_key, cells["material"].strip(), cells["value"].strip(), units)
                )
        return records

    # -----------------------------------------------------------------------
    # Per-table conversation (real document tables; no verification)
    # -----------------------------------------------------------------------

    async def _process_table(self, doc_idx, table_text, attr_key, prop) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": ""}]

        ans = await self._ask(messages, TABLE_CLASSIFY_Q.format(prop=prop) + table_text, yes_no=True)
        if "yes" not in ans.lower():
            return []

        extracted = await self._ask(messages, TABLE_EXTRACT_Q.format(prop=prop) + table_text, yes_no=False)
        rows = self._parse_table_rows(extracted)

        records: list[dict] = []
        for material, value, unit in rows:
            value = str(value).strip()
            if not value or "none" in value.lower():
                continue
            material = None if (not material.strip() or "none" in material.lower()) else material.strip()
            units = None if (not unit.strip() or "none" in unit.lower()) else unit.strip()
            records.append(self._make_record(doc_idx, attr_key, material, value, units))
        return records

    # -----------------------------------------------------------------------
    # Table parsing
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_table_rows(text: str) -> list[tuple[str, str, str]]:
        """Parse a model-emitted Material/Value/Unit table into (mat, val, unit) rows.

        Handles both CSV-style and Markdown-pipe tables (split on ``[,|]``, drop
        empty cells — ChatExtract.py line 162-163), drops the header row and any
        Markdown separator (``|---|---|``) row, and reads the first three cells of
        each remaining row positionally as Material, Value, Unit.
        """
        parsed: list[list[str]] = []
        for line in text.strip().splitlines():
            cells = [c.strip() for c in re.split(r"[,|]", line) if c.strip()]
            if len(cells) >= 3:
                parsed.append(cells)

        rows: list[tuple[str, str, str]] = []
        for idx, cells in enumerate(parsed):
            if idx == 0:  # header row
                continue
            if all(set(c) <= set("-: ") for c in cells):  # markdown separator
                continue
            rows.append((cells[0], cells[1], cells[2]))
        return rows

    # -----------------------------------------------------------------------
    # Record construction
    # -----------------------------------------------------------------------

    @staticmethod
    def _slug(material: str | None) -> str:
        if not material:
            return "none"
        slug = re.sub(r"[^a-z0-9]+", "_", material.lower()).strip("_")
        return slug or "none"

    def _make_record(self, doc_idx: int, attribute: str, material: str | None, value: str, units: str | None) -> dict:
        """Build one extraction record in the standard flat schema.

        ``entity_id`` keys on the document + normalized material so `_deduplicate`
        merges the same material+attribute+value mentioned across sentences.
        """
        item = {
            "name": material,
            "identifiers": None,
            "location": None,
            "ecosystem": None,   # pond fuzzy-match field (unpopulated by ChatExtract)
            "site_type": None,   # nfix fuzzy-match field (unpopulated by ChatExtract)
            "date": None,
            "additional_details": None,
            "attribute": attribute,
            "value": value,
            "units": units,
        }
        entity_id = f"doc_{doc_idx}_{attribute}_{self._slug(material)}"
        return {"document_id": doc_idx} | item | {"entity_id": entity_id, "attribute_terms": []}

    # -----------------------------------------------------------------------
    # Extraction driver
    # -----------------------------------------------------------------------

    def _extract_records(self, documents: list[str], titles: list[str]) -> list[dict]:
        """Run the full ChatExtract conversation for every (text-unit × property).

        Each work item is an independent stateful conversation; they run
        concurrently under a semaphore (mirroring `_call_batch`'s pattern), but
        each conversation's own turns are sequential.
        """
        doc_units = [self._prepare_document(doc, titles[i]) for i, doc in enumerate(documents)]
        property_items = self._property_items()

        total_sentences = sum(len(u["sentences"]) for u in doc_units)
        total_tables = sum(len(u["tables"]) for u in doc_units)
        print(
            f"Prepared {total_sentences} candidate sentences and {total_tables} tables "
            f"across {len(documents)} documents; {len(property_items)} properties "
            f"→ {(total_sentences + (total_tables if self.extract_tables else 0)) * len(property_items)} conversations."
        )

        async def _run():
            sem = asyncio.Semaphore(self.max_concurrent)

            async def _guarded(coro):
                async with sem:
                    try:
                        return await coro
                    except Exception as e:  # never let one conversation sink the run
                        print(f"ChatExtract work item failed: {e}")
                        return []

            tasks = []
            for doc_idx, units in enumerate(doc_units):
                for attr_key, prop in property_items:
                    for sentence, passage in units["sentences"]:
                        tasks.append(_guarded(self._process_sentence(doc_idx, sentence, passage, attr_key, prop)))
                    if self.extract_tables:
                        for table_text in units["tables"]:
                            tasks.append(_guarded(self._process_table(doc_idx, table_text, attr_key, prop)))

            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        return [record for record_list in results for record in record_list]

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def fit(self, documents: list[str], titles: list[str]) -> list[dict]:
        """Run ChatExtract over the given documents.

        Like the NuExtract baseline, this deliberately skips `_standardize` (an
        extra MeasurementLM-specific LLM pass) and applies only the programmatic
        `_deduplicate` for a fair, non-conflating cleanup of duplicate mentions.
        """
        if len(titles) != len(documents):
            raise ValueError("titles must be the same length as documents")
        self.data = [{"document_id": i} for i in range(len(documents))]
        self.data = self._extract_records(documents, titles)
        self.data = self._deduplicate(self.data)
        return self.data
