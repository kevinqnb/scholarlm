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

from pydantic import BaseModel

from .measurementlm import MeasurementLM, response_validator
from .utils.sentences import split_sentences

# ---------------------------------------------------------------------------
# OCR tag regexes (canonical copies live in utils/page_attribution.py).
# ---------------------------------------------------------------------------
_PAGE_RE = re.compile(r'<page number="(\d+)">(.*?)</page>', re.DOTALL)
_TABLE_RE = re.compile(r'<table number="(\d+)">.*?</table>', re.DOTALL)

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

# value, unit, material — asked in this order. ``{noun}``/``{Noun}`` are the
# dataset-supplied replacement for the reference script's materials-science
# "material"/"compound" wording (see DatasetConfig.chatextract_entity_noun);
# ``{Noun}`` is the title-cased form used in table-column headers.
SINGLE_Q = [
    'Give the number only without units, do not use a full sentence. If the value is not present in the text, type "None". What is the value of the {prop} in the following text?\n\n',
    'Give the unit only, do not use a full sentence. If the unit is not present in the text, type "None". What is the unit of the {prop} in the following text?\n\n',
    'Give the name of the {noun} only, do not use a full sentence. If the name of the {noun} is not present in the text, type "None". What is the {noun} for which the {prop} is given in the following text?\n\n',
]
# Redundant follow-up verification (two-part; the extracted answer is inserted
# between the parts). The reference script defines these but only ever runs the
# multi-valued verifications (ChatExtract.py:184-203); the single-valued branch
# is gated solely by a literal "none" check with no verification call. We match
# that by default; ``include_single_verification`` optionally enables single-
# branch verification as an explicit, non-faithful ablation.
SINGLE_FOLLOWUP_Q = [
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the value of the {prop} for the {noun} in the following text?\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the unit of the value of {prop} in the following text?\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is "', '" the {noun} for which the value of {prop} is given in the following text? Make sure it is a real {noun}.\n\n'],
]

TAB_Q = 'Use only data present in the text. If data is not present in the text, type "None". Summarize the values of {prop} in the following text in a form of a table consisting of: {Noun}, Value, Unit\n\n'
# Three-part templates: cell value, ordinal, and passage are spliced in.
TAB_FOLLOWUP_Q = [
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is "', '" the ', ' {noun} for which the value of {prop} is given in the following text? Make sure it is a real {noun}.\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the value of the {prop} for the ', ' {noun} in the following text?\n\n'],
    ['There is a possibility that the data you extracted is incorrect. Answer "Yes" or "No" only. Be very strict. Is ', ' the unit of the ', ' value of {prop} in the following text?\n\n'],
]

# Real-document-table workflow (paper Sec. II.C; not in the reference script).
TABLE_CLASSIFY_Q = 'Answer "Yes" or "No" only. Does the following table contain values of {prop}?\n\n'
TABLE_EXTRACT_Q = (
    'Use only data given in the table and its caption. Extract the {noun} names and '
    'values of {prop} from the following table. Present these values as a JSON object '
    'with a "rows" list, where each item has "material", "value", and "unit" fields.\n\n'
)

# Appended (not merged into TAB_Q, which is part of the verbatim-copied block
# above) so TAB_Q's own text stays exactly what ChatExtract.py asked for.
# `_TABLE_RESPONSE_FORMAT` (below) already forces valid JSON regardless of
# wording, but every other JSON-structured step in this codebase also states
# the expected shape explicitly (see instruction_prompts.py) -- match that
# convention rather than relying on response_format silently.
_TABLE_JSON_INSTRUCTION = (
    'Structure your response as a JSON object with a "rows" list, where each '
    'item has "material", "value", and "unit" fields.\n\n'
)

# Token budgets by call type (ChatExtract.py: 6 for yes/no, 500 otherwise). We
# use a slightly larger yes/no budget to tolerate models that prefix a word.
_YN_TOKENS = 12
_EXTRACT_TOKENS = 512

# ---------------------------------------------------------------------------
# Structured table-extraction schema. TAB_Q/TABLE_EXTRACT_Q still ask (in the
# paper's own wording) for a "table"; guided decoding is what actually parses
# the answer, replacing a hand-rolled CSV/pipe split that broke on any comma
# inside a cell (e.g. "1,000" parsed as two cells) and assumed a header row
# was always present.
# ---------------------------------------------------------------------------


class _ChatExtractTableRow(BaseModel):
    material: str
    value: str
    unit: str


class _ChatExtractTableResponse(BaseModel):
    rows: list[_ChatExtractTableRow]


_TABLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "chatextract_table",
        "schema": _ChatExtractTableResponse.model_json_schema(),
    },
}


def _is_yes(text: str) -> bool:
    """True iff `text`'s first word is "yes" -- anchored so "eyes" or "Yesterday"
    don't false-match the way a bare substring check would."""
    m = re.match(r"\s*([A-Za-z]+)", text)
    return bool(m) and m.group(1).lower() == "yes"


def _is_no(text: str) -> bool:
    """True iff `text`'s first word is "no" -- anchored so "know", "not", or
    "none" don't false-match the way a bare substring check would."""
    m = re.match(r"\s*([A-Za-z]+)", text)
    return bool(m) and m.group(1).lower() == "no"


class MeasurementLMChatExtract(MeasurementLM):
    """Conversational sentence-by-sentence extraction baseline (ChatExtract)."""

    def __init__(
        self,
        *args,
        attribute_property_names: dict[str, str] | None = None,
        entity_noun: str | None = None,
        include_single_verification: bool = False,
        extract_tables: bool = True,
        max_concurrent: int = 32,
        **kwargs,
    ):
        # ChatExtract reads OCR text directly; never run the image-based table
        # cleaning pass (mirrors the other baselines).
        kwargs.setdefault("clean_tables", False)
        super().__init__(*args, max_concurrent=max_concurrent, **kwargs)
        self.attribute_property_names = attribute_property_names or {}
        # Replaces the reference script's materials-science "material"/"compound"
        # wording throughout the prompt templates -- see
        # DatasetConfig.chatextract_entity_noun. "material" is the fallback
        # (collapsing the original material/compound split onto one word) so an
        # unset dataset still gets a grammatical, if paper-flavored, default.
        self.entity_noun = entity_noun or "material"
        self.include_single_verification = include_single_verification
        self.extract_tables = extract_tables
        # One entry per work item (sentence or table conversation) whose
        # coroutine raised -- populated by `_extract_records`, read by the
        # runner to write a count (and the individual errors) into
        # run_metadata.json rather than letting them vanish into stdout.
        self.failures: list[dict] = []

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

        Returns ``{"sentences": [(sentence, passage, page_num), ...],
        "tables": [(table_text, page_num), ...]}``.  Prose is gathered per page
        (in page order), tables are pulled out and handled separately, and each
        candidate sentence's ``passage`` is built as ``title + preceding
        sentence + target sentence``.  Sentences with no digit are dropped up
        front (values are always numeric).  ``page_num`` is the 0-indexed OCR
        ``<page number="N">`` the unit came from (``None`` if the text is
        untagged), and is carried through to each record's ``page_number`` so the
        judge can limit its context to the source page.
        """
        title = (title or "").strip()

        # (page_num, sentence) pairs so a sentence's page survives the
        # preceding-sentence lookup below (prev may live on an earlier page).
        prose_sentences: list[tuple[int | None, str]] = []
        tables: list[tuple[str, int | None]] = []

        page_matches = list(_PAGE_RE.finditer(context))
        # Fall back to treating the whole context as one untagged page.
        page_bodies: list[tuple[int | None, str]] = (
            [(int(m.group(1)), m.group(2)) for m in page_matches]
            if page_matches else [(None, context)]
        )

        for page_num, body in page_bodies:
            for tm in _TABLE_RE.finditer(body):
                # Cleaned OCR tags a <caption> element inside the table block;
                # raw OCR has none. Either way the table is passed through as
                # OCR'd -- no caption is guessed from surrounding text.
                tables.append((tm.group(0), page_num))
            # Prose = page body with table blocks removed, then sentence-split.
            prose = _TABLE_RE.sub(" ", body)
            prose_sentences.extend((page_num, s) for s in split_sentences(prose))

        sentences: list[tuple[str, str, int | None]] = []
        for i, (page_num, sentence) in enumerate(prose_sentences):
            if not re.search(r"\d", sentence):  # digit pre-filter
                continue
            prev = prose_sentences[i - 1][1] if i > 0 else ""
            passage = self._build_passage(title, prev, sentence)
            sentences.append((sentence, passage, page_num))

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

    # -----------------------------------------------------------------------
    # Conversation helpers (stateful; reuse _acall)
    # -----------------------------------------------------------------------

    async def _acall_retry(
        self, messages, max_tokens, max_retries: int = 3, response_format: dict | None = None
    ) -> str:
        """`_acall` at temperature 0 with a light retry on empty responses."""
        answer = ""
        for attempt in range(max_retries):
            answer = await self._acall(
                messages, response_format=response_format, temperature=0.0, max_tokens=max_tokens
            )
            if answer:
                return answer
            await asyncio.sleep(2 ** attempt)
        return answer

    async def _ask(
        self, messages: list[dict], question: str, yes_no: bool, response_format: dict | None = None
    ) -> str:
        """Append a user turn, call the model, append the assistant turn, return it."""
        messages.append({"role": "user", "content": question})
        answer = await self._acall_retry(
            messages, max_tokens=_YN_TOKENS if yes_no else _EXTRACT_TOKENS, response_format=response_format
        )
        messages.append({"role": "assistant", "content": answer})
        return answer

    # -----------------------------------------------------------------------
    # Per-sentence conversation (single/multi branching)
    # -----------------------------------------------------------------------

    async def _process_sentence(self, doc_idx, sentence, passage, attr_key, prop, page_num) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": ""}]

        # Stage A: classify the bare sentence.
        ans = await self._ask(messages, CLASSIF_Q.format(prop=prop) + sentence, yes_no=True)
        if not _is_yes(ans):
            return []

        # Stage B gate: single vs. multiple values (on the passage).
        ans = await self._ask(messages, IFMULTI_Q.format(prop=prop) + passage, yes_no=True)
        if _is_no(ans):
            return await self._extract_single(messages, doc_idx, passage, attr_key, prop, page_num)
        if _is_yes(ans):
            return await self._extract_multi(messages, doc_idx, passage, attr_key, prop, page_num)
        return []  # ambiguous gate → extract nothing (matches reference behavior)

    async def _extract_single(self, messages, doc_idx, passage, attr_key, prop, page_num) -> list[dict]:
        """Ask value → unit → material, each with an optional strict verification."""
        fields = ["value", "unit", "material"]
        extracted: dict[str, str] = {}
        valid: dict[str, bool] = {}

        for idx, field in enumerate(fields):
            ans = (await self._ask(messages, SINGLE_Q[idx].format(prop=prop, noun=self.entity_noun) + passage, yes_no=False)).strip()
            extracted[field] = ans
            ok = bool(ans) and "none" not in ans.lower()
            if ok and self.include_single_verification:
                pre, post = SINGLE_FOLLOWUP_Q[idx]
                followup = pre + ans + post.format(prop=prop, noun=self.entity_noun) + passage
                verdict = await self._ask(messages, followup, yes_no=True)
                if _is_no(verdict):
                    ok = False
            valid[field] = ok

        if not valid["value"]:
            return []
        material = extracted["material"] if valid["material"] else None
        units = extracted["unit"] if valid["unit"] else None
        return [self._make_record(doc_idx, attr_key, material, extracted["value"], units, page_num)]

    async def _extract_multi(self, messages, doc_idx, passage, attr_key, prop, page_num) -> list[dict]:
        """Ask for a Material/Value/Unit table, then verify each cell strictly."""
        table_text = await self._ask(
            messages,
            TAB_Q.format(prop=prop, Noun=self.entity_noun.title()) + _TABLE_JSON_INSTRUCTION + passage,
            yes_no=False, response_format=_TABLE_RESPONSE_FORMAT,
        )
        rows = self._parse_table_response(table_text)

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
                followup = p0 + cell + p1 + ordinal + p2.format(prop=prop, noun=self.entity_noun) + passage
                verdict = await self._ask(messages, followup, yes_no=True)
                if _is_no(verdict):
                    valid[col] = False
                    row_ok = False
                else:
                    valid[col] = True

            if valid.get("material") and valid.get("value"):
                units = cells["unit"].strip() if valid.get("unit") else None
                records.append(
                    self._make_record(doc_idx, attr_key, cells["material"].strip(), cells["value"].strip(), units, page_num)
                )
        return records

    # -----------------------------------------------------------------------
    # Per-table conversation (real document tables; no verification)
    # -----------------------------------------------------------------------

    async def _process_table(self, doc_idx, table_text, attr_key, prop, page_num) -> list[dict]:
        messages: list[dict] = [{"role": "system", "content": ""}]

        ans = await self._ask(messages, TABLE_CLASSIFY_Q.format(prop=prop) + table_text, yes_no=True)
        if not _is_yes(ans):
            return []

        extracted = await self._ask(
            messages,
            TABLE_EXTRACT_Q.format(prop=prop, noun=self.entity_noun, Noun=self.entity_noun.title()) + table_text,
            yes_no=False,
            response_format=_TABLE_RESPONSE_FORMAT,
        )
        rows = self._parse_table_response(extracted)

        records: list[dict] = []
        for material, value, unit in rows:
            value = str(value).strip()
            if not value or "none" in value.lower():
                continue
            material = None if (not material.strip() or "none" in material.lower()) else material.strip()
            units = None if (not unit.strip() or "none" in unit.lower()) else unit.strip()
            records.append(self._make_record(doc_idx, attr_key, material, value, units, page_num))
        return records

    # -----------------------------------------------------------------------
    # Table parsing
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_table_response(text: str) -> list[tuple[str, str, str]]:
        """Parse a guided-JSON table response into (material, value, unit) rows.

        ``response_format=_TABLE_RESPONSE_FORMAT`` constrains the model's answer
        to this schema, so parsing is exact regardless of a comma or pipe inside
        a cell (e.g. ``"1,000"``) or a missing header row -- both broke the
        CSV/pipe-splitting this replaced. Raises if the model's response doesn't
        validate against the schema; the caller's `_guarded` wrapper counts that
        as a failed work item rather than silently returning no rows.
        """
        parsed = response_validator(_ChatExtractTableResponse, text)
        return [(row["material"], row["value"], row["unit"]) for row in parsed["rows"]]

    # -----------------------------------------------------------------------
    # Record construction
    # -----------------------------------------------------------------------

    @staticmethod
    def _slug(material: str | None) -> str:
        if not material:
            return "none"
        slug = re.sub(r"[^a-z0-9]+", "_", material.lower()).strip("_")
        return slug or "none"

    def _make_record(self, doc_idx: int, attribute: str, material: str | None, value: str, units: str | None, page_num: int | None) -> dict:
        """Build one extraction record in the standard flat schema.

        ``entity_id`` keys on the document + normalized material, matching the
        rest of the record schema's ID scheme -- `fit()` skips `_deduplicate`
        (see its docstring), so distinct mentions of the same material+attribute+
        value stay as separate records, one per (sentence or table row).
        """
        item = {
            "name": material,
            "identifiers": None,
            "location": None,
            "ecosystem": None,   # pond fuzzy-match field (unpopulated by ChatExtract)
            "site_type": None,   # nfix fuzzy-match field (unpopulated by ChatExtract)
            "property": None,    # measeval fuzzy-match field (unpopulated by ChatExtract)
            # measeval entity field under its quantity-first design (column
            # parity only -- ChatExtract never enumerates quantities; see the
            # ChatExtract note in experiments/dataset-configs/measeval.py).
            "quantity": None,
            "date": None,
            "additional_details": None,
            "attribute": attribute,
            "value": value,
            "units": units,
        }
        entity_id = f"doc_{doc_idx}_{attribute}_{self._slug(material)}"
        return {"document_id": doc_idx} | item | {
            "entity_id": entity_id, "attribute_terms": [], "page_number": page_num,
        }

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

        self.failures = []

        async def _run():
            sem = asyncio.Semaphore(self.max_concurrent)

            async def _guarded(label, coro):
                async with sem:
                    try:
                        return await coro
                    except Exception as e:  # never let one conversation sink the run
                        self.failures.append({"item": label, "error": f"{type(e).__name__}: {e}"})
                        return []

            tasks = []
            for doc_idx, units in enumerate(doc_units):
                for attr_key, prop in property_items:
                    for sentence, passage, page_num in units["sentences"]:
                        label = f"doc={doc_idx} attr={attr_key} kind=sentence page={page_num}"
                        tasks.append(_guarded(
                            label, self._process_sentence(doc_idx, sentence, passage, attr_key, prop, page_num)
                        ))
                    if self.extract_tables:
                        for table_idx, (table_text, page_num) in enumerate(units["tables"]):
                            label = f"doc={doc_idx} attr={attr_key} kind=table table_idx={table_idx} page={page_num}"
                            tasks.append(_guarded(
                                label, self._process_table(doc_idx, table_text, attr_key, prop, page_num)
                            ))

            return await asyncio.gather(*tasks)

        results = asyncio.run(_run())
        if self.failures:
            print(f"ChatExtract: {len(self.failures)} of {len(results)} work items failed (see self.failures):")
            for failure in self.failures:
                print(f"  {failure['item']}: {failure['error']}")
        return [record for record_list in results for record in record_list]

    # -----------------------------------------------------------------------
    # Full pipeline
    # -----------------------------------------------------------------------

    def fit(self, documents: list[str], titles: list[str]) -> list[dict]:
        """Run ChatExtract over the given documents.

        Like the NuExtract3 and LangExtract baselines, this skips both
        `_standardize()` and `_deduplicate()`: both are part of the actual
        MeasurementLM pipeline's own contribution, not something a reimplemented
        reference method should be credited with.
        """
        if len(titles) != len(documents):
            raise ValueError("titles must be the same length as documents")
        self.data = self._extract_records(documents, titles)
        return self.data
