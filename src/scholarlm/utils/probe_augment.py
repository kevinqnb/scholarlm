"""
Dataset-agnostic machinery for synthetic probe-dataset augmentation.

Used by ``data/{pond,nfix,supermat}/create_probe_dataset.py`` when run with the
``--augment`` family of flags.  The per-dataset scripts own their dataset-specific
rules (entity fields, shared-unit attribute groups, fabricated-name lists,
supermat's "never touch the formula" rule); this module owns everything that is
the same across datasets:

  * ``AugmentCache``        — content-hash cache of gpt-oss responses so that two
                             regenerations at the same seed are byte-identical
                             (the LLM is only ever called on a cache miss).
  * ``GptOssClient``        — batched async OpenAI-compatible client for a locally
                             served ``openai/gpt-oss-120b``.
  * ``StubAugmentClient``   — deterministic canned edits, for unit tests and the
                             no-server smoke rung.
  * value / unit perturbation helpers for hard negatives.
  * ``apply_verified_edit`` — string replace that asserts the span it is told to
                             replace actually occurs (fail-loud).
  * ``balance_and_cap``     — per-source cap, exact 50/50 prevalence, target size.
  * ``inline_context_overrides`` / ``emit_context_diff_report`` — stamp the public
                             ``context_override`` field onto edited rows (read by
                             ``judge_common.prepare_chat_entries``) and the
                             spot-check diff report.

Nothing here reads a config or a path by itself — every input is passed in.
"""
from __future__ import annotations

import asyncio
import difflib
import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# ---------------------------------------------------------------------------
# Page extraction (kept byte-identical to experiments/judge_common.extract_page_text
# — copied here to avoid pulling judge_common's import chain into the generators)
# ---------------------------------------------------------------------------

_PAGE_BLOCK_RE = re.compile(r'<page number="(\d+)">.*?</page>', re.DOTALL)


def extract_page_text(document: str, page_numbers: list[int]) -> str:
    """Extract ``<page number="N">…</page>`` blocks for ``page_numbers``.

    Falls back to the full document when there are no page numbers, none are
    non-None, or no matching page block is found — identical to
    ``judge_common.extract_page_text`` so the augmentation pipeline sees exactly
    the context the judge would.
    """
    if not page_numbers:
        return document
    target = {pn for pn in page_numbers if pn is not None}
    if not target:
        return document
    parts = [m.group(0) for m in _PAGE_BLOCK_RE.finditer(document)
             if int(m.group(1)) in target]
    return "\n\n".join(parts) if parts else document


# ---------------------------------------------------------------------------
# gpt-oss response cache (reproducibility)
# ---------------------------------------------------------------------------


def _cache_key(op: str, payload: dict) -> str:
    """Stable content hash for a gpt-oss call.

    Keyed on the *semantic* inputs (operation + payload), never on row order or
    ``measurement_id``, so the cache survives dataset regeneration and the
    per-file ``measurement_id`` renumbering.

    The payload must carry everything that changes the model's expected output.
    In particular ``rewrite`` payloads carry ``protocol: 2`` (the diff protocol):
    an old-protocol full-page ``rewrite`` entry has a different key and is a
    permanent miss, so it can never be read back by the new ``edits`` parser.
    ``event_fill`` payloads are unaffected and keep hitting.
    """
    blob = json.dumps({"op": op, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AugmentCache:
    """JSON-backed ``{cache_key: response}`` store.

    ``response`` is whatever the client method returns (a dict for event-fill, a
    string for context rewrites).  Load/save are explicit; the client checks
    ``get`` before every call and ``put`` after every miss.
    """

    def __init__(self, path: Path | None):
        self.path = Path(path) if path is not None else None
        self._store: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0
        if self.path is not None and self.path.exists():
            with open(self.path) as f:
                self._store = json.load(f)

    def get(self, key: str) -> Any | None:
        if key in self._store:
            self._hits += 1
            return self._store[key]
        return None

    def put(self, key: str, value: Any) -> None:
        self._misses += 1
        self._store[key] = value

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self._store, f, indent=2, ensure_ascii=False, sort_keys=True)
        tmp.replace(self.path)

    @property
    def stats(self) -> str:
        return f"cache: {self._hits} hit / {self._misses} miss ({len(self._store)} entries)"


# ---------------------------------------------------------------------------
# Client protocol + implementations
# ---------------------------------------------------------------------------

_EVENT_FILL_SYS = (
    "You are a careful scientific data annotator. You are given a page of text "
    "from a scientific paper and one measurement extracted from it. Your job is "
    "to report ONLY the measurement-event fields that the page text explicitly "
    "states for THIS specific measurement. Never guess. If the page does not "
    "state a field for this measurement, return null for it. Respond with a "
    "single JSON object and nothing else."
)

_REWRITE_SYS = (
    "You are a careful scientific text editor. You are given a page of text from "
    "a scientific paper and asked to change one specific claim in it while "
    "changing as little else as possible. You do NOT rewrite or re-emit the "
    "page. Instead you return the minimal list of exact-text edits that make the "
    "change. Respond with a single JSON object and nothing else.\n"
    "If the change can be made, respond with:\n"
    '{\"feasible\": true, \"edits\": [{\"find\": \"<verbatim substring of the '
    'page>\", \"replace\": \"<what it becomes>\"}, ...]}\n'
    "Rules for every edit:\n"
    "- \"find\" MUST be copied character-for-character from the PAGE TEXT below "
    "(same whitespace, casing, LaTeX and markdown) so it can be located by an "
    "exact string search.\n"
    "- \"find\" SHOULD carry enough surrounding words to be unambiguous (e.g. "
    "\"a critical temperature of 5.2 K\", not \"5.2\"), so applying the edit "
    "cannot corrupt an unrelated occurrence (a year, a different measurement, a "
    "substring of a larger number).\n"
    "- Return one edit per site. If the edited claim appears in the running "
    "text, a table cell and a figure caption, that is three edits. Change every "
    "place the edited claim appears and nothing else.\n"
    "If making the change consistently would require rewriting large parts of "
    "the page, or would contradict other statements or tables on the page, "
    'respond with {\"feasible\": false, \"reason\": \"...\"} instead.'
)


class AugmentClient(Protocol):
    """Interface the generators depend on. Both real and stub satisfy it."""

    def event_fill(
        self, *, context: str, measurement: dict, event_prompt: str, event_fields: list[str]
    ) -> dict[str, str | None]: ...

    def rewrite_context(
        self, *, context: str, instruction: str, expected: list[tuple[str, str]],
    ) -> tuple[bool, str, list[tuple[str, str]], str]: ...

    def flush(self) -> None: ...


def _dump_bad_response(raw: str, op: str, dump_dir: Path | None) -> Path | None:
    """Write an unparseable gpt-oss response to ``dump_dir`` for post-mortem.

    Called only on a path that is about to raise -- this makes the failure
    inspectable, it does not soften it. The raw text is not recoverable
    otherwise: nothing upstream logs it and the response cache is flushed to
    disk only at end of run, so an exception here would take the response
    with it.
    """
    if dump_dir is None:
        return None
    dump_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    path = dump_dir / f"bad_response_{op}_{digest}.txt"
    path.write_text(raw, encoding="utf-8")
    return path


# gpt-oss copies `find` spans verbatim from the page, and OCR'd pages sometimes
# carry literal LaTeX-style math delimiters (`\( n = 3 \)`); the model reproduces
# the backslash unescaped, which is invalid JSON (`\(` is not a recognized escape)
# though the response is otherwise fine. This regex matches every backslash
# escape *as a single unit* (valid two-char escape, `\uXXXX`, or a lone
# backslash) so an already-correct `\\` pair is consumed whole and never
# miscounted, character by character, as two separate invalid backslashes --
# a bug the first version of this regex actually had.
_JSON_ESCAPE_UNIT_RE = re.compile(r'\\u[0-9a-fA-F]{4}|\\.|\\', re.DOTALL)


def _repair_invalid_escapes(text: str) -> str:
    """Double any backslash that isn't part of a valid JSON escape sequence.

    Last-resort repair, tried only after a normal parse has already failed
    (see ``_extract_json_object``). It targets one specific, understood
    JSON-encoding quirk (see the comment above), not generation errors in
    general: if the repaired text still doesn't parse, the caller dumps and
    raises. This narrows what counts as a failure, it doesn't remove the
    fail-loud path.
    """
    def _fix(m: re.Match) -> str:
        s = m.group(0)
        if s.startswith("\\u") or (len(s) == 2 and s[1] in '"\\/bfnrt'):
            return s  # already a valid escape -- leave untouched
        return "\\\\" + s[1:]  # lone or invalid backslash -- double it

    return _JSON_ESCAPE_UNIT_RE.sub(_fix, text)


def _strip_json_line_comments(text: str) -> str:
    """Delete ``// ...`` line comments that sit *outside* any JSON string.

    Last-resort repair, tried only after a normal parse has already failed
    (see ``_extract_json_object``), for one understood, verified quirk: gpt-oss
    sometimes annotates a ``rewrite`` response's ``edits`` array with JS-style
    ``// unchanged`` comments -- not legal JSON, though the response is
    otherwise well-formed (first seen on the old-protocol ``replacements``
    array, cached nfix key ``bf3746a2...``). The scan is
    string-aware, so a ``//`` *inside* a value (a ``http://`` URL echoed from
    OCR'd page text) is never touched. Like ``_repair_invalid_escapes``: if the
    stripped text still doesn't parse the caller dumps and raises -- this
    narrows what counts as a failure, it does not remove the fail-loud path.
    """
    out: list[str] = []
    i, n = 0, len(text)
    in_str = esc = False
    while i < n:
        c = text[i]
        if in_str:
            out.append(c)
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
        elif c == '"':
            in_str = True
            out.append(c)
            i += 1
        elif c == "/" and i + 1 < n and text[i + 1] == "/":
            while i < n and text[i] not in "\r\n":   # skip to end of line
                i += 1
        else:
            out.append(c)
            i += 1
    return "".join(out)


# Ordered last-resort repairs for a gpt-oss JSON response that failed a plain
# ``json.loads``. Each targets one understood, verified-against-real-output
# quirk (unescaped LaTeX backslashes copied from OCR'd page spans; JS-style
# ``//`` comments on the ``edits`` array); the third entry is for a response
# that has both. Cheapest/safest first. Kept as one list so the admission-time probe in
# ``GptOssClient._run_batch`` and the raise-time path in ``_extract_json_object``
# run the identical ladder.
_JSON_LAST_RESORT_REPAIRS: list[tuple[str, Callable[[str], str]]] = [
    ("repaired an invalid backslash escape", _repair_invalid_escapes),
    ("stripped one or more // line comments", _strip_json_line_comments),
    ("stripped // line comments and repaired an invalid backslash escape",
     lambda s: _repair_invalid_escapes(_strip_json_line_comments(s))),
]


def _try_json_object(text: str):
    """Best-effort parse of a gpt-oss JSON-*object* response, with repairs.

    Returns ``(obj, repair_note, detail)``:
      * ``(dict, None, None)``       — parsed as-is;
      * ``(dict, "<note>", <err>)``  — parsed only after a last-resort repair;
        ``<err>`` is the original ``JSONDecodeError`` (kept for the location it
        carries). The caller MUST surface ``<note>``: repaired text lands in the
        dataset and must never be rewritten silently;
      * ``(None, None, "not-json")``   — no ``{...}`` span to try;
      * ``(None, None, "not-object")`` — parsed, but the value is not an object;
      * ``(None, None, <JSONDecodeError>)`` — had a ``{...}`` span, nothing parsed.
    """
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
        return (obj, None, None) if isinstance(obj, dict) else (None, None, "not-object")
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, None, "not-json"
    trimmed = text[start : end + 1]
    try:
        obj = json.loads(trimmed)
        return (obj, None, None) if isinstance(obj, dict) else (None, None, "not-object")
    except json.JSONDecodeError as first:
        for note, fn in _JSON_LAST_RESORT_REPAIRS:
            try:
                obj = json.loads(fn(trimmed))
            except json.JSONDecodeError:
                continue
            return (obj, note, first) if isinstance(obj, dict) else (None, None, "not-object")
        return None, None, first


def _looks_like_json_object(text: str) -> bool:
    """True if ``text`` yields a JSON object as-is or after a last-resort repair.

    The admission gate in ``GptOssClient._run_batch`` uses this to keep a
    malformed response out of the cache, where it would otherwise be flushed as
    a "success" and only detonate hours later in the strict-cache real pass
    (nfix Rung-4, and pond incident #3 before it). Same parse ladder as
    ``_extract_json_object`` -- ``_try_json_object`` is the shared core.
    """
    obj, _note, _detail = _try_json_object(text)
    return obj is not None


def _extract_json_object(text: str, *, op: str, dump_dir: Path | None) -> dict:
    """Parse the first top-level JSON object in ``text``; fail loud otherwise.

    ``op`` and ``dump_dir`` are only used to label/save the raw response via
    ``_dump_bad_response`` when parsing fails -- they play no role in a
    successful parse.
    """
    obj, repair_note, detail = _try_json_object(text)
    if obj is not None:
        if repair_note:
            # a repair rewrites text that lands in the dataset -- never silent
            loc = (f" ({detail.msg} at char {detail.pos})"
                   if isinstance(detail, json.JSONDecodeError) else "")
            print(f"  probe_augment: {repair_note} in a gpt-oss {op} response{loc}")
        return obj

    path = _dump_bad_response(text, op, dump_dir)
    snippet = text.strip()[:400]
    if detail == "not-json":
        raise ValueError(f"gpt-oss {op} response is not JSON: {snippet!r}")
    if detail == "not-object":
        raise ValueError(f"gpt-oss {op} response is JSON but not an object: {snippet!r}")
    e = detail  # the original json.JSONDecodeError
    where = f"; full raw response written to {path}" if path else ""
    raise ValueError(
        f"gpt-oss {op} response is not valid JSON even after brace-trimming "
        f"({e.msg} at line {e.lineno} column {e.colno}, char {e.pos}){where}"
    ) from e


@dataclass
class GptOssClient:
    """Batched async client for a locally served ``openai/gpt-oss-120b``.

    Mirrors ``experiments/run_judge_local.py``'s ``AsyncOpenAI`` + ``Semaphore``
    pattern.  Every call is checked against ``cache`` first; the model is only
    hit on a miss.  ``event_fill`` / ``rewrite_context`` are synchronous from the
    caller's point of view — batching happens because the generator collects many
    calls and the OS event loop runs them concurrently via ``_run_batch``.  For
    simplicity (the generators are straight-line loops) each public method runs a
    one-shot event loop; call ``prewarm`` with a list of payloads to parallelise.
    """

    api_base: str
    cache: AugmentCache
    model_id: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    max_concurrent: int = 32
    # Both ops now emit a small JSON object (event-fill: a handful of fields;
    # rewrite under the diff protocol: a ~100-token `edits` list), so this cap is
    # generous headroom, not a real constraint. It was 20k when the rewrite op
    # re-emitted the whole page; kept oversized deliberately -- truncation at the
    # cap is exactly what the protocol change set out to eliminate.
    max_tokens: int = 20000
    # gpt-oss's harmony chat template reads `reasoning_effort` (low/medium/high,
    # default "medium" if omitted); it has no `enable_thinking` variable at all,
    # unlike Qwen3 (checked against the cached chat_template.jinja). "low" is set
    # deliberately so the whole cache is generated under one known regime. It also
    # predates the diff protocol: at the default effort, reasoning could compete
    # with the old full-page rewrite output for `max_tokens` and exhaust it before
    # the answer -- a plausible cause of the truncated-JSON and empty-message
    # crashes on the initial generation run. Far less likely now the output is
    # tiny, but the regime stays pinned so the cache is single-regime.
    reasoning_effort: str = "low"
    # Record mode: `_resolve` collects every (key, messages) it is asked for into
    # `_pending` and returns a benign canned response instead of calling the
    # model, so the orchestrator can gather the whole run's prompts in one RNG-
    # matched dry pass and `prewarm` them concurrently (see `run_and_write`).
    record: bool = False
    # Set True after `prewarm`: a cache miss on the real pass is then a hard
    # error rather than a silent fall-through to the serial one-shot path (which
    # would surface only as a blown job walltime).
    strict_cache: bool = False
    _pending: dict[str, list[dict]] = field(default_factory=dict, repr=False)

    @property
    def _bad_response_dir(self) -> Path | None:
        """Where to dump a response that fails JSON parsing, or None if the
        cache is path-less (e.g. in unit tests) and there's nowhere durable to
        put it."""
        return self.cache.path.parent / "bad_responses" if self.cache.path is not None else None

    # -- low level -----------------------------------------------------------

    async def _one(self, client, messages: list[dict]) -> str:
        resp = await client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            extra_body={"chat_template_kwargs": {"reasoning_effort": self.reasoning_effort}},
        )
        choice = resp.choices[0]
        content = choice.message.content
        if not content:
            # Diagnostic, not a fallback: still raises. `finish_reason="length"`
            # plus a non-trivial `reasoning_content` would confirm the
            # reasoning-exhausted-the-budget hypothesis (see `reasoning_effort`);
            # nothing upstream logs either field otherwise.
            reasoning = getattr(choice.message, "reasoning_content", None)
            raise ValueError(
                f"gpt-oss returned an empty message (finish_reason="
                f"{choice.finish_reason!r}, reasoning_content_len="
                f"{len(reasoning) if reasoning else 0})"
            )
        if choice.finish_reason == "length":
            # Non-empty but truncated at `max_tokens`: the tail of the JSON is
            # gone, so a downstream parse failure is guaranteed. Catch it here,
            # at the call site, with the diagnostic fields attached -- rather
            # than two layers down as an opaque `not valid JSON`. Still a hard
            # error, never a fallback. Under the diff protocol the output is
            # tiny, so this should never fire; if it does, something is wrong
            # upstream (a runaway repetition loop, a bad `max_tokens`).
            reasoning = getattr(choice.message, "reasoning_content", None)
            raise ValueError(
                f"gpt-oss response was truncated at max_tokens={self.max_tokens} "
                f"(finish_reason='length', content_len={len(content)}, "
                f"reasoning_content_len={len(reasoning) if reasoning else 0})"
            )
        return content

    async def _run_batch(self, jobs: list[tuple[str, list[dict]]]) -> dict[str, str]:
        """Run ``jobs`` concurrently, caching each success as it lands.

        ``asyncio.gather`` without ``return_exceptions=True`` would propagate
        the first failure and discard every sibling response already computed
        in the same batch -- for a ``prewarm`` batch, potentially thousands of
        GPU calls lost because one came back malformed. Every success is cached,
        and the cache saved if any job failed, before this still fails loud on
        the failures.

        A response that ``_one`` returns but that does not parse as a JSON
        object (even after ``_JSON_LAST_RESORT_REPAIRS``) is treated as a
        failure here and kept OUT of the cache: otherwise it is flushed as a
        "success" and only detonates hours later in the strict-cache real pass,
        with no way to regenerate just that key without a manual cache edit
        (nfix Rung-4; pond incident #3).

        The gate only checks JSON-*object*-ness, not the ``rewrite`` schema, so
        a response that is a well-formed object with a malformed ``edits`` array
        (not a list, an element missing ``find``/``replace``) still gets cached
        and then raises in ``rewrite_context`` on the real pass. Known residual,
        unchanged from the old protocol (which cached ``feasible: true`` objects
        with no ``context`` the same way); the diff protocol just has more such
        shapes. ``rewrite_context`` fails loud on all of them.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="EMPTY", base_url=self.api_base, timeout=600.0)
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _guarded(key: str, messages: list[dict]) -> tuple[str, str | BaseException]:
            async with sem:
                try:
                    return key, await self._one(client, messages)
                except Exception as e:
                    return key, e

        pairs = await asyncio.gather(*(_guarded(k, m) for k, m in jobs))
        results: dict[str, str] = {}
        failures: list[tuple[str, BaseException]] = []
        for key, value in pairs:
            if isinstance(value, BaseException):
                failures.append((key, value))
            elif not _looks_like_json_object(value):
                _dump_bad_response(value, "prewarm", self._bad_response_dir)
                failures.append((key, ValueError(
                    "gpt-oss response is not a JSON object even after last-resort "
                    f"repairs: {value.strip()[:200]!r}"
                )))
            else:
                results[key] = value
                self.cache.put(key, value)
        if failures:
            self.cache.save()
            raise RuntimeError(
                f"{len(failures)}/{len(jobs)} gpt-oss call(s) failed in this "
                f"batch; {len(results)} succeeded and are already cached. "
                f"First failure (key {failures[0][0][:12]}...): {failures[0][1]!r}"
            ) from failures[0][1]
        return results

    # Canned responses returned during a record pass — parseable by the callers
    # so control flow (and RNG consumption) matches a normal run up to the point
    # where the collected prompts are all that matters.
    _RECORD_RESPONSES = {
        "event_fill": "{}",                                  # -> every field null
        "rewrite": '{"feasible": false, "reason": "record pass"}',
    }

    def _resolve(self, key: str, op: str, messages: list[dict]) -> str:
        """Return raw model text for ``key``, using the cache.

        On a cache miss outside record mode this runs a one-shot event loop for a
        single call — fully serial.  A full generation run therefore does a
        record pass first (``record=True``) so every ``(key, messages)`` is
        collected and ``prewarm``-ed concurrently; the real pass then only ever
        hits the cache.  ``run_and_write`` wires this up.
        """
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        if self.record:
            self._pending[key] = messages
            if op not in self._RECORD_RESPONSES:
                raise ValueError(f"no canned record-pass response for op {op!r}")
            return self._RECORD_RESPONSES[op]
        if self.strict_cache:
            raise RuntimeError(
                f"gpt-oss cache miss for op {op!r} on the real pass, after "
                f"prewarm — the record pass did not collect this prompt. This "
                f"means an earlier phase's output changed a later phase's "
                f"prompt: today the only such dependency is event-fill results "
                f"feeding the pos_event rewrite instruction, so it fires only "
                f"with --augment-pos-axes pos_event. Run prewarm in two rounds "
                f"(event-fill, then the rest) or drop that axis."
            )
        return asyncio.run(self._run_batch([(key, messages)]))[key]

    def prewarm(self, jobs: list[tuple[str, list[dict]]]) -> None:
        """Fill the cache for a list of ``(cache_key, messages)`` in one batch.

        ``_run_batch`` caches each success itself, so nothing further to do
        here on the happy path.
        """
        misses = [(k, m) for k, m in jobs if self.cache.get(k) is None]
        if not misses:
            return
        asyncio.run(self._run_batch(misses))

    def flush(self) -> None:
        self.cache.save()

    # -- public ops --------------------------------------------------------

    def _event_fill_messages(self, context: str, measurement: dict, event_prompt: str,
                             event_fields: list[str]) -> list[dict]:
        user = (
            f"{event_prompt}\n\n"
            f"Report only these fields: {event_fields}. "
            f"Return a JSON object with exactly those keys; use null where the "
            f"page does not state the field for this measurement.\n\n"
            f"## MEASUREMENT\n{json.dumps(measurement, ensure_ascii=False)}\n\n"
            f"## PAGE TEXT\n{context}"
        )
        return [{"role": "system", "content": _EVENT_FILL_SYS},
                {"role": "user", "content": user}]

    def event_fill(self, *, context, measurement, event_prompt, event_fields):
        payload = {"context": context, "measurement": measurement,
                   "event_prompt": event_prompt, "event_fields": sorted(event_fields)}
        key = _cache_key("event_fill", payload)
        raw = self._resolve(key, "event_fill",
                            self._event_fill_messages(context, measurement, event_prompt, event_fields))
        obj = _extract_json_object(raw, op="event_fill", dump_dir=self._bad_response_dir)
        out: dict[str, str | None] = {}
        for f in event_fields:
            v = obj.get(f)
            out[f] = v if (isinstance(v, str) and v.strip()) else None
        return out

    def _rewrite_messages(self, context: str, instruction: str) -> list[dict]:
        user = (
            f"{instruction}\n\n"
            f"Return only the minimal exact-text edits needed. Do not alter any "
            f"other fact, number, unit, name, table cell, or sentence.\n\n"
            f"## PAGE TEXT\n{context}"
        )
        return [{"role": "system", "content": _REWRITE_SYS},
                {"role": "user", "content": user}]

    def rewrite_context(self, *, context, instruction, expected):
        """Diff-protocol context edit: ask the model only for the edits, apply
        them ourselves, verify.

        The model returns ``{"feasible": true, "edits": [{"find": ..., "replace":
        ...}]}`` where each ``find`` is a verbatim substring of ``context``. We
        apply the edits by exact string replacement (``count=1`` per edit; the
        prompt asks for one edit per site) and run the ``expected``-span sanity
        check against the locally-patched text. Returns
        ``(applied, new_context, edits, reason)``:

          * ``(True, new_ctx, [(find, replace), ...], "")`` on success;
          * ``(False, context, [], reason)`` when the model says infeasible, when
            a proposed ``find`` span is not in the context (same category as
            infeasible -- skip, don't force, per build note §4.1), or when the
            ``expected`` check fails.

        Fails loud (``ValueError``) only on a schema violation: ``feasible: true``
        with no usable ``edits`` list.
        """
        payload = {"context": context, "instruction": instruction, "protocol": 2}
        key = _cache_key("rewrite", payload)
        raw = self._resolve(key, "rewrite", self._rewrite_messages(context, instruction))
        obj = _extract_json_object(raw, op="rewrite", dump_dir=self._bad_response_dir)
        if not obj.get("feasible", False):
            return False, context, [], str(obj.get("reason", "infeasible"))
        edits_raw = obj.get("edits")
        if not isinstance(edits_raw, list):
            raise ValueError(
                f"gpt-oss rewrite marked feasible but 'edits' is not a list: {raw[:400]!r}"
            )
        if not edits_raw:
            raise ValueError(
                f"gpt-oss rewrite marked feasible but proposed no edits: {raw[:400]!r}"
            )
        edits: list[tuple[str, str]] = []
        for e in edits_raw:
            if not (isinstance(e, dict)
                    and isinstance(e.get("find"), str)
                    and isinstance(e.get("replace"), str)):
                raise ValueError(
                    f"gpt-oss rewrite edit is not a {{'find': str, 'replace': str}} "
                    f"object: {e!r}"
                )
            edits.append((e["find"], e["replace"]))
        # Apply the model's edits to the ORIGINAL context. `apply_verified_edit`
        # checks each `find` against the running (progressively-patched) text and
        # raises if it is absent -- which also covers the case where an earlier
        # edit's replacement destroyed a later edit's `find` span. A missing span
        # is "the model proposed something that doesn't apply": same category as
        # an infeasible rewrite, so we skip rather than force.
        try:
            new_ctx = apply_verified_edit(context, edits)
        except ValueError as exc:
            return False, context, [], f"rewrite edit not applicable: {exc}"
        # Sanity-check the model did (roughly) what was asked, against the
        # locally-patched text: every expected old-span gone, every expected
        # new-span present. (The `expected` old-span is the ground-truth
        # measurement string, which is often not a verbatim page substring at
        # all -- in that case the first check passes vacuously and this reduces
        # to "did the intended new value get introduced somewhere".)
        for old, new in expected:
            if old and old in new_ctx and old != new:
                return False, context, [], f"rewrite left the original span in place: {old!r}"
            if new and new not in new_ctx:
                return False, context, [], f"rewrite did not introduce the target span: {new!r}"
        return True, new_ctx, edits, ""


class StubAugmentClient:
    """Deterministic canned client for tests and the no-server smoke rung.

    ``event_fill`` returns null for every field (no event claimed).
    ``rewrite_context`` applies the ``expected`` ``(old, new)`` replacements
    directly, marking infeasible when an ``old`` span is absent — no LLM, no
    prompt parsing.  It is deliberately *stricter* than the real client: it
    patches the bare ``expected`` span, where the real client applies
    model-chosen multi-word spans and only uses ``expected`` for the final
    sanity check.  So on real OCR (where the GT measurement string is usually
    not a verbatim page substring) the stub skips almost everything — fine for
    plumbing/determinism tests, not representative of real yield.
    """

    def __init__(self, cache: AugmentCache | None = None):
        self.cache = cache or AugmentCache(None)

    def event_fill(self, *, context, measurement, event_prompt, event_fields):
        return {f: None for f in event_fields}

    def rewrite_context(self, *, context, instruction, expected):
        if not expected:
            return False, context, [], "stub: nothing to replace"
        new_ctx = context
        applied: list[tuple[str, str]] = []
        for old, new in expected:
            if old and old not in new_ctx:
                return False, context, [], f"stub: span {old!r} not in context"
            if old:
                new_ctx = new_ctx.replace(old, new, 1)
            applied.append((old, new))
        return True, new_ctx, applied, ""

    def flush(self):
        pass


# ---------------------------------------------------------------------------
# Edit application (fail-loud)
# ---------------------------------------------------------------------------


def apply_verified_edit(context: str, replacements: list[tuple[str, str]]) -> str:
    """Apply ``(old, new)`` replacements in order, asserting each ``old`` occurs
    in the *running* (progressively-patched) text first.

    Each replacement is ``count=1`` (first occurrence); the diff-protocol prompt
    asks the model for one edit per site, so a claim that recurs comes back as
    one edit per occurrence. Checking against the running text (not the original)
    also catches an earlier edit's replacement having destroyed a later edit's
    ``old`` span.

    Raises ``ValueError`` if a span is not found. Whether that is fatal or a
    skip is the caller's call: ``GptOssClient.rewrite_context`` catches it and
    skips (a model-proposed span that doesn't apply is the infeasible category);
    a caller passing spans it computed itself should let it crash.
    """
    out = context
    for old, new in replacements:
        if old == new:
            continue
        if old not in out:
            raise ValueError(f"replacement span not found in context: {old!r}")
        out = out.replace(old, new, 1)
    return out


# ---------------------------------------------------------------------------
# Value / unit perturbation (hard negatives)
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"^[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?$")


def _decimals(s: str) -> int:
    return len(s.split(".", 1)[1]) if "." in s else 0


def perturb_value(value_str: str, rng: random.Random, *, lo: float = 0.05, hi: float = 0.25) -> str | None:
    """Return a numerically-close but wrong value string, or ``None`` if non-numeric.

    The relative offset is drawn from ``±U(lo, hi)``; decimal precision of the
    original is preserved; the returned string is guaranteed to differ from the
    input.
    """
    s = value_str.strip()
    if not _NUM_RE.match(s):
        return None
    val = float(s)
    ndec = _decimals(s)
    for _ in range(50):
        frac = rng.uniform(lo, hi) * (1 if rng.random() < 0.5 else -1)
        cand = val * (1 + frac)
        if ndec:
            out = f"{cand:.{ndec}f}"
        else:
            out = str(int(round(cand)))
        if out != s and _NUM_RE.match(out) and float(out) != val:
            return out
    return None


def alt_unit(current: str | None, units: list[str], rng: random.Random,
             *, family_bias: float = 0.7) -> str | None:
    """Pick a different unit from ``units`` for a hard-negative unit error.

    With probability ``family_bias`` prefer a unit that shares a leading
    metric-prefix / measure token with ``current`` (e.g. µg/L → mg/L, m → cm),
    which is a harder error than an out-of-family swap.
    """
    cands = [u for u in units if current is None or u.lower() != current.lower()]
    if not cands:
        return None
    if current is not None and rng.random() < family_bias:
        cur_tokens = set(re.findall(r"[A-Za-zµμ]+", current.lower()))
        near = [u for u in cands if set(re.findall(r"[A-Za-zµμ]+", u.lower())) & cur_tokens]
        if near:
            return rng.choice(sorted(near))
    return rng.choice(sorted(cands))


# ---------------------------------------------------------------------------
# source_group_id + measurement_id bookkeeping
# ---------------------------------------------------------------------------


def stamp_source_group_id(rows: list[dict], *, key: str = "gt_row_index") -> None:
    """Set ``row['source_group_id']`` = the originating GT row index.

    Every augmented row must carry this so CV / analysis can group near-duplicate
    derivatives.  For a GT-derived row it equals ``gt_row_index``; a row already
    carrying ``source_group_id`` (a derivative of a derivative) is left alone.
    """
    for r in rows:
        r.setdefault("source_group_id", r[key])


def assign_measurement_ids(rows: list[dict], *, start: int = 0) -> None:
    """Assign a contiguous per-file ``measurement_id`` (positional, like the
    baseline generator's ``enumerate(combined)``)."""
    for i, r in enumerate(rows, start=start):
        r["measurement_id"] = i


# ---------------------------------------------------------------------------
# Balancing + per-source cap + target size
# ---------------------------------------------------------------------------


@dataclass
class BalanceReport:
    n_positive: int
    n_negative: int
    n_capped_dropped: int
    target: int
    floor: int
    hit_target: bool
    hit_floor: bool

    def __str__(self) -> str:
        return (
            f"balanced: {self.n_positive} pos / {self.n_negative} neg "
            f"= {self.n_positive + self.n_negative} rows "
            f"({self.n_capped_dropped} dropped by per-source cap); "
            f"target {self.target} {'MET' if self.hit_target else 'not met'}, "
            f"floor {self.floor} {'MET' if self.hit_floor else 'NOT MET'}"
        )


def balance_and_cap(
    positives: list[dict],
    negatives: list[dict],
    *,
    target: int,
    floor: int,
    max_derived_per_source: int,
    rng: random.Random,
) -> tuple[list[dict], BalanceReport]:
    """Enforce the per-source cap, then trim to exact 50/50 and to ``target``.

    ``positives`` / ``negatives`` each carry ``source_group_id``.  The per-source
    cap counts *derived* rows only — a row flagged ``_is_base`` (an original GT
    valid kept verbatim) is never dropped by the cap.  After capping, the
    majority class is randomly down-sampled to match the minority, then both
    classes are trimmed evenly so the total does not exceed ``target``.  Falling
    below ``floor`` is reported, not raised (the caller decides — nfix is
    allowed to).
    """

    def _cap(rows: list[dict]) -> tuple[list[dict], int]:
        by_src: dict[Any, list[dict]] = {}
        for r in rows:  # insertion order is deterministic given deterministic input
            by_src.setdefault(r["source_group_id"], []).append(r)
        kept: list[dict] = []
        dropped = 0
        for sid in by_src:
            group = by_src[sid]
            base = [r for r in group if r.get("_is_base")]
            derived = [r for r in group if not r.get("_is_base")]
            rng.shuffle(derived)
            keep_derived = derived[:max_derived_per_source]
            dropped += len(derived) - len(keep_derived)
            kept.extend(base + keep_derived)
        return kept, dropped

    pos, d_pos = _cap(positives)
    neg, d_neg = _cap(negatives)

    n = min(len(pos), len(neg))
    # trim to target (must stay even between classes)
    per_class_cap = target // 2
    n = min(n, per_class_cap) if target else n

    rng.shuffle(pos)
    rng.shuffle(neg)
    pos, neg = pos[:n], neg[:n]

    out = pos + neg
    rng.shuffle(out)

    report = BalanceReport(
        n_positive=len(pos),
        n_negative=len(neg),
        n_capped_dropped=d_pos + d_neg,
        target=target,
        floor=floor,
        hit_target=(len(out) >= target) if target else True,
        hit_floor=len(out) >= floor,
    )
    return out, report


# ---------------------------------------------------------------------------
# Side-car assembly + diff report
# ---------------------------------------------------------------------------

_CTX_ORIG_KEY = "_context_original"
_CTX_EDIT_KEY = "_context_override"
_CTX_PUBLIC_KEY = "context_override"   # kept on the output row (see strip_internal_fields)


def inline_context_overrides(rows: list[dict]) -> int:
    """Stamp the public ``context_override`` field onto rows whose context was
    genuinely edited (in place). Returns the count stamped.

    ``judge_common.prepare_chat_entries`` reads this field directly off the row
    — no side-car, no measurement_id-keyed lookup. A no-op edit (edited text
    identical to the original) is not stamped, same filter the retired
    side-car builder used.
    """
    n = 0
    for r in rows:
        edited = r.get(_CTX_EDIT_KEY)
        if edited is not None and edited != r.get(_CTX_ORIG_KEY):
            r[_CTX_PUBLIC_KEY] = edited
            n += 1
    return n


def emit_context_diff_report(rows: list[dict], path: Path) -> int:
    """Write a unified diff (original → edited context) per edited row. Returns count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for r in rows:
            orig = r.get(_CTX_ORIG_KEY)
            edit = r.get(_CTX_EDIT_KEY)
            if orig is None or edit is None or orig == edit:
                continue
            n += 1
            f.write(f"### measurement_id={r['measurement_id']} "
                    f"source_group_id={r['source_group_id']} "
                    f"axis={r.get('augment_axis')} mod={r.get('modification_type')}\n")
            diff = difflib.unified_diff(
                orig.splitlines(keepends=False), edit.splitlines(keepends=False),
                fromfile="original", tofile="edited", lineterm="", n=1,
            )
            f.write("\n".join(diff) + "\n\n")
    return n


def strip_internal_fields(rows: list[dict], gt_cols: list[str], extra_keep: list[str]) -> list[dict]:
    """Project rows onto the output schema (GT cols + probe bookkeeping), dropping
    every ``_``-prefixed internal field (contexts, page-number lists, …)."""
    keep = list(gt_cols) + list(extra_keep)
    return [{k: r[k] for k in keep if k in r} for r in rows]


# ---------------------------------------------------------------------------
# Dataset rules + axis orchestration
# ---------------------------------------------------------------------------

# axis-2 sub-axes, each its own opt-in flag on the generator CLI.
# DEFAULT_POS_AXES are the equivalence-preserving ones enabled by --augment-pos;
# pos_value / pos_units / pos_event carry a higher label-noise risk (a perturbed
# value or a cross-measurand unit can make the "valid" label wrong) and ship
# off-by-default — the enabled experiment turns them on once the diagnostic file
# shows whether the safe axes hold up.
POS_AXES = ("pos_entity", "pos_attribute", "pos_value", "pos_units", "pos_event")
DEFAULT_POS_AXES = ("pos_entity", "pos_attribute")
HARD_NEG_KINDS = ("hard_value", "hard_units", "hard_entity")


@dataclass
class DatasetAugmentRules:
    """Everything dataset-specific the orchestrator needs, assembled by each
    ``create_probe_dataset.py`` from its own config."""

    name: str
    # entity handling
    entity_name_field: str                      # "name" — the only field an entity
                                                # swap touches (resolution D: no
                                                # wholesale donor-field copy)
    fabricated_names_by_type: dict[str, list[str]]   # type-token -> candidate names
    fabricated_names_any: list[str]             # fallback pool
    # maps a record to a coarse entity-type token used to pick a same-type
    # fabricated name (pond: pond/lake/wetland/pool/reservoir from the messy
    # free-text ``ecosystem`` field; nfix: ``site_type``; supermat: None)
    entity_type_token: Callable[[dict], str | None]
    # trailing sentence of the pos_entity rewrite instruction — what must stay
    # fixed while the entity is renamed. Dataset-specific: pond/nfix talk about
    # "the ecosystem type", supermat about the measured Tc and its conditions.
    entity_swap_preserve_clause: str
    # judge-visible entity fields (besides entity_name_field) that go stale after
    # a pos_entity rename and are nulled on the positive row + any hard negative
    # derived from it. supermat: ("identifiers",) — the abbreviation catalogue no
    # longer matches the fabricated formula. pond/nfix: () (identifiers ~always
    # null on their GT rows).
    entity_swap_clear_fields: tuple[str, ...]
    # attribute handling
    attr_units: dict[str, list[str]]            # attribute -> canonical units list
    shared_unit_groups: list[list[str]]         # groups of attributes safe to swap between
    entity_field_locked: bool                   # supermat: never touch entity_name_field
    # event handling (judge-VISIBLE fields only)
    event_fields: list[str]
    event_prompt: str
    # pos_event: the single field to synthesise into + the canned value to use
    # when the source measurement has no event at all.  None disables pos_event
    # for this dataset.
    event_synth_field: str | None
    event_synth_value: str | None
    # output schema
    gt_cols: list[str]

    def fabricated_name(self, record: dict, rng: random.Random) -> str | None:
        cur = record.get(self.entity_name_field)
        token = self.entity_type_token(record)
        pool = self.fabricated_names_by_type.get(token) if token else None
        pool = pool or self.fabricated_names_any
        cands = sorted(n for n in pool if n != cur)
        return rng.choice(cands) if cands else None

    def attribute_swap_target(self, attribute: str, rng: random.Random) -> str | None:
        for group in self.shared_unit_groups:
            if attribute in group:
                cands = sorted(a for a in group if a != attribute)
                if cands:
                    return rng.choice(cands)
        return None


def event_fill_records(
    records: list[dict], client: AugmentClient, rules: DatasetAugmentRules,
) -> int:
    """Fill judge-visible event fields on each record from its context (pond only).

    Measurement fields only — the context is never edited here.  A field already
    non-null on the record is left as-is (it came from reviewed GT).  Returns the
    number of (record, field) pairs newly filled.
    """
    if not rules.event_fields:
        return 0
    filled = 0
    for r in records:
        need = [f for f in rules.event_fields if not r.get(f)]
        if not need:
            continue
        measurement = {
            "entity": r.get(rules.entity_name_field),
            "attribute": r.get("attribute"),
            "value": r.get("value"),
            "units": r.get("units"),
        }
        got = client.event_fill(
            context=r[_CTX_ORIG_KEY], measurement=measurement,
            event_prompt=rules.event_prompt, event_fields=need,
        )
        for f, v in got.items():
            if v:
                r[f] = v
                filled += 1
    return filled


def _new_derived_row(src: dict, *, label: str, mod_type: str | None, axis: str | None) -> dict:
    """Copy a source record into a fresh derived row, carrying provenance.

    Deliberately *drops* fields that must be set explicitly per derivation:
    ``_context_override`` (only present if the caller edits the context),
    ``augment_axis`` (set from the ``axis`` arg), ``measurement_id`` (assigned
    per output file, after balancing), and ``_is_base`` (only the verbatim GT
    valids are base).  ``_context_original`` is kept — it is the source page and
    is needed for the diff report and for building negatives on the original
    context.
    """
    row = dict(src)
    for k in (_CTX_EDIT_KEY, "measurement_id", "_is_base"):
        row.pop(k, None)
    row["label"] = label
    row["modification_type"] = mod_type
    row["augment_axis"] = axis
    row["donor_gt_row_index"] = None
    row["source_group_id"] = src.get("source_group_id", src["gt_row_index"])
    return row


def make_axis2_positive(
    src: dict, sub_axis: str, rules: DatasetAugmentRules,
    client: AugmentClient, rng: random.Random,
) -> dict | None:
    """Attempt one equivalent (context, measurement) edit → a new valid row.

    Returns ``None`` when the edit is infeasible (skip, don't force).  On success
    the row carries ``_context_override`` (the locally-patched page) and the
    updated measurement field(s); ``label='valid'``.
    """
    ctx = src[_CTX_ORIG_KEY]

    def _finish(axis: str, expected: list[tuple[str, str]], instr: str,
                field_updates: dict) -> dict | None:
        ok, new_ctx, _edits, _reason = client.rewrite_context(
            context=ctx, instruction=instr, expected=expected,
        )
        if not ok:
            return None
        row = _new_derived_row(src, label="valid", mod_type=None, axis=axis)
        row.update(field_updates)
        row[_CTX_EDIT_KEY] = new_ctx
        return row

    if sub_axis == "pos_entity":
        if rules.entity_field_locked:
            return None
        old = src.get(rules.entity_name_field)
        new = rules.fabricated_name(src, rng)
        if not old or not new:
            return None
        instr = (f'Rewrite this page so that the measurement currently attributed to '
                 f'"{old}" is instead attributed to "{new}". '
                 f'{rules.entity_swap_preserve_clause}')
        updates: dict = {rules.entity_name_field: new}
        for f in rules.entity_swap_clear_fields:
            updates[f] = None
        return _finish("pos_entity", [(old, new)], instr, updates)

    if sub_axis == "pos_attribute":
        tgt = rules.attribute_swap_target(src.get("attribute"), rng)
        old_a = src.get("attribute")
        if tgt is None or not old_a:
            return None
        instr = (f'Rewrite this page so that the measurement currently reported as '
                 f'"{old_a}" is instead a measurement of "{tgt}" (same units, same '
                 f'entity, same value). Change the attribute name/description wherever '
                 f'it appears.')
        return _finish("pos_attribute", [(old_a, tgt)], instr, {"attribute": tgt})

    if sub_axis == "pos_value":
        old_v = str(src.get("value"))
        new_v = perturb_value(old_v, rng, lo=0.15, hi=0.6)
        if new_v is None:
            return None
        instr = (f'Rewrite this page so that the measured value {old_v} for this '
                 f'measurement is instead {new_v}, keeping the same units and entity.')
        return _finish("pos_value", [(old_v, new_v)], instr, {"value": new_v})

    if sub_axis == "pos_units":
        old_u = src.get("units")
        units = rules.attr_units.get(src.get("attribute"), [])
        new_u = alt_unit(old_u, units, rng, family_bias=0.0)
        if not old_u or new_u is None:
            return None
        instr = (f'Rewrite this page so that the units of this measurement, currently '
                 f'"{old_u}", are instead "{new_u}" — assume the reported number is '
                 f'correct in the new units.')
        return _finish("pos_units", [(old_u, new_u)], instr, {"units": new_u})

    if sub_axis == "pos_event":
        if rules.event_synth_field is None:
            return None
        field_ = rules.event_synth_field
        cur = src.get(field_)
        if cur:
            new_ev = f"{cur} (re-sampled)"
            instr = (f'Rewrite this page so that the measurement event detail for this '
                     f'measurement changes from "{cur}" to "{new_ev}".')
            expected = [(str(cur), new_ev)]
        else:
            new_ev = rules.event_synth_value
            val = str(src.get("value"))
            if not new_ev or val not in ctx:
                return None
            sentence = f"This measurement was taken in {new_ev}."
            instr = (f'Add the sentence "{sentence}" immediately after the reported '
                     f'measurement value {val} so that its timing is explicit.')
            expected = [(val, f"{val} ({sentence})")]
        return _finish("pos_event", expected, instr, {field_: new_ev})

    raise ValueError(f"unknown sub_axis {sub_axis!r}")


def make_hard_negative(
    src: dict, kind: str, rules: DatasetAugmentRules, rng: random.Random,
    *, on_edited_context: bool = False,
) -> dict | None:
    """Corrupt only the measurement of ``src`` → a deliberately-hard invalid row.

    ``on_edited_context=True`` builds the negative on top of an axis-2 edited
    context (``src`` is an axis-2 positive); its ``_context_override`` is carried
    through so edited contexts appear on both labels (leakage control).
    """
    if kind == "hard_value":
        new_v = perturb_value(str(src.get("value")), rng, lo=0.05, hi=0.25)
        if new_v is None:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="hard_value",
                               axis=src.get("augment_axis") if on_edited_context else None)
        row["value"] = new_v
    elif kind == "hard_units":
        units = rules.attr_units.get(src.get("attribute"), [])
        new_u = alt_unit(src.get("units"), units, rng, family_bias=0.7)
        if new_u is None:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="hard_units",
                               axis=src.get("augment_axis") if on_edited_context else None)
        row["units"] = new_u
    elif kind == "hard_entity":
        if rules.entity_field_locked:
            return None
        # A hard negative must corrupt a *stated* claim, not invent one. If the
        # source row's entity field is empty (e.g. supermat's sample_details is
        # null for ~82% of GT rows), overwriting it with a fabricated value would
        # label "the paper says X" invalid on a row where the paper says nothing
        # — which may actually be true. Skip instead.
        if not src.get(rules.entity_name_field):
            return None
        new_name = rules.fabricated_name(src, rng)
        if not new_name:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="hard_entity",
                               axis=src.get("augment_axis") if on_edited_context else None)
        row[rules.entity_name_field] = new_name
        # Same staleness as a pos_entity rename: a fabricated formula in `name`
        # leaves supermat's `identifiers` catalogue ("YBCO; Y-123") pointing at
        # the real compound, which the unedited page still supports — that would
        # be a valid-looking claim on an invalid-labelled row. Null it.
        for f in rules.entity_swap_clear_fields:
            row[f] = None
    else:
        raise ValueError(f"unknown hard-negative kind {kind!r}")

    if on_edited_context and _CTX_EDIT_KEY in src:
        row[_CTX_EDIT_KEY] = src[_CTX_EDIT_KEY]
    return row


# ---------------------------------------------------------------------------
# Three-file orchestration
# ---------------------------------------------------------------------------


@dataclass
class AugmentFlags:
    augment_events: bool = False
    pos_axes: tuple[str, ...] = DEFAULT_POS_AXES
    hard_negatives: bool = True
    target_rows: int = 10000
    floor_rows: int = 5000
    max_derived_per_source: int = 4
    diag_pos_event: bool = True   # keep the pos_event slice in the diagnostic file


def _prep_base_valid(src: dict, rules: DatasetAugmentRules) -> dict:
    """A verbatim GT valid, ready as a positive row (train / primary-test)."""
    row = dict(src)
    row.pop(_CTX_EDIT_KEY, None)
    row["label"] = "valid"
    row["modification_type"] = None
    row["augment_axis"] = None
    row["donor_gt_row_index"] = None
    row["source_group_id"] = src["gt_row_index"]
    row["_is_base"] = True
    return row


def _axis2_positives(valids: list[dict], rules: DatasetAugmentRules,
                     client: AugmentClient, rng: random.Random,
                     axes: tuple[str, ...]) -> list[dict]:
    out: list[dict] = []
    for v in valids:
        for axis in axes:
            row = make_axis2_positive(v, axis, rules, client, rng)
            if row is not None:
                out.append(row)
    return out


def _gt_hard_negatives(valids: list[dict], rules: DatasetAugmentRules,
                       rng: random.Random) -> list[dict]:
    out: list[dict] = []
    for v in valids:
        for kind in HARD_NEG_KINDS:
            row = make_hard_negative(v, kind, rules, rng, on_edited_context=False)
            if row is not None:
                out.append(row)
    return out


def _matched_hard_negatives(positives: list[dict], rules: DatasetAugmentRules,
                            rng: random.Random) -> list[dict]:
    """One hard negative per axis-2 positive, sitting on that positive's edited
    context (so edited contexts appear on both labels)."""
    out: list[dict] = []
    for p in positives:
        for kind in rng.sample(list(HARD_NEG_KINDS), k=len(HARD_NEG_KINDS)):
            row = make_hard_negative(p, kind, rules, rng, on_edited_context=True)
            if row is not None:
                out.append(row)
                break
    return out


def build_augmented_files(
    *,
    xv_train: list[dict],
    xv_test: list[dict],
    rng: random.Random,
    client: AugmentClient,
    rules: DatasetAugmentRules,
    flags: AugmentFlags,
    quiet: bool = False,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Build the three augmented output files.

    Each input record must carry ``_context_original`` (the original page text),
    ``gt_row_index``, the entity / attribute / value / units fields, and any GT
    event fields.  Returns ``{file: (output_rows, rows_with_contexts)}`` where
    ``rows_with_contexts`` still has the internal ``_context_*`` fields for the
    diff report; ``output_rows`` is already projected onto the file schema and
    carries the public ``context_override`` field on rows whose context was
    edited (read directly by ``judge_common.prepare_chat_entries`` — no side-car).

    **Phase ordering.**  Every step that calls the client (event-fill, axis-2
    rewrites) runs first, contiguously, for all three splits — *before* any
    hard-negative or balancing work.  ``run_and_write`` relies on this: it runs
    this function once in a ``record`` dry pass to collect every gpt-oss prompt
    for one concurrent ``prewarm``, and that pass only makes the same RNG draws
    as the real pass up to the point where control flow first depends on a
    client return value (``len(axis_pos)``, which the record pass gets wrong
    because it can't know which rewrites the model would accept).  Keeping all
    client calls ahead of that divergence point is what makes the collected
    prompt set correct.  ``quiet`` silences the per-split progress prints for
    the record pass (its counts are all zero and would mislead a log reader).
    """
    extra_keep = ["label", "modification_type", "gt_row_index", "donor_gt_row_index",
                  "measurement_id", "source_group_id", "augment_axis"]

    def _say(msg: str) -> None:
        if not quiet:
            print(msg)

    def _finalize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
        assign_measurement_ids(rows)
        inline_context_overrides(rows)
        clean = strip_internal_fields(rows, rules.gt_cols, extra_keep + [_CTX_PUBLIC_KEY])
        return clean, rows

    # ---- client-calling phases (run first, contiguously — see the docstring) --
    # Three independent shallow-copy passes over the two input lists.  The
    # primary-test valids MUST NOT share a list with the diagnostic-test valids:
    # `event_fill_records` mutates rows in place and the primary test split is
    # exempt from event-fill (its GT event fields are the headline metric's
    # ground truth).
    train_valids = [_prep_base_valid(v, rules) for v in xv_train]
    ptest_valids = [_prep_base_valid(v, rules) for v in xv_test]
    dtest_valids = [_prep_base_valid(v, rules) for v in xv_test]

    if flags.augment_events:
        n = event_fill_records(train_valids, client, rules)
        _say(f"  [train] event-fill: {n} (record, field) pairs filled")
        m = event_fill_records(dtest_valids, client, rules)
        _say(f"  [diagnostic-test] event-fill: {m} (record, field) pairs filled")

    train_axis_pos = _axis2_positives(train_valids, rules, client, rng, flags.pos_axes)
    _say(f"  [train] axis-2 positives: {len(train_axis_pos)}")
    diag_axes = tuple(a for a in flags.pos_axes
                      if flags.diag_pos_event or a != "pos_event")
    dtest_axis_pos = _axis2_positives(dtest_valids, rules, client, rng, diag_axes)
    _say(f"  [diagnostic-test] axis-2 positives: {len(dtest_axis_pos)}")

    # ---- assembly (no client calls past this point; a record pass may diverge
    #      in RNG here and it does not matter — nothing below is collected) -----
    # TRAIN
    gt_neg = _gt_hard_negatives(train_valids, rules, rng)
    edit_neg = _matched_hard_negatives(train_axis_pos, rules, rng)
    train_rows, train_report = balance_and_cap(
        train_valids + train_axis_pos, gt_neg + edit_neg,
        target=flags.target_rows, floor=flags.floor_rows,
        max_derived_per_source=flags.max_derived_per_source, rng=rng,
    )
    _say(f"  [train] {train_report}")

    # PRIMARY TEST (no event-fill, no axis-2, GT-derived negatives)
    ptest_neg = _gt_hard_negatives(ptest_valids, rules, rng)
    ptest_rows, ptest_report = balance_and_cap(
        ptest_valids, ptest_neg, target=0, floor=0,
        max_derived_per_source=flags.max_derived_per_source, rng=rng,
    )
    _say(f"  [primary-test] {ptest_report}")

    # DIAGNOSTIC TEST (axis-2 positives + matched negatives on edited contexts)
    dtest_neg = _matched_hard_negatives(dtest_axis_pos, rules, rng)
    dtest_rows, dtest_report = balance_and_cap(
        dtest_axis_pos, dtest_neg, target=0, floor=0,
        max_derived_per_source=flags.max_derived_per_source, rng=rng,
    )
    _say(f"  [diagnostic-test] {dtest_report}")

    return {
        "train": _finalize(train_rows),
        "primary_test": _finalize(ptest_rows),
        "diagnostic_test": _finalize(dtest_rows),
    }


_OUT_NAMES = {
    "train": "probe_dataset{s}.json",
    "primary_test": "probe_dataset_test{s}.json",
    "diagnostic_test": "probe_dataset_test{s}_diag.json",
}


def assert_wellformed(key: str, rows: list[dict]) -> None:
    """Fail loud on any malformed output file."""
    assert rows, f"[{key}] no rows"
    n_pos = sum(1 for r in rows if r["label"] == "valid")
    n_neg = len(rows) - n_pos
    assert n_pos == n_neg, f"[{key}] not balanced: {n_pos} valid / {n_neg} invalid"
    mids = [r["measurement_id"] for r in rows]
    assert mids == list(range(len(rows))), f"[{key}] measurement_id not contiguous 0..N-1"
    for r in rows:
        assert "source_group_id" in r and r["source_group_id"] is not None, \
            f"[{key}] row {r['measurement_id']} missing source_group_id"
        override = r.get(_CTX_PUBLIC_KEY)
        assert override is None or (isinstance(override, str) and override), \
            f"[{key}] row {r['measurement_id']} has a non-string/empty {_CTX_PUBLIC_KEY}"


def run_and_write(
    *,
    base_dir: Path,
    out_suffix: str,
    ocr_dir: Path,
    xv_train: list[dict],
    xv_test: list[dict],
    rules: DatasetAugmentRules,
    flags: AugmentFlags,
    client: AugmentClient,
    rng: random.Random,
    paper_code_key: str = "_paper_code",
    page_numbers_key: str = "_page_numbers",
) -> dict[str, Path]:
    """End-to-end: attach original contexts, build the three files, write them
    (+ diff reports for splits with edited rows), assert well-formed. Returns
    ``{key: data_path}``. Edited-context rows carry their edited text inline as
    the public ``context_override`` field — no side-car file."""
    base_dir = Path(base_dir)
    ocr_dir = Path(ocr_dir)

    doc_cache: dict[str, str] = {}
    for v in (*xv_train, *xv_test):
        code = v[paper_code_key]
        if code not in doc_cache:
            doc_cache[code] = (ocr_dir / f"{code}.txt").read_text(encoding="utf-8")
        v[_CTX_ORIG_KEY] = extract_page_text(doc_cache[code], v.get(page_numbers_key) or [])

    # gpt-oss: gather every prompt in an RNG-matched dry pass, then fill the cache
    # in one concurrent batch. Without this the real pass hits the model serially
    # (one event loop per call) and blows the generation-job walltime.
    if isinstance(client, GptOssClient):
        rng_state = rng.getstate()
        client.record = True
        build_augmented_files(
            xv_train=xv_train, xv_test=xv_test, rng=rng, client=client,
            rules=rules, flags=flags, quiet=True,
        )
        client.record = False
        jobs = list(client._pending.items())
        client._pending.clear()
        rng.setstate(rng_state)
        print(f"  prewarm: {len(jobs)} gpt-oss call(s) to batch "
              f"({client.cache.stats})")
        client.prewarm(jobs)
        # persist the GPU-generated responses before the real pass: it makes no
        # further model calls (strict_cache), so a later crash must not lose them
        client.flush()
        client.strict_cache = True   # a real-pass miss is now a hard error
        print(f"  prewarm done ({client.cache.stats})")

    bundles = build_augmented_files(
        xv_train=xv_train, xv_test=xv_test, rng=rng, client=client, rules=rules, flags=flags,
    )
    client.flush()

    written: dict[str, Path] = {}
    for key, data_fmt in _OUT_NAMES.items():
        rows, raw_rows = bundles[key]
        assert_wellformed(key, rows)
        data_path = base_dir / data_fmt.format(s=out_suffix)
        with open(data_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        written[key] = data_path
        n_edited = sum(1 for r in rows if r.get(_CTX_PUBLIC_KEY) is not None)
        print(f"  wrote {len(rows):,} rows ({n_edited:,} with an edited context) -> "
              f"{data_path.name}")
        # Diff report is data-derived, not tied to a fixed set of split names: any
        # split with edited rows gets one (today that's train + diagnostic_test;
        # the primary test structurally never has axis-2 edits).
        if n_edited:
            n_diff = emit_context_diff_report(raw_rows, base_dir / (data_path.name + ".diff.txt"))
            print(f"  wrote diff report for {n_diff} edited row(s) -> {data_path.name}.diff.txt")
    return written
