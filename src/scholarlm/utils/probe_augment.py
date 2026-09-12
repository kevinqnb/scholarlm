"""
Dataset-agnostic machinery for synthetic probe-dataset augmentation.

Used by ``data/{pond,nfix,supermat}/create_probe_dataset.py`` when run with the
``--augment`` family of flags.  The per-dataset scripts own their dataset-specific
rules (entity fields, fabricated-name / value / event alternative pools, the
ecosystem-type map); this module owns everything that is the same across
datasets:

  * ``AugmentCache``        — content-hash cache of gpt-oss responses so that two
                             regenerations at the same seed are byte-identical
                             (the LLM is only ever called on a cache miss).
  * ``GptOssClient``        — batched async OpenAI-compatible client for a locally
                             served ``openai/gpt-oss-120b``.  Protocol 3: the
                             model is given the full paper and picks BOTH the new
                             property value and the minimal ``{find, replace}``
                             edits that make the whole paper consistent with it.
  * ``StubAugmentClient``   — deterministic canned edits, for unit tests and the
                             no-server smoke rung.
  * ``make_axis2_positive`` — one equivalence-preserving rewrite (entity name,
                             value, or event) → a synthetic valid row.
  * ``make_typed_negative`` — one direct measurement swap from a pre-generated
                             per-type alternative pool → an invalid row.
  * ``fill_positive_target`` / ``fill_negative_quota`` — resample to the valid
                             floor; fill each error type to its quota, no repeats.
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
import itertools
import json
import math
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

# The augmentation edits and stores the FULL paper text (not the measurement's
# page) — gpt-oss sees the whole document, and every output row carries it as
# ``context_override`` so the judge reads one consistent context scope for valid
# and invalid, edited and unedited rows alike (see ``run_and_write``). Recorded
# in the rewrite cache key so a later scope change can never be served a
# page-scoped cache entry.
_CONTEXT_SCOPE = "full"

# ---------------------------------------------------------------------------
# gpt-oss response cache (reproducibility)
# ---------------------------------------------------------------------------


def _cache_key(op: str, payload: dict) -> str:
    """Stable content hash for a gpt-oss call.

    Keyed on the *semantic* inputs (operation + payload), never on row order or
    ``measurement_id``, so the cache survives dataset regeneration and the
    per-file ``measurement_id`` renumbering.

    The payload must carry everything that changes the model's expected output.
    ``rewrite`` payloads carry ``protocol: 3`` (the model picks the replacement
    value AND the edits), ``context_scope`` (``"full"`` — the prompt and the
    text to edit are the whole paper, not one page), plus ``attempt`` (the
    resample round): an entry from an earlier protocol, a different scope, or a
    different resample round has a different key and is a permanent miss, so a
    stale response can never be read back by the current parser.
    """
    blob = json.dumps({"op": op, "payload": payload}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class AugmentCache:
    """JSON-backed ``{cache_key: response}`` store.

    ``response`` is the raw model text for a ``rewrite`` call.  Load/save are
    explicit; the client checks ``get`` before every call and ``put`` after
    every miss.
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

_REWRITE_SYS = (
    "You are a careful scientific text editor. You are given the full text of a "
    "scientific paper and asked to change one specific property of one "
    "measurement it reports, while changing as little else as possible. You do "
    "NOT rewrite or re-emit the paper. You choose a concrete new value for the "
    "property and return the minimal list of exact-text edits that make the "
    "whole paper consistent with it. Respond with a single JSON object and "
    "nothing else.\n"
    "If the change can be made, respond with:\n"
    '{\"feasible\": true, \"replacement\": \"<the new value, ready to paste into '
    'the measurement record verbatim>\", \"edits\": [{\"find\": \"<verbatim '
    'substring of the paper>\", \"replace\": \"<what it becomes>\"}, ...]}\n'
    "Rules:\n"
    "- \"replacement\" is the new property value on its own (e.g. a site name, a "
    "number, a date) — not a sentence, and not the old value.\n"
    "- Every \"find\" MUST be copied character-for-character from the PAPER TEXT "
    "below (same whitespace, casing, LaTeX and markdown) so it can be located by "
    "an exact string search.\n"
    "- \"find\" SHOULD carry enough surrounding words to be unambiguous (e.g. "
    "\"a critical temperature of 5.2 K\", not \"5.2\"), so applying the edit "
    "cannot corrupt an unrelated occurrence (a year, a different measurement, a "
    "substring of a larger number).\n"
    "- Return one edit per site, and cover EVERY site. The edited claim may "
    "recur many times across a full paper — the abstract, the results text, one "
    "or more table cells, a figure caption, the discussion. Find every "
    "occurrence and emit one edit for each; change nothing else.\n"
    "If making the change consistently would require rewriting large parts of "
    "the paper, or would contradict other statements or tables in it, "
    'respond with {\"feasible\": false, \"reason\": \"...\"} instead.'
)


class AugmentClient(Protocol):
    """Interface the generators depend on. Both real and stub satisfy it."""

    def rewrite_context(
        self, *, context: str, instruction: str, original: str, attempt: int = 0,
    ) -> "RewriteResult": ...

    def flush(self) -> None: ...


@dataclass
class RewriteResult:
    """Outcome of one context-rewrite attempt.

    ``applied`` False is a clean skip (model declined, a ``find`` span did not
    apply, or the local verification failed) — never an error. ``unverified_span``
    is True when ``original`` was not a verbatim substring of the source context,
    so the "old value is gone from the page" check could not run and the row
    rests on the model's self-consistency alone (reported per split).
    """

    applied: bool
    new_context: str
    replacement: str
    edits: list[tuple[str, str]]
    reason: str
    unverified_span: bool = False


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


# `_run_batch` outcome markers (never a valid `_one` return, which is always a
# `str`): a call dropped after the retry budget, and one abandoned unrun because
# the circuit breaker had already tripped.
_DROPPED = object()
_NEVER_RAN = object()

# Served by `_resolve` for a dropped key. Parses as a model decline, so
# `rewrite_context` -> `make_axis2_positive` skips the row; `_skip_category` maps
# this exact reason to the "dropped" bucket in the per-split skip tally.
_DROP_REASON = "dropped: gpt-oss prewarm generation failed after retries"
_DROP_DECLINE = f'{{"feasible": false, "reason": "{_DROP_REASON}"}}'


@dataclass
class GptOssClient:
    """Batched async client for a locally served ``openai/gpt-oss-120b``.

    Mirrors ``experiments/run_judge_local.py``'s ``AsyncOpenAI`` + ``Semaphore``
    pattern.  Every call is checked against ``cache`` first; the model is only
    hit on a miss.  ``rewrite_context`` is synchronous from the caller's point of
    view — batching happens because the generator collects many calls and the OS
    event loop runs them concurrently via ``_run_batch``.  For
    simplicity (the generators are straight-line loops) each public method runs a
    one-shot event loop; call ``prewarm`` with a list of payloads to parallelise.
    """

    api_base: str
    cache: AugmentCache
    model_id: str = "openai/gpt-oss-120b"
    # One fixed temperature for the whole augment run. Higher than a judge run's
    # 0.2 on purpose: `fill_positive_target` resamples the same (source, axis)
    # across rounds and needs the model to give a genuinely different replacement
    # each time. The resample round is in the cache key (`attempt`), so rounds
    # stay individually reproducible; the temperature is the documented "schedule".
    temperature: float = 0.7
    max_concurrent: int = 32
    # The rewrite response is a small JSON object (a `replacement` string + a
    # ~100-token `edits` list) — truncation at the cap is what the diff protocol
    # set out to eliminate. Since context_scope: full, this reservation is now
    # load-bearing: vLLM rejects at admission on `prompt + max_tokens >
    # max_model_len`, so the binding constraint is `largest_paper_tokens +
    # max_tokens < serve max_model_len` (supermat's ~52k-token paper + 20k =
    # 72k, under the 90k served). Raising this back toward "generous" would
    # start 400-ing the largest papers — leave it unless rung 3 shows real
    # truncation.
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
    # would surface only as a blown job walltime). A key in `_dropped_detail` is
    # the one exception — see `_resolve`.
    strict_cache: bool = False
    # A prewarm call that comes back truncated / empty / unparseable is resampled
    # up to this many times before it is dropped. temperature > 0, so a resample
    # of the same prompt can land differently.
    prewarm_max_retries: int = 2
    # Circuit breaker: if the number of dropped calls exceeds this fraction of the
    # batch, `_run_batch` stops the batch and raises instead of finishing with a
    # gutted corpus (a wedged server would otherwise grind through every retry
    # before failing). A single opportunistic call can always be dropped.
    # Residual: a server that accepts and hangs still costs up to
    # ceiling x (1 + retries) x request-timeout / max_concurrent before the
    # breaker fires (vs one timeout for connection-refused) — lower this to abort
    # such a run sooner.
    prewarm_drop_ceiling: float = 0.02
    _pending: dict[str, list[dict]] = field(default_factory=dict, repr=False)
    # {cache_key: repr(last error)} for calls dropped in prewarm after the retry
    # budget. `_resolve` serves these a `feasible: false` decline so the real pass
    # skips them cleanly; also written to `dropped_prewarm.json` beside the cache.
    _dropped_detail: dict[str, str] = field(default_factory=dict, repr=False)

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
            # Empty message, usually `finish_reason="length"` with the whole
            # budget spent on reasoning (see `reasoning_effort`). `_run_batch`
            # retries this a few times and drops the call if it keeps happening;
            # the diagnostic fields are attached here because nothing upstream
            # logs them.
            reasoning = getattr(choice.message, "reasoning_content", None)
            raise ValueError(
                f"gpt-oss returned an empty message (finish_reason="
                f"{choice.finish_reason!r}, reasoning_content_len="
                f"{len(reasoning) if reasoning else 0})"
            )
        if choice.finish_reason == "length":
            # Non-empty but truncated at `max_tokens`: the tail of the JSON is
            # gone, so a downstream parse failure is guaranteed. Raised here, with
            # the diagnostic fields attached, rather than surfacing two layers
            # down as an opaque `not valid JSON`. `_run_batch` retries then drops
            # (temperature > 0, so a resample can come back whole). Under the diff
            # protocol the output is tiny, so a persistent truncation points at
            # something upstream — a runaway repetition loop, a bad `max_tokens`.
            reasoning = getattr(choice.message, "reasoning_content", None)
            raise ValueError(
                f"gpt-oss response was truncated at max_tokens={self.max_tokens} "
                f"(finish_reason='length', content_len={len(content)}, "
                f"reasoning_content_len={len(reasoning) if reasoning else 0})"
            )
        return content

    def _write_dropped_detail(self) -> None:
        """Persist ``_dropped_detail`` (dropped cache key -> last error repr) next
        to the cache, so a post-mortem can see which rewrites were dropped and
        why. No-op when the cache is path-less (unit tests)."""
        if self.cache.path is None:
            return
        path = self.cache.path.parent / "dropped_prewarm.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._dropped_detail, f, indent=2, sort_keys=True)

    async def _run_batch(self, jobs: list[tuple[str, list[dict]]]) -> dict[str, str]:
        """Run ``jobs`` concurrently, caching each success as it lands; return
        ``{key: response}`` for the successes.

        ``asyncio.gather`` without ``return_exceptions=True`` would propagate the
        first failure and discard every sibling response already computed in the
        same batch. Instead a failed call -- ``_one`` raised (truncated / empty
        response, transport error), or the body does not parse as a JSON object
        even after ``_JSON_LAST_RESORT_REPAIRS`` -- is resampled up to
        ``prewarm_max_retries`` times (temperature > 0, so a resample of the same
        prompt can land differently).

        A call still failing after that is DROPPED: recorded in
        ``_dropped_detail`` (and ``dropped_prewarm.json``), kept OUT of the cache,
        and served a ``feasible: false`` decline by ``_resolve`` on the real pass.
        One persistently bad case is then a skipped row, not an aborted run; a
        resubmit re-attempts only the drops (the cache persists). Downstream, the
        valid / negative floors still enforce a hard minimum.

        Circuit breaker: once the drop count exceeds ``prewarm_drop_ceiling`` x
        the batch size, the remaining jobs are abandoned without touching the
        model and this raises -- a wedged or mis-served model still fails fast
        instead of burning the whole walltime on retries.

        The JSON gate checks object-ness, not the ``rewrite`` schema: a
        well-formed object with a bad ``edits`` array or a missing ``replacement``
        is cached here and ``rewrite_context`` fails loud on it on the real pass,
        unchanged. (A ``feasible: true`` object with an *empty* ``edits`` list is
        a model decline, not an error.)
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="EMPTY", base_url=self.api_base, timeout=600.0)
        sem = asyncio.Semaphore(self.max_concurrent)
        drop_ceiling = max(1, math.ceil(self.prewarm_drop_ceiling * len(jobs)))
        tripped = asyncio.Event()
        dropped: dict[str, BaseException] = {}

        async def _guarded(key: str, messages: list[dict]):
            # The failure is retried, never swallowed: after the budget it is
            # recorded in `dropped` (surfaced in the warning, the drop file, and
            # the circuit-breaker raise) and the row is skipped downstream.
            last_exc: BaseException | None = None
            for _ in range(1 + self.prewarm_max_retries):
                if tripped.is_set():
                    return key, _NEVER_RAN
                async with sem:
                    if tripped.is_set():
                        return key, _NEVER_RAN
                    try:
                        text = await self._one(client, messages)
                    except Exception as e:
                        last_exc = e
                        continue
                if _looks_like_json_object(text):
                    return key, text
                _dump_bad_response(text, "prewarm", self._bad_response_dir)
                last_exc = ValueError(
                    "gpt-oss response is not a JSON object even after last-resort "
                    f"repairs: {text.strip()[:200]!r}")
            assert last_exc is not None
            # mutation + check is synchronous, so atomic under asyncio — no lock
            dropped[key] = last_exc
            if len(dropped) > drop_ceiling:
                tripped.set()
            return key, _DROPPED

        pairs = await asyncio.gather(*(_guarded(k, m) for k, m in jobs))
        results: dict[str, str] = {}
        n_never_ran = 0
        for key, value in pairs:
            if value is _DROPPED:
                continue
            if value is _NEVER_RAN:
                n_never_ran += 1
                continue
            results[key] = value
            self.cache.put(key, value)

        if dropped:
            self._dropped_detail.update({k: repr(e) for k, e in dropped.items()})
            self.cache.save()
            self._write_dropped_detail()
            first_k, first_e = next(iter(dropped.items()))
            print(
                f"\n  probe_augment: WARNING -- {len(dropped)}/{len(jobs)} gpt-oss "
                f"prewarm call(s) still failed after {self.prewarm_max_retries} "
                f"retries and were DROPPED ({len(results)} cached). The affected "
                f"rewrites become skipped rows (skip bucket 'dropped'); the "
                f"valid / negative floors still enforce a hard minimum. First "
                f"drop (key {first_k[:12]}...): {first_e!r}"
            )
        if tripped.is_set():
            self.cache.save()
            first_k, first_e = next(iter(dropped.items()))
            raise RuntimeError(
                f"gpt-oss prewarm aborted by the drop circuit breaker: "
                f"{len(dropped)} dropped call(s) over the ceiling of {drop_ceiling} "
                f"({self.prewarm_drop_ceiling:.0%} of {len(jobs)}); {n_never_ran} "
                f"call(s) abandoned when the breaker tripped, {len(results)} "
                f"cached. Fix the server and resubmit -- the cache persists (the "
                f"abandoned calls are not recorded as drops, so they re-run). "
                f"First drop (key {first_k[:12]}...): {first_e!r}"
            ) from first_e
        return results

    # Canned response returned during a record pass — parseable by the caller so
    # control flow (and RNG consumption) matches a normal run up to the point
    # where the collected prompts are all that matters.
    _RECORD_RESPONSES = {
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
        if key in self._dropped_detail:
            # Dropped in prewarm after the retry budget — a clean skip on the
            # real pass, not a crash (`_run_batch` already warned).
            return _DROP_DECLINE
        if self.strict_cache:
            raise RuntimeError(
                f"gpt-oss cache miss for op {op!r} on the real pass, after "
                f"prewarm — the record pass did not collect this prompt. Every "
                f"rewrite prompt is a pure function of (source row, axis, "
                f"attempt) with no dependence on a prior model call, so the "
                f"record pass should collect all of them; a miss here means the "
                f"record dry pass and the real pass diverged in their RNG-driven "
                f"(source, axis, attempt) enumeration."
            )
        batch = asyncio.run(self._run_batch([(key, messages)]))
        return batch.get(key, _DROP_DECLINE)

    def prewarm(self, jobs: list[tuple[str, list[dict]]]) -> None:
        """Fill the cache for a list of ``(cache_key, messages)`` in one batch.

        ``_run_batch`` caches each success itself and records the drops (calls
        still failing after the retry budget) in ``_dropped_detail`` for
        ``_resolve`` to skip on the real pass — so on the happy path, and on a
        few-drops path, there is nothing further to do here. It still raises if
        the drop circuit breaker tripped.
        """
        misses = [(k, m) for k, m in jobs if self.cache.get(k) is None]
        if not misses:
            return
        asyncio.run(self._run_batch(misses))

    def flush(self) -> None:
        self.cache.save()

    # -- public ops --------------------------------------------------------

    _RESAMPLE_NUDGE = (
        "\n\nThis is an ALTERNATIVE rewrite of the same paper. Choose a "
        "noticeably different, less obvious replacement value than the first "
        "one that comes to mind, while still keeping it plausible for this "
        "measurement."
    )

    def _rewrite_messages(self, context: str, instruction: str, attempt: int) -> list[dict]:
        nudge = self._RESAMPLE_NUDGE if attempt else ""
        user = (
            f"{instruction}{nudge}\n\n"
            f"Return the new value and only the minimal exact-text edits needed. "
            f"Do not alter any other fact, number, unit, name, table cell, or "
            f"sentence.\n\n"
            f"## PAPER TEXT\n{context}"
        )
        return [{"role": "system", "content": _REWRITE_SYS},
                {"role": "user", "content": user}]

    @staticmethod
    def _parse_edits(obj: dict, raw: str) -> list[tuple[str, str]]:
        """Validate a non-empty ``edits`` array; hard error on any schema
        violation. A missing or empty ``edits`` list is the caller's to handle
        (``rewrite_context`` treats it as a model decline, not a schema error)."""
        edits_raw = obj.get("edits")
        if not isinstance(edits_raw, list):
            raise ValueError(f"gpt-oss rewrite feasible but 'edits' is not a list: {raw[:400]!r}")
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
        return edits

    @staticmethod
    def verify_rewrite(orig_ctx: str, new_ctx: str, original: str, replacement: str,
                       edits: list[tuple[str, str]]) -> RewriteResult:
        """Local, model-independent check that the rewrite did what was asked.

        ``original`` is the property's old surface value (a name, a number, a
        date), or ``""`` for the pos_event *inject* case where the source paper
        stated no event. Shared by ``GptOssClient`` and ``StubAugmentClient`` so
        both admit exactly the same rows.
        """
        if replacement not in new_ctx:
            return RewriteResult(
                False, orig_ctx, "", [],
                f"rewrite did not introduce the replacement {replacement!r}")
        if not original:                       # inject case (pos_event, no prior event)
            if replacement in orig_ctx:
                return RewriteResult(False, orig_ctx, "", [],
                                     f"replacement {replacement!r} was already in the paper")
            # the added event is not grounded in the source paper by construction
            return RewriteResult(True, new_ctx, replacement, edits, "", unverified_span=True)
        if orig_ctx.count(original) == 0:
            # `original` is not a verbatim paper span, so "old value is gone" can't
            # be checked -- the row rests on the model's self-consistency alone.
            return RewriteResult(True, new_ctx, replacement, edits, "", unverified_span=True)
        if new_ctx.count(original) and original != replacement:
            return RewriteResult(False, orig_ctx, "", [],
                                 f"rewrite left the original {original!r} in the paper")
        return RewriteResult(True, new_ctx, replacement, edits, "")

    def rewrite_context(self, *, context, instruction, original, attempt=0,
                        stub_replacement=None):
        """Protocol-3 context edit: the model picks the new value AND the edits.

        Returns a ``RewriteResult``. ``applied=False`` is a clean skip (model
        declined outright, declared the edit feasible but proposed no edits, a
        ``find`` span did not apply, or ``verify_rewrite`` failed). Fails loud
        (``ValueError``) only on a genuine schema violation: ``feasible: true``
        with no ``replacement`` string, or an ``edits`` value that is present
        but malformed (not a list, or an element that is not
        ``{find: str, replace: str}``).
        ``stub_replacement`` is ignored here — it only steers ``StubAugmentClient``.
        """
        payload = {"context": context, "instruction": instruction,
                   "protocol": 3, "context_scope": _CONTEXT_SCOPE, "attempt": attempt}
        key = _cache_key("rewrite", payload)
        raw = self._resolve(key, "rewrite",
                            self._rewrite_messages(context, instruction, attempt))
        obj = _extract_json_object(raw, op="rewrite", dump_dir=self._bad_response_dir)
        if not obj.get("feasible", False):
            return RewriteResult(False, context, "", [], str(obj.get("reason", "infeasible")))
        replacement = obj.get("replacement")
        if not (isinstance(replacement, str) and replacement.strip()):
            raise ValueError(
                f"gpt-oss rewrite feasible but 'replacement' is missing/empty: {raw[:400]!r}"
            )
        replacement = replacement.strip()
        edits_raw = obj.get("edits")
        if edits_raw is None or (isinstance(edits_raw, list) and not edits_raw):
            # feasible + a replacement but zero edits: the model claims the
            # change is possible yet supplies no way to make it. In practice the
            # GT surface form just isn't a verbatim span of the paper, so this is
            # functionally a decline — a clean skip, like `feasible: false`, not
            # a schema error. (Rare: 1/359 on nfix rung 3, 0/415 on pond.)
            return RewriteResult(False, context, "", [], "feasible but proposed no edits")
        edits = self._parse_edits(obj, raw)
        try:
            new_ctx = apply_verified_edit(context, edits)
        except ValueError as exc:
            return RewriteResult(False, context, "", [], f"rewrite edit not applicable: {exc}")
        return self.verify_rewrite(context, new_ctx, original, replacement, edits)


class StubAugmentClient:
    """Deterministic canned client for tests and the no-server smoke rung.

    ``rewrite_context`` replaces the caller-supplied ``original`` span with the
    caller's ``stub_replacement`` hint (which stands in for the model's proposed
    replacement) everywhere it occurs in the context — no LLM, no prompt
    parsing.  The pos_event inject case (``original == ""``) appends a short
    sentence naming the replacement.  Marks the attempt infeasible when there is
    no hint or the ``original`` span is absent, so on real OCR (where the GT
    surface form is often not a verbatim page substring) the stub skips more
    than the real client — fine for plumbing/determinism tests, not
    representative of real yield.
    """

    def __init__(self, cache: AugmentCache | None = None):
        self.cache = cache or AugmentCache(None)

    def rewrite_context(self, *, context, instruction, original, attempt=0,
                        stub_replacement=None):
        if not stub_replacement:
            return RewriteResult(False, context, "", [], "stub: no stub_replacement hint")
        if not original:                       # pos_event inject case
            sentence = f" This measurement's event was {stub_replacement}."
            new_ctx = context + sentence
            edits = [(context, new_ctx)]
        else:
            if original not in context:
                return RewriteResult(False, context, "", [],
                                     f"stub: span {original!r} not in context")
            new_ctx = context.replace(original, stub_replacement)
            edits = [(original, stub_replacement)]
        return GptOssClient.verify_rewrite(context, new_ctx, original, stub_replacement, edits)

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
# Numeric value perturbation (pos_value stub hint)
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


# ---------------------------------------------------------------------------
# measurement_id bookkeeping
# ---------------------------------------------------------------------------


def assign_measurement_ids(rows: list[dict], *, start: int = 0) -> None:
    """Assign a contiguous per-file ``measurement_id`` (positional, like the
    baseline generator's ``enumerate(combined)``)."""
    for i, r in enumerate(rows, start=start):
        r["measurement_id"] = i


# ---------------------------------------------------------------------------
# File assembly
# ---------------------------------------------------------------------------


def even_quota(total: int, types: list[str]) -> dict[str, int]:
    """Split ``total`` across ``types`` as evenly as possible (deterministic —
    the first ``total % len(types)`` types get one extra)."""
    assert types, "even_quota: no error types"
    base, rem = divmod(total, len(types))
    return {t: base + (1 if i < rem else 0) for i, t in enumerate(types)}


def assemble_file(positives: list[dict], negatives: list[dict],
                  rng: random.Random) -> list[dict]:
    """Concatenate the two classes, shuffle, assign contiguous measurement_ids.

    Prevalence and per-class counts are the caller's responsibility (the
    positive filler and the negative quota already produce a balanced pair);
    this only interleaves them reproducibly.
    """
    out = list(positives) + list(negatives)
    rng.shuffle(out)
    assign_measurement_ids(out)
    return out


# ---------------------------------------------------------------------------
# Context-override inlining + diff report
# ---------------------------------------------------------------------------

_CTX_ORIG_KEY = "_context_original"
_CTX_EDIT_KEY = "_context_override"
_CTX_PUBLIC_KEY = "context_override"   # kept on the output row (see strip_internal_fields)


def inline_context_overrides(rows: list[dict]) -> int:
    """Stamp the public ``context_override`` field onto *every* row (in place):
    the edited paper where the augmentation edited it, the verbatim source paper
    otherwise. Returns the count of rows whose stamped context is a genuine edit.

    ``judge_common.prepare_chat_entries`` reads this field directly off the row
    — no side-car, no measurement_id-keyed lookup. Because every augmented row
    carries it, the judge reads one consistent context scope (the full paper)
    for valid and invalid, edited and unedited rows alike — the within-file and
    train/primary-test comparability the leakage controls depend on.
    """
    n_edited = 0
    for r in rows:
        orig = r.get(_CTX_ORIG_KEY)
        assert orig, f"row {r.get('measurement_id')} carries no source context to stamp"
        edited = r.get(_CTX_EDIT_KEY)
        if edited is not None and edited != orig:
            r[_CTX_PUBLIC_KEY] = edited
            n_edited += 1
        else:
            r[_CTX_PUBLIC_KEY] = orig
    return n_edited


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

# Positive axes: for every ground-truth valid we attempt an equivalence-preserving
# rewrite along each of these — the entity name, the measured value, the
# measurement event. Attribute and units are deliberately excluded: a consistent
# change to either usually needs a much larger context edit (a whole table) than
# the diff protocol can make safely.
POS_AXES = ("pos_entity", "pos_value", "pos_event")

# Invalid error types, in the canonical order used for even quotas. `attribute`
# is only constructible where the dataset reports more than one attribute (pond);
# `event` needs a stated (GT) or synthesised (pos_event) event on the source
# valid to contradict.
ERROR_TYPES = ("entity", "attribute", "value", "units", "event")


@dataclass
class DatasetAugmentRules:
    """Everything dataset-specific the orchestrator needs, assembled by each
    ``create_probe_dataset.py`` from its own config."""

    name: str
    # --- entity ---------------------------------------------------------------
    entity_name_field: str                     # "name" — the only field a rename touches
    entity_noun: str                           # "site", "material" — for the instruction
    fabricated_names_by_type: dict[str, list[str]]   # type token -> candidate names
    fabricated_names_any: list[str]            # fallback pool (also the negative pool)
    entity_type_token: Callable[[dict], str | None]  # record -> coarse type or None
    name_suffix_to_type: dict[str, str]        # last-word -> type; {} = cannot verify a name
    entity_preserve_clause: str                # what a rename must hold fixed
    entity_swap_clear_fields: tuple[str, ...]  # judge-visible fields nulled on a rename
    # --- attribute / units --------------------------------------------------
    attr_units: dict[str, list[str]]           # attribute -> canonical units list
    attribute_pool: list[str]                  # every attribute key; < 2 -> no attribute error
    value_pool_by_attr: dict[str, list[str]]   # attribute -> distinct GT value strings
    # --- event ------------------------------------------------------------
    event_field: str | None                    # measurement field a pos_event / bad_event
                                               # writes; None = the dataset has no
                                               # judge-visible event field at all
                                               # (supermat: date absent, pressure filtered
                                               # out of the judge prompt) — no pos_event
                                               # axis, no event error type
    event_noun: str                            # "measurement date", "applied pressure"
    event_pool: list[str]                      # distinct plausible event values
    event_allow_inject: bool                   # pos_event may ADD an event to a page
                                               # that states none (pond/nfix: a date
                                               # is always injectable; supermat: no —
                                               # unstated pressure means "ambient")
    # --- output schema --------------------------------------------------
    gt_cols: list[str]

    def __post_init__(self) -> None:
        # A dataset with no event field must also carry no event pool and disallow
        # injection; a half-set trio is a config error, not a silently event-less run.
        if self.event_field is None:
            assert not self.event_pool and not self.event_allow_inject, (
                f"{self.name}: event_field is None but event_pool="
                f"{self.event_pool!r} / event_allow_inject={self.event_allow_inject!r}"
            )

    def entity_candidates(self, record: dict) -> list[str]:
        tok = self.entity_type_token(record)
        pool = self.fabricated_names_by_type.get(tok) if tok else None
        return list(pool or self.fabricated_names_any)

    def fabricated_name(self, record: dict, rng: random.Random) -> str | None:
        cur = record.get(self.entity_name_field)
        cands = sorted(n for n in self.entity_candidates(record) if n != cur)
        return rng.choice(cands) if cands else None

    def name_type(self, name: str) -> str | None:
        return self.name_suffix_to_type.get(name.rsplit(" ", 1)[-1].lower())

    def entity_type_ok(self, record: dict, proposed_name: str) -> bool:
        """Does a model-proposed replacement name preserve the source's type?

        Returns True unconditionally where the dataset gives no name→type map
        (nfix, supermat). Where it does (pond), the proposed name must have a
        recognised suffix, and it must match the source record's type token.
        """
        if not self.name_suffix_to_type:
            return True
        want = self.entity_type_token(record)
        got = self.name_type(proposed_name)
        return got is not None if want is None else got == want


@dataclass
class AugmentFlags:
    pos_axes: tuple[str, ...] = POS_AXES
    # FLOORS, not caps. Every GT valid is carried through and gets one rewrite
    # attempt per axis (round 0, always kept); resampling only kicks in if the
    # distinct-synthetic count is still below the floor. The train / diagnostic
    # files therefore usually end up LARGER than the floor, balanced 1:1 with
    # negatives at whatever size the positives reach.
    valid_floor: int = 5000
    diag_valid_floor: int = 1000
    # Attempt budget = len(gt_valids) * len(axes) * prompt_budget_multiple.
    # multiple 1 == round 0 only (no resampling); raise it (per dataset, in the
    # config) when round 0 alone doesn't clear the floor. Falling short inside
    # the budget is a hard error.
    prompt_budget_multiple: int = 1


# ---------------------------------------------------------------------------
# Provenance-stamped rows
# ---------------------------------------------------------------------------

_INTERNAL_DERIVED_KEYS = (_CTX_EDIT_KEY, "measurement_id",
                          "_unverified_span", "_augment_edits", "augment_attempt")


def _prep_base_valid(src: dict, rules: DatasetAugmentRules) -> dict:
    """A verbatim GT valid, carried through as a positive row (every file)."""
    row = dict(src)
    row.pop(_CTX_EDIT_KEY, None)
    row["label"] = "valid"
    row["modification_type"] = None
    row["augment_axis"] = None
    row["augment_attempt"] = None
    row["donor_gt_row_index"] = None
    row["source_group_id"] = src["gt_row_index"]
    return row


def _new_derived_row(src: dict, *, label: str, mod_type: str | None,
                     axis: str | None) -> dict:
    """Copy a source record into a fresh derived row, carrying provenance and
    dropping every field that must be set per derivation."""
    row = dict(src)
    for k in _INTERNAL_DERIVED_KEYS:
        row.pop(k, None)
    row["label"] = label
    row["modification_type"] = mod_type
    row["augment_axis"] = axis
    row["augment_attempt"] = None
    row["donor_gt_row_index"] = None
    row["source_group_id"] = src.get("source_group_id", src["gt_row_index"])
    return row


def _row_signature(row: dict, gt_cols: list[str]) -> tuple:
    """Content identity of a row, for de-duplication: every GT-schema field the
    probe sees plus the edited paper text (``None`` when the augmentation did
    not edit this row). Only ever called during assembly, before ``_finalize``
    stamps the verbatim full paper onto every row's ``context_override`` — so an
    unedited row still signs as ``None`` here regardless of that later stamp.
    Two rows with the same signature are the same example regardless of
    provenance; only the first is kept."""
    ctx = row.get(_CTX_EDIT_KEY) if _CTX_EDIT_KEY in row else row.get(_CTX_PUBLIC_KEY)
    vals = (tuple(v) if isinstance(v, list) else v for v in (row.get(c) for c in gt_cols))
    return (*vals, ctx)


def _dedup_rows(rows: list[dict], gt_cols: list[str]) -> list[dict]:
    seen: set = set()
    out: list[dict] = []
    for r in rows:
        sig = _row_signature(r, gt_cols)
        if sig not in seen:
            seen.add(sig)
            out.append(r)
    return out


def _draw_alt(pool, current, rng: random.Random, context: str) -> str | None:
    """Pick a pool entry that differs from ``current`` and does not already
    occur verbatim in ``context`` (the mislabel guard — a "wrong" value that the
    page happens to state elsewhere is not wrong). Deterministic given ``rng``."""
    cands = sorted({str(x) for x in pool if str(x) != str(current) and str(x) not in context})
    return rng.choice(cands) if cands else None


# ---------------------------------------------------------------------------
# Axis-2 synthetic positives — the model picks the new value AND the edits
# ---------------------------------------------------------------------------


def _axis_entity(src, rules, rng):
    old = src.get(rules.entity_name_field)
    if not old:
        return None
    tok = rules.entity_type_token(src)
    examples = ", ".join(f'"{n}"' for n in sorted(rules.entity_candidates(src))[:3])
    type_hint = f" of the same kind ({tok})" if tok else ""
    instr = (
        f'This paper reports a measurement attributed to the {rules.entity_noun} '
        f'"{old}". Rewrite the paper so the identical measurement is attributed to '
        f'a different, made-up {rules.entity_noun}{type_hint} that is not a real '
        f'place' + (f' — for example {examples}' if examples else '') + '. '
        f'Replace the name everywhere it appears in the paper. '
        f'{rules.entity_preserve_clause} Report the new name alone as "replacement".'
    )
    updates = {f: None for f in rules.entity_swap_clear_fields}
    return old, instr, rules.fabricated_name(src, rng), rules.entity_name_field, updates


def _axis_value(src, rules, rng):
    old_v = str(src.get("value")).strip()
    if not _NUM_RE.match(old_v):
        return None
    attr = src.get("attribute")
    units = src.get("units") or ""
    instr = (
        f'This paper reports a {attr} measurement of "{old_v}" {units} for '
        f'"{src.get(rules.entity_name_field)}". Rewrite the paper so the identical '
        f'measurement instead reads a different, plausible {attr} value — same '
        f'units, same entity, same measurement event — changing the number every '
        f'place it appears in the paper (running text, tables, abstract). Report '
        f'the new number alone as "replacement".'
    )
    return old_v, instr, perturb_value(old_v, rng, lo=0.15, hi=0.6), "value", {}


def _axis_event(src, rules, rng):
    field = rules.event_field
    if field is None:                          # dataset has no judge-visible event
        return None
    ctx = src[_CTX_ORIG_KEY]
    cur = src.get(field)
    if cur:
        instr = (
            f'This paper states the {rules.event_noun} for this measurement is '
            f'"{cur}". Rewrite the paper so it instead states a different but '
            f'plausible {rules.event_noun} for the same measurement (same entity, '
            f'value and units), everywhere it appears. Report the new '
            f'{rules.event_noun} alone as "replacement".'
        )
        return str(cur), instr, _draw_alt(rules.event_pool, cur, rng, ctx), field, {}
    if not rules.event_allow_inject:
        return None
    val = str(src.get("value")).strip()
    if not val or val not in ctx:
        return None
    instr = (
        f'This paper reports a measurement (value "{val}") but does not state its '
        f'{rules.event_noun}. Add a short phrase, next to where the paper reports '
        f'the value "{val}", giving a plausible {rules.event_noun} for this '
        f'measurement, and report that {rules.event_noun} alone as "replacement". '
        f'Change nothing else.'
    )
    return "", instr, _draw_alt(rules.event_pool, None, rng, ctx), field, {}


_AXIS_BUILDERS = {
    "pos_entity": _axis_entity,
    "pos_value": _axis_value,
    "pos_event": _axis_event,
}

# Known skip reasons, collapsed for the per-split tally (drops the row-specific
# quoted value from the message so counts aggregate).
_SKIP_CATEGORIES = (
    "axis not applicable to this row", "no pool alternative",
    "wrong entity type", "edit not applicable", "did not introduce the replacement",
    "left the original", "already in the paper", "not valid JSON",
    "feasible but proposed no edits",
)


def _skip_category(reason: str) -> str:
    # Match the drop sentinel exactly, not the bare word: gpt-oss's own free-text
    # `feasible: false` reason can legitimately contain "dropped" ("dropped from
    # Table 2") and must not be miscounted as an infrastructure drop.
    if reason == _DROP_REASON:
        return "dropped"
    for pat in _SKIP_CATEGORIES:
        if pat in reason:
            return pat
    return (reason.split(":", 1)[0].strip() or "infeasible")[:48]


def make_axis2_positive(
    src: dict, sub_axis: str, rules: DatasetAugmentRules,
    client: AugmentClient, rng: random.Random, *, attempt: int = 0,
    skips: "Counter | None" = None,
) -> dict | None:
    """One equivalence-preserving (context, measurement) edit → a new valid row,
    or ``None`` when the edit is infeasible (skip, don't force). When ``skips``
    is given, the reason for a skip is tallied into it.

    RNG is consumed only in the axis builder (the ``stub_replacement`` hint / the
    event-pool draw), never after the client call — so a ``record`` dry pass and
    the real pass consume identical RNG here regardless of what the client
    returns.
    """
    if sub_axis not in _AXIS_BUILDERS:
        raise ValueError(f"unknown sub_axis {sub_axis!r}")

    def _skip(reason: str) -> None:
        if skips is not None:
            skips[f"{sub_axis}: {_skip_category(reason)}"] += 1

    spec = _AXIS_BUILDERS[sub_axis](src, rules, rng)
    if spec is None:
        _skip("axis not applicable to this row")
        return None
    original, instr, stub_replacement, field, extra = spec
    if stub_replacement is None:
        _skip("no pool alternative")
        return None
    res = client.rewrite_context(
        context=src[_CTX_ORIG_KEY], instruction=instr, original=original,
        attempt=attempt, stub_replacement=stub_replacement,
    )
    if not res.applied:
        _skip(res.reason)
        return None
    if sub_axis == "pos_entity" and not rules.entity_type_ok(src, res.replacement):
        _skip("proposed name is the wrong entity type")
        return None
    row = _new_derived_row(src, label="valid", mod_type=sub_axis, axis=sub_axis)
    row[field] = res.replacement
    for k, v in extra.items():
        row[k] = v
    row[_CTX_EDIT_KEY] = res.new_context
    row["_unverified_span"] = res.unverified_span
    row["_augment_edits"] = tuple(sorted(res.edits))
    row["augment_attempt"] = attempt
    return row


# ---------------------------------------------------------------------------
# Typed hard negatives — one direct measurement swap from a pre-generated pool
# ---------------------------------------------------------------------------


def make_typed_negative(
    src: dict, err_type: str, rules: DatasetAugmentRules, rng: random.Random,
) -> dict | None:
    """Corrupt exactly one measurement field of ``src`` with an alternative
    drawn from the matching pre-generated pool → an invalid row, or ``None`` when
    no usable alternative exists for this source (skip).

    The context is never edited. A negative built on a synthetic positive keeps
    that positive's edited paper and axis tag, so edited contexts appear on both
    labels (the diagnostic file's leakage control). ``ctx`` here (the collision
    guard for ``_draw_alt``) is the full paper — a pool alternative already
    stated anywhere in it is rejected."""
    ctx = src.get(_CTX_EDIT_KEY) or src[_CTX_ORIG_KEY]

    if err_type == "entity":
        cur = src.get(rules.entity_name_field)
        if not cur:
            return None                    # never invent a claim the page omits
        alt = _draw_alt(rules.entity_candidates(src), cur, rng, ctx)
        if alt is None:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="bad_entity", axis=None)
        row[rules.entity_name_field] = alt
        for f in rules.entity_swap_clear_fields:
            row[f] = None
    elif err_type == "attribute":
        cur = src.get("attribute")
        # No context-collision guard here: on a multi-measurement page the other
        # attribute names are always present, but "entity X's measurement is
        # attribute A" is still false — attribute alone doesn't define the claim.
        others = sorted(a for a in rules.attribute_pool if a != cur)
        if not others:
            return None
        alt = rng.choice(others)
        row = _new_derived_row(src, label="invalid", mod_type="bad_attribute", axis=None)
        row["attribute"] = alt
    elif err_type == "value":
        cur = str(src.get("value"))
        pool = rules.value_pool_by_attr.get(src.get("attribute"), [])
        alt = _draw_alt([v for v in pool if v != cur], cur, rng, ctx)
        if alt is None:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="bad_value", axis=None)
        row["value"] = alt
    elif err_type == "units":
        cur = src.get("units")
        pool = rules.attr_units.get(src.get("attribute"), [])
        alt = _draw_alt([u for u in pool if not cur or u.lower() != cur.lower()],
                        cur, rng, ctx)
        if alt is None:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="bad_units", axis=None)
        row["units"] = alt
    elif err_type == "event":
        if rules.event_field is None:
            return None                    # dataset has no judge-visible event
        cur = src.get(rules.event_field)
        if not cur:
            return None                    # nothing stated to contradict
        alt = _draw_alt([e for e in rules.event_pool if e != str(cur)],
                        str(cur), rng, ctx)
        if alt is None:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="bad_event", axis=None)
        row[rules.event_field] = alt
    else:
        raise ValueError(f"unknown error type {err_type!r}")

    if _CTX_EDIT_KEY in src:
        row[_CTX_EDIT_KEY] = src[_CTX_EDIT_KEY]
        row["augment_axis"] = src.get("augment_axis")
    return row


# ---------------------------------------------------------------------------
# Fillers — resample to the positive floor, fill each error type to its quota
# ---------------------------------------------------------------------------


def active_error_types(rules: DatasetAugmentRules, valids: list[dict]) -> list[str]:
    """The error types constructible for this dataset and this valid pool:
    ``attribute`` only when the dataset has >1 attribute; ``event`` only when at
    least one valid in the pool carries a (GT or synthesised) event."""
    have_attr = len(rules.attribute_pool) >= 2
    have_event = (rules.event_field is not None and bool(rules.event_pool)
                  and any(v.get(rules.event_field) for v in valids))
    return [t for t in ERROR_TYPES
            if not (t == "attribute" and not have_attr)
            and not (t == "event" and not have_event)]


def fill_positive_target(
    gt_valids: list[dict], rules: DatasetAugmentRules, client: AugmentClient,
    rng: random.Random, *, floor: int, axes: tuple[str, ...],
    prompt_budget_multiple: int, seen_sigs: set, label: str = "",
    quiet: bool = False,
) -> list[dict]:
    """One equivalence-preserving rewrite per (GT valid, axis) in round 0 — every
    accepted result kept — then resample randomly-ordered (valid, axis) pairs at
    the raised temperature / nudge until ``floor`` *distinct* synthetic valids
    are accepted. Every accepted row is distinct (by ``_row_signature``) from
    every other and from ``seen_sigs`` (the carried GT valids, pre-seeded by the
    caller; this function mutates it).

    Attempt budget = ``len(gt_valids) * len(axes) * prompt_budget_multiple``;
    running out before ``floor`` is a hard error. Both the ``record`` dry pass
    and the real pass iterate the full budget so their RNG stays aligned — the
    real pass just stops *collecting* resamples once the floor is met (round 0
    always collects)."""
    recording = getattr(client, "record", False)
    per_round = max(1, len(gt_valids) * len(axes))
    budget = per_round * max(1, prompt_budget_multiple)
    accepted: list[dict] = []
    skips: Counter = Counter()
    attempts = 0
    round_i = 0
    while attempts < budget:
        order = list(itertools.product(range(len(gt_valids)), axes))
        rng.shuffle(order)
        for vi, axis in order:
            if attempts >= budget:
                break
            attempts += 1
            row = make_axis2_positive(gt_valids[vi], axis, rules, client, rng,
                                      attempt=round_i, skips=skips)
            if row is None:
                continue
            row.pop("_augment_edits", None)
            # round 0 keeps every result; resample rounds stop at the floor
            if not recording and round_i > 0 and len(accepted) >= floor:
                continue                       # RNG already spent; don't collect
            sig = _row_signature(row, rules.gt_cols)
            if sig in seen_sigs:
                skips[f"{axis}: duplicate row"] += 1
                continue
            seen_sigs.add(sig)
            accepted.append(row)
        round_i += 1
    if not recording and not quiet:
        tag = f" [{label}]" if label else ""
        print(f"  positive fill{tag}: {len(accepted)} distinct synthetic / {attempts} "
              f"attempts ({round_i} rounds); skips: "
              f"{dict(sorted(skips.items(), key=lambda kv: -kv[1]))}")
    if not recording and len(accepted) < floor:
        raise RuntimeError(
            f"positive floor not met{f' [{label}]' if label else ''}: "
            f"{len(accepted)}/{floor} distinct synthetic valids in {attempts} "
            f"attempts (budget {budget}). Raise prompt_budget_multiple, lower "
            f"--augment-valid-floor for this dataset, or the model yield / "
            f"distinct-resample rate is too low."
        )
    return accepted


def fill_negative_quota(
    valids: list[dict], rules: DatasetAugmentRules, rng: random.Random, *,
    quota_by_type: dict[str, int], seen_sigs: set, label: str = "",
) -> list[dict]:
    """Fill each error type to its quota by sampling ``valids`` with replacement.
    Every accepted negative is distinct (by ``_row_signature``) from every other
    and from ``seen_sigs`` (pre-seed it with the positive-row signatures so a
    "negative" that coincides with a real valid is dropped, not mislabelled).
    Fails loud if a type's pool is missing (only misses) or too shallow for its
    quota (only duplicate draws)."""
    out: list[dict] = []
    for err_type, quota in quota_by_type.items():
        if quota <= 0:
            continue
        got: list[dict] = []
        order = list(range(len(valids)))
        rng.shuffle(order)
        cursor = misses = dups = 0
        cap = max(5000, 50 * len(valids))
        while len(got) < quota:
            if cursor >= len(order):
                rng.shuffle(order)
                cursor = 0
            neg = make_typed_negative(valids[order[cursor]], err_type, rules, rng)
            cursor += 1
            if neg is None:
                misses += 1
                if misses > cap:
                    raise RuntimeError(
                        f"negative quota not met{f' [{label}]' if label else ''} "
                        f"for {err_type!r}: {len(got)}/{quota}; {misses} sources had "
                        f"no usable {err_type} alternative — the pool is missing."
                    )
                continue
            sig = _row_signature(neg, rules.gt_cols)
            if sig in seen_sigs:
                dups += 1
                if dups > cap:
                    raise RuntimeError(
                        f"negative quota not met{f' [{label}]' if label else ''} "
                        f"for {err_type!r}: {len(got)}/{quota}; {dups} duplicate "
                        f"draws — too few distinct (source, {err_type}-alternative) "
                        f"combinations for this quota."
                    )
                continue
            seen_sigs.add(sig)
            got.append(neg)
        out.extend(got)
    return out


# ---------------------------------------------------------------------------
# Three-file orchestration
# ---------------------------------------------------------------------------

_EXTRA_KEEP = ["label", "modification_type", "gt_row_index", "donor_gt_row_index",
               "measurement_id", "source_group_id", "augment_axis", "augment_attempt"]


def build_augmented_files(
    *, xv_train: list[dict], xv_test: list[dict], rng: random.Random,
    client: AugmentClient, rules: DatasetAugmentRules, flags: AugmentFlags,
    quiet: bool = False,
) -> dict[str, tuple[list[dict], list[dict]]]:
    """Build the train / primary-test / diagnostic-test files.

    Each input record must carry ``_context_original`` (the full source paper
    text), ``gt_row_index``, the entity / attribute / value / units fields and
    any GT event field. Returns ``{file: (output_rows, rows_with_contexts)}`` —
    the second element keeps the internal ``_context_*`` fields for the diff
    report.

    **Phase ordering.** Both ``fill_positive_target`` calls (the only client-
    calling code) run first, contiguously; every negative and every assembly
    step is pure and RNG-deterministic after that. ``run_and_write`` relies on
    this: it runs this function once in a ``record`` dry pass to collect every
    gpt-oss prompt for one concurrent ``prewarm``."""

    def _say(msg: str) -> None:
        if not quiet:
            print(msg)

    def _finalize(rows: list[dict]) -> tuple[list[dict], list[dict]]:
        inline_context_overrides(rows)
        clean = strip_internal_fields(rows, rules.gt_cols, _EXTRA_KEEP + [_CTX_PUBLIC_KEY])
        return clean, rows

    train_gt = _dedup_rows([_prep_base_valid(v, rules) for v in xv_train], rules.gt_cols)
    dtest_gt = _dedup_rows([_prep_base_valid(v, rules) for v in xv_test], rules.gt_cols)
    ptest_gt = _dedup_rows([_prep_base_valid(v, rules) for v in xv_test], rules.gt_cols)
    _say(f"  GT valids after dedup: train {len(train_gt)}, test {len(dtest_gt)} "
         f"(from {len(xv_train)} / {len(xv_test)})")

    # A dataset with no judge-visible event field cannot run pos_event even if the
    # flag (or the CLI default) still lists it — drop it here so the axis set the
    # fillers see always matches what the rules can actually build.
    pos_axes = tuple(a for a in flags.pos_axes
                     if not (a == "pos_event" and rules.event_field is None))
    if pos_axes != flags.pos_axes:
        _say(f"  pos_event dropped ({rules.name} has no judge-visible event field); "
             f"axes: {list(pos_axes)}")

    # ---- client-calling phase (runs first, contiguously) -------------------
    # `seen_sigs` seeded with the carried GT valids so a synthetic can't reproduce
    # one; the filler mutates it.
    train_syn = fill_positive_target(
        train_gt, rules, client, rng,
        floor=max(0, flags.valid_floor - len(train_gt)), axes=pos_axes,
        prompt_budget_multiple=flags.prompt_budget_multiple,
        seen_sigs={_row_signature(r, rules.gt_cols) for r in train_gt},
        label="train", quiet=quiet)
    diag_syn = fill_positive_target(
        dtest_gt, rules, client, rng,
        floor=max(0, flags.diag_valid_floor - len(dtest_gt)), axes=pos_axes,
        prompt_budget_multiple=flags.prompt_budget_multiple,
        seen_sigs={_row_signature(r, rules.gt_cols) for r in dtest_gt},
        label="diagnostic", quiet=quiet)

    # The record dry pass exists only to collect the gpt-oss prompts above; the
    # pure assembly below would just operate on empty synthetic sets.
    if getattr(client, "record", False):
        return {}

    # ---- assembly (no client calls past this point). Each file: all its valids
    #      (GT carried + distinct synthetic) balanced 1:1 with distinct negatives,
    #      the negative count split evenly across the constructible error types.
    def _build(pos: list[dict], name: str) -> list[dict]:
        types = active_error_types(rules, pos)
        quota = even_quota(len(pos), types)
        neg = fill_negative_quota(
            pos, rules, rng, quota_by_type=quota,
            seen_sigs={_row_signature(r, rules.gt_cols) for r in pos}, label=name)
        rows = assemble_file(pos, neg, rng)
        _say(f"  [{name}] {len(pos)} valid + {len(neg)} invalid = {len(rows)} rows; "
             f"negative types {quota}")
        return rows

    train_rows = _build(train_gt + train_syn, "train")
    ptest_rows = _build(ptest_gt, "primary-test")             # zero gpt-oss content
    diag_rows = _build(dtest_gt + diag_syn, "diagnostic-test")  # built like train

    return {
        "train": _finalize(train_rows),
        "primary_test": _finalize(ptest_rows),
        "diagnostic_test": _finalize(diag_rows),
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
        assert r.get("source_group_id") is not None, \
            f"[{key}] row {r['measurement_id']} missing source_group_id"
        override = r.get(_CTX_PUBLIC_KEY)
        assert isinstance(override, str) and override, \
            f"[{key}] row {r['measurement_id']} missing/empty {_CTX_PUBLIC_KEY} " \
            f"— every augmented row must carry the full-paper context"


def run_and_write(
    *, base_dir: Path, out_suffix: str, ocr_dir: Path,
    xv_train: list[dict], xv_test: list[dict], rules: DatasetAugmentRules,
    flags: AugmentFlags, client: AugmentClient, rng: random.Random,
    paper_code_key: str = "_paper_code",
) -> dict[str, Path]:
    """End-to-end: attach the full source paper as each record's context, build
    the three files, write them (+ diff reports for splits with edited rows),
    assert well-formed. Returns ``{key: data_path}``."""
    base_dir = Path(base_dir)
    ocr_dir = Path(ocr_dir)

    doc_cache: dict[str, str] = {}
    for v in (*xv_train, *xv_test):
        code = v[paper_code_key]
        if code not in doc_cache:
            path = ocr_dir / f"{code}.txt"
            text = path.read_text(encoding="utf-8")
            assert text.strip(), f"OCR file is empty: {path}"
            doc_cache[code] = text
        v[_CTX_ORIG_KEY] = doc_cache[code]

    # gpt-oss: gather every prompt in an RNG-matched dry pass, then fill the cache
    # in one concurrent batch. Without this the real pass hits the model serially
    # (one event loop per call) and blows the generation-job walltime.
    if isinstance(client, GptOssClient):
        rng_state = rng.getstate()
        client.record = True
        build_augmented_files(xv_train=xv_train, xv_test=xv_test, rng=rng,
                              client=client, rules=rules, flags=flags, quiet=True)
        client.record = False
        jobs = list(client._pending.items())
        client._pending.clear()
        rng.setstate(rng_state)
        print(f"  prewarm: {len(jobs)} gpt-oss call(s) to batch ({client.cache.stats})")
        client.prewarm(jobs)
        client.flush()          # persist GPU-generated responses before the real pass
        client.strict_cache = True
        n_dropped = len(client._dropped_detail)
        drop_note = f", {n_dropped} dropped after retries" if n_dropped else ""
        print(f"  prewarm done ({client.cache.stats}{drop_note})")

    bundles = build_augmented_files(xv_train=xv_train, xv_test=xv_test, rng=rng,
                                    client=client, rules=rules, flags=flags)
    client.flush()

    # Every output row now carries `context_override` (the full paper); a row is
    # "edited" only when the augmentation actually changed that paper.
    def _is_edited(r: dict) -> bool:
        e = r.get(_CTX_EDIT_KEY)
        return e is not None and e != r.get(_CTX_ORIG_KEY)

    written: dict[str, Path] = {}
    for key, data_fmt in _OUT_NAMES.items():
        rows, raw_rows = bundles[key]
        assert_wellformed(key, rows)
        data_path = base_dir / data_fmt.format(s=out_suffix)
        with open(data_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        written[key] = data_path
        n_edited = sum(1 for r in raw_rows if _is_edited(r))
        n_syn_pos = sum(1 for r in raw_rows
                        if r["label"] == "valid" and r.get(_CTX_EDIT_KEY))
        n_unverified = sum(1 for r in raw_rows if r.get("_unverified_span"))
        print(f"  wrote {len(rows):,} rows ({n_edited:,} with an edited paper) -> "
              f"{data_path.name}")
        if n_syn_pos:
            print(f"    {n_unverified:,}/{n_syn_pos:,} synthetic positives rest on "
                  f"the model's self-consistency alone (original surface form not "
                  f"a verbatim paper span)")
        # Leakage check for the diagnostic file: if edited-context rows cluster on
        # one label, a probe can score above chance on edited-ness alone.
        if n_edited and any(r["label"] == "invalid" and _is_edited(r) for r in raw_rows):
            for lab in ("valid", "invalid"):
                lab_rows = [r for r in raw_rows if r["label"] == lab]
                n_lab_edited = sum(1 for r in lab_rows if _is_edited(r))
                print(f"    {lab:>7}: {n_lab_edited}/{len(lab_rows)} on an edited "
                      f"paper ({100 * n_lab_edited / len(lab_rows):.0f}%)")
        if n_edited:
            n_diff = emit_context_diff_report(raw_rows, base_dir / (data_path.name + ".diff.txt"))
            print(f"  wrote diff report for {n_diff} edited row(s) -> {data_path.name}.diff.txt")
    return written
