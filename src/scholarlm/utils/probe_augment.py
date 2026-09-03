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
  * ``build_context_overrides`` / ``emit_context_diff_report`` — side-car assembly
                             and the spot-check diff report.

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
    "You are a careful scientific text editor. You rewrite a page of text from a "
    "scientific paper so that one specific claim changes, while changing as "
    "little else as possible. Preserve layout, tables, sentence structure, and "
    "every other fact. If the requested change cannot be made without extensive "
    "rewriting or by introducing contradictions, respond with the JSON object "
    '{\"feasible\": false, \"reason\": \"...\"}. Otherwise respond with '
    '{\"feasible\": true, \"context\": \"<full rewritten page>\", '
    '\"replacements\": [[\"old span\", \"new span\"], ...]} and nothing else.'
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


def _extract_json_object(text: str) -> dict:
    """Parse the first top-level JSON object in ``text``; fail loud otherwise."""
    text = text.strip()
    # strip a ```json ... ``` fence if present
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError(f"gpt-oss response is not JSON: {text[:400]!r}")
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError(f"gpt-oss response is JSON but not an object: {text[:400]!r}")
    return obj


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
    max_tokens: int = 20000
    _pending: dict[str, tuple[str, list[dict]]] = field(default_factory=dict, repr=False)

    # -- low level -----------------------------------------------------------

    async def _one(self, client, messages: list[dict]) -> str:
        resp = await client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        content = resp.choices[0].message.content
        if not content:
            raise ValueError("gpt-oss returned an empty message")
        return content

    async def _run_batch(self, jobs: list[tuple[str, list[dict]]]) -> dict[str, str]:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key="EMPTY", base_url=self.api_base, timeout=600.0)
        sem = asyncio.Semaphore(self.max_concurrent)

        async def _guarded(key: str, messages: list[dict]) -> tuple[str, str]:
            async with sem:
                return key, await self._one(client, messages)

        results = await asyncio.gather(*(_guarded(k, m) for k, m in jobs))
        return dict(results)

    def _resolve(self, key: str, op: str, messages: list[dict]) -> str:
        """Return raw model text for ``key``, using the cache.

        NOTE: on a cache miss this runs a one-shot event loop for a single call
        — fully serial.  For a full generation run the orchestrator MUST call
        ``prewarm`` with every ``(key, messages)`` first so this only ever hits
        the cache (see the build note's remaining-work item 4).
        """
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        raw = asyncio.run(self._run_batch([(key, messages)]))[key]
        self.cache.put(key, raw)
        return raw

    def prewarm(self, jobs: list[tuple[str, list[dict]]]) -> None:
        """Fill the cache for a list of ``(cache_key, messages)`` in one batch."""
        misses = [(k, m) for k, m in jobs if self.cache.get(k) is None]
        if not misses:
            return
        for k, raw in asyncio.run(self._run_batch(misses)).items():
            self.cache.put(k, raw)

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
        obj = _extract_json_object(raw)
        out: dict[str, str | None] = {}
        for f in event_fields:
            v = obj.get(f)
            out[f] = v if (isinstance(v, str) and v.strip()) else None
        return out

    def _rewrite_messages(self, context: str, instruction: str) -> list[dict]:
        user = (
            f"{instruction}\n\n"
            f"Change as little as possible. Do not alter any other fact, number, "
            f"unit, name, table cell, or sentence.\n\n## PAGE TEXT\n{context}"
        )
        return [{"role": "system", "content": _REWRITE_SYS},
                {"role": "user", "content": user}]

    def rewrite_context(self, *, context, instruction, expected):
        payload = {"context": context, "instruction": instruction}
        key = _cache_key("rewrite", payload)
        raw = self._resolve(key, "rewrite", self._rewrite_messages(context, instruction))
        obj = _extract_json_object(raw)
        if not obj.get("feasible", False):
            return False, context, [], str(obj.get("reason", "infeasible"))
        new_ctx = obj.get("context")
        if not isinstance(new_ctx, str) or not new_ctx.strip():
            raise ValueError(f"gpt-oss rewrite marked feasible but returned no context: {raw[:400]!r}")
        reps_raw = obj.get("replacements", [])
        reps: list[tuple[str, str]] = []
        for pair in reps_raw:
            if not (isinstance(pair, list) and len(pair) == 2):
                raise ValueError(f"gpt-oss replacement is not a [old, new] pair: {pair!r}")
            reps.append((str(pair[0]), str(pair[1])))
        # Sanity-check the model actually did (roughly) what was asked: every
        # expected old-span must be gone from the rewrite and the new-span present.
        for old, new in expected:
            if old and old in new_ctx and old != new:
                return False, context, [], f"rewrite left the original span in place: {old!r}"
            if new and new not in new_ctx:
                return False, context, [], f"rewrite did not introduce the target span: {new!r}"
        return True, new_ctx, reps, ""


class StubAugmentClient:
    """Deterministic canned client for tests and the no-server smoke rung.

    ``event_fill`` returns null for every field (no event claimed).
    ``rewrite_context`` applies the ``expected`` ``(old, new)`` replacements
    directly, marking infeasible when an ``old`` span is absent — no LLM, no
    prompt parsing.  This is the same ``expected`` list the real client verifies
    the model's output against, so both clients take the same code path.
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
    """Apply ``(old, new)`` replacements, asserting each ``old`` occurs first.

    Raises ``ValueError`` if a span is not found — a claimed edit that does not
    land is a silent-wrong-result bug, so it must crash.
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


def build_context_overrides(rows: list[dict]) -> dict[str, str]:
    """``{str(measurement_id): edited_page_text}`` for rows whose context was edited."""
    out: dict[str, str] = {}
    for r in rows:
        edited = r.get(_CTX_EDIT_KEY)
        if edited is not None and edited != r.get(_CTX_ORIG_KEY):
            out[str(r["measurement_id"])] = edited
    return out


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
    the row carries ``_context_override`` (the rewritten page) and the updated
    measurement field(s); ``label='valid'``.
    """
    ctx = src[_CTX_ORIG_KEY]

    def _finish(axis: str, expected: list[tuple[str, str]], instr: str,
                field_updates: dict) -> dict | None:
        ok, new_ctx, _reps, _reason = client.rewrite_context(
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
                 f'"{old}" is instead attributed to "{new}". Keep the ecosystem type '
                 f'and every measured quantity identical.')
        return _finish("pos_entity", [(old, new)], instr, {rules.entity_name_field: new})

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
        new_name = rules.fabricated_name(src, rng)
        if not new_name:
            return None
        row = _new_derived_row(src, label="invalid", mod_type="hard_entity",
                               axis=src.get("augment_axis") if on_edited_context else None)
        row[rules.entity_name_field] = new_name
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
) -> dict[str, tuple[list[dict], dict[str, str], list[dict]]]:
    """Build the three augmented output files.

    Each input record must carry ``_context_original`` (the original page text),
    ``gt_row_index``, the entity / attribute / value / units fields, and any GT
    event fields.  Returns ``{file: (output_rows, side_car, rows_with_contexts)}``
    where ``rows_with_contexts`` still has the ``_context_*`` fields for the diff
    report; ``output_rows`` is already projected onto the file schema.
    """
    extra_keep = ["label", "modification_type", "gt_row_index", "donor_gt_row_index",
                  "measurement_id", "source_group_id", "augment_axis"]

    def _finalize(rows: list[dict]) -> tuple[list[dict], dict[str, str], list[dict]]:
        assign_measurement_ids(rows)
        side_car = build_context_overrides(rows)
        clean = strip_internal_fields(rows, rules.gt_cols, extra_keep)
        return clean, side_car, rows

    # ---- TRAIN --------------------------------------------------------------
    train_valids = [_prep_base_valid(v, rules) for v in xv_train]
    if flags.augment_events:
        n = event_fill_records(train_valids, client, rules)
        print(f"  [train] event-fill: {n} (record, field) pairs filled")
    axis_pos = _axis2_positives(train_valids, rules, client, rng, flags.pos_axes)
    print(f"  [train] axis-2 positives: {len(axis_pos)}")
    gt_neg = _gt_hard_negatives(train_valids, rules, rng)
    edit_neg = _matched_hard_negatives(axis_pos, rules, rng)
    positives = train_valids + axis_pos
    negatives = gt_neg + edit_neg
    train_rows, train_report = balance_and_cap(
        positives, negatives, target=flags.target_rows, floor=flags.floor_rows,
        max_derived_per_source=flags.max_derived_per_source, rng=rng,
    )
    print(f"  [train] {train_report}")

    # ---- PRIMARY TEST (no event-fill, no axis-2, GT-derived negatives) -----
    ptest_valids = [_prep_base_valid(v, rules) for v in xv_test]
    ptest_neg = _gt_hard_negatives(ptest_valids, rules, rng)
    ptest_rows, ptest_report = balance_and_cap(
        ptest_valids, ptest_neg, target=0, floor=0,
        max_derived_per_source=flags.max_derived_per_source, rng=rng,
    )
    print(f"  [primary-test] {ptest_report}")

    # ---- DIAGNOSTIC TEST (axis-2 positives + matched negatives on edits) --
    dtest_valids = [_prep_base_valid(v, rules) for v in xv_test]
    if flags.augment_events:
        event_fill_records(dtest_valids, client, rules)
    diag_axes = tuple(a for a in flags.pos_axes
                      if flags.diag_pos_event or a != "pos_event")
    dtest_pos = _axis2_positives(dtest_valids, rules, client, rng, diag_axes)
    dtest_neg = _matched_hard_negatives(dtest_pos, rules, rng)
    dtest_rows, dtest_report = balance_and_cap(
        dtest_pos, dtest_neg, target=0, floor=0,
        max_derived_per_source=flags.max_derived_per_source, rng=rng,
    )
    print(f"  [diagnostic-test] {dtest_report}")

    return {
        "train": _finalize(train_rows),
        "primary_test": _finalize(ptest_rows),
        "diagnostic_test": _finalize(dtest_rows),
    }


_OUT_NAMES = {
    # key: (data filename fmt, side-car filename fmt or None)
    "train": ("probe_dataset{s}.json", "probe_context_overrides{s}.json"),
    "primary_test": ("probe_dataset_test{s}.json", None),
    "diagnostic_test": ("probe_dataset_test{s}_diag.json",
                        "probe_context_overrides_test{s}_diag.json"),
}


def assert_wellformed(key: str, rows: list[dict], side_car: dict[str, str]) -> None:
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
    mid_set = {str(m) for m in mids}
    assert set(side_car) <= mid_set, \
        f"[{key}] side-car has keys not in this file: {sorted(set(side_car) - mid_set)[:5]}"


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
    (+ side-cars + diff reports), assert well-formed. Returns ``{key: data_path}``."""
    base_dir = Path(base_dir)
    ocr_dir = Path(ocr_dir)

    doc_cache: dict[str, str] = {}
    for v in (*xv_train, *xv_test):
        code = v[paper_code_key]
        if code not in doc_cache:
            doc_cache[code] = (ocr_dir / f"{code}.txt").read_text(encoding="utf-8")
        v[_CTX_ORIG_KEY] = extract_page_text(doc_cache[code], v.get(page_numbers_key) or [])

    bundles = build_augmented_files(
        xv_train=xv_train, xv_test=xv_test, rng=rng, client=client, rules=rules, flags=flags,
    )
    client.flush()

    written: dict[str, Path] = {}
    for key, (data_fmt, sc_fmt) in _OUT_NAMES.items():
        rows, side_car, rows_ctx = bundles[key]
        assert_wellformed(key, rows, side_car)
        data_path = base_dir / data_fmt.format(s=out_suffix)
        with open(data_path, "w") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        written[key] = data_path
        print(f"  wrote {len(rows):,} rows -> {data_path.name}")
        if sc_fmt is not None:
            sc_path = base_dir / sc_fmt.format(s=out_suffix)
            with open(sc_path, "w") as f:
                json.dump(side_car, f, indent=2, ensure_ascii=False, sort_keys=True)
            n_diff = emit_context_diff_report(rows_ctx, base_dir / (data_path.name + ".diff.txt"))
            print(f"  wrote {len(side_car):,} side-car contexts -> {sc_path.name} "
                  f"({n_diff} in the diff report)")
    return written
