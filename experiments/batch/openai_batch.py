"""OpenAI Batch API implementation.

Three-step flow:
    requests  = build_requests(chat_entries, model)
    batch_ids = submit_batch(requests, client=client)   # list[str], one per chunk
    poll_batch(batch_ids, client=client)
    results   = fetch_results(batch_ids, client=client, model=model)

Input size limit: 200 MB per JSONL file, 40 M enqueued tokens across all active
batches. submit_batch() splits automatically and waits for in-flight batches to
drain before submitting new chunks when the token budget is tight.
"""
from __future__ import annotations

import io
import json
import time
from typing import Any

from openai import OpenAI

from .common import chunk_by_size, normalize_bool_text

MAX_BYTES = 200 * 1024 * 1024  # 200 MB per file
MAX_TOKENS = 40_000_000        # 40 M enqueued-token queue limit

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}

# ─── Token estimation ────────────────────────────────────────────────────────


def _estimate_tokens(request: dict) -> int:
    """Rough token estimate for one OpenAI batch request (4 chars ≈ 1 token).

    Counts characters in all message content fields and divides by 4, then
    adds max_completion_tokens for the expected output budget.
    """
    body = request.get("body", {})
    messages = body.get("messages", [])
    char_count = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            char_count += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict):
                    char_count += len(part.get("text", ""))
                elif isinstance(part, str):
                    char_count += len(part)
    input_tokens = max(1, char_count // 4)
    output_tokens = body.get("max_completion_tokens", 2048)
    return input_tokens + output_tokens


def _chunk_by_tokens(requests: list[dict], max_tokens: int) -> list[list[dict]]:
    """Split requests so each chunk's estimated token total stays within max_tokens.

    A single request that on its own exceeds max_tokens is placed in its own
    chunk (with a warning) rather than dropped.
    """
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for req in requests:
        req_tokens = _estimate_tokens(req)
        if current and current_tokens + req_tokens > max_tokens:
            chunks.append(current)
            current = []
            current_tokens = 0
        if not current and req_tokens > max_tokens:
            print(
                f"  Warning: single request estimates {req_tokens:,} tokens "
                f"(>{max_tokens:,}); submitting alone."
            )
        current.append(req)
        current_tokens += req_tokens

    if current:
        chunks.append(current)

    return chunks


def _poll_until_token_budget(
    in_flight: list[tuple[str, int]],
    needed_tokens: int,
    *,
    client: OpenAI,
    max_tokens: int,
    interval: int = 60,
) -> list[tuple[str, int]]:
    """Block until in-flight batches have drained enough to fit needed_tokens.

    Returns the updated in_flight list (completed batches removed).
    """
    while sum(t for _, t in in_flight) + needed_tokens > max_tokens:
        enqueued = sum(t for _, t in in_flight)
        print(
            f"  Token budget: {enqueued:,} enqueued + {needed_tokens:,} needed "
            f"> {max_tokens:,} limit. Waiting {interval}s for batches to drain ..."
        )
        time.sleep(interval)
        still_in_flight: list[tuple[str, int]] = []
        for batch_id, tokens in in_flight:
            batch = client.batches.retrieve(batch_id)
            if batch.status not in TERMINAL_STATUSES:
                still_in_flight.append((batch_id, tokens))
            else:
                print(f"  Batch {batch_id} finished (status={batch.status}), freeing ~{tokens:,} tokens.")
        in_flight = still_in_flight
    return in_flight


# ─── Request building ─────────────────────────────────────────────────────────


def build_requests(
    chat_entries: list[dict[str, Any]],
    model: str,
    *,
    max_completion_tokens: int = 100,
    temperature: float | None = None,
) -> list[dict]:
    """Convert chat entries to OpenAI batch JSONL records.

    ``temperature`` is omitted from the request body when ``None`` (the
    default) so that reasoning models that reject the parameter are not
    broken.
    """
    requests = []
    for entry in chat_entries:
        body: dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": entry["system"]},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": entry["user"]}],
                },
            ],
            "max_completion_tokens": max_completion_tokens,
        }
        if temperature is not None:
            body["temperature"] = temperature
        requests.append(
            {
                "custom_id": entry["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
    return requests


# ─── Submission ───────────────────────────────────────────────────────────────


def _submit_one_chunk(
    chunk: list[dict],
    *,
    client: OpenAI,
    chunk_label: str,
    metadata: dict | None,
) -> str:
    """Upload one JSONL chunk and create a batch job. Returns batch_id."""
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in chunk)
    buf = io.BytesIO(lines.encode("utf-8"))
    buf.name = "batch_requests.jsonl"

    file_obj = client.files.create(file=buf, purpose="batch")
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata=metadata or {},
    )
    print(f"  {chunk_label}: file={file_obj.id}  batch={batch.id}  status={batch.status}")
    return batch.id


def submit_batch(
    requests: list[dict],
    *,
    client: OpenAI,
    max_bytes: int = MAX_BYTES,
    max_tokens: int = MAX_TOKENS,
    metadata: dict | None = None,
) -> list[str]:
    """Split requests into chunks respecting the 200 MB file limit and the
    40 M enqueued-token limit, upload each chunk, and create batch jobs.

    Chunks are first split by byte size, then any chunk that still exceeds
    max_tokens is split further.  When the running token total of in-flight
    batches would exceed max_tokens, submission pauses until earlier batches
    complete before continuing.

    Returns a list of batch ids (one per chunk).
    """
    # 1. Split by file size.
    byte_chunks = chunk_by_size(requests, max_bytes)

    # 2. Further split any byte-chunk that exceeds the token limit.
    chunks: list[list[dict]] = []
    for bc in byte_chunks:
        sub = _chunk_by_tokens(bc, max_tokens)
        if len(sub) > 1:
            print(f"  One size-based chunk split into {len(sub)} sub-chunks to stay within token limit.")
        chunks.extend(sub)

    n = len(chunks)
    print(
        f"Submitting {len(requests)} requests to OpenAI Batch API"
        + (f" in {n} chunks" if n > 1 else "")
        + " ..."
    )

    # 3. Submit chunks, pausing when the enqueued-token budget is exhausted.
    batch_ids: list[str] = []
    in_flight: list[tuple[str, int]] = []  # (batch_id, estimated_tokens)

    for i, chunk in enumerate(chunks):
        chunk_tokens = sum(_estimate_tokens(r) for r in chunk)
        label = f"chunk {i + 1}/{n} ({len(chunk)} requests, ~{chunk_tokens:,} tokens)"

        in_flight = _poll_until_token_budget(
            in_flight, chunk_tokens, client=client, max_tokens=max_tokens
        )

        batch_id = _submit_one_chunk(chunk, client=client, chunk_label=label, metadata=metadata)
        batch_ids.append(batch_id)
        in_flight.append((batch_id, chunk_tokens))

    return batch_ids


# ─── Polling ─────────────────────────────────────────────────────────────────


def poll_batch(
    batch_ids: list[str],
    *,
    client: OpenAI,
    interval: int = 60,
    timeout: int = 86400,
) -> None:
    """Block until all batches complete (raises on any failure or timeout)."""
    pending = set(batch_ids)
    elapsed = 0

    while pending and elapsed < timeout:
        still_pending: set[str] = set()
        for batch_id in sorted(pending):
            batch = client.batches.retrieve(batch_id)
            status = batch.status
            counts = batch.request_counts
            print(
                f"  [{elapsed:>6}s] {batch_id}: status={status}"
                f"  total={counts.total}"
                f"  completed={counts.completed}"
                f"  failed={counts.failed}"
            )
            if status not in TERMINAL_STATUSES:
                still_pending.add(batch_id)
            elif status != "completed":
                raise RuntimeError(f"Batch {batch_id} ended with status={status}")

        pending = still_pending
        if pending:
            time.sleep(interval)
            elapsed += interval

    if pending:
        raise TimeoutError(
            f"{len(pending)} batch(es) did not complete within {timeout}s: {pending}"
        )


# ─── Result processing ────────────────────────────────────────────────────────


def _fetch_one_chunk(
    batch_id: str,
    *,
    client: OpenAI,
    model: str,
) -> dict[str, dict[str, Any]]:
    """Download and parse results for a single batch."""
    batch = client.batches.retrieve(batch_id)
    if not batch.output_file_id:
        raise RuntimeError(
            f"Batch {batch_id} has no output file (status={batch.status})"
        )

    content = client.files.content(batch.output_file_id)
    results: dict[str, dict[str, Any]] = {}
    n_errors = 0

    for line in content.text.splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        custom_id = rec["custom_id"]

        if rec.get("error"):
            n_errors += 1
            results[custom_id] = {
                "judgement": None,
                "prob": None,
                "model": model,
                "raw_text": "",
            }
            continue

        body = (rec.get("response") or {}).get("body") or {}
        choices = body.get("choices") or []
        raw = (
            ((choices[0].get("message") or {}).get("content") or "").strip()
            if choices
            else ""
        )
        results[custom_id] = {
            "judgement": normalize_bool_text(raw),
            "prob": None,
            "model": body.get("model", model),
            "raw_text": raw,
        }

    if n_errors:
        print(f"  Warning: {n_errors} failed requests in batch {batch_id}")
    return results


def fetch_results(
    batch_ids: list[str],
    *,
    client: OpenAI,
    model: str,
) -> dict[str, dict[str, Any]]:
    """Download and merge results from all batch chunks.

    Returns a dict keyed by custom_id:
        {"judgement": bool|None, "prob": None, "model": str, "raw_text": str}
    """
    merged: dict[str, dict[str, Any]] = {}
    for batch_id in batch_ids:
        print(f"Fetching results for {batch_id} ...")
        merged.update(_fetch_one_chunk(batch_id, client=client, model=model))
    return merged
