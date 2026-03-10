"""Anthropic Message Batches API implementation.

Three-step flow:
    requests  = build_requests(chat_entries, model)
    batch_ids = submit_batch(requests, client=client)   # list[str], one per chunk
    poll_batch(batch_ids, client=client)
    results   = fetch_results(batch_ids, client=client, model=model)

Input size limit: 256 MB per batch (total JSON payload). submit_batch() splits automatically.
"""
from __future__ import annotations

import time
from typing import Any

import anthropic

from .common import chunk_by_size, normalize_bool_text

MAX_BYTES = 256 * 1024 * 1024  # 256 MB per batch payload

# ─── Request building ─────────────────────────────────────────────────────────


def build_requests(
    chat_entries: list[dict[str, Any]],
    model: str,
    *,
    max_tokens: int = 5,
    temperature: float = 0.0,
) -> list[dict]:
    """Convert chat entries to Anthropic batch request objects."""
    requests = []
    for entry in chat_entries:
        requests.append(
            {
                "custom_id": entry["custom_id"],
                "params": {
                    "model": model,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "system": entry["system"],
                    "messages": [{"role": "user", "content": entry["user"]}],
                },
            }
        )
    return requests


# ─── Submission ───────────────────────────────────────────────────────────────


def _submit_one_chunk(
    chunk: list[dict],
    *,
    client: anthropic.Anthropic,
    chunk_label: str,
) -> str:
    """Create one Anthropic message batch. Returns the batch id."""
    batch = client.messages.batches.create(requests=chunk)
    print(f"  {chunk_label}: batch={batch.id}  status={batch.processing_status}")
    return batch.id


def submit_batch(
    requests: list[dict],
    *,
    client: anthropic.Anthropic,
    max_bytes: int = MAX_BYTES,
) -> list[str]:
    """Split requests into ≤max_bytes chunks and submit each as a separate batch.

    Returns a list of batch ids (one per chunk).
    """
    chunks = chunk_by_size(requests, max_bytes)
    n = len(chunks)
    print(
        f"Submitting {len(requests)} requests to Anthropic Batch API"
        + (f" in {n} chunks" if n > 1 else "")
        + " ..."
    )

    batch_ids: list[str] = []
    for i, chunk in enumerate(chunks):
        label = f"chunk {i + 1}/{n} ({len(chunk)} requests)"
        batch_ids.append(_submit_one_chunk(chunk, client=client, chunk_label=label))
    return batch_ids


# ─── Polling ─────────────────────────────────────────────────────────────────


def poll_batch(
    batch_ids: list[str],
    *,
    client: anthropic.Anthropic,
    interval: int = 30,
    timeout: int = 86400,
) -> None:
    """Block until all batches end (raises on timeout)."""
    pending = set(batch_ids)
    elapsed = 0

    while pending and elapsed < timeout:
        still_pending: set[str] = set()
        for batch_id in sorted(pending):
            batch = client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            counts = batch.request_counts
            print(
                f"  [{elapsed:>6}s] {batch_id}: status={status}"
                f"  processing={counts.processing}"
                f"  succeeded={counts.succeeded}"
                f"  errored={counts.errored}"
                f"  canceled={counts.canceled}"
                f"  expired={counts.expired}"
            )
            if status != "ended":
                still_pending.add(batch_id)

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
    client: anthropic.Anthropic,
    model: str,
) -> dict[str, dict[str, Any]]:
    """Stream and parse results for a single batch."""
    results: dict[str, dict[str, Any]] = {}
    n_errors = 0

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        outcome = result.result

        if outcome.type == "succeeded":
            raw = ""
            for block in outcome.message.content or []:
                if hasattr(block, "text"):
                    raw += block.text
            raw = raw.strip()
            results[custom_id] = {
                "judgement": normalize_bool_text(raw),
                "prob": None,
                "model": outcome.message.model,
                "raw_text": raw,
            }
        else:
            # errored / canceled / expired
            n_errors += 1
            results[custom_id] = {
                "judgement": None,
                "prob": None,
                "model": model,
                "raw_text": "",
            }

    if n_errors:
        print(f"  Warning: {n_errors} non-succeeded requests in batch {batch_id}")
    return results


def fetch_results(
    batch_ids: list[str],
    *,
    client: anthropic.Anthropic,
    model: str,
) -> dict[str, dict[str, Any]]:
    """Stream and merge results from all batch chunks.

    Returns a dict keyed by custom_id:
        {"judgement": bool|None, "prob": None, "model": str, "raw_text": str}
    """
    merged: dict[str, dict[str, Any]] = {}
    for batch_id in batch_ids:
        print(f"Fetching results for {batch_id} ...")
        merged.update(_fetch_one_chunk(batch_id, client=client, model=model))
    return merged
