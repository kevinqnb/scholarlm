"""OpenAI Batch API implementation.

Three-step flow:
    requests  = build_requests(chat_entries, model)
    batch_ids = submit_batch(requests, client=client)   # list[str], one per chunk
    poll_batch(batch_ids, client=client)
    results   = fetch_results(batch_ids, client=client, model=model)

Input size limit: 200 MB per JSONL file. submit_batch() splits automatically.
"""
from __future__ import annotations

import io
import json
import time
from typing import Any

from openai import OpenAI

from .common import chunk_by_size, normalize_bool_text

MAX_BYTES = 200 * 1024 * 1024  # 200 MB per file

# ─── Request building ─────────────────────────────────────────────────────────


def build_requests(
    chat_entries: list[dict[str, Any]],
    model: str,
    *,
    max_completion_tokens: int = 2048,
) -> list[dict]:
    """Convert chat entries to OpenAI batch JSONL records."""
    requests = []
    for entry in chat_entries:
        requests.append(
            {
                "custom_id": entry["custom_id"],
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": entry["system"]},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": entry["user"]}],
                        },
                    ],
                    "max_completion_tokens": max_completion_tokens,
                },
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
    metadata: dict | None = None,
) -> list[str]:
    """Split requests into ≤max_bytes chunks, upload each, and create batch jobs.

    Returns a list of batch ids (one per chunk).
    """
    chunks = chunk_by_size(requests, max_bytes)
    n = len(chunks)
    print(
        f"Submitting {len(requests)} requests to OpenAI Batch API"
        + (f" in {n} chunks" if n > 1 else "")
        + " ..."
    )

    batch_ids: list[str] = []
    for i, chunk in enumerate(chunks):
        label = f"chunk {i + 1}/{n} ({len(chunk)} requests)"
        batch_ids.append(
            _submit_one_chunk(chunk, client=client, chunk_label=label, metadata=metadata)
        )
    return batch_ids


# ─── Polling ─────────────────────────────────────────────────────────────────

TERMINAL_STATUSES = {"completed", "failed", "expired", "cancelled"}


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
