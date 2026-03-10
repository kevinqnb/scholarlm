"""Google Gemini Batch API implementation.

Three-step flow:
    requests    = build_requests(chat_entries, model)
    batch_names = submit_batch(requests, model, dest_gcs="gs://bucket/out/")
    poll_batch(batch_names)
    results     = fetch_results(batch_names, model, dest_gcs="gs://bucket/out/")

Input size limit: 2 GB per source file. submit_batch() splits automatically.

Requirements:
  - GEMINI_API_KEY env var
  - A GCS bucket for output: Gemini writes results under dest_gcs.
    Each chunk creates its own subdirectory keyed by job name, so all
    chunks under the same dest_gcs prefix are found by fetch_results.
  - uv add google-genai google-cloud-storage
"""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any

from .common import chunk_by_size, normalize_bool_text

MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB per source file


# ─── Request building ─────────────────────────────────────────────────────────


def build_requests(
    chat_entries: list[dict[str, Any]],
    model: str,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 5,
) -> list[dict]:
    """Convert chat entries to Gemini batch JSONL records.

    The ``model`` parameter is accepted for a consistent interface but is set
    at the job level in submit_batch(), not per-record.
    """
    requests = []
    for entry in chat_entries:
        requests.append(
            {
                "key": entry["custom_id"],
                "request": {
                    "contents": [
                        {"role": "user", "parts": [{"text": entry["user"]}]}
                    ],
                    "systemInstruction": {
                        "parts": [{"text": entry["system"]}]
                    },
                    "generationConfig": {
                        "temperature": temperature,
                        "maxOutputTokens": max_output_tokens,
                        "responseMimeType": "text/plain",
                        "candidateCount": 1,
                    },
                },
            }
        )
    return requests


# ─── Submission ───────────────────────────────────────────────────────────────


def _submit_one_chunk(
    chunk: list[dict],
    model: str,
    *,
    client: Any,
    dest_gcs: str,
    display_name: str,
    chunk_label: str,
) -> str:
    """Upload one JSONL chunk to Files API and create a batch job. Returns job name."""
    from google.genai import types  # type: ignore

    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in chunk)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(lines)
        tmp_path = tmp.name

    try:
        uploaded = client.files.upload(
            file=Path(tmp_path),
            config=types.UploadFileConfig(
                display_name=display_name,
                mime_type="application/jsonl",
            ),
        )
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    prefixed_model = model if model.startswith("models/") else f"models/{model}"
    batch = client.batches.create(
        model=prefixed_model,
        src=uploaded.uri,
        config=types.CreateBatchJobConfig(dest=dest_gcs),
    )
    print(f"  {chunk_label}: file={uploaded.uri}  job={batch.name}  state={batch.state}")
    return batch.name


def submit_batch(
    requests: list[dict],
    model: str,
    *,
    dest_gcs: str,
    display_name: str = "judge_batch",
    max_bytes: int = MAX_BYTES,
) -> list[str]:
    """Split requests into ≤max_bytes chunks, upload each, and create batch jobs.

    Returns a list of batch job names (one per chunk).

    Args:
        requests:     list of dicts from build_requests().
        model:        Gemini model, e.g. "gemini-2.5-flash-lite".
        dest_gcs:     GCS URI for results, e.g. "gs://my-bucket/output/".
        display_name: Human-readable label for Files API uploads.
        max_bytes:    Per-file size cap; defaults to the 2 GB API limit.
    """
    import os

    from google import genai  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)
    chunks = chunk_by_size(requests, max_bytes)
    n = len(chunks)
    print(
        f"Submitting {len(requests)} requests to Gemini Batch API"
        + (f" in {n} chunks" if n > 1 else "")
        + " ..."
    )

    batch_names: list[str] = []
    for i, chunk in enumerate(chunks):
        label = f"chunk {i + 1}/{n} ({len(chunk)} requests)"
        chunk_display = f"{display_name}_{i + 1}" if n > 1 else display_name
        batch_names.append(
            _submit_one_chunk(
                chunk,
                model,
                client=client,
                dest_gcs=dest_gcs,
                display_name=chunk_display,
                chunk_label=label,
            )
        )
    return batch_names


# ─── Polling ─────────────────────────────────────────────────────────────────

TERMINAL_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_PAUSED",
}


def poll_batch(
    batch_names: list[str],
    *,
    interval: int = 60,
    timeout: int = 86400,
) -> None:
    """Block until all batch jobs complete (raises on failure or timeout)."""
    import os

    from google import genai  # type: ignore

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)
    pending = set(batch_names)
    elapsed = 0

    while pending and elapsed < timeout:
        still_pending: set[str] = set()
        for name in sorted(pending):
            batch = client.batches.get(name=name)
            state_key = str(batch.state).split(".")[-1]
            print(f"  [{elapsed:>6}s] {name}: state={state_key}")
            if state_key not in TERMINAL_STATES:
                still_pending.add(name)
            elif state_key != "JOB_STATE_SUCCEEDED":
                raise RuntimeError(f"Batch {name} ended with state={state_key}")

        pending = still_pending
        if pending:
            time.sleep(interval)
            elapsed += interval

    if pending:
        raise TimeoutError(
            f"{len(pending)} batch job(s) did not complete within {timeout}s: {pending}"
        )


# ─── Result processing ────────────────────────────────────────────────────────


def fetch_results(
    batch_names: list[str],
    model: str,
    *,
    dest_gcs: str,
) -> dict[str, dict[str, Any]]:
    """Download and parse results for all chunks from GCS.

    Each batch job writes its output under dest_gcs keyed by job name, so a
    single prefix scan captures results from all chunks. The ``key`` field in
    each output record matches the ``custom_id`` used throughout the pipeline.

    Returns a dict keyed by key/custom_id:
        {"judgement": bool|None, "prob": None, "model": str, "raw_text": str}
    """
    from google.cloud import storage  # type: ignore

    if not dest_gcs.startswith("gs://"):
        raise ValueError(f"dest_gcs must start with gs://, got: {dest_gcs}")

    bucket_name, _, prefix = dest_gcs[5:].partition("/")
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)

    prefix = prefix.rstrip("/") + "/"
    blobs = [b for b in bucket.list_blobs(prefix=prefix) if b.name.endswith(".jsonl")]
    if not blobs:
        raise RuntimeError(f"No output JSONL files found under {dest_gcs}")

    print(f"Found {len(blobs)} output file(s) under {dest_gcs}")
    results: dict[str, dict[str, Any]] = {}
    n_errors = 0

    for blob in blobs:
        for line in blob.download_as_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            key = rec.get("key", "")

            status_code = (rec.get("status") or {}).get("code", 0)
            if status_code != 0:
                n_errors += 1
                results[key] = {
                    "judgement": None,
                    "prob": None,
                    "model": model,
                    "raw_text": "",
                }
                continue

            candidates = (rec.get("response") or {}).get("candidates") or []
            raw = ""
            if candidates:
                parts = ((candidates[0].get("content") or {}).get("parts") or [])
                raw = "".join(p.get("text", "") for p in parts).strip()

            results[key] = {
                "judgement": normalize_bool_text(raw),
                "prob": None,
                "model": model,
                "raw_text": raw,
            }

    if n_errors:
        print(f"Warning: {n_errors} failed requests across all chunks")
    return results
