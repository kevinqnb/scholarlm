"""Google Gemini Batch API implementation.

Three-step flow:
    requests    = build_requests(chat_entries, model)
    batch_names = submit_batch(requests, model, dest_gcs="gs://bucket/out/",
                               project="my-project", location="us-central1")
    poll_batch(batch_names, project="my-project", location="us-central1")
    results     = fetch_results(batch_names, model, dest_gcs="gs://bucket/out/")

Input size limit: 2 GB per source file. submit_batch() splits automatically.

Authentication:
    The Gemini Batch API requires Google Cloud credentials (OAuth / service
    account), NOT a plain GEMINI_API_KEY. Run once to set up local credentials:

        gcloud auth application-default login

    Then pass your GCP project and location to submit_batch() and poll_batch(),
    or set GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION env vars.

    Input JSONL chunks are uploaded to {dest_gcs}input/ via google-cloud-storage
    (also uses ADC). Output is written by Gemini under dest_gcs.

    uv add google-genai google-cloud-storage
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from .common import chunk_by_size, normalize_bool_text

MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB per source file


# ─── Client helpers ───────────────────────────────────────────────────────────


def _make_client(project: str, location: str) -> Any:
    """Build a Vertex-AI-mode genai client using Application Default Credentials."""
    from google import genai  # type: ignore

    return genai.Client(vertexai=True, project=project, location=location)


def _gcs_upload(content: str, gcs_uri: str, project: str) -> None:
    """Upload UTF-8 text to a GCS URI using Application Default Credentials."""
    from google.cloud import storage  # type: ignore

    bucket_name, _, blob_path = gcs_uri[5:].partition("/")
    gcs_client = storage.Client(project=project)
    gcs_client.bucket(bucket_name).blob(blob_path).upload_from_string(
        content.encode("utf-8"), content_type="application/jsonl"
    )


def _resolve_project_location(
    project: str | None, location: str | None
) -> tuple[str, str]:
    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT", "")
    location = location or os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    if not project:
        raise RuntimeError(
            "GCP project is required. Pass --gcp-project or set GOOGLE_CLOUD_PROJECT."
        )
    return project, location


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
    project: str,
    src_gcs_prefix: str,
    dest_gcs: str,
    chunk_label: str,
    chunk_index: int,
) -> str:
    """Upload one JSONL chunk to GCS and create a batch job. Returns job name."""
    from google.genai import types  # type: ignore

    src_uri = f"{src_gcs_prefix.rstrip('/')}/chunk_{chunk_index:04d}.jsonl"
    lines = "\n".join(json.dumps(r, ensure_ascii=False) for r in chunk)
    print(f"  {chunk_label}: uploading input to {src_uri} ...")
    _gcs_upload(lines, src_uri, project)

    prefixed_model = model if model.startswith("models/") else f"models/{model}"
    batch = client.batches.create(
        model=prefixed_model,
        src=src_uri,
        config=types.CreateBatchJobConfig(dest=dest_gcs),
    )
    print(f"  {chunk_label}: job={batch.name}  state={batch.state}")
    return batch.name


def submit_batch(
    requests: list[dict],
    model: str,
    *,
    dest_gcs: str,
    project: str | None = None,
    location: str | None = None,
    display_name: str = "judge_batch",
    max_bytes: int = MAX_BYTES,
) -> list[str]:
    """Split requests into ≤max_bytes chunks, upload each to GCS, and create batch jobs.

    Input chunks are written to {dest_gcs}input/ so only one GCS path is needed.
    Returns a list of batch job names (one per chunk).

    Args:
        requests:     list of dicts from build_requests().
        model:        Gemini model, e.g. "gemini-2.5-flash-lite".
        dest_gcs:     GCS URI for results, e.g. "gs://my-bucket/output/".
        project:      GCP project id (or set GOOGLE_CLOUD_PROJECT env var).
        location:     GCP region, default "us-central1" (or GOOGLE_CLOUD_LOCATION).
        display_name: Label prefix for input files in GCS.
        max_bytes:    Per-file size cap; defaults to the 2 GB API limit.
    """
    project, location = _resolve_project_location(project, location)
    client = _make_client(project, location)

    src_gcs_prefix = f"{dest_gcs.rstrip('/')}/input/{display_name}"
    chunks = chunk_by_size(requests, max_bytes)
    n = len(chunks)
    print(
        f"Submitting {len(requests)} requests to Gemini Batch API"
        + (f" in {n} chunks" if n > 1 else "")
        + f" (project={project}, location={location}) ..."
    )

    batch_names: list[str] = []
    for i, chunk in enumerate(chunks):
        label = f"chunk {i + 1}/{n} ({len(chunk)} requests)"
        batch_names.append(
            _submit_one_chunk(
                chunk,
                model,
                client=client,
                project=project,
                src_gcs_prefix=src_gcs_prefix,
                dest_gcs=dest_gcs,
                chunk_label=label,
                chunk_index=i,
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
    project: str | None = None,
    location: str | None = None,
    interval: int = 60,
    timeout: int = 86400,
) -> None:
    """Block until all batch jobs complete (raises on failure or timeout)."""
    project, location = _resolve_project_location(project, location)
    client = _make_client(project, location)

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
    project: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Download and parse results for all chunks from GCS.

    Scans the entire dest_gcs prefix for output JSONL files, excluding the
    input/ subdirectory where source files were uploaded.

    Returns a dict keyed by key/custom_id:
        {"judgement": bool|None, "prob": None, "model": str, "raw_text": str}
    """
    from google.cloud import storage  # type: ignore

    if not dest_gcs.startswith("gs://"):
        raise ValueError(f"dest_gcs must start with gs://, got: {dest_gcs}")

    project = project or os.environ.get("GOOGLE_CLOUD_PROJECT") or None
    bucket_name, _, prefix = dest_gcs[5:].partition("/")
    gcs_client = storage.Client(project=project)
    bucket = gcs_client.bucket(bucket_name)

    output_prefix = prefix.rstrip("/") + "/"
    input_prefix = output_prefix + "input/"

    blobs = [
        b for b in bucket.list_blobs(prefix=output_prefix)
        if b.name.endswith(".jsonl") and not b.name.startswith(input_prefix)
    ]
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
