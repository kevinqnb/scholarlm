"""
Quick script to retrieve an OpenAI batch job and inspect its status/errors.

Usage:
    python check_batch.py <batch_id>
    python check_batch.py              # lists recent batches to pick from
"""

import sys
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

client = OpenAI()  # reads OPENAI_API_KEY from environment


def retrieve_batch(batch_id: str):
    batch = client.batches.retrieve(batch_id)

    print(f"Batch ID:    {batch.id}")
    print(f"Status:      {batch.status}")
    print(f"Endpoint:    {batch.endpoint}")
    print(f"Created:     {batch.created_at}")
    print(f"Expires:     {batch.expires_at}")
    print()

    counts = batch.request_counts
    print(f"Requests:    {counts.total} total, {counts.completed} completed, {counts.failed} failed")
    print()

    if batch.errors and batch.errors.data:
        print("=== ERRORS ===")
        for err in batch.errors.data:
            print(f"  Code:    {err.code}")
            print(f"  Message: {err.message}")
            if err.line is not None:
                print(f"  Line:    {err.line}")
            print()
    else:
        print("No batch-level errors.")

    if batch.output_file_id:
        print(f"\nOutput file: {batch.output_file_id}")
    if batch.error_file_id:
        print(f"Error file:  {batch.error_file_id}")


def list_recent_batches(limit: int = 10):
    print(f"Recent batches (last {limit}):\n")
    for batch in client.batches.list(limit=limit):
        counts = batch.request_counts
        print(f"  {batch.id}  status={batch.status}  "
              f"total={counts.total} completed={counts.completed} failed={counts.failed}")
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        retrieve_batch(sys.argv[1])
    else:
        print("No batch_id provided. Listing recent batches...\n")
        list_recent_batches()
        print("Re-run with:  python check_batch.py <batch_id>")