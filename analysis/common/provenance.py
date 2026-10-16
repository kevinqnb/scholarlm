"""File provenance helpers: content hashes and repo-relative paths.

Recorded in analysis manifests and cache sidecars so stale inputs can be detected later.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent


def repo_relative(path: Path) -> str:
    """Path as a string relative to the repo root, so sidecars compare across checkouts.

    Args:
        path: File path to convert.

    Returns:
        Repo-relative path string, or ``str(path)`` if it is outside the repo
        (e.g. a test tmp dir).
    """
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's contents, read in 64 KiB chunks.

    Args:
        path: File to hash.

    Returns:
        Hex digest string.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
