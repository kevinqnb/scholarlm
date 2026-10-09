"""File provenance helpers: content hashes and repo-relative paths, recorded in every
analysis manifest / cache sidecar so inputs can be checked for staleness later."""
from __future__ import annotations

import hashlib
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent


def repo_relative(path: Path) -> str:
    """path's string relative to the repo root, or its plain string if it
    isn't under the repo root (e.g. a tmp_path in tests) -- used for the
    match_cache.meta.json sidecar so it stays comparable across machines
    with the repo checked out at different absolute paths.
    """
    try:
        return str(path.relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    """Hex sha256 digest of a file's contents, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
