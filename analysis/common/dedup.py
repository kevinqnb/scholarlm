"""Deduplication caches and outputs: where analysis/deduplicate_cache.py writes each
extraction's pairwise duplicate-candidate graph and analysis/deduplication.py its
clustering, the two config loaders, and the cache reader. Shared with
analysis/common/meta_inputs.py (rows: deduplicated) and analysis/_resolve_job.py.

Building either artefact stays in its script; nothing here computes edges or clusters.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent.parent
for _p in (_REPO_ROOT / "src", _REPO_ROOT / "experiments", _REPO_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from analysis.common.config import ANALYSIS_CONFIGS_ROOT, _load_envelope, get_section  # noqa: E402
from analysis.common.matching import edges_above_threshold  # noqa: E402

DEDUP_CACHE_ROOT = _REPO_ROOT / "analysis" / "results" / "deduplicate_cache"

CACHE_SECTION = "deduplicate_cache"

CACHE_SECTION_KEYS = (
    "extraction_file", "strict_fields", "fuzzy_fields", "numeric_coerce",
    "join_list_fields", "summary_thresholds",
)

EXTRACTION_FILES = ("final.json", "postprocessed.json")

def deduplicate_cache_path(config_id: str, experiment_id: str) -> Path:
    """DEDUP_CACHE_ROOT/<config id>/<experiment id>/deduplicate_cache.pkl -- the
    one place this path is built."""
    return DEDUP_CACHE_ROOT / config_id / experiment_id / "deduplicate_cache.pkl"

def deduplicate_cache_meta_path(config_id: str, experiment_id: str) -> Path:
    return deduplicate_cache_path(config_id, experiment_id).with_name("deduplicate_cache.meta.json")

def load_deduplicate_cache_config(path: Path) -> dict:
    """Load and validate an analysis config for deduplicate_cache.

    Envelope (id == filename stem, id/project/description/seed/params) as for
    every analysis config; ``seed`` is required but unused (the build is
    deterministic). ``params`` holds exactly ``experiment_ids`` (non-empty,
    unique strings) and the ``deduplicate_cache`` section (CACHE_SECTION_KEYS, no
    defaults, no extras).

    Raises:
        ValueError / KeyError on anything malformed.
    """
    cfg = _load_envelope(path)
    unexpected = set(cfg["params"]) - {"experiment_ids", CACHE_SECTION}
    if unexpected:
        raise ValueError(f"{path}: unexpected params key(s) {sorted(unexpected)}")
    ids = cfg["params"].get("experiment_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"{path}: params.experiment_ids must be a non-empty list of strings, got {ids!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: params.experiment_ids has duplicates")

    sec = get_section(cfg, CACHE_SECTION, CACHE_SECTION_KEYS)
    if sec["extraction_file"] not in EXTRACTION_FILES:
        raise ValueError(f"{path}: extraction_file must be one of {EXTRACTION_FILES}, got {sec['extraction_file']!r}")
    for key in ("strict_fields", "fuzzy_fields", "numeric_coerce", "join_list_fields"):
        v = sec[key]
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v) or len(set(v)) != len(v):
            raise ValueError(f"{path}: {CACHE_SECTION}.{key} must be a list of unique strings, got {v!r}")
    if not sec["strict_fields"] or not sec["fuzzy_fields"]:
        raise ValueError(f"{path}: {CACHE_SECTION}.strict_fields and fuzzy_fields must be non-empty")
    both = set(sec["strict_fields"]) & set(sec["fuzzy_fields"])
    if both:
        raise ValueError(f"{path}: columns both strict and fuzzy: {sorted(both)}")
    if not set(sec["numeric_coerce"]) <= set(sec["strict_fields"]):
        raise ValueError(f"{path}: numeric_coerce must be a subset of strict_fields")
    if not set(sec["join_list_fields"]) <= set(sec["strict_fields"]) | set(sec["fuzzy_fields"]):
        raise ValueError(f"{path}: join_list_fields must be a subset of strict_fields + fuzzy_fields")
    th = sec["summary_thresholds"]
    if not isinstance(th, list) or not th or not all(
        isinstance(t, (int, float)) and not isinstance(t, bool) and 0.0 <= t <= 1.0 for t in th
    ):
        raise ValueError(f"{path}: {CACHE_SECTION}.summary_thresholds must be a non-empty list of numbers in [0, 1]")
    return cfg

def load_deduplicate_cache(config_id: str, experiment_id: str, fuzzy_threshold: float | None = None):
    """Load a built cache. Returns the dict {"edges","edge_weights","n_rows"},
    or, with ``fuzzy_threshold``, just the edges with weight >= it. Never
    builds anything; raises if the cache has not been built."""
    path = deduplicate_cache_path(config_id, experiment_id)
    if not path.exists():
        raise FileNotFoundError(
            f"No deduplicate cache for ({config_id!r}, {experiment_id!r}) at {path}; "
            f"run `python analysis/deduplicate_cache.py --config analysis/analysis-configs/{config_id}.yaml`"
        )
    with open(path, "rb") as f:
        data = pickle.load(f)
    if fuzzy_threshold is None:
        return data
    return edges_above_threshold(data["edges"], data["edge_weights"], fuzzy_threshold)


DEDUP_ROOT = _REPO_ROOT / "analysis" / "results" / "deduplication"

DEDUP_SECTION = "deduplication"

DEDUP_SECTION_KEYS = ("deduplicate_cache_config_id", "solver_time_limit_s", "provenance_fields")

def deduplication_dir(config_id: str, experiment_id: str) -> Path:
    """DEDUP_ROOT/<config id>/<experiment id>/ -- the one place this path is built."""
    return DEDUP_ROOT / config_id / experiment_id

def load_deduplication_config(path: Path) -> dict:
    """Load and validate an analysis config for deduplication.

    ``params`` holds exactly ``experiment_ids`` and the ``deduplication`` section
    (DEDUP_SECTION_KEYS, no defaults, no extras). The named deduplicate_cache config must
    exist and list every experiment id here; the provenance fields must not be match
    fields of that config (a field cannot both decide duplication and be merged).
    ``seed`` is required by the envelope and unused (the solve is deterministic).
    """
    cfg = _load_envelope(path)
    unexpected = set(cfg["params"]) - {"experiment_ids", DEDUP_SECTION}
    if unexpected:
        raise ValueError(f"{path}: unexpected params key(s) {sorted(unexpected)}")
    ids = cfg["params"].get("experiment_ids")
    if not isinstance(ids, list) or not ids or not all(isinstance(x, str) for x in ids):
        raise ValueError(f"{path}: params.experiment_ids must be a non-empty list of strings, got {ids!r}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: params.experiment_ids has duplicates")

    sec = get_section(cfg, DEDUP_SECTION, DEDUP_SECTION_KEYS)
    cache_id = sec["deduplicate_cache_config_id"]
    if not isinstance(cache_id, str) or not cache_id:
        raise ValueError(f"{path}: {DEDUP_SECTION}.deduplicate_cache_config_id must be a non-empty string")
    tl = sec["solver_time_limit_s"]
    if isinstance(tl, bool) or not isinstance(tl, (int, float)) or tl <= 0:
        raise ValueError(f"{path}: {DEDUP_SECTION}.solver_time_limit_s must be a positive number, got {tl!r}")
    pf = sec["provenance_fields"]
    if not isinstance(pf, list) or not pf or not all(isinstance(x, str) for x in pf) or len(set(pf)) != len(pf):
        raise ValueError(f"{path}: {DEDUP_SECTION}.provenance_fields must be a non-empty list of unique strings")

    cache_cfg_path = ANALYSIS_CONFIGS_ROOT / f"{cache_id}.yaml"
    if not cache_cfg_path.exists():
        raise FileNotFoundError(f"{path}: deduplicate_cache config {cache_cfg_path} does not exist")
    cache_cfg = load_deduplicate_cache_config(cache_cfg_path)
    not_cached = sorted(set(ids) - set(cache_cfg["params"]["experiment_ids"]))
    if not_cached:
        raise ValueError(f"{path}: experiment ids not in cache config {cache_id}: {not_cached}")
    match_fields = set(cache_cfg["params"][CACHE_SECTION]["strict_fields"]) | set(
        cache_cfg["params"][CACHE_SECTION]["fuzzy_fields"])
    both = sorted(set(pf) & match_fields)
    if both:
        raise ValueError(f"{path}: provenance_fields are also match fields of {cache_id}: {both}")
    return cfg
