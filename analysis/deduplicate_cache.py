"""Compute and cache within-extraction duplicate edges, by experiment id.

The deduplication counterpart of analysis/match_cache.py. The only job of this
script: given an analysis config (analysis/analysis-configs/<id>.yaml) listing
experiment ids and a ``params.deduplicate_cache`` section, load each run's
extraction file, apply the section's explicit pre-normalisation, score EVERY
pair of rows that strict-matches under the same pairwise rule as
``scholarlm.utils.deduplication.pair_score`` (itself a mirror of
``match_datasets``' edge rule), and write the edges + weights to
analysis/results/deduplicate_cache/<config id>/<experiment id>/
deduplicate_cache.pkl, plus a deduplicate_cache.meta.json sidecar recording the
extraction file (repo-relative path + sha256), row count and the exact section
the cache was built with.

Like match_cache.py the cache is built at fuzzy_threshold=0.0: it stores every
strict-equal pair once with its mean fuzzy score, and a threshold is applied
afterwards with ``edges_above_threshold`` (inclusive ``>=``, the same boundary
as ``deduplicate_records`` and ``match_datasets``). Which deduplication
*technique* is applied to those edges (greedy, connected components, complete
linkage, ...) is deliberately NOT decided here -- this script only builds the
pair graph those techniques reason about. Always recomputes and overwrites.

Unlike match_cache.py the fields come from the analysis config, not a
DatasetConfig (there is no dataset-level notion of "the dedup fields"), so the
cache is keyed by (config id, experiment id): two configs with different field
sets over the same run never overwrite each other.

Edge indices are positional row indices (0..n_rows-1) into the extraction file
as loaded; the sidecar's sha256 pins which file that was. Edges are (i, j) with
i < j, sorted.

Pre-normalisation (the dedup function compares exactly what it is given, so it
is explicit and recorded, never inferred):
  * ``join_list_fields``: a list-valued column (e.g. ``qualifiers``) becomes the
    sorted '|'-joined string of its elements (empty list -> '' == null). Null
    stays null; anything else is a hard error.
  * ``numeric_coerce``: a strict column whose values are numeric strings in some
    rows and floats in others (``point_value``) is parsed with match_cache's
    ``parse_numeric``. Unlike match_cache, a value that does not parse is KEPT
    as its original string rather than turned into NaN -- NaN would make two
    different garbage strings (both "null") strict-equal to each other, which
    for dedup means merging unrelated rows. The count of kept-raw values is
    printed and stored in the sidecar.

Usage
-----
    python analysis/deduplicate_cache.py --config analysis/analysis-configs/<id>.yaml
    bash analysis/submit.sh deduplicate_cache <id> --walltime HH:MM:SS --omp N
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
import time
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from scholarlm.utils.deduplication import _block_key, _is_null, _validate, pair_score
from analysis.common.config import get_section
from analysis.common.dedup import (
    CACHE_SECTION as SECTION, CACHE_SECTION_KEYS as SECTION_KEYS, deduplicate_cache_meta_path,
    deduplicate_cache_path, load_deduplicate_cache_config,
)
from analysis.common.matching import edges_above_threshold, parse_numeric
from analysis.common.provenance import repo_relative, sha256_file
import utils as paths


def _join_list(v):
    if isinstance(v, list):
        if not all(isinstance(x, str) for x in v):
            raise TypeError(f"join_list_fields element is not all str: {v!r}")
        return "|".join(sorted(v))
    if _is_null(v):
        return None
    raise TypeError(f"join_list_fields value is neither list nor null: {v!r}")


def prepare_frame(records: list[dict], sec: dict, label: str) -> pd.DataFrame:
    """Extraction records -> the frame whose rows are scored (positional index).

    Applies the section's explicit pre-normalisation, then the same
    scalar-type validation ``deduplicate_records`` runs. Only strict+fuzzy
    columns are kept.
    """
    df = pd.DataFrame(records).reset_index(drop=True)
    cols = sec["strict_fields"] + sec["fuzzy_fields"]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{label}: columns missing from extraction file: {missing}")
    df = df[cols].copy()
    for c in sec["join_list_fields"]:
        df[c] = pd.Series([_join_list(v) for v in df[c]], index=df.index, dtype=object)
    for c in sec["numeric_coerce"]:
        out, kept_raw = [], []
        for v in df[c]:
            p = parse_numeric(v)
            if pd.isna(p) and not _is_null(v):
                kept_raw.append(v)
                out.append(v)
            else:
                out.append(None if pd.isna(p) else p)
        df[c] = pd.Series(out, index=df.index, dtype=object)
        if kept_raw:
            d = sorted(set(map(str, kept_raw)))
            print(f"{label}: {c!r}: {len(kept_raw)} value(s) ({len(d)} distinct) did not parse, kept as raw "
                  f"strings: {d[:10]}" + (" ..." if len(d) > 10 else ""))
    _validate(df, sec["strict_fields"], sec["fuzzy_fields"], 0.0, [])
    return df


def compute_edges(df: pd.DataFrame, strict_fields: list, fuzzy_fields: list) -> tuple:
    """Every pair (i < j) of rows with all strict fields equal, scored at
    threshold 0.0 by ``pair_score``. Returns (edges, weights, block_sizes)."""
    records = df.to_dict("records")
    blocks: dict = collections.defaultdict(list)
    for i, r in enumerate(records):
        blocks[_block_key(r, strict_fields)].append(i)
    block_sizes = sorted((len(b) for b in blocks.values()), reverse=True)
    assert sum(block_sizes) == len(df)
    n_pairs = sum(s * (s - 1) // 2 for s in block_sizes)
    print(f"  {len(blocks)} strict blocks, largest {block_sizes[0]}, {n_pairs} within-block pairs to score", flush=True)

    edges, weights = [], []
    for members in blocks.values():
        for a in range(len(members)):
            for b in range(a + 1, len(members)):
                i, j = members[a], members[b]
                eligible, score = pair_score(
                    records[i], records[j], strict_fields=strict_fields, fuzzy_fields=fuzzy_fields
                )
                if eligible:
                    edges.append((i, j))
                    weights.append(float(score))
    order = sorted(range(len(edges)), key=edges.__getitem__)
    edges = [edges[k] for k in order]
    weights = [weights[k] for k in order]
    assert all(i < j for i, j in edges) and len(set(edges)) == len(edges)
    assert all(0.0 <= w <= 1.0 for w in weights)
    return edges, weights, block_sizes, n_pairs


def summarise(n_rows: int, edges: list, weights: list, threshold: float) -> dict:
    """Connected-component structure of the graph at ``threshold``."""
    kept = edges_above_threshold(edges, weights, threshold)
    parent = list(range(n_rows))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in kept:
        parent[find(i)] = find(j)
    members: dict = collections.defaultdict(list)
    for r in range(n_rows):
        members[find(r)].append(r)
    n_edges_in = collections.Counter(find(i) for i, _ in kept)
    multi = {root: m for root, m in members.items() if len(m) > 1}
    non_clique = [root for root, m in multi.items() if n_edges_in[root] < len(m) * (len(m) - 1) // 2]
    return {
        "threshold": threshold,
        "n_edges": len(kept),
        "n_components": len(members),
        "n_components_size_gt1": len(multi),
        "largest_component": max(len(m) for m in members.values()),
        "n_non_clique_components": len(non_clique),
        "rows_in_non_clique_components": sum(len(multi[r]) for r in non_clique),
    }


def build_deduplicate_cache(config_id: str, experiment_id: str, sec: dict) -> Path:
    """Compute and cache the within-extraction duplicate graph for one id."""
    result_dir = paths.find_result_dir(experiment_id)
    extraction_file = result_dir / sec["extraction_file"]
    if not extraction_file.exists():
        raise FileNotFoundError(f"{experiment_id}: {sec['extraction_file']} does not exist at {extraction_file}")
    with open(extraction_file) as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{experiment_id}: {extraction_file} is not a non-empty list of records")

    t0 = time.time()
    print(f"{experiment_id}: {len(records)} rows from {repo_relative(extraction_file)}", flush=True)
    df = prepare_frame(records, sec, experiment_id)
    assert len(df) == len(records)
    edges, weights, block_sizes, n_pairs = compute_edges(df, sec["strict_fields"], sec["fuzzy_fields"])
    assert len(edges) <= n_pairs

    cache_path = deduplicate_cache_path(config_id, experiment_id)
    meta_path = deduplicate_cache_meta_path(config_id, experiment_id)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.unlink(missing_ok=True)  # never leave a stale sidecar next to a new pkl
    with open(cache_path, "wb") as f:
        pickle.dump({"edges": edges, "edge_weights": weights, "n_rows": len(df)}, f)

    summary = [summarise(len(df), edges, weights, t) for t in sec["summary_thresholds"]]
    meta = {
        "analysis_config_id": config_id,
        "experiment_id": experiment_id,
        "extraction_file": repo_relative(extraction_file),
        "extraction_sha256": sha256_file(extraction_file),
        "n_rows": len(df),
        "section": sec,
        "n_blocks": len(block_sizes),
        "largest_block": block_sizes[0],
        "n_candidate_pairs": n_pairs,
        "n_edges_at_threshold_0": len(edges),
        "summary": summary,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"{experiment_id}: {len(edges)} strict-equal edges cached at threshold=0.0 -> {cache_path} "
          f"({time.time() - t0:.0f}s)")
    for s in summary:
        print(f"{experiment_id}:   threshold {s['threshold']}: {s['n_edges']} edges, "
              f"{s['n_components_size_gt1']} multi-row components (largest {s['largest_component']}), "
              f"{s['n_non_clique_components']} non-clique ({s['rows_in_non_clique_components']} rows), "
              f"{s['n_components']} components total")
    return cache_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True,
                        help="analysis-configs/<id>.yaml with params.experiment_ids and params.deduplicate_cache")
    args = parser.parse_args()
    cfg = load_deduplicate_cache_config(args.config)
    sec = get_section(cfg, SECTION, SECTION_KEYS)
    for experiment_id in cfg["params"]["experiment_ids"]:
        build_deduplicate_cache(cfg["id"], experiment_id, sec)


if __name__ == "__main__":
    main()
