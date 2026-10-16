"""
Build the 3 matching validation sets (pond, nfix, supermat; gemma27b full
pipeline): 100 candidate ground-truth <-> extraction pairs each, drawn at random
from the run's cached match edges.

An *edge* is one entry of ``edges`` in
``analysis/results/match_cache/<experiment_id>/match_cache.pkl`` -- a
``(gt_index, extraction_index)`` pair that agrees on every strict field, cached
at fuzzy_threshold=0.0 together with its fuzzy score (``edge_weights``). The
sample is uniform over *all* cached edges (not just those at/above the dataset's
selected fuzzy threshold), so each pair also records its ``weight``,
``above_threshold`` and ``in_matching`` (kept by the one-to-one matching) for
later stratified analysis. The validator is shown none of that -- only the two
rows side by side.

Indices are positional into the files named in the cache's
``match_cache.meta.json`` (``postprocessed.json`` extraction; the ground-truth
file the cache was built with, e.g. supermat's ``ground_truth_qualifiers.json``),
both read with plain ``json.load``. The sha256s in the sidecar are asserted, and
every sampled edge's strict fields are re-checked for agreement (a failure there
means the indexing is off).

Output, one per run, next to the dataset's other files:
    data/{dataset}/matching_validation_set_<experiment_id>.json
Seeded (SEED) -- re-running reproduces the same files; do not re-run to
"refresh" a set that is already live.

Usage::

    python data/validation/build_matching_validation_sets.py [dataset ...] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import random
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT))

from analysis.match_cache import (  # noqa: E402
    _parse_numeric,
    get_matching_config,
    match_cache_meta_path,
    match_cache_path,
)
from experiments.run_extraction import load_dataset_config  # noqa: E402

SEED = 20261004
N_SAMPLE = 100

RUNS: dict[str, str] = {
    "pond": "2026-09-22-pond-extraction-gemma27b-parseqty-standardization-valueonly-01",
    "nfix": "2026-09-21-nfix-extraction-gemma27b-full-01",
    "supermat": "2026-09-21-supermat-extraction-gemma27b-full-01",
}

# Row keys that are bulky free text and never shown to the validator.
DROP_KEYS = {"context"}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(v):
    """JSON-safe: NaN/inf -> None."""
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v


def _row(r: dict) -> dict:
    return {k: _clean(v) for k, v in r.items() if k not in DROP_KEYS}


def _isnull(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _same_strict(gt_v, ex_v, numeric: bool) -> bool:
    if numeric:
        a, b = _parse_numeric(gt_v), _parse_numeric(ex_v)
        return a == b or (math.isnan(a) and math.isnan(b))  # pandas merge keys match NaN to NaN
    if _isnull(gt_v) and _isnull(ex_v):
        return True
    return gt_v == ex_v


def build(dataset: str, experiment_id: str, out_dir: Path) -> Path:
    meta = json.loads(match_cache_meta_path(experiment_id).read_text())
    gt_path = _REPO_ROOT / meta["ground_truth_file"]
    ex_path = _REPO_ROOT / meta["extraction_file"]
    pkl_path = match_cache_path(experiment_id)
    assert _sha256(gt_path) == meta["ground_truth_sha256"], f"{experiment_id}: GT changed since cache"
    assert _sha256(ex_path) == meta["extraction_sha256"], f"{experiment_id}: extraction changed since cache"

    gt = json.loads(gt_path.read_text())
    ex = json.loads(ex_path.read_text())
    assert len(gt) == meta["n_gt"], f"{experiment_id}: {len(gt)} GT rows, cache says {meta['n_gt']}"

    with open(pkl_path, "rb") as f:
        matching, edges, weights = pickle.load(f)
    assert len(edges) == len(weights)
    in_matching = {tuple(e) for e in matching}

    cfg = get_matching_config(load_dataset_config(dataset))
    strict, fuzzy, thr = cfg["strict"], cfg["fuzzy"], cfg["fuzzy_threshold"]
    numeric = set(cfg["numeric_coerce"])

    order = sorted(range(len(edges)))
    chosen = random.Random(SEED).sample(order, N_SAMPLE)
    chosen.sort(key=lambda i: tuple(edges[i]))

    pairs = []
    for pair_id, i in enumerate(chosen, start=1):
        gi, ei = edges[i]
        g, e = gt[gi], ex[ei]
        for gcol, ecol in strict.items():
            assert _same_strict(g.get(gcol), e.get(ecol), gcol in numeric), (
                f"{experiment_id}: edge {(gi, ei)} disagrees on strict field {gcol!r}: "
                f"{g.get(gcol)!r} vs {e.get(ecol)!r} -- indexing is off"
            )
        pairs.append({
            "pair_id": pair_id,
            "gt_index": gi,
            "extraction_index": ei,
            "weight": float(weights[i]),
            "above_threshold": bool(weights[i] >= thr),
            "in_matching": (gi, ei) in in_matching,
            "gt": _row(g),
            "extraction": _row(e),
        })

    payload = {
        "dataset": f"{dataset}__matching-gemma27b",
        "probe_dataset": dataset,
        "method_key": "matching-gemma27b",
        "method_label": "Matching: gemma27b full pipeline",
        "experiment_id": experiment_id,
        "seed": SEED,
        "n_sample": N_SAMPLE,
        "n_edges": len(edges),
        "n_edges_above_threshold": sum(1 for w in weights if w >= thr),
        "fuzzy_threshold": thr,
        "strict_fields": [{"gt": g, "extraction": e} for g, e in strict.items()],
        "fuzzy_fields": [{"gt": g, "extraction": e} for g, e in fuzzy.items()],
        "match_cache": str(pkl_path.relative_to(_REPO_ROOT)),
        "match_cache_sha256": _sha256(pkl_path),
        "ground_truth_file": meta["ground_truth_file"],
        "ground_truth_sha256": meta["ground_truth_sha256"],
        "extraction_file": meta["extraction_file"],
        "extraction_sha256": meta["extraction_sha256"],
        "pairs": pairs,
    }
    out = out_dir / dataset / f"matching_validation_set_{experiment_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")
    n_above = sum(p["above_threshold"] for p in pairs)
    n_match = sum(p["in_matching"] for p in pairs)
    print(f"[{dataset}] {len(edges)} edges ({payload['n_edges_above_threshold']} >= {thr:.4f}) -> "
          f"{N_SAMPLE} sampled ({n_above} above threshold, {n_match} in matching); wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("datasets", nargs="*", default=list(RUNS))
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data")
    args = ap.parse_args()
    for dataset in args.datasets:
        build(dataset, RUNS[dataset], args.out_dir)


if __name__ == "__main__":
    main()
