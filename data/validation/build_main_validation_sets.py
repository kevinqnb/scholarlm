"""Build the main-track validation sets (pond, supermat; 2026-10-01) that extend
the 100-ish pilot sets to ~TARGET_TOTAL measurements each.

Same extraction run as the pilot (``cfg.final_json`` from
build_pilot_validation_sets.py, asserted equal to the pilot file's recorded
``extraction_final_json``), same table-cleaned OCR. The main set is disjoint
from the pilot: no pilot measurement_id is ever drawn, so pilot + main together
make the ~500-point set, and the pilot's live judgements are unaffected.

Sampling, per dataset (all draws from one ``Random(SEED)`` stream, papers visited
in sorted order over the *whole* test split):
  1. Paper in the pilot (``sampled_document_ids``): draw up to PER_PILOT_PAPER
     non-pilot rows; paper not in the pilot: up to PER_NEW_PAPER rows. Fewer
     available -> take them all.
  2. If n_pilot + n_main < TARGET_TOTAL: for each of the r remaining slots,
     pick a paper uniformly from those with undrawn rows left and draw one row.

Papers with no rows in final.json contribute nothing.

Usage::

    python data/validation/build_main_validation_sets.py pond supermat
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from build_pilot_validation_sets import (
    NO_PROVENANCE_PAGE,
    SEED,
    TEST_SPLIT_FILE,
    TRAIN_SPLIT_FILE,
    _REPO_ROOT,
    _configs,
    _configs_other,
    _get_page_text,
    _load_split_doc_ids,
    _normalize_page_number,
)

METHOD_KEY = "main-gemma27b-full"
PILOT_METHOD_KEY = "pilot-gemma27b-full"
TARGET_TOTAL = 500
# (rows per paper already in the pilot, rows per paper not in the pilot)
PER_PAPER = {"pond": (3, 8), "supermat": (2, 7)}


def build(dataset: str, data_dir: Path = _REPO_ROOT / "data") -> None:
    cfg = next(c for c in _configs() + _configs_other() if c.dataset == dataset and c.method_key == PILOT_METHOD_KEY)
    ds_dir = _REPO_ROOT / "data" / dataset  # split files
    out_dir = data_dir / dataset  # pilot set read from / main set written to
    pilot = json.loads((out_dir / f"validation_set_{PILOT_METHOD_KEY}.json").read_text())
    assert pilot["extraction_final_json"] == str(cfg.final_json.resolve().relative_to(_REPO_ROOT.resolve())), (
        "pilot file was built from a different extraction than the config"
    )
    per_pilot, per_new = PER_PAPER[dataset]

    test_ids = _load_split_doc_ids(ds_dir / TEST_SPLIT_FILE)
    train_ids = _load_split_doc_ids(ds_dir / TRAIN_SPLIT_FILE)
    pilot_docs = set(pilot["sampled_document_ids"])
    pilot_mids = {m["measurement_id"] for m in pilot["measurements"]}
    assert pilot_docs <= test_ids
    n_pilot = len(pilot_mids)

    final = json.loads(cfg.final_json.read_text())
    pool: dict[str, list[dict]] = {}  # test-split rows not in the pilot, per paper
    for r in final:
        if r["document_id"] in test_ids and r["measurement_id"] not in pilot_mids:
            row = dict(r)
            row["page_number"] = _normalize_page_number(
                row.get("page_number"), method_key=METHOD_KEY, measurement_id=row["measurement_id"]
            )
            pool.setdefault(row["document_id"], []).append(row)
    for rows in pool.values():
        rows.sort(key=lambda r: r["measurement_id"])

    rng = random.Random(SEED)
    chosen: list[dict] = []
    for doc in sorted(test_ids):
        rows = pool.get(doc, [])
        k = min(per_pilot if doc in pilot_docs else per_new, len(rows))
        picked = rng.sample(rows, k)
        chosen.extend(picked)
        picked_ids = {r["measurement_id"] for r in picked}
        pool[doc] = [r for r in rows if r["measurement_id"] not in picked_ids]
    n_main_loop = len(chosen)

    remaining = TARGET_TOTAL - n_pilot - n_main_loop
    n_extra = 0
    while n_extra < remaining:
        open_docs = sorted(d for d, rows in pool.items() if rows)
        assert open_docs, f"{dataset}: ran out of rows with {remaining - n_extra} slots still to fill"
        doc = rng.choice(open_docs)
        r = rng.choice(pool[doc])
        pool[doc] = [x for x in pool[doc] if x["measurement_id"] != r["measurement_id"]]
        chosen.append(r)
        n_extra += 1

    chosen.sort(key=lambda r: r["measurement_id"])
    mids = [r["measurement_id"] for r in chosen]
    assert len(mids) == len(set(mids)) and not set(mids) & pilot_mids
    docs_used = {r["document_id"] for r in chosen}
    assert docs_used <= test_ids and docs_used.isdisjoint(train_ids)

    ocr_dir = _REPO_ROOT / cfg.ocr_dir
    documents: dict[str, dict] = {}
    texts: dict[str, str] = {}
    for r in chosen:
        doc = r["document_id"]
        text = texts.setdefault(doc, (ocr_dir / f"{doc}.txt").read_text())
        pages = documents.setdefault(doc, {"pages": {}, "full_text": text})["pages"]
        for p in r["page_number"]:
            pages[str(p)] = text if p == NO_PROVENANCE_PAGE else _get_page_text(text, p)
            assert pages[str(p)], f"empty page text: {doc} page {p}"

    measurements = []
    for r in chosen:
        m = {k: v for k, v in r.items() if k != "context"}
        m["sampled"] = True
        measurements.append(m)

    payload = {
        "dataset": f"{dataset}__{METHOD_KEY}",
        "probe_dataset": dataset,
        "method_key": METHOD_KEY,
        "method_label": "Main: full pipeline (gemma-3-27b)",
        "extraction_model": cfg.extraction_model,
        "extraction_date": cfg.extraction_date,
        "extraction_final_json": pilot["extraction_final_json"],
        "seed": SEED,
        "n_sample": len(measurements),
        "n_pool": sum(len(v) for v in pool.values()) + len(measurements),
        "n_sampled": len(measurements),
        "n_included": len(measurements),
        "n_pilot_excluded": n_pilot,
        "n_main_loop": n_main_loop,
        "n_main_extra": n_extra,
        "n_papers_with_measurements": len(docs_used),
        "sampled_document_ids": sorted(docs_used),
        "pilot_document_ids": sorted(pilot_docs),
        "documents": documents,
        "measurements": measurements,
    }
    out = out_dir / f"validation_set_{METHOD_KEY}.json"
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    short = [d for d in sorted(test_ids) if sum(r["document_id"] == d for r in chosen)
             < (per_pilot if d in pilot_docs else per_new)]
    print(f"[{dataset}] test papers {len(test_ids)} (pilot {len(pilot_docs)}); pilot n={n_pilot}")
    print(f"[{dataset}] main loop {n_main_loop}, extra {n_extra}, main total {len(measurements)}, "
          f"pilot+main {n_pilot + len(measurements)}")
    print(f"[{dataset}] papers short of quota in the loop: {len(short)}; papers with a main row: {len(docs_used)}")
    print(f"[{dataset}] wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="+", choices=sorted(PER_PAPER))
    ap.add_argument("--data-dir", type=Path, default=_REPO_ROOT / "data",
                    help="root holding <dataset>/validation_set_pilot-*.json (read) and receiving "
                         "validation_set_main-*.json (default: the repo's data/)")
    args = ap.parse_args()
    for ds in args.datasets:
        build(ds, args.data_dir)
