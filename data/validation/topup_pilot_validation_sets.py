"""Top up the gemma27b pilot validation sets to >= MIN_TOTAL measurements
(2026-09-29), *additively*: the existing sampled papers/measurements are kept
byte-for-byte (live judgements key on (dataset, measurement_id)), and extra
papers are drawn from the same test split until the total reaches MIN_TOTAL.

Why: build_pilot_validation_sets.py draws a fixed 20 papers, and papers with
<5 extracted rows leave the pilots short (pond 97, nfix 84, supermat 84).

Per dataset: candidate papers = test-split papers not already in the pilot and
with >=1 row in that method's final.json (papers with no rows can't help).
Shuffle candidates with ``Random(SEED)``; take them one at a time, drawing up
to N_PER_PAPER rows each (``Random(SEED)`` stream, sorted by measurement_id),
stopping once the total is >= MIN_TOTAL -- so the last paper may overshoot by
up to N_PER_PAPER-1.

Rewrites ``validation_set_<method_key>.json`` in place as a strict superset of
the old file (asserted); the old file is saved next to it as
``.pre-topup``. Load into a live DB with ``ingest --append``.

Usage::

    python data/validation/topup_pilot_validation_sets.py pond nfix supermat
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from pathlib import Path

from build_pilot_validation_sets import (
    N_PER_PAPER,
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

MIN_TOTAL = 100
METHOD_KEY = "pilot-gemma27b-full"


def topup(dataset: str) -> None:
    cfg = next(c for c in _configs() + _configs_other() if c.dataset == dataset and c.method_key == METHOD_KEY)
    ds_dir = _REPO_ROOT / "data" / dataset
    path = ds_dir / f"validation_set_{METHOD_KEY}.json"
    old = json.loads(path.read_text())
    assert not old.get("topup_document_ids"), f"{path} was already topped up"
    assert old["extraction_final_json"] == str(cfg.final_json.resolve().relative_to(_REPO_ROOT.resolve()))

    test_ids = _load_split_doc_ids(ds_dir / TEST_SPLIT_FILE)
    train_ids = _load_split_doc_ids(ds_dir / TRAIN_SPLIT_FILE)
    have_docs = set(old["sampled_document_ids"])
    have_mids = {m["measurement_id"] for m in old["measurements"]}
    total = len(old["measurements"])

    final = json.loads(cfg.final_json.read_text())
    by_doc: dict[str, list[dict]] = {}
    for r in final:
        if r["document_id"] in test_ids and r["document_id"] not in have_docs:
            row = dict(r)
            row["page_number"] = _normalize_page_number(
                row.get("page_number"), method_key=METHOD_KEY, measurement_id=row["measurement_id"]
            )
            by_doc.setdefault(row["document_id"], []).append(row)
    candidates = sorted(by_doc)
    rng = random.Random(SEED)
    rng.shuffle(candidates)

    ocr_dir = _REPO_ROOT / cfg.ocr_dir
    new_docs: list[str] = []
    new_meas: list[dict] = []
    documents = {d: dict(v) for d, v in old["documents"].items()}
    for doc in candidates:
        if total >= MIN_TOTAL:
            break
        rows = sorted(by_doc[doc], key=lambda r: r["measurement_id"])
        take = sorted(rng.sample(rows, min(N_PER_PAPER, len(rows))), key=lambda r: r["measurement_id"])
        text = (ocr_dir / f"{doc}.txt").read_text()
        pages: dict[str, str] = {}
        for r in take:
            for p in r["page_number"]:
                pages[str(p)] = text if p == NO_PROVENANCE_PAGE else _get_page_text(text, p)
                assert pages[str(p)], f"empty page text: {doc} page {p}"
        documents[doc] = {"pages": pages, "full_text": text}
        for r in take:
            m = {k: v for k, v in r.items() if k != "context"}
            m["sampled"] = True
            assert m["measurement_id"] not in have_mids
            new_meas.append(m)
        new_docs.append(doc)
        total += len(take)
    assert total >= MIN_TOTAL, f"{dataset}: only reached {total} after exhausting {len(candidates)} candidate papers"

    added = {m["document_id"] for m in new_meas}
    assert added <= test_ids and added.isdisjoint(train_ids) and added.isdisjoint(have_docs)

    shutil.copy2(path, path.with_name(path.name + ".pre-topup"))
    new = dict(old)
    new["measurements"] = old["measurements"] + new_meas  # existing rows untouched, in order
    new["documents"] = documents
    new["sampled_document_ids"] = sorted(have_docs | set(new_docs))
    new["topup_document_ids"] = sorted(new_docs)
    new["topup_seed"] = SEED
    new["topup_min_total"] = MIN_TOTAL
    new["n_sampled"] = new["n_included"] = len(new["measurements"])
    new["n_papers_with_measurements"] = old["n_papers_with_measurements"] + len(new_docs)
    assert new["measurements"][: len(old["measurements"])] == old["measurements"]
    for d, v in old["documents"].items():
        assert new["documents"][d] == v
    with open(path, "w") as f:
        json.dump(new, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(f"[{dataset}] {len(old['measurements'])} -> {len(new['measurements'])} measurements; "
          f"+{len(new_docs)} papers {new_docs} (+{[sum(m['document_id']==d for m in new_meas) for d in new_docs]} rows)")


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        topup(ds)
