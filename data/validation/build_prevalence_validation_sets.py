"""
Build the 9 prevalence-estimation validation sets (3 datasets x 3 extraction
models), one per full-pipeline extraction run.

Each set is a simple random sample of N_SAMPLE measurements drawn uniformly
from *all* rows of the run's final.json whose document is in the TRAIN split --
the papers of ``data/{dataset}/probe_dataset_v3.json`` (the train half; its
held-out counterpart is ``probe_dataset_test_v3.json``). Not stratified by
paper, so the sample is unbiased for prevalence (the share of extracted
measurements that are valid).

Runs are the gold full-pipeline extraction ids from
``analysis/analysis-configs/2026-09-23-{dataset}-full-pipeline-vs-baselines-recovery-01.yaml``
(the first three experiment_ids of each; confirmed with the user 2026-10-03).

Output, one per run, next to the dataset's other files:
    data/{dataset}/prevalence_validation_set_<experiment_id>.json
in the same payload shape as the pilot/main validation sets, so the validation
site ingests it unchanged. Seeded (SEED) -- re-running reproduces the same
files; do not re-run to "refresh" a set that is already live.

Usage::

    python data/validation/build_prevalence_validation_sets.py [dataset ...] [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from build_pilot_validation_sets import (
    NO_PROVENANCE_PAGE,
    SEED,
    _REPO_ROOT,
    _get_page_text,
    _normalize_page_number,
)

N_SAMPLE = 20
SPLIT_FILE = "probe_dataset_v3.json"  # train papers
TEST_SPLIT_FILE = "probe_dataset_test_v3.json"

# dataset -> [(model short name, experiment id)]
RUNS: dict[str, list[tuple[str, str]]] = {
    "pond": [
        ("llama8b", "2026-09-26-pond-extraction-llama8b-reppen-full-01"),
        ("gemma27b", "2026-09-22-pond-extraction-gemma27b-parseqty-standardization-valueonly-01"),
        ("gptoss120b", "2026-09-21-pond-extraction-gptoss120b-full-01"),
    ],
    "nfix": [
        ("llama8b", "2026-09-26-nfix-extraction-llama8b-reppen-full-01"),
        ("gemma27b", "2026-09-21-nfix-extraction-gemma27b-full-01"),
        ("gptoss120b", "2026-09-25-nfix-extraction-gptoss120b-full-02"),
    ],
    "supermat": [
        ("llama8b", "2026-09-26-supermat-extraction-llama8b-reppen-full-01"),
        ("gemma27b", "2026-09-21-supermat-extraction-gemma27b-full-01"),
        ("gptoss120b", "2026-09-21-supermat-extraction-gptoss120b-full-01"),
    ],
}


def _split_ids(path: Path) -> set[str]:
    rows = json.loads(path.read_text())
    assert isinstance(rows, list) and rows, f"empty or non-list split file: {path}"
    return {r["document_id"] for r in rows}


def build(dataset: str, model: str, experiment_id: str, out_dir: Path) -> Path:
    ds_dir = _REPO_ROOT / "data" / dataset
    train_ids = _split_ids(ds_dir / SPLIT_FILE)
    test_ids = _split_ids(ds_dir / TEST_SPLIT_FILE)
    assert train_ids.isdisjoint(test_ids), f"{dataset}: v3 train/test splits overlap"

    run_dir = _REPO_ROOT / "experiments" / "results" / dataset / "extraction" / experiment_id
    final = json.loads((run_dir / "final.json").read_text())
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    assert meta["dataset"] == dataset, f"{experiment_id}: run_metadata dataset={meta['dataset']!r}"
    ocr_dir = _REPO_ROOT / meta["ocr_dir"]

    mids = [r["measurement_id"] for r in final]
    assert len(mids) == len(set(mids)), f"{experiment_id}: measurement_id not unique"

    pool = []
    for r in final:
        if r["document_id"] in train_ids:
            row = dict(r)
            row["page_number"] = _normalize_page_number(
                row.get("page_number"), method_key=experiment_id, measurement_id=row["measurement_id"]
            )
            pool.append(row)
    pool.sort(key=lambda r: r["measurement_id"])
    assert len(pool) >= N_SAMPLE, f"{experiment_id}: only {len(pool)} train-split rows"

    chosen = random.Random(SEED).sample(pool, N_SAMPLE)
    chosen.sort(key=lambda r: r["measurement_id"])
    docs_used = {r["document_id"] for r in chosen}
    assert docs_used <= train_ids and docs_used.isdisjoint(test_ids)

    documents: dict[str, dict] = {}
    for r in chosen:
        doc = r["document_id"]
        if doc not in documents:
            documents[doc] = {"pages": {}, "full_text": (ocr_dir / f"{doc}.txt").read_text()}
        text = documents[doc]["full_text"]
        for p in r["page_number"]:
            documents[doc]["pages"][str(p)] = text if p == NO_PROVENANCE_PAGE else _get_page_text(text, p)
            assert documents[doc]["pages"][str(p)], f"empty page text: {doc} page {p}"

    measurements = []
    for r in chosen:
        m = {k: v for k, v in r.items() if k != "context"}
        m["sampled"] = True
        measurements.append(m)

    method_key = f"prevalence-{model}"
    payload = {
        "dataset": f"{dataset}__{method_key}",
        "probe_dataset": dataset,
        "method_key": method_key,
        "method_label": f"Prevalence: full pipeline ({model})",
        "experiment_id": experiment_id,
        "extraction_model": meta["model"],
        "extraction_date": experiment_id[:10],
        "extraction_final_json": str((run_dir / "final.json").relative_to(_REPO_ROOT)),
        "split": "train",
        "split_file": SPLIT_FILE,
        "seed": SEED,
        "n_sample": len(measurements),
        "n_pool": len(pool),
        "n_sampled": len(measurements),
        "n_included": len(measurements),
        "n_papers_with_measurements": len(docs_used),
        "sampled_document_ids": sorted(docs_used),
        "documents": documents,
        "measurements": measurements,
    }
    out = out_dir / dataset / f"prevalence_validation_set_{experiment_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")
    print(f"[{dataset}/{model}] pool {len(pool)} train rows over {len(train_ids)} train papers -> "
          f"{len(measurements)} sampled from {len(docs_used)} papers; wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("datasets", nargs="*", default=list(RUNS))
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data")
    args = ap.parse_args()
    for dataset in args.datasets:
        for model, experiment_id in RUNS[dataset]:
            build(dataset, model, experiment_id, args.out_dir)


if __name__ == "__main__":
    main()
