"""Build a seeded, reproducible manual-validation set for one dataset.

Phase 1 of build note ``notes/scholarlm/builds/2026-09-03-manual-validation-sets-01.md``.

Draws a simple random sample (without replacement) of ``n_sample`` extracted
measurements from the pinned gemma-3-27b MeasurementLM extraction of a dataset,
restricted to the probe-dataset *test* split, then writes a self-contained
``data/{dataset}/validation_set.json``:

* every sampled measurement (``sampled: true``);
* every further measurement reached by a **page-closure fixpoint** over the
  sampled rows' ``page_number`` sets (``sampled: false``) -- so any page shown to
  a validator carries its complete extraction set;
* the full verbatim page text (``<page number="N">...</page>`` body from the OCR
  the run consumed) for every page any included row touches.

The per-measurement ``context`` retrieval chunk is deliberately NOT stored: when
it is prose it equals the page text verbatim, when it is a table it is one
already-identified table (``table_number`` on the row), and it is re-derivable
from ``extraction_final_json`` keyed on ``measurement_id``.

Fail-loud throughout (CLAUDE.md): a missing config key, a pool smaller than
``n_sample``, a ``page_number`` with no matching ``<page>`` tag, an empty page
body, or a violated boundary invariant is a hard error.

Usage::

    python data/validation/create_validation_set.py --dataset pond
    python data/validation/create_validation_set.py --dataset supermat --n 50 --seed 0 --out /tmp/x.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(_REPO_ROOT / "experiments"))
from utils import load_config  # noqa: E402  (experiments/config.yaml loader)


def _get_page_text(document_text: str, page_number: int) -> str:
    """Verbatim body of ``<page number="{page_number}">...</page>``, stripped.

    Mirrors ``scholarlm.measurementlm.MeasurementLM._get_page_text`` but **raises**
    instead of returning ``""`` on a missing tag -- a ``page_number`` that does not
    resolve to a page in the OCR is a page-index misalignment, not a soft miss.
    """
    tag = f'<page number="{page_number}">'
    start = document_text.find(tag)
    if start == -1:
        raise KeyError(f"no {tag!r} in document text")
    start += len(tag)
    end = document_text.find("</page>", start)
    if end == -1:
        raise KeyError(f"unterminated {tag!r} in document text")
    return document_text[start:end].strip()


def _load_split_doc_ids(path: Path) -> set[str]:
    with open(path) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and rows, f"empty or non-list split file: {path}"
    ids = {r["document_id"] for r in rows}
    assert ids, f"no document_id values in {path}"
    return ids


def build(
    dataset: str,
    *,
    repo_root: Path,
    experiment_config: Path,
    out_path: Path,
    n_sample: int | None = None,
    seed_override: int | None = None,
) -> dict:
    """Build the validation set for ``dataset`` and write it to ``out_path``.

    Returns the written payload (also for tests).
    """
    # ---- config -----------------------------------------------------------
    with open(experiment_config) as f:
        exp = yaml.safe_load(f)
    params = exp["params"]                      # KeyError if absent -> fail loud
    assert dataset in params["datasets"], f"{dataset!r} not in params.datasets"

    repo_seed = load_config(repo_root / "experiments" / "config.yaml")["defaults"]["seed"]
    assert exp["seed"] == repo_seed, (
        f"config seed {exp['seed']} != experiments/config.yaml defaults.seed {repo_seed}"
    )
    seed = seed_override if seed_override is not None else repo_seed
    n_sample = n_sample if n_sample is not None else params["n_sample"]

    model = params["extraction_model"]
    extraction_date = params["extraction_dates"][dataset]
    test_file = params["test_split_file"]
    train_file = params["train_split_file"]

    # ---- inputs ---------------------------------------------------------
    ds_dir = repo_root / "data" / dataset
    test_ids = _load_split_doc_ids(ds_dir / test_file)
    train_ids = _load_split_doc_ids(ds_dir / train_file)
    assert test_ids.isdisjoint(train_ids), (
        f"test/train split overlap: {sorted(test_ids & train_ids)}"
    )

    ext_dir = repo_root / "data" / "experiments" / dataset / "extraction" / model / extraction_date
    final_path = ext_dir / "final.json"
    with open(final_path) as f:
        final = json.load(f)
    assert isinstance(final, list) and final, f"empty final.json: {final_path}"

    mids = [r["measurement_id"] for r in final]
    assert len(mids) == len(set(mids)), "measurement_id not unique in final.json"
    for r in final:
        pn = r["page_number"]
        assert isinstance(pn, list) and pn and all(isinstance(p, int) and p >= 0 for p in pn), (
            f"malformed page_number on measurement_id={r['measurement_id']}: {pn!r}"
        )

    run_meta = json.loads((ext_dir / "run_metadata.json").read_text())
    ocr_abs = Path(run_meta["ocr_dir"])
    assert ocr_abs.is_absolute(), f"run_metadata ocr_dir is not absolute: {ocr_abs}"
    ocr_rel = ocr_abs.resolve().relative_to(repo_root.resolve())   # ValueError if outside repo
    ocr_dir = repo_root / ocr_rel
    assert ocr_dir.is_dir(), f"OCR dir does not exist: {ocr_dir}"

    # ---- pool + sample -------------------------------------------------
    pool = [r for r in final if r["document_id"] in test_ids]
    print(f"[{dataset}] final.json rows : {len(final)}")
    print(f"[{dataset}] test-split pool : {len(pool)}")
    assert len(pool) >= n_sample, (
        f"pool has only {len(pool)} test-split rows; need n_sample={n_sample}"
    )
    pool.sort(key=lambda r: r["measurement_id"])          # sampling must not depend on file order

    rng = random.Random(seed)
    sampled = rng.sample(pool, n_sample)
    sampled_ids = {r["measurement_id"] for r in sampled}
    assert len(sampled_ids) == n_sample

    # ---- page-closure fixpoint ---------------------------------------
    by_doc: dict[str, list[dict]] = {}
    for r in pool:
        by_doc.setdefault(r["document_id"], []).append(r)

    def pages_of(r: dict) -> set[tuple[str, int]]:
        return {(r["document_id"], p) for p in r["page_number"]}

    id_to_row = {r["measurement_id"]: r for r in pool}
    included_ids = set(sampled_ids)
    while True:
        touched: set[tuple[str, int]] = set()
        for mid in included_ids:
            touched |= pages_of(id_to_row[mid])
        added = False
        for (doc, _p) in touched:
            for r in by_doc[doc]:
                if r["measurement_id"] in included_ids:
                    continue
                if pages_of(r) & touched:
                    included_ids.add(r["measurement_id"])
                    added = True
        if not added:
            break

    included = sorted((id_to_row[mid] for mid in included_ids), key=lambda r: r["measurement_id"])
    assert sampled_ids <= included_ids

    all_pages: set[tuple[str, int]] = set()
    for r in included:
        all_pages |= pages_of(r)

    # ---- page text --------------------------------------------------
    doc_text_cache: dict[str, str] = {}
    documents: dict[str, dict] = {}
    for (doc, p) in sorted(all_pages):
        if doc not in doc_text_cache:
            doc_text_cache[doc] = (ocr_dir / f"{doc}.txt").read_text()
        text = _get_page_text(doc_text_cache[doc], p)
        assert text, f"empty page text: {doc} page {p}"
        documents.setdefault(doc, {"pages": {}})["pages"][str(p)] = text

    # ---- measurements list -----------------------------------------
    measurements = []
    for r in included:
        row = {k: v for k, v in r.items() if k != "context"}
        row["sampled"] = r["measurement_id"] in sampled_ids
        measurements.append(row)
    measurements.sort(key=lambda m: m["measurement_id"])

    # ---- boundary asserts (CLAUDE.md) -----------------------------
    n_true = sum(1 for m in measurements if m["sampled"])
    assert n_true == n_sample, f"{n_true} sampled:true rows, expected {n_sample}"

    sampled_docs = {r["document_id"] for r in sampled}
    assert sampled_docs <= test_ids, f"sampled doc not in test split: {sorted(sampled_docs - test_ids)}"
    assert sampled_docs.isdisjoint(train_ids), (
        f"leakage: sampled doc in train split: {sorted(sampled_docs & train_ids)}"
    )

    seen = [m["measurement_id"] for m in measurements]
    assert len(seen) == len(set(seen)), "measurement_id appears twice in output"
    assert not any("context" in m for m in measurements), "context leaked into output"

    for m in measurements:
        for p in m["page_number"]:
            assert str(p) in documents[m["document_id"]]["pages"], (
                f"page {p} of measurement_id={m['measurement_id']} missing from documents[]"
            )

    # page-closure invariant: every pool row on any output page is present
    out_pages = {(doc, int(p)) for doc, d in documents.items() for p in d["pages"]}
    present = set(seen)
    for r in pool:
        if pages_of(r) & out_pages:
            assert r["measurement_id"] in present, (
                f"page-closure violated: measurement_id={r['measurement_id']} on an output page "
                f"but not in output"
            )

    # ---- write -----------------------------------------------------
    payload = {
        "dataset": dataset,
        "extraction_model": model,
        "extraction_date": extraction_date,
        "extraction_final_json": str(final_path.resolve().relative_to(repo_root.resolve())),
        "ocr_dir": str(ocr_rel),
        "seed": seed,
        "n_sample": n_sample,
        "n_pool": len(pool),
        "n_sampled": n_true,
        "n_included": len(measurements),
        "documents": documents,
        "measurements": measurements,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print(f"[{dataset}] sampled          : {n_true}")
    print(f"[{dataset}] included (closure): {len(measurements)}")
    print(f"[{dataset}] documents / pages : {len(documents)} / {len(out_pages)}")
    print(f"[{dataset}] wrote            : {out_path}  ({out_path.stat().st_size / 1e6:.2f} MB)")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", required=True, choices=["pond", "nfix", "supermat"])
    ap.add_argument("--n", type=int, default=None, help="Override params.n_sample (tests / ladder only).")
    ap.add_argument("--seed", type=int, default=None, help="Override the repo seed (tests / ladder only).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Override output path (default: data/{dataset}/validation_set.json).")
    ap.add_argument("--experiment-config", type=Path, default=None,
                    help="Override the experiment config path (tests only).")
    args = ap.parse_args(argv)

    experiment_config = args.experiment_config or (
        _REPO_ROOT / "configs" / "2026-09-03-manual-validation-sets-01.yaml"
    )
    out_path = args.out or (_REPO_ROOT / "data" / args.dataset / "validation_set.json")
    if args.out is None and (args.n is not None or args.seed is not None):
        raise SystemExit(
            "refusing to overwrite the committed data/{ds}/validation_set.json with an "
            "--n/--seed override; pass --out explicitly for ladder / test runs"
        )
    build(
        args.dataset,
        repo_root=_REPO_ROOT,
        experiment_config=experiment_config,
        out_path=out_path,
        n_sample=args.n,
        seed_override=args.seed,
    )


if __name__ == "__main__":
    main()
