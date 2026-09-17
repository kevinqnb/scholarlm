"""Build validation_set.json files for the multi-method validation site
expansion: 9 extraction methods x 3 probe datasets (pond/nfix/supermat),
100 samples each.

Generalizes create_validation_set.py's sampling + page-closure algorithm
(kept untouched -- this script does not import or modify it) to run over
explicit (dataset, method) result directories rather than the single
pinned gemma-3-27b MeasurementLM extraction:

* an explicit final.json path per (dataset, method), instead of resolving
  through experiments/config.yaml;
* a shared, per-dataset OCR directory (experiments/results/{ds}/ocr/<id>/),
  since baseline and ablation1 runs don't record run_metadata["ocr_dir"] --
  OCR is identical across methods for a given probe dataset;
* rows with no page_number (ablation1: a single whole-document LLM call
  records no provenance at all). These get a synthetic page_number [-1]
  per row before sampling/closure runs -- every row in a document then
  shares the same synthetic page, so the existing closure fixpoint becomes
  "include every measurement in this document" for free, and the site
  renders the whole paper with no page tinted (approved design).

Fail-loud throughout, matching create_validation_set.py.

Usage::

    python data/validation/build_method_validation_sets.py
    python data/validation/build_method_validation_sets.py --only pond:baseline-gliner
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 342
N_SAMPLE = 100
TEST_SPLIT_FILE = "probe_dataset_test.json"
TRAIN_SPLIT_FILE = "probe_dataset.json"

_RESULTS = _REPO_ROOT / "experiments" / "results"


@dataclass(frozen=True)
class MethodConfig:
    dataset: str
    method_key: str
    method_label: str
    extraction_model: str
    final_json: Path
    has_page_provenance: bool

    @property
    def extraction_date(self) -> str:
        # Experiment-id directories are named "<date>-...-01"; the date
        # prefix is exactly the parent directory name's first 10 chars.
        return self.final_json.parent.name[:10]


def _configs() -> list[MethodConfig]:
    def extraction(ds: str, exp_id: str, model: str, method_key: str, label: str) -> MethodConfig:
        return MethodConfig(
            dataset=ds,
            method_key=method_key,
            method_label=label,
            extraction_model=model,
            final_json=_RESULTS / ds / "extraction" / exp_id / "final.json",
            has_page_provenance=True,
        )

    def ablation1(ds: str, exp_id: str, model: str) -> MethodConfig:
        return MethodConfig(
            dataset=ds,
            method_key=f"ablation1-{model}",
            method_label=f"Ablation 1: direct extraction ({model})",
            extraction_model=model,
            final_json=_RESULTS / ds / "ablation" / exp_id / "final.json",
            has_page_provenance=False,
        )

    def baseline(ds: str, subdir: str, exp_id: str, model: str, method_key: str, label: str) -> MethodConfig:
        return MethodConfig(
            dataset=ds,
            method_key=method_key,
            method_label=label,
            extraction_model=model,
            final_json=_RESULTS / ds / subdir / exp_id / "final.json",
            has_page_provenance=True,
        )

    configs: list[MethodConfig] = []

    # --- pond ---
    configs += [
        extraction("pond", "2026-05-04-pond-llama-3.1-8b-extraction-01", "llama-3.1-8b",
                    "measurementlm-llama-3.1-8b", "MeasurementLM (llama-3.1-8b)"),
        extraction("pond", "2026-05-05-pond-gemma-3-27b-extraction-01", "gemma-3-27b",
                    "measurementlm-gemma-3-27b", "MeasurementLM (gemma-3-27b)"),
        extraction("pond", "2026-05-02-pond-gpt-oss-120b-extraction-01", "gpt-oss-120b",
                    "measurementlm-gpt-oss-120b", "MeasurementLM (gpt-oss-120b)"),
        ablation1("pond", "2026-05-04-pond-llama-3.1-8b-ablation1-01", "llama-3.1-8b"),
        ablation1("pond", "2026-05-04-pond-gemma-3-27b-ablation1-01", "gemma-3-27b"),
        ablation1("pond", "2026-05-03-pond-gpt-oss-120b-ablation1-01", "gpt-oss-120b"),
        baseline("pond", "baseline_gliner", "2026-07-11-pond-baseline-gliner-01", "gliner-large-v1",
                 "baseline-gliner", "Baseline: GLiNER"),
        baseline("pond", "baseline_nuextract", "2026-07-11-pond-baseline-nuextract-01", "nuextract-2.0-8b",
                 "baseline-nuextract", "Baseline: NuExtract"),
        baseline("pond", "baseline_chatextract", "2026-07-11-pond-baseline-chatextract-01", "gemma-3-27b",
                 "baseline-chatextract", "Baseline: ChatExtract"),
    ]

    # --- nfix ---
    configs += [
        extraction("nfix", "2026-05-05-nfix-llama-3.1-8b-extraction-01", "llama-3.1-8b",
                    "measurementlm-llama-3.1-8b", "MeasurementLM (llama-3.1-8b)"),
        extraction("nfix", "2026-05-06-nfix-gemma-3-27b-extraction-01", "gemma-3-27b",
                    "measurementlm-gemma-3-27b", "MeasurementLM (gemma-3-27b)"),
        extraction("nfix", "2026-05-03-nfix-gpt-oss-120b-extraction-01", "gpt-oss-120b",
                    "measurementlm-gpt-oss-120b", "MeasurementLM (gpt-oss-120b)"),
        ablation1("nfix", "2026-05-07-nfix-llama-3.1-8b-ablation1-01", "llama-3.1-8b"),
        ablation1("nfix", "2026-05-07-nfix-gemma-3-27b-ablation1-01", "gemma-3-27b"),
        ablation1("nfix", "2026-05-06-nfix-gpt-oss-120b-ablation1-01", "gpt-oss-120b"),
        baseline("nfix", "baseline_gliner", "2026-07-11-nfix-baseline-gliner-01", "gliner-large-v1",
                 "baseline-gliner", "Baseline: GLiNER"),
        baseline("nfix", "baseline_nuextract", "2026-07-11-nfix-baseline-nuextract-01", "nuextract-2.0-8b",
                 "baseline-nuextract", "Baseline: NuExtract"),
        baseline("nfix", "baseline_chatextract", "2026-07-11-nfix-baseline-chatextract-01", "gemma-3-27b",
                 "baseline-chatextract", "Baseline: ChatExtract"),
    ]

    # --- supermat ---
    configs += [
        extraction("supermat", "2026-07-13-supermat-llama-3.1-8b-extraction-01", "llama-3.1-8b",
                    "measurementlm-llama-3.1-8b", "MeasurementLM (llama-3.1-8b)"),
        extraction("supermat", "2026-07-09-supermat-gemma-3-27b-extraction-01", "gemma-3-27b",
                    "measurementlm-gemma-3-27b", "MeasurementLM (gemma-3-27b)"),
        extraction("supermat", "2026-07-13-supermat-gpt-oss-120b-extraction-01", "gpt-oss-120b",
                    "measurementlm-gpt-oss-120b", "MeasurementLM (gpt-oss-120b)"),
        ablation1("supermat", "2026-08-13-supermat-llama-3.1-8b-ablation1-01", "llama-3.1-8b"),
        ablation1("supermat", "2026-08-13-supermat-gemma-3-27b-ablation1-01", "gemma-3-27b"),
        ablation1("supermat", "2026-08-13-supermat-gpt-oss-120b-ablation1-01", "gpt-oss-120b"),
        baseline("supermat", "baseline_gliner", "2026-07-13-supermat-baseline-gliner-01", "gliner-large-v1",
                 "baseline-gliner", "Baseline: GLiNER"),
        baseline("supermat", "baseline_nuextract", "2026-07-13-supermat-baseline-nuextract-01", "nuextract-2.0-8b",
                 "baseline-nuextract", "Baseline: NuExtract"),
        baseline("supermat", "baseline_chatextract", "2026-07-13-supermat-baseline-chatextract-01", "gemma-3-27b",
                 "baseline-chatextract", "Baseline: ChatExtract"),
    ]

    return configs


def _ocr_dir(dataset: str) -> Path:
    ocr_root = _RESULTS / dataset / "ocr"
    candidates = sorted(p for p in ocr_root.iterdir() if p.is_dir())
    assert len(candidates) == 1, (
        f"{dataset}: expected exactly one OCR results dir under {ocr_root}, "
        f"found {[c.name for c in candidates]}"
    )
    return candidates[0]


def _load_split_doc_ids(path: Path) -> set[str]:
    with open(path) as f:
        rows = json.load(f)
    assert isinstance(rows, list) and rows, f"empty or non-list split file: {path}"
    ids = {r["document_id"] for r in rows}
    assert ids, f"no document_id values in {path}"
    return ids


def _get_page_text(document_text: str, page_number: int) -> str:
    tag = f'<page number="{page_number}">'
    start = document_text.find(tag)
    if start == -1:
        raise KeyError(f"no {tag!r} in document text")
    start += len(tag)
    end = document_text.find("</page>", start)
    if end == -1:
        raise KeyError(f"unterminated {tag!r} in document text")
    return document_text[start:end].strip()


# Sentinel "page" assigned to every row of an ablation1 (no-provenance) run.
# Real OCR page numbers are >= 0 (verified), so -1 never collides.
NO_PROVENANCE_PAGE = -1


def build(cfg: MethodConfig, *, ds_dir: Path, ocr_dir: Path, out_path: Path) -> dict:
    test_ids = _load_split_doc_ids(ds_dir / TEST_SPLIT_FILE)
    train_ids = _load_split_doc_ids(ds_dir / TRAIN_SPLIT_FILE)
    assert test_ids.isdisjoint(train_ids), (
        f"{cfg.dataset}: test/train split overlap: {sorted(test_ids & train_ids)}"
    )

    with open(cfg.final_json) as f:
        final = json.load(f)
    assert isinstance(final, list) and final, f"empty final.json: {cfg.final_json}"

    mids = [r["measurement_id"] for r in final]
    assert len(mids) == len(set(mids)), (
        f"{cfg.dataset}/{cfg.method_key}: measurement_id not unique in final.json"
    )

    working: list[dict] = []
    for r in final:
        row = dict(r)
        pn = row.get("page_number")
        if cfg.has_page_provenance:
            assert isinstance(pn, list) and pn and all(isinstance(p, int) and p >= 0 for p in pn), (
                f"{cfg.dataset}/{cfg.method_key}: expected page-provenance row "
                f"but measurement_id={row['measurement_id']} has page_number={pn!r}"
            )
        else:
            assert pn is None, (
                f"{cfg.dataset}/{cfg.method_key}: expected no-provenance row (page_number=None) "
                f"but measurement_id={row['measurement_id']} has page_number={pn!r}"
            )
            row["page_number"] = [NO_PROVENANCE_PAGE]
        working.append(row)

    pool = [r for r in working if r["document_id"] in test_ids]
    print(f"[{cfg.dataset}/{cfg.method_key}] final.json rows : {len(working)}")
    print(f"[{cfg.dataset}/{cfg.method_key}] test-split pool : {len(pool)}")
    assert len(pool) >= N_SAMPLE, (
        f"{cfg.dataset}/{cfg.method_key}: pool has only {len(pool)} test-split rows; "
        f"need n_sample={N_SAMPLE}"
    )
    pool.sort(key=lambda r: r["measurement_id"])

    rng = random.Random(SEED)
    sampled = rng.sample(pool, N_SAMPLE)
    sampled_ids = {r["measurement_id"] for r in sampled}
    assert len(sampled_ids) == N_SAMPLE

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

    doc_text_cache: dict[str, str] = {}
    documents: dict[str, dict] = {}

    def doc_text(doc: str) -> str:
        if doc not in doc_text_cache:
            doc_text_cache[doc] = (ocr_dir / f"{doc}.txt").read_text()
        return doc_text_cache[doc]

    for (doc, p) in sorted(all_pages):
        if p == NO_PROVENANCE_PAGE:
            text = doc_text(doc)  # whole document -- no single page to point to
        else:
            text = _get_page_text(doc_text(doc), p)
        assert text, f"empty page text: {doc} page {p}"
        documents.setdefault(doc, {"pages": {}})["pages"][str(p)] = text

    for doc in documents:
        documents[doc]["full_text"] = doc_text(doc)

    measurements = []
    for r in included:
        row = {k: v for k, v in r.items() if k != "context"}
        row["sampled"] = r["measurement_id"] in sampled_ids
        measurements.append(row)
    measurements.sort(key=lambda m: m["measurement_id"])

    n_true = sum(1 for m in measurements if m["sampled"])
    assert n_true == N_SAMPLE, f"{n_true} sampled:true rows, expected {N_SAMPLE}"

    sampled_docs = {r["document_id"] for r in sampled}
    assert sampled_docs <= test_ids, (
        f"sampled doc not in test split: {sorted(sampled_docs - test_ids)}"
    )
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

    out_pages = {(doc, int(p)) for doc, d in documents.items() for p in d["pages"]}
    present = set(seen)
    for r in pool:
        if pages_of(r) & out_pages:
            assert r["measurement_id"] in present, (
                f"page-closure violated: measurement_id={r['measurement_id']} on an output "
                f"page but not in output"
            )

    payload = {
        "dataset": f"{cfg.dataset}__{cfg.method_key}",
        "probe_dataset": cfg.dataset,
        "method_key": cfg.method_key,
        "method_label": cfg.method_label,
        "extraction_model": cfg.extraction_model,
        "extraction_date": cfg.extraction_date,
        "extraction_final_json": str(cfg.final_json.resolve().relative_to(_REPO_ROOT.resolve())),
        "seed": SEED,
        "n_sample": N_SAMPLE,
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

    print(f"[{cfg.dataset}/{cfg.method_key}] sampled           : {n_true}")
    print(f"[{cfg.dataset}/{cfg.method_key}] included (closure): {len(measurements)}")
    print(f"[{cfg.dataset}/{cfg.method_key}] documents / pages : {len(documents)} / {len(out_pages)}")
    print(f"[{cfg.dataset}/{cfg.method_key}] wrote             : {out_path} "
          f"({out_path.stat().st_size / 1e6:.2f} MB)")
    return payload


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--only", action="append", default=None,
        help="Restrict to one or more 'dataset:method_key' configs (repeatable). Default: all 27.",
    )
    ap.add_argument(
        "--out-dir", type=Path, default=None,
        help="Override output directory (default: data/{dataset}/, one file per method).",
    )
    args = ap.parse_args(argv)

    configs = _configs()
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if f"{c.dataset}:{c.method_key}" in wanted]
        missing = wanted - {f"{c.dataset}:{c.method_key}" for c in configs}
        if missing:
            raise SystemExit(f"--only referenced unknown configs: {sorted(missing)}")

    ocr_dirs: dict[str, Path] = {}
    for ds in {c.dataset for c in configs}:
        ocr_dirs[ds] = _ocr_dir(ds)
        print(f"[{ds}] OCR dir: {ocr_dirs[ds]}", file=sys.stderr)

    for cfg in configs:
        ds_dir = _REPO_ROOT / "data" / cfg.dataset
        out_dir = args.out_dir or ds_dir
        out_path = out_dir / f"validation_set_{cfg.method_key}.json"
        build(cfg, ds_dir=ds_dir, ocr_dir=ocr_dirs[cfg.dataset], out_path=out_path)


if __name__ == "__main__":
    main()
