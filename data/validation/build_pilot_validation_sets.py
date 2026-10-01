"""Build the pilot validation_set.json files for the manual-validation site
(pond, nfix, supermat): a paper-first sample.

Phase 1: sample ``N_PAPERS`` papers once from the dataset's test split
(seeded), then for *each* pilot method independently draw up to ``N_PER_PAPER``
measurements from that paper's rows in that method's own ``final.json`` -- a
paper absent from a method's output, or with fewer than ``N_PER_PAPER`` rows
there, contributes fewer (down to zero) measurements.

Phase 2 (top-up): if a method's phase-1 total is below ``TARGET_MEASUREMENTS``,
extra test-split papers are drawn (fresh ``Random(SEED)`` shuffle of the
remaining candidates, up to ``N_PER_PAPER`` rows each) until the target is
reached; the last paper may overshoot. Phase-1 rows are never changed, so a
set that needed no top-up is identical to its phase-1 output. Top-up papers
are recorded under ``topup_document_ids`` in the output.

The whole build is deterministic given the pinned ``final.json`` files and
OCR dir below: re-running reproduces the deployed sets byte-for-byte
(judgements on the site key on ``(dataset, measurement_id)``, so never change
the seeds, constants, or pinned inputs for a set that is already live).

Every row in the output is ``sampled: true``: no non-sampled sibling
measurements that share a page are pulled in (a page-closure approach was used
for the retired phase-1 sets). At this scale (~100 rows per method) it isn't
needed and would balloon file size for no benefit.

Pilot methods:
  - pond / pilot-gemma27b-full: full MeasurementLM pipeline, gemma-3-27b
    (experiments/results/pond/extraction/2026-09-22-pond-extraction-gemma27b-parseqty-standardization-valueonly-01/final.json)
  - pond / pilot-langextract-gemma27b: LangExtract baseline, gemma-3-27b, no
    qualifiers
    (experiments/results/pond/baseline_langextract/2026-09-25-pond-langextract-gemma27b-noqualifiers-full-01/final.json)
  - nfix, supermat / pilot-gemma27b-full: full pipeline, gemma-3-27b, run
    2026-09-21 (see ``_configs_other``).

The pond experiments were run against the same table-cleaned OCR dir
(verified against each run's run_metadata.json / experiment yaml
2026-09-27):
    experiments/results/pond/drop_references/2026-09-18-pond-table-cleaning-drop-references-01

The validation site shows validators this same table-cleaned OCR (2026-09-28
decision, reversing a brief switch to the raw OCR): they should judge against
the text the extraction methods actually saw. This holds for every future
dataset added to the site too -- point ``ocr_dir`` at the OCR dir the
extraction ran against, never the raw one.

``page_number`` shape varies by method -- confirmed by inspection, not
assumed:
  - pilot-gemma27b-full: always a non-empty list[int >= 0].
  - Methods whose rows carry no page_number at all (e.g. NuExtract3: one API
    call per whole document -- see run_baseline_nuextract3.py's docstring).
  - pilot-langextract-gemma27b: a bare int >= 0 on most rows (6040/6204 in
    the full final.json), but ``None`` on a minority (164/6204) where
    LangExtract couldn't localize the span to one page.
``_normalize_page_number`` below handles all three per-row, not per-config:
int/list[int] rows keep their real page(s); everything else (missing key,
None, or a None inside what would otherwise be a list) gets the same
sentinel-page-(-1) treatment (``NO_PROVENANCE_PAGE``), where the whole
document stands in as "the page".

Fail-loud throughout.

Usage::

    python data/validation/build_pilot_validation_sets.py              # all three datasets, in place
    python data/validation/build_pilot_validation_sets.py pond --out-dir /tmp/x   # scratch copy
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]

SEED = 342
N_PAPERS = 20
N_PER_PAPER = 5
# Each pilot is topped up to at least this many measurements (see _top_up).
TARGET_MEASUREMENTS = 100
TEST_SPLIT_FILE = "probe_dataset_test.json"
TRAIN_SPLIT_FILE = "probe_dataset.json"

# Real OCR page numbers are always >= 0 (verified) -- this sentinel never
# collides.
NO_PROVENANCE_PAGE = -1

# Table-cleaned OCR (references dropped): the text every extraction method
# ran against, and so what validators must read. See module docstring.
_OCR_DIR = "experiments/results/pond/drop_references/2026-09-18-pond-table-cleaning-drop-references-01"


@dataclass(frozen=True)
class PilotConfig:
    dataset: str
    method_key: str
    method_label: str
    extraction_model: str
    final_json: Path
    ocr_dir: str

    @property
    def extraction_date(self) -> str:
        return self.final_json.parent.name[:10]


def _configs() -> list[PilotConfig]:
    results = _REPO_ROOT / "experiments" / "results"
    return [
        PilotConfig(
            dataset="pond",
            method_key="pilot-gemma27b-full",
            method_label="Pilot: full pipeline (gemma-3-27b)",
            extraction_model="gemma-3-27b",
            final_json=results / "pond" / "extraction"
            / "2026-09-22-pond-extraction-gemma27b-parseqty-standardization-valueonly-01" / "final.json",
            ocr_dir=_OCR_DIR,
        ),
        PilotConfig(
            dataset="pond",
            method_key="pilot-langextract-gemma27b",
            method_label="Pilot: LangExtract (gemma-3-27b)",
            extraction_model="gemma-3-27b",
            final_json=results / "pond" / "baseline_langextract"
            / "2026-09-25-pond-langextract-gemma27b-noqualifiers-full-01" / "final.json",
            ocr_dir=_OCR_DIR,
        ),
    ]


def _configs_other() -> list[PilotConfig]:
    """nfix/supermat pilots (added 2026-09-29): full pipeline, gemma-3-27b, run
    2026-09-21 against each dataset's table-cleaned OCR (verified from each
    run's run_metadata.json ``ocr_dir``)."""
    results = _REPO_ROOT / "experiments" / "results"
    return [
        PilotConfig(
            dataset=ds,
            method_key="pilot-gemma27b-full",
            method_label="Pilot: full pipeline (gemma-3-27b)",
            extraction_model="gemma-3-27b",
            final_json=results / ds / "extraction"
            / f"2026-09-21-{ds}-extraction-gemma27b-full-01" / "final.json",
            ocr_dir=f"experiments/results/{ds}/drop_references/2026-09-18-{ds}-table-cleaning-drop-references-01",
        )
        for ds in ("nfix", "supermat")
    ]


def _normalize_page_number(pn, *, method_key: str, measurement_id: int) -> list[int]:
    """A row's page_number, normalized to a non-empty list[int] -- real page(s)
    if we have them, else [NO_PROVENANCE_PAGE] (see module docstring for the
    three shapes this collapses: list[int], bare int, or missing/None)."""
    if pn is None:
        return [NO_PROVENANCE_PAGE]
    if isinstance(pn, int):
        assert pn >= 0, f"{method_key}: measurement_id={measurement_id} has negative page_number={pn!r}"
        return [pn]
    if isinstance(pn, list):
        assert pn and all(isinstance(p, int) and p >= 0 for p in pn), (
            f"{method_key}: measurement_id={measurement_id} has malformed page_number={pn!r}"
        )
        return pn
    raise AssertionError(f"{method_key}: measurement_id={measurement_id} has unrecognized page_number={pn!r}")


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


def sample_papers(ds_dir: Path, *, n_papers: int, seed: int) -> tuple[list[str], set[str], set[str]]:
    test_ids = _load_split_doc_ids(ds_dir / TEST_SPLIT_FILE)
    train_ids = _load_split_doc_ids(ds_dir / TRAIN_SPLIT_FILE)
    assert test_ids.isdisjoint(train_ids), f"test/train split overlap: {sorted(test_ids & train_ids)}"
    assert len(test_ids) >= n_papers, f"test split has only {len(test_ids)} papers; need {n_papers}"
    rng = random.Random(seed)
    papers = rng.sample(sorted(test_ids), n_papers)
    return papers, test_ids, train_ids


def _top_up(
    working: list[dict], *, taken_docs: set[str], test_ids: set[str], total: int
) -> tuple[list[str], list[dict]]:
    """Phase 2: papers beyond the first ``N_PAPERS`` until ``total`` reaches
    ``TARGET_MEASUREMENTS``. Candidates are test-split papers not already
    sampled with >=1 row in this method's final.json, shuffled with a *fresh*
    ``Random(SEED)``; each contributes up to ``N_PER_PAPER`` rows (sorted by
    measurement_id), so the last paper may overshoot the target by up to
    ``N_PER_PAPER - 1``. Returns (added paper ids in draw order, added rows)."""
    by_doc: dict[str, list[dict]] = {}
    for r in working:
        if r["document_id"] in test_ids and r["document_id"] not in taken_docs:
            by_doc.setdefault(r["document_id"], []).append(r)
    candidates = sorted(by_doc)
    rng = random.Random(SEED)
    rng.shuffle(candidates)

    new_docs: list[str] = []
    new_rows: list[dict] = []
    for doc in candidates:
        if total >= TARGET_MEASUREMENTS:
            break
        rows = sorted(by_doc[doc], key=lambda r: r["measurement_id"])
        take = sorted(rng.sample(rows, min(N_PER_PAPER, len(rows))), key=lambda r: r["measurement_id"])
        new_rows.extend(take)
        new_docs.append(doc)
        total += len(take)
    assert total >= TARGET_MEASUREMENTS, (
        f"only reached {total} measurements after exhausting {len(candidates)} candidate papers"
    )
    return new_docs, new_rows


def build(
    cfg: PilotConfig,
    *,
    paper_ids: list[str],
    test_ids: set[str],
    train_ids: set[str],
    ocr_dir: Path,
    out_path: Path,
) -> dict:
    with open(cfg.final_json) as f:
        final = json.load(f)
    assert isinstance(final, list) and final, f"empty final.json: {cfg.final_json}"

    mids = [r["measurement_id"] for r in final]
    assert len(mids) == len(set(mids)), f"{cfg.method_key}: measurement_id not unique in final.json"

    working: list[dict] = []
    for r in final:
        row = dict(r)
        row["page_number"] = _normalize_page_number(
            row.get("page_number"), method_key=cfg.method_key, measurement_id=row["measurement_id"]
        )
        working.append(row)

    paper_id_set = set(paper_ids)
    pool = [r for r in working if r["document_id"] in paper_id_set]
    assert {r["document_id"] for r in pool} <= test_ids, "pool leaked outside the test split"

    by_doc: dict[str, list[dict]] = {}
    for r in pool:
        by_doc.setdefault(r["document_id"], []).append(r)

    rng = random.Random(SEED)
    sampled: list[dict] = []
    n_papers_with_rows = 0
    for doc in paper_ids:  # fixed order -> deterministic rng consumption
        rows = sorted(by_doc.get(doc, []), key=lambda r: r["measurement_id"])
        if not rows:
            continue
        n_papers_with_rows += 1
        take = min(N_PER_PAPER, len(rows))
        sampled.extend(rng.sample(rows, take))

    sampled.sort(key=lambda r: r["measurement_id"])
    # Phase 2: top up to TARGET_MEASUREMENTS. Phase-1 rows keep their sorted
    # order; top-up rows are appended paper by paper after them.
    topup_docs: list[str] = []
    if len(sampled) < TARGET_MEASUREMENTS:
        topup_docs, extra = _top_up(working, taken_docs=paper_id_set, test_ids=test_ids, total=len(sampled))
        sampled.extend(extra)
    sampled_ids = {r["measurement_id"] for r in sampled}
    assert len(sampled_ids) == len(sampled), "measurement_id collision in pilot sample"

    def pages_of(r: dict) -> set[tuple[str, int]]:
        return {(r["document_id"], p) for p in r["page_number"]}

    all_pages: set[tuple[str, int]] = set()
    for r in sampled:
        all_pages |= pages_of(r)

    doc_text_cache: dict[str, str] = {}

    def doc_text(doc: str) -> str:
        if doc not in doc_text_cache:
            doc_text_cache[doc] = (ocr_dir / f"{doc}.txt").read_text()
        return doc_text_cache[doc]

    documents: dict[str, dict] = {}
    for (doc, p) in sorted(all_pages):
        text = doc_text(doc) if p == NO_PROVENANCE_PAGE else _get_page_text(doc_text(doc), p)
        assert text, f"empty page text: {doc} page {p}"
        documents.setdefault(doc, {"pages": {}})["pages"][str(p)] = text
    for doc in documents:
        documents[doc]["full_text"] = doc_text(doc)

    measurements = []
    for r in sampled:
        row = {k: v for k, v in r.items() if k != "context"}
        row["sampled"] = True
        measurements.append(row)

    # ---- boundary asserts ------------------------------------------------
    assert not any("context" in m for m in measurements), "context leaked into output"
    sampled_docs = {m["document_id"] for m in measurements}
    assert sampled_docs <= test_ids, f"sampled doc not in test split: {sorted(sampled_docs - test_ids)}"
    assert sampled_docs.isdisjoint(train_ids), (
        f"leakage: sampled doc in train split: {sorted(sampled_docs & train_ids)}"
    )
    for m in measurements:
        for p in m["page_number"]:
            assert str(p) in documents[m["document_id"]]["pages"], (
                f"page {p} of measurement_id={m['measurement_id']} missing from documents[]"
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
        "n_sample": N_PAPERS * N_PER_PAPER,
        "n_pool": len(pool),
        "n_sampled": len(measurements),
        "n_included": len(measurements),
        "n_papers_target": N_PAPERS,
        "n_papers_with_measurements": n_papers_with_rows + len(topup_docs),
        "sampled_document_ids": sorted(paper_id_set | set(topup_docs)),
        "documents": documents,
        "measurements": measurements,
    }
    if topup_docs:
        payload["topup_document_ids"] = sorted(topup_docs)
        payload["topup_seed"] = SEED
        payload["topup_min_total"] = TARGET_MEASUREMENTS
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")

    print(f"[{cfg.method_key}] papers sampled       : {N_PAPERS}")
    print(f"[{cfg.method_key}] papers with >=1 row  : {n_papers_with_rows}")
    print(f"[{cfg.method_key}] top-up papers added  : {len(topup_docs)}")
    print(f"[{cfg.method_key}] measurements sampled : {len(measurements)} (target >= {TARGET_MEASUREMENTS})")
    print(f"[{cfg.method_key}] documents / pages     : {len(documents)} / {len(all_pages)}")
    print(f"[{cfg.method_key}] wrote                 : {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    return payload


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("datasets", nargs="*", default=["pond", "nfix", "supermat"],
                    help="datasets to build (default: all three)")
    ap.add_argument("--out-dir", type=Path, default=_REPO_ROOT / "data",
                    help="root for <out-dir>/<dataset>/validation_set_<method_key>.json "
                         "(default: the repo's data/ -- overwrites the sets in place)")
    args = ap.parse_args()

    all_configs = _configs() + _configs_other()
    for dataset in args.datasets:
        configs = [c for c in all_configs if c.dataset == dataset]
        assert configs, f"no pilot configs for dataset {dataset!r}"
        ds_dir = _REPO_ROOT / "data" / dataset  # split files always come from the repo
        paper_ids, test_ids, train_ids = sample_papers(ds_dir, n_papers=N_PAPERS, seed=SEED)
        print(f"[{dataset}] sampled {N_PAPERS} papers (seed={SEED}): {paper_ids}")

        for cfg in configs:
            ocr_dir = _REPO_ROOT / cfg.ocr_dir
            assert ocr_dir.is_dir(), f"{cfg.method_key}: OCR dir does not exist: {ocr_dir}"
            build(
                cfg,
                paper_ids=paper_ids,
                test_ids=test_ids,
                train_ids=train_ids,
                ocr_dir=ocr_dir,
                out_path=args.out_dir / dataset / f"validation_set_{cfg.method_key}.json",
            )


if __name__ == "__main__":
    main()
