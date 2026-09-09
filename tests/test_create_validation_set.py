"""Model-free unit tests for data/validation/create_validation_set.py (ladder rung 1).

A hand-built fixture repo whose seed=0 draw is verifiable by inspection:

  pool (test-split, sorted by measurement_id): m0 m1 m2 m3 m4 m6
  random.Random(0).sample(pool, 2) -> [m3, m6]

  m3 = doc A page [2]        m6 = doc C page [0]
  page-closure fixpoint from {(A,2), (C,0)}:
    (A,2) also carries m2 (pages [1,2]) -> pulls in m2 and page (A,1)
    (C,0) also carries m4               -> pulls in m4
  included = {m2, m3, m4, m6}; m0/m1 (page A,0) are never touched.
  documents: A -> pages {1, 2};  C -> page {0}
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "data" / "validation" / "create_validation_set.py"

_spec = importlib.util.spec_from_file_location("create_validation_set", _SCRIPT)
cvs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cvs)


DATE = "2026_05_05"
MODEL = "gemma-3-27b"
OCR_SUBDIR = "ocr_output_cleaned_qwen-3.5-27b"

OCR_TEXT = {
    "A": (
        '<page number="0">alpha page zero</page>\n'
        '<page number="1">beta page one</page>\n'
        '<page number="2">gamma page two</page>\n'
    ),
    "B": '<page number="0">bee page zero</page>\n',
    "C": '<page number="0">gamma cee zero</page>\n',
}

# (measurement_id, document_id, page_number, extra table_number)
FINAL_ROWS = [
    ("m0", "A", [0], None),
    ("m1", "A", [0], None),   # sibling of m0 on page A/0
    ("m2", "A", [1, 2], 4),   # multi-page row
    ("m3", "A", [2], 4),      # on page A/2 only -- reachable from m2's closure
    ("m4", "C", [0], None),
    ("m5", "B", [0], None),   # TRAIN doc -- must never appear
    ("m6", "C", [0], None),   # sibling of m4
]


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _make_repo(tmp_path: Path, *, test_docs=("A", "C"), train_docs=("B",)) -> Path:
    repo = tmp_path / "repo"
    (repo / "experiments").mkdir(parents=True)
    _write_json(repo / "experiments" / "config.yaml", {})  # overwritten below as YAML
    (repo / "experiments" / "config.yaml").write_text("defaults:\n  seed: 342\n")

    exp_cfg = repo / "configs" / "cfg.yaml"
    exp_cfg.parent.mkdir(parents=True)
    exp_cfg.write_text(
        "id: test\nproject: scholarlm\nseed: 342\n"
        "params:\n"
        f"  extraction_model: {MODEL}\n"
        "  datasets: [pond, nfix, supermat]\n"
        "  extraction_dates:\n"
        f"    pond: \"{DATE}\"\n    nfix: \"{DATE}\"\n    supermat: \"{DATE}\"\n"
        "  n_sample: 1000\n"
        "  test_split_file: probe_dataset_test.json\n"
        "  train_split_file: probe_dataset.json\n"
    )

    ds_dir = repo / "data" / "pond"
    _write_json(ds_dir / "probe_dataset_test.json", [{"document_id": d} for d in test_docs])
    _write_json(ds_dir / "probe_dataset.json", [{"document_id": d} for d in train_docs])

    for doc, text in OCR_TEXT.items():
        (ds_dir / OCR_SUBDIR / f"{doc}.txt").parent.mkdir(parents=True, exist_ok=True)
        (ds_dir / OCR_SUBDIR / f"{doc}.txt").write_text(text)

    ext_dir = repo / "data" / "experiments" / "pond" / "extraction" / MODEL / DATE
    final = [
        {
            "measurement_id": mid,
            "document_id": doc,
            "page_number": pages,
            "table_number": tbl,
            "attribute": "x",
            "value": "1",
            "units": "u",
            "context": [f"CONTEXT for {mid}"] * len(pages),
        }
        for (mid, doc, pages, tbl) in FINAL_ROWS
    ]
    _write_json(ext_dir / "final.json", final)
    _write_json(
        ext_dir / "run_metadata.json",
        {"ocr_dir": str((ds_dir / OCR_SUBDIR).resolve())},
    )
    return repo


def _build(repo: Path, **kw):
    out = repo / "out.json"
    payload = cvs.build(
        "pond",
        repo_root=repo,
        experiment_config=repo / "configs" / "cfg.yaml",
        out_path=out,
        n_sample=kw.get("n_sample", 2),
        seed_override=kw.get("seed_override", 0),
    )
    return payload, out


# --------------------------------------------------------------------------


def test_get_page_text_raises_on_missing_tag():
    assert cvs._get_page_text('<page number="0">hi</page>', 0) == "hi"
    with pytest.raises(KeyError):
        cvs._get_page_text('<page number="0">hi</page>', 3)
    with pytest.raises(KeyError):
        cvs._get_page_text('<page number="0">hi', 0)  # unterminated


def test_known_answer_seed0(tmp_path):
    payload, _ = _build(_make_repo(tmp_path))

    by_id = {m["measurement_id"]: m for m in payload["measurements"]}
    assert set(by_id) == {"m2", "m3", "m4", "m6"}
    assert {m for m, r in by_id.items() if r["sampled"]} == {"m3", "m6"}
    assert payload["n_sampled"] == 2
    assert payload["n_included"] == 4
    assert payload["n_pool"] == 6

    # m5 (train doc B) never appears
    assert "m5" not in by_id
    # m0/m1 on page A/0 are never touched
    assert "m0" not in by_id and "m1" not in by_id

    assert payload["documents"]["A"]["pages"] == {"1": "beta page one", "2": "gamma page two"}
    assert payload["documents"]["C"]["pages"] == {"0": "gamma cee zero"}

    # full_text is the whole OCR document verbatim (every page, including
    # ones no included measurement touches -- e.g. A/0)
    assert payload["documents"]["A"]["full_text"] == OCR_TEXT["A"]
    assert payload["documents"]["C"]["full_text"] == OCR_TEXT["C"]
    assert "B" not in payload["documents"]  # train doc, still absent


def test_no_context_key_in_output(tmp_path):
    payload, _ = _build(_make_repo(tmp_path))
    assert all("context" not in m for m in payload["measurements"])
    assert all("context_index" not in m for m in payload["measurements"])


def test_page_closure_invariant(tmp_path):
    payload, _ = _build(_make_repo(tmp_path))
    out_pages = {
        (doc, int(p)) for doc, d in payload["documents"].items() for p in d["pages"]
    }
    present = {m["measurement_id"] for m in payload["measurements"]}
    for (mid, doc, pages, _tbl) in FINAL_ROWS:
        if any((doc, p) in out_pages for p in pages):
            assert mid in present, f"{mid} sits on an output page but is absent"


def test_ocr_dir_stored_repo_relative(tmp_path):
    payload, _ = _build(_make_repo(tmp_path))
    assert payload["ocr_dir"] == f"data/pond/{OCR_SUBDIR}"
    assert payload["extraction_final_json"] == (
        f"data/experiments/pond/extraction/{MODEL}/{DATE}/final.json"
    )


def test_seed_determinism(tmp_path):
    repo = _make_repo(tmp_path)
    _, out1 = _build(repo)
    b1 = out1.read_bytes()
    _, out2 = _build(repo)
    b2 = out2.read_bytes()
    # no wall-clock fields in the payload -> byte-identical
    assert b1 == b2


def test_pool_smaller_than_n_sample_raises(tmp_path):
    repo = _make_repo(tmp_path)
    with pytest.raises(AssertionError, match="pool has only"):
        _build(repo, n_sample=10)


def test_test_train_overlap_raises(tmp_path):
    repo = _make_repo(tmp_path, test_docs=("A", "B", "C"), train_docs=("B",))
    with pytest.raises(AssertionError, match="overlap"):
        _build(repo)


def test_seed_mismatch_raises(tmp_path):
    repo = _make_repo(tmp_path)
    (repo / "experiments" / "config.yaml").write_text("defaults:\n  seed: 999\n")
    with pytest.raises(AssertionError, match="defaults.seed"):
        _build(repo)
