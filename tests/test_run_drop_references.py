"""Rung-1 unit tests for experiments/run_drop_references.py.

Two hand-built fixtures under a monkeypatched RESULTS_ROOT: one plain
reference-only tail (no resume) and one with a resume-eligible bare
appendix heading -- both with known-by-inspection expected output, so a
passing test demonstrates the runner's file I/O and dataset/source_type
cross-checks are wired correctly, not just that drop_references_section
itself works (that's tests/test_references.py's job).
"""
import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import utils
import run_drop_references as rdr


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(rdr.paths, "RESULTS_ROOT", tmp_path / "results")
    return tmp_path / "results"


PAPER_A = (
    '<page number="0">\n\nBody text A.\nREFERENCES\nSmith, J. (2020). A citation.\n\n</page>\n\n'
)
PAPER_B = (
    '<page number="0">\n\nBody text B.\nREFERENCES\nJones, A. (2020). A citation.\n'
    'APPENDIX A\nSupplementary methods B.\n\n</page>\n\n'
)


def _write_source(fake_root, dataset, source_type, source_id):
    d = fake_root / dataset / source_type / source_id
    d.mkdir(parents=True)
    (d / "paper_a.txt").write_text(PAPER_A)
    (d / "paper_b.txt").write_text(PAPER_B)
    return d


# ---------------------------------------------------------------------------
# run_drop_references()
# ---------------------------------------------------------------------------


def test_writes_dropped_text_and_diagnostics_report(fake_root, tmp_path):
    _write_source(fake_root, "pond", "ocr", "2026-01-01-fake-ocr-01")
    output_dir = tmp_path / "out"

    summary = rdr.run_drop_references(
        dataset="pond", source_type="ocr", source_id="2026-01-01-fake-ocr-01",
        output_dir=output_dir,
    )

    assert summary["papers_processed"] == 2
    assert summary["reference_heading_found"] == 2
    assert summary["resume_heading_found"] == 1
    assert summary["total_dropped_chars"] > 0
    assert summary["total_rescued_chars"] > 0

    a_out = (output_dir / "paper_a.txt").read_text()
    assert "Body text A." in a_out
    assert "Smith, J." not in a_out

    b_out = (output_dir / "paper_b.txt").read_text()
    assert "Body text B." in b_out
    assert "Jones, A." not in b_out
    assert "APPENDIX A" in b_out
    assert "Supplementary methods B." in b_out

    report = json.loads((output_dir / "drop_references_report.json").read_text())
    by_paper = {r["paper"]: r for r in report}
    assert by_paper["paper_a"]["resume_heading_found"] is False
    assert by_paper["paper_a"]["rescued_chars"] == 0
    assert by_paper["paper_b"]["resume_heading_found"] is True
    assert by_paper["paper_b"]["resume_heading_text"] == "APPENDIX A"
    assert by_paper["paper_b"]["rescued_chars"] == summary["total_rescued_chars"]


def test_paper_subset_filters_to_named_papers(fake_root, tmp_path):
    _write_source(fake_root, "pond", "ocr", "2026-01-01-fake-ocr-01")
    output_dir = tmp_path / "out"

    summary = rdr.run_drop_references(
        dataset="pond", source_type="ocr", source_id="2026-01-01-fake-ocr-01",
        output_dir=output_dir, paper_subset_override=["paper_a"],
    )

    assert summary["papers_processed"] == 1
    assert (output_dir / "paper_a.txt").exists()
    assert not (output_dir / "paper_b.txt").exists()


def test_dataset_mismatch_raises(fake_root, tmp_path):
    _write_source(fake_root, "pond", "ocr", "2026-01-01-fake-ocr-01")
    with pytest.raises(ValueError, match="does not match params.dataset"):
        rdr.run_drop_references(
            dataset="nfix", source_type="ocr", source_id="2026-01-01-fake-ocr-01",
            output_dir=tmp_path / "out",
        )


def test_source_type_mismatch_raises(fake_root, tmp_path):
    _write_source(fake_root, "pond", "ocr", "2026-01-01-fake-ocr-01")
    with pytest.raises(ValueError, match="does not match params.source_type"):
        rdr.run_drop_references(
            dataset="pond", source_type="table_cleaning", source_id="2026-01-01-fake-ocr-01",
            output_dir=tmp_path / "out",
        )


def test_empty_subset_raises(fake_root, tmp_path):
    _write_source(fake_root, "pond", "ocr", "2026-01-01-fake-ocr-01")
    with pytest.raises(ValueError, match="No .txt files found"):
        rdr.run_drop_references(
            dataset="pond", source_type="ocr", source_id="2026-01-01-fake-ocr-01",
            output_dir=tmp_path / "out", paper_subset_override=["nonexistent"],
        )


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


def _write_cfg(path: Path, exp_id: str, seed: int, params: dict) -> Path:
    cfg_path = path / f"{exp_id}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(
            {"id": exp_id, "project": "scholarlm", "description": "test", "seed": seed, "params": params},
            f,
        )
    return cfg_path


_REPO_SEED = utils.load_config()["defaults"]["seed"]


def test_main_dispatches_with_resolved_params_and_output_dir(fake_root, tmp_path):
    cfg_path = _write_cfg(
        tmp_path, "2026-09-18-test-drop-references-01", _REPO_SEED,
        {"dataset": "pond", "source_type": "ocr", "source_id": "2026-01-01-fake-ocr-01"},
    )
    with patch("run_drop_references.run_drop_references") as m:
        m.return_value = {
            "papers_processed": 0, "reference_heading_found": 0,
            "resume_heading_found": 0, "total_dropped_chars": 0, "total_rescued_chars": 0,
        }
        rdr.main([str(cfg_path)])
        kw = m.call_args.kwargs
        assert kw["dataset"] == "pond"
        assert kw["source_type"] == "ocr"
        assert kw["source_id"] == "2026-01-01-fake-ocr-01"
        assert str(kw["output_dir"]).endswith("results/pond/drop_references/2026-09-18-test-drop-references-01")


def test_main_requires_source_type_and_source_id(tmp_path):
    cfg_path = _write_cfg(
        tmp_path, "2026-09-18-test-drop-references-02", _REPO_SEED,
        {"dataset": "pond"},  # missing source_type, source_id
    )
    with pytest.raises(ValueError, match="missing required params key"):
        rdr.main([str(cfg_path)])


def test_main_rejects_invalid_source_type(tmp_path):
    cfg_path = _write_cfg(
        tmp_path, "2026-09-18-test-drop-references-03", _REPO_SEED,
        {"dataset": "pond", "source_type": "extraction", "source_id": "2026-01-01-fake-ocr-01"},
    )
    with pytest.raises(ValueError, match="must be one of"):
        rdr.main([str(cfg_path)])


def test_main_enforces_repo_seed_consistency(tmp_path):
    cfg_path = _write_cfg(
        tmp_path, "2026-09-18-test-drop-references-04", _REPO_SEED + 1,
        {"dataset": "pond", "source_type": "ocr", "source_id": "2026-01-01-fake-ocr-01"},
    )
    with pytest.raises(ValueError, match="does not match"):
        rdr.main([str(cfg_path)])
