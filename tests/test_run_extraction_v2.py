"""Rung 1: run_extraction_v2's context-length-exceeded metadata record, on hand-built inputs."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from run_extraction_v2 import _context_length_exceeded_record

TEXT_INFO = [{"document_id": f"paper_{i}"} for i in range(5)]


def test_no_overflow_records_empty_lists(tmp_path):
    rec = _context_length_exceeded_record(tmp_path, set(), TEXT_INFO)
    assert rec == {"context_length_exceeded_docs": [], "context_length_exceeded_document_ids": []}


def test_overflow_records_sorted_indices_and_names(tmp_path):
    rec = _context_length_exceeded_record(tmp_path, {3, 1}, TEXT_INFO)
    assert rec["context_length_exceeded_docs"] == [1, 3]
    assert rec["context_length_exceeded_document_ids"] == ["paper_1", "paper_3"]


def test_unions_with_earlier_invocation_instead_of_erasing_it(tmp_path):
    (tmp_path / "run_metadata.json").write_text(json.dumps({"context_length_exceeded_docs": [0, 3]}))
    rec = _context_length_exceeded_record(tmp_path, {3, 4}, TEXT_INFO)
    assert rec["context_length_exceeded_docs"] == [0, 3, 4]
    assert rec["context_length_exceeded_document_ids"] == ["paper_0", "paper_3", "paper_4"]


def test_resumed_invocation_with_no_new_overflow_keeps_earlier_record(tmp_path):
    (tmp_path / "run_metadata.json").write_text(json.dumps({"context_length_exceeded_docs": [2]}))
    assert _context_length_exceeded_record(tmp_path, set(), TEXT_INFO)["context_length_exceeded_docs"] == [2]


def test_index_outside_paper_list_is_a_hard_error(tmp_path):
    with pytest.raises(AssertionError):
        _context_length_exceeded_record(tmp_path, {5}, TEXT_INFO)


def _write_cfg(tmp_path, params):
    import yaml
    path = tmp_path / "2026-01-01-testset-extractionv2-model-full-01.yaml"
    path.write_text(yaml.safe_dump({
        "id": path.stem, "project": "scholarlm", "description": "t", "seed": 342, "params": params,
    }))
    return path


def test_runner_requires_standardize_context(tmp_path):
    import run_extraction_v2 as r
    cfg = _write_cfg(tmp_path, {"dataset": "pond", "model": "gemma-3-27b", "ocr_dir": "x"})
    with pytest.raises(ValueError, match="standardize_context"):
        r.main([str(cfg)])


def test_runner_rejects_inert_parse_quantities_context(tmp_path):
    import run_extraction_v2 as r
    cfg = _write_cfg(tmp_path, {"dataset": "pond", "model": "gemma-3-27b", "ocr_dir": "x",
                                "standardize_context": "value_only", "parse_quantities_context": "value_only"})
    with pytest.raises(ValueError, match="parse_quantities_context"):
        r.main([str(cfg)])


def test_carried_forward_prefers_this_invocation(tmp_path):
    from run_extraction_v2 import _carried_forward
    (tmp_path / "run_metadata.json").write_text(json.dumps({"contextualization_stats": {"sent": 1}}))
    assert _carried_forward(tmp_path, "contextualization_stats", {"sent": 9}) == {"sent": 9}


def test_carried_forward_keeps_earlier_value_when_step_skipped_on_resume(tmp_path):
    from run_extraction_v2 import _carried_forward
    (tmp_path / "run_metadata.json").write_text(json.dumps({"contextualization_stats": {"sent": 1}}))
    assert _carried_forward(tmp_path, "contextualization_stats", None) == {"sent": 1}


def test_carried_forward_is_none_when_nothing_ever_ran(tmp_path):
    from run_extraction_v2 import _carried_forward
    assert _carried_forward(tmp_path, "contextualization_stats", None) is None
