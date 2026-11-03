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
