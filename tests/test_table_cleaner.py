"""Unit tests for TableCleaner.clean() (rung 1: hand-built fixture, no network).

TableCleaner was split out of MeasurementLM._clean_tables in the table-cleaning
separation refactor (see devlog/ and notes/scholarlm/builds/ for the id). These
tests exist to prove the split didn't quietly weaken behavior: a stubbed
_call_batch (not a trimmed re-implementation) verifies the real retry/
concurrency kwargs are still passed, per-page splicing only touches the page it
cleaned, a ContextLengthExceededError slot leaves that document's page
untouched, and a missing processed_pdf_dir still fails loud.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.measurementlm import ContextLengthExceededError
from scholarlm.table_cleaner import TableCleaner


def _make_cleaner(**overrides):
    kwargs = dict(
        model_name="test-model",
        sampling_params={"temperature": 0.5},
        api_base="http://localhost:0/v1",
        use_extra_body=False,
    )
    kwargs.update(overrides)
    return TableCleaner(**kwargs)


def _write_b64_pages(dir_path: Path, page_indices: list[int]) -> str:
    """Write dummy .b64 files for the given page indices; return the dir as str.

    clean() never decodes the contents -- it only embeds them verbatim into a
    data: URL -- so any placeholder text is fine.
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in page_indices:
        (dir_path / f"{i}.b64").write_text(f"FAKE_IMAGE_{i}")
    return str(dir_path)


_DOC_ONE_TABLE = (
    '<page number="0">Intro text without a table.</page>'
    '<page number="1">Some prose. <table number="1">'
    "<table><tbody><tr><td>index</td><td>x</td></tr></tbody></table>"
    "</table></page>"
)


def test_clean_sends_message_only_for_pages_with_tables(tmp_path):
    cleaner = _make_cleaner()
    proc_dir = _write_b64_pages(tmp_path / "doc0", [0, 1])

    captured = {}

    def fake_call_batch(self, message_sets, **kwargs):
        captured["message_sets"] = message_sets
        captured["kwargs"] = kwargs
        return ["CLEANED PAGE 1 TEXT"]

    import types
    cleaner._call_batch = types.MethodType(fake_call_batch, cleaner)

    cleaned = cleaner.clean([_DOC_ONE_TABLE], [proc_dir])

    # Only the one page with a <table> tag produced a message.
    assert len(captured["message_sets"]) == 1
    # Real retry/isolation kwargs must still be forwarded, not a trimmed subset.
    assert captured["kwargs"]["max_retries"] == 4
    assert captured["kwargs"]["max_concurrent"] == 2

    assert len(cleaned) == 1
    assert '<page number="0">Intro text without a table.</page>' in cleaned[0]
    assert "CLEANED PAGE 1 TEXT" in cleaned[0]
    assert "Some prose." not in cleaned[0]  # page 1's original text was replaced


def test_clean_splices_only_between_page_tags(tmp_path):
    """Untouched pages must be byte-identical; only the cleaned page's inner
    text changes, and the <page>/</page> tags themselves are preserved."""
    doc = (
        '<page number="0">Page zero, unrelated to tables.</page>'
        '<page number="1">Has a table. <table number="1">'
        "<table><tbody><tr><td>a</td></tr></tbody></table></table></page>"
        '<page number="2">Page two, also unrelated.</page>'
    )
    cleaner = _make_cleaner()
    proc_dir = _write_b64_pages(tmp_path / "doc0", [0, 1, 2])

    import types
    cleaner._call_batch = types.MethodType(
        lambda self, message_sets, **kwargs: ["NORMALIZED TABLE TEXT"], cleaner
    )

    cleaned = cleaner.clean([doc], [proc_dir])[0]

    assert '<page number="0">Page zero, unrelated to tables.</page>' in cleaned
    assert '<page number="2">Page two, also unrelated.</page>' in cleaned
    assert '<page number="1">\nNORMALIZED TABLE TEXT\n</page>' in cleaned


def test_clean_context_length_exceeded_leaves_page_unchanged(tmp_path):
    cleaner = _make_cleaner()
    proc_dir = _write_b64_pages(tmp_path / "doc0", [0, 1])

    import types
    cleaner._call_batch = types.MethodType(
        lambda self, message_sets, **kwargs: [ContextLengthExceededError("too long")],
        cleaner,
    )

    cleaned = cleaner.clean([_DOC_ONE_TABLE], [proc_dir])

    assert cleaned[0] == _DOC_ONE_TABLE  # untouched
    assert cleaner.context_length_exceeded_docs == {0}


def test_clean_no_tables_returns_documents_unchanged_without_calling_llm(tmp_path):
    doc = '<page number="0">No tables anywhere on this page.</page>'
    cleaner = _make_cleaner()
    proc_dir = _write_b64_pages(tmp_path / "doc0", [0])

    def fail_if_called(self, *args, **kwargs):
        raise AssertionError("must not be called when no page has a <table> tag")

    import types
    cleaner._call_batch = types.MethodType(fail_if_called, cleaner)

    cleaned = cleaner.clean([doc], [proc_dir])

    assert cleaned == [doc]


def test_clean_missing_processed_pdf_dir_raises(tmp_path):
    cleaner = _make_cleaner()
    missing_dir = str(tmp_path / "does_not_exist")

    with pytest.raises(FileNotFoundError):
        cleaner.clean([_DOC_ONE_TABLE], [missing_dir])


def test_clean_writes_output_dir(tmp_path):
    out_dir = tmp_path / "cleaned_out"
    cleaner = _make_cleaner(output_dir=str(out_dir))
    proc_dir = _write_b64_pages(tmp_path / "paperA", [0, 1])

    import types
    cleaner._call_batch = types.MethodType(
        lambda self, message_sets, **kwargs: ["CLEANED"], cleaner
    )

    cleaner.clean([_DOC_ONE_TABLE], [proc_dir])

    out_file = out_dir / "paperA.txt"
    assert out_file.exists()
    assert "CLEANED" in out_file.read_text()
