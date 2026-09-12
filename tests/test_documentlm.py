"""Unit tests for DocumentLM's fast mode (see
notes/scholarlm/builds/2026-08-13-documentlm-fast-mode-01.md) and per-document
formatting isolation / drop_references (see
notes/scholarlm/builds/2026-08-20-per-document-isolation-01.md).

Mocks scholarlm.documentlm.process_pdf (no real PDFs rendered) and
DocumentLM._call_batch_with_usage (no network calls). Verifies:
  * fast=True passes target_longest_dim=1024 to process_pdf; fast=False (the
    default) still passes 2048 -- normal-mode behavior is unchanged.
  * fast=True skips the temperature-escalation retry loop entirely, even for
    a page that would trigger a retry under normal mode (max_tokens exceeded,
    or is_table=True with no <table> tag): only one _call_batch_with_usage
    call happens, and the unmodified first-pass text ends up in self.text.
  * fast=False (unchanged behavior) still retries those same trigger cases.
  * A chandra-ocr-2 document whose formatted text hits an unrecognized
    data-label does not take down its batch-mates: self.text[i] is None and
    format_errors[i] records the failure for that document alone, while every
    other document's text is unaffected.
  * save() skips writing a file for a document in format_errors (with a loud
    warning) instead of crashing on file.write(None), and still writes every
    unaffected document normally.
  * drop_references=True removes an unrecognized-label div sitting after a
    references heading before it ever reaches chandra's label validation;
    drop_references=False (the default) leaves that failure in place.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import scholarlm.documentlm as documentlm_module
from scholarlm.documentlm import DocumentLM


MODEL_NAME = "allenai/olmOCR-2-7B-1025"
CHANDRA_MODEL_NAME = "datalab-to/chandra-ocr-2"


def _fake_process_pdf(calls):
    def _process_pdf(pdf_path, target_longest_dim=2048):
        calls.append(target_longest_dim)
        return ["fake_b64_page"]
    return _process_pdf


def _make_doclm(monkeypatch, fast, max_tokens=100):
    process_pdf_calls = []
    monkeypatch.setattr(
        documentlm_module, "process_pdf", _fake_process_pdf(process_pdf_calls)
    )
    doclm = DocumentLM(
        model_name=MODEL_NAME,
        sampling_params={"temperature": 0.1, "max_tokens": max_tokens},
        api_base="http://localhost:0/v1",
        fast=fast,
    )
    return doclm, process_pdf_calls


def test_fast_mode_renders_pages_at_1024(monkeypatch):
    doclm, process_pdf_calls = _make_doclm(monkeypatch, fast=True)
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: [("page text", 5)] * len(message_sets),
    )

    doclm.fit(["paper.pdf"])

    assert process_pdf_calls == [1024]


def test_normal_mode_renders_pages_at_2048(monkeypatch):
    doclm, process_pdf_calls = _make_doclm(monkeypatch, fast=False)
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: [("page text", 5)] * len(message_sets),
    )

    doclm.fit(["paper.pdf"])

    assert process_pdf_calls == [2048]


def test_fast_mode_skips_retry_on_max_tokens_exceeded(monkeypatch):
    doclm, _ = _make_doclm(monkeypatch, fast=True, max_tokens=10)
    call_count = []

    def fake_call_batch(self, message_sets, temperature=None, max_tokens=None):
        call_count.append(len(message_sets))
        # completion_tokens >= max_tokens (10) triggers a retry under normal mode.
        return [("truncated page text", 10)]

    monkeypatch.setattr(DocumentLM, "_call_batch_with_usage", fake_call_batch)

    result = doclm.fit(["paper.pdf"])

    assert call_count == [1]  # single pass only, no retry call
    assert "truncated page text" in result[0]


def test_fast_mode_skips_retry_on_missing_table_tags(monkeypatch):
    doclm, _ = _make_doclm(monkeypatch, fast=True)
    call_count = []

    def fake_call_batch(self, message_sets, temperature=None, max_tokens=None):
        call_count.append(len(message_sets))
        # is_table=True with no <table> tag triggers a retry under normal mode.
        text = "---\nis_table: true\n---\nNo table tags here."
        return [(text, 5)]

    monkeypatch.setattr(DocumentLM, "_call_batch_with_usage", fake_call_batch)

    result = doclm.fit(["paper.pdf"])

    assert call_count == [1]  # single pass only, no retry call
    assert "No table tags here." in result[0]


def test_normal_mode_still_retries_on_max_tokens_exceeded(monkeypatch):
    doclm, _ = _make_doclm(monkeypatch, fast=False, max_tokens=10)
    call_count = []

    def fake_call_batch(self, message_sets, temperature=None, max_tokens=None):
        call_count.append(len(message_sets))
        if len(call_count) == 1:
            return [("truncated page text", 10)]
        return [("retried page text", 3)]

    monkeypatch.setattr(DocumentLM, "_call_batch_with_usage", fake_call_batch)

    result = doclm.fit(["paper.pdf"])

    assert len(call_count) == 2  # initial pass + one retry
    assert "retried page text" in result[0]


def _make_chandra_doclm(monkeypatch, **overrides):
    monkeypatch.setattr(documentlm_module, "process_pdf", _fake_process_pdf([]))
    kwargs = dict(
        model_name=CHANDRA_MODEL_NAME,
        sampling_params={"temperature": 0.1, "max_tokens": 100},
        api_base="http://localhost:0/v1",
    )
    kwargs.update(overrides)
    return DocumentLM(**kwargs)


_GOOD_CHANDRA_PAGE = '<div data-bbox="0 0 1 1" data-label="Text">Fine content.</div>'
_BAD_CHANDRA_PAGE = '<div data-bbox="0 0 1 1" data-label="Weird-Unknown-Label">Bad content.</div>'


def test_chandra_formatting_failure_isolated_to_its_document(monkeypatch):
    doclm = _make_chandra_doclm(monkeypatch)
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: (
            [(_BAD_CHANDRA_PAGE, 5), (_GOOD_CHANDRA_PAGE, 5)]
        ),
    )

    result = doclm.fit(["bad.pdf", "good.pdf"])

    assert set(doclm.format_errors.keys()) == {0}
    assert "Unrecognized chandra-ocr-2 data-label" in doclm.format_errors[0]
    assert result[0] is None
    assert result[1] is not None
    assert "Fine content." in result[1]


def test_save_skips_document_with_format_error_and_warns(monkeypatch, tmp_path):
    doclm = _make_chandra_doclm(monkeypatch)
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: (
            [(_BAD_CHANDRA_PAGE, 5), (_GOOD_CHANDRA_PAGE, 5)]
        ),
    )
    doclm.fit(["bad.pdf", "good.pdf"])

    out0 = tmp_path / "bad.txt"
    out1 = tmp_path / "good.txt"
    with pytest.warns(UserWarning, match="formatting failed"):
        doclm.save([str(out0), str(out1)])

    assert not out0.exists()
    assert out1.exists()
    assert "Fine content." in out1.read_text()


_REFERENCES_CHANDRA_PAGE = (
    '<div data-bbox="0 0 1 1" data-label="Text">Good intro.</div>'
    '<div data-bbox="0 0 1 1" data-label="Section-Header"><h2>References</h2></div>'
    '<div data-bbox="0 0 1 1" data-label="Weird-Unknown-Label">Should be dropped.</div>'
)


def test_drop_references_true_avoids_unrecognized_label_after_heading(monkeypatch):
    doclm = _make_chandra_doclm(monkeypatch, drop_references=True)
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: (
            [(_REFERENCES_CHANDRA_PAGE, 5)]
        ),
    )

    result = doclm.fit(["paper.pdf"])

    assert doclm.format_errors == {}
    assert "Good intro." in result[0]
    assert "Should be dropped." not in result[0]


def test_drop_references_default_false_still_fails_on_unrecognized_label(monkeypatch):
    doclm = _make_chandra_doclm(monkeypatch)  # drop_references defaults to False
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: (
            [(_REFERENCES_CHANDRA_PAGE, 5)]
        ),
    )

    result = doclm.fit(["paper.pdf"])

    assert 0 in doclm.format_errors
    assert result[0] is None


# ---------------------------------------------------------------------------
# unknown_label_policy (see 2026-09-01-chandra-unknown-label-fallback-01.md)
# ---------------------------------------------------------------------------

_COERCE_BAD_PAGE = (
    '<div data-bbox="0 0 1 1" data-label="Text">Recognized body.</div>'
    '<div data-bbox="0 0 1 1" data-label="Chemical-Block"><p>H2O + CO2</p></div>'
)


def test_invalid_unknown_label_policy_raises_at_construction(monkeypatch):
    with pytest.raises(ValueError, match="unknown_label_policy"):
        _make_chandra_doclm(monkeypatch, unknown_label_policy="drop")


def test_coerce_policy_recovers_document_and_records_coerced_labels(monkeypatch):
    doclm = _make_chandra_doclm(monkeypatch, unknown_label_policy="coerce")
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: (
            [(_COERCE_BAD_PAGE, 5), (_GOOD_CHANDRA_PAGE, 5)]
        ),
    )

    result = doclm.fit(["chem.pdf", "plain.pdf"])

    assert doclm.format_errors == {}
    assert result[0] is not None and "Recognized body." in result[0]
    assert result[1] is not None and "Fine content." in result[1]
    assert doclm.coerced_labels == {0: {"Chemical-Block": 1}}


def test_default_raise_policy_still_isolates_unknown_label_to_format_errors(monkeypatch):
    doclm = _make_chandra_doclm(monkeypatch)  # unknown_label_policy defaults to "raise"
    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: (
            [(_COERCE_BAD_PAGE, 5)]
        ),
    )

    result = doclm.fit(["chem.pdf"])

    assert result[0] is None
    assert 0 in doclm.format_errors
    assert doclm.coerced_labels == {}


def test_coerced_labels_reset_between_fit_calls(monkeypatch):
    doclm = _make_chandra_doclm(monkeypatch, unknown_label_policy="coerce")

    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: [(_COERCE_BAD_PAGE, 5)],
    )
    doclm.fit(["chem.pdf"])
    assert doclm.coerced_labels == {0: {"Chemical-Block": 1}}

    monkeypatch.setattr(
        DocumentLM, "_call_batch_with_usage",
        lambda self, message_sets, temperature=None, max_tokens=None: [(_GOOD_CHANDRA_PAGE, 5)],
    )
    doclm.fit(["plain.pdf"])
    assert doclm.coerced_labels == {}
