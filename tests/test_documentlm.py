"""Unit tests for DocumentLM's fast mode (see
notes/scholarlm/builds/2026-08-13-documentlm-fast-mode-01.md).

Mocks scholarlm.documentlm.process_pdf (no real PDFs rendered) and
DocumentLM._call_batch_with_usage (no network calls). Verifies:
  * fast=True passes target_longest_dim=1024 to process_pdf; fast=False (the
    default) still passes 2048 -- normal-mode behavior is unchanged.
  * fast=True skips the temperature-escalation retry loop entirely, even for
    a page that would trigger a retry under normal mode (max_tokens exceeded,
    or is_table=True with no <table> tag): only one _call_batch_with_usage
    call happens, and the unmodified first-pass text ends up in self.text.
  * fast=False (unchanged behavior) still retries those same trigger cases.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import scholarlm.documentlm as documentlm_module
from scholarlm.documentlm import DocumentLM


MODEL_NAME = "allenai/olmOCR-2-7B-1025"


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
