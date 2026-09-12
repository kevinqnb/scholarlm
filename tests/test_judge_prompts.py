"""Rung-1 unit tests for ``experiments/judge_common.prepare_chat_entries``.

Hand-built fixture, model-free.  The invariant under test: the judge's
``## CONTEXT`` is always the whole OCR paper — never a page slice — unless the
row carries an explicit ``context_override``.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import judge_common


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

_DOC = (
    '<page number="0">Intro. Site A described here.</page>\n\n'
    '<page number="1">Total phosphorus was 5 ug/L at Site A.</page>\n\n'
    '<page number="2">References.</page>'
)


def _cfg():
    """Minimal stand-in for ``DatasetConfig`` — only the attributes
    ``prepare_chat_entries`` reads."""
    return SimpleNamespace(
        judge_filter_fields=["identifiers"],
        entity_schema=SimpleNamespace(model_fields={"name": None, "identifiers": None}),
        measurement_event_schema=None,
        attribute_info_dict={"tp": {"description": "Total phosphorus concentration."}},
        entity_type_description="a sampled water body",
        judge_instructions="JUDGE",
    )


def _row(**over):
    base = {
        "document_id": "X", "measurement_id": 0, "attribute": "tp",
        "value": "5", "units": "ug/L", "name": "Site A",
        "page_number": [1], "attribute_terms": [],
    }
    base.update(over)
    return base


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_context_is_the_whole_paper():
    entries = judge_common.prepare_chat_entries([_row()], {"X": _DOC}, _cfg())
    assert len(entries) == 1
    e = entries[0]
    assert e["context_text"] == _DOC
    # every page block survives — nothing is sliced out
    for pn in (0, 1, 2):
        assert f'<page number="{pn}">' in e["context_text"]
    assert e["user"] == f"## CONTEXT:\n{_DOC}\n\n## QUERY:\n{e['user_query']}"
    assert "page_text" not in e


def test_page_number_is_not_consulted():
    """A row whose page_number points nowhere (or is absent) still gets the
    full paper, byte-identical to a row with a valid page_number."""
    good = judge_common.prepare_chat_entries([_row(page_number=[1])], {"X": _DOC}, _cfg())
    bogus = judge_common.prepare_chat_entries([_row(page_number=[99])], {"X": _DOC}, _cfg())
    missing = judge_common.prepare_chat_entries([_row(page_number=None)], {"X": _DOC}, _cfg())
    assert good[0]["context_text"] == bogus[0]["context_text"] == missing[0]["context_text"] == _DOC


def test_context_override_wins():
    entries = judge_common.prepare_chat_entries(
        [_row(context_override="EDITED PAPER")], {"X": _DOC}, _cfg()
    )
    assert entries[0]["context_text"] == "EDITED PAPER"
    assert "EDITED PAPER" in entries[0]["user"]


def test_null_override_falls_back_to_full_paper():
    entries = judge_common.prepare_chat_entries(
        [_row(context_override=None)], {"X": _DOC}, _cfg()
    )
    assert entries[0]["context_text"] == _DOC
