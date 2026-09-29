"""Ablation 1's params.max_items: a JSON-Schema maxItems on the response's top-level
"items" list, plus bookkeeping of which documents hit the cap or failed validation.

Rung-1 checks on a hand-built fixture (no model, no server): _call_batch is stubbed
to return canned response texts, so every expected output can be verified by eye.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "experiments"))

from scholarlm.measurementlm_ablation1 import MeasurementLMAblation1  # noqa: E402
import run_ablation  # noqa: E402


class _Item(BaseModel):
    attribute: str
    value: str | None


def _make(max_items, monkeypatch, responses):
    """Build an ablation-1 instance whose _call_batch returns `responses` and
    records the response_format it was given."""
    mlm = MeasurementLMAblation1.__new__(MeasurementLMAblation1)
    # Bypass MeasurementLM.__init__ (needs a server); set only what
    # _extract_triples touches.
    mlm.direct_extraction_schema = _Item
    mlm.direct_extraction_prompt = "prompt"
    mlm.direct_extraction_instructions = "instructions"
    mlm.max_items = max_items
    mlm.capped_document_ids = []
    mlm.failed_document_ids = []
    mlm.data = [{"document_id": i, "context": f"doc {i}"} for i in range(len(responses))]
    seen = {}

    def fake_call_batch(messages, **kwargs):
        seen["response_format"] = kwargs["response_format"]
        return responses

    monkeypatch.setattr(mlm, "_call_batch", fake_call_batch)
    return mlm, seen


def _resp(n):
    return json.dumps({"items": [{"attribute": "a", "value": str(k)} for k in range(n)]})


def test_schema_carries_maxitems_only_when_set(monkeypatch):
    mlm, seen = _make(3, monkeypatch, [_resp(1)])
    mlm._extract_triples()
    assert seen["response_format"]["json_schema"]["schema"]["properties"]["items"]["maxItems"] == 3

    mlm, seen = _make(None, monkeypatch, [_resp(1)])
    mlm._extract_triples()
    assert "maxItems" not in seen["response_format"]["json_schema"]["schema"]["properties"]["items"]


def test_capped_and_failed_documents_are_recorded(monkeypatch):
    # doc 0: under the cap (2), doc 1: exactly at the cap (3), doc 2: unparseable,
    # doc 3: empty list.
    mlm, _ = _make(3, monkeypatch, [_resp(2), _resp(3), "not json", _resp(0)])
    out = mlm._extract_triples()
    assert mlm.capped_document_ids == [1]
    assert mlm.failed_document_ids == [2]
    assert [t["document_id"] for t in out] == [0, 0, 1, 1, 1]  # 2 + 3 records, none from doc 2/3


def test_response_over_the_cap_is_a_validation_failure_not_silently_truncated(monkeypatch):
    # A server that ignored maxItems would return 4 items; that must be flagged,
    # not accepted.
    mlm, _ = _make(3, monkeypatch, [_resp(4)])
    out = mlm._extract_triples()
    assert out == []
    assert mlm.failed_document_ids == [0]
    assert mlm.capped_document_ids == []


def test_uncapped_records_nothing_as_capped(monkeypatch):
    mlm, _ = _make(None, monkeypatch, [_resp(50)])
    mlm._extract_triples()
    assert mlm.capped_document_ids == []


@pytest.mark.parametrize("bad", [0, -1, 2.5, "10", True])
def test_bad_max_items_raises(bad):
    with pytest.raises(ValueError, match="max_items"):
        MeasurementLMAblation1(
            model_name="m", entity_identification_prompt="p",
            entity_identification_schema=_Item, attribute_info_dict={},
            sampling_params={}, api_base="http://localhost:1/v1", api_key="x",
            measurement_event_schema=_Item, measurement_event_prompt="p",
            use_extra_body=True, collect_attribute_terms=False, max_items=bad,
        )


def test_max_items_rejected_for_other_ablations():
    with pytest.raises(ValueError, match="only applies to ablation '1'"):
        run_ablation.run_ablation(
            dataset_config=None, model_config=None, ablation="2",
            output_dir=Path("/nonexistent"), max_items=10,
        )
