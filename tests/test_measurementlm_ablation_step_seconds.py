"""Unit tests for per-step timing (mlm.step_seconds) in ablations 2 and 3's
own fit() overrides.

Ablations 4, 5, and 6 don't override fit() at all -- they inherit the base
MeasurementLM.fit(), so they get step_seconds "entities"/"entity_prov"/
"attributes"/"attribute_prov"/"events"/"values"/"final" for free, already
covered by test_measurementlm.py's
test_pipeline_mode_default_construction_dispatches_original_seven_steps.
Ablations 2 and 3 instead merge or drop specific steps (that's the point of
the ablation), so their fit() overrides needed their own _timed_step calls --
these tests confirm the resulting step_seconds keys reflect exactly which
steps each ablation actually kept, merged, or dropped, using stubbed internal
methods (no LLM calls).
"""
import sys
from pathlib import Path

from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scholarlm.measurementlm_ablation2 import MeasurementLMAblation2
from scholarlm.measurementlm_ablation3 import MeasurementLMAblation3


class _EntityAttrSchema(BaseModel):
    name: str | None = None
    location: str | None = None
    attribute: str | None = None
    attribute_terms: list[str] | None = None


_ATTRIBUTE_INFO = {"depth": {"description": "Maximum depth", "units": ["m"]}}


def _stub(*args, **kwargs):
    return []


def test_ablation2_fit_records_its_own_merged_step_names(monkeypatch):
    """Ablation 2 merges entity+attribute detection into one step and both
    provenance steps into one -- step_seconds must use names that reflect
    that merge (entity_attribute_pairs, pair_provenance), not the base
    pipeline's separate entities/attributes/entity_prov/attribute_prov keys,
    since those steps don't exist as separate calls in this ablation."""
    mlm = MeasurementLMAblation2(
        model_name="test-model",
        entity_identification_prompt="Identify entity-attribute pairs.",
        entity_identification_schema=_EntityAttrSchema,
        attribute_info_dict=_ATTRIBUTE_INFO,
        api_base="http://localhost:0/v1",
        use_extra_body=False,
    )
    monkeypatch.setattr(mlm, "_extract_entity_attribute_pairs", lambda: [])
    monkeypatch.setattr(mlm, "_entity_attribute_provenance", lambda pair_data: {})
    monkeypatch.setattr(mlm, "_extract_values_from_text", _stub)
    monkeypatch.setattr(mlm, "_extract_values_from_tables", _stub)
    monkeypatch.setattr(mlm, "_standardize", lambda: [])
    monkeypatch.setattr(mlm, "_parse_quantities", lambda: [])
    monkeypatch.setattr(mlm, "_deduplicate", lambda data: [])

    result = mlm.fit(["doc text"])

    assert result == []
    assert set(mlm.step_seconds) == {
        "entity_attribute_pairs", "pair_provenance", "events", "values_text", "values_tables", "final",
    }


def test_ablation3_fit_keeps_entities_attributes_but_merges_provenance(monkeypatch):
    """Ablation 3 keeps entity extraction and attribute detection as separate
    steps (unlike ablation 2) but replaces both provenance steps with one
    full-document call -- step_seconds must show entities/attributes still
    separate, plus a single pair_provenance_full_context key instead of
    entity_prov + attribute_prov. Named differently from ablation 2's
    pair_provenance since the two replace provenance with different
    mechanisms (per-page combined query vs. one full-document call)."""
    mlm = MeasurementLMAblation3(
        model_name="test-model",
        entity_identification_prompt="Identify entities.",
        entity_identification_schema=_EntityAttrSchema,
        attribute_info_dict=_ATTRIBUTE_INFO,
        api_base="http://localhost:0/v1",
        use_extra_body=False,
    )
    monkeypatch.setattr(mlm, "_extract_entities", lambda: [])
    monkeypatch.setattr(mlm, "_detect_attributes", lambda: {})
    monkeypatch.setattr(mlm, "_pair_provenance_full_context", lambda entity_data, doc_attributes: {})
    monkeypatch.setattr(mlm, "_extract_values_from_text", _stub)
    monkeypatch.setattr(mlm, "_extract_values_from_tables", _stub)
    monkeypatch.setattr(mlm, "_standardize", lambda: [])
    monkeypatch.setattr(mlm, "_parse_quantities", lambda: [])
    monkeypatch.setattr(mlm, "_deduplicate", lambda data: [])

    result = mlm.fit(["doc text"])

    assert result == []
    assert set(mlm.step_seconds) == {
        "entities", "attributes", "pair_provenance_full_context", "events", "values_text", "values_tables", "final",
    }
