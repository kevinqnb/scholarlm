"""Integration check for run_ablation.py's write_run_metadata wiring of
mlm.step_seconds -- the same class of check as
test_run_extraction_metadata_wiring.py, for the ablation runner instead.

Ablation 4 is used because it inherits the base MeasurementLM.fit() unchanged
(no ablation-specific fit() override to also exercise -- that's covered
directly against the class in test_measurementlm_ablation_step_seconds.py for
ablations 2/3, and implicitly for 4/5/6 by the base-fit() test in
test_measurementlm.py). What this test adds is confidence that
run_ablation.py's own plumbing -- reading mlm.step_seconds after fit()
returns and passing it into write_run_metadata -- isn't a step ahead of or
behind what fit() actually populates.

Predicted result before running: run_metadata.json for the ablation-4 output
dir has a step_seconds key with exactly {entities, entity_prov, attributes,
attribute_prov, events, values_text, values_tables, final}, all ran=True, and
total_step_seconds equal to their sum.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import run_ablation
from scholarlm.measurementlm import MeasurementLM
from scholarlm.measurementlm_ablation4 import MeasurementLMAblation4
from scholarlm.config import DatasetConfig, ModelConfig


def _make_fixture(tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "doc1.txt").write_text("Site A depth 10m.")
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"doc1": {}}))

    from pydantic import BaseModel

    class _EntitySchema(BaseModel):
        name: str | None = None

    dataset_config = DatasetConfig(
        name="fixture",
        data_dir=str(tmp_path),
        metadata_file=str(metadata_file),
        entity_schema=_EntitySchema,
        entity_identification_prompt="Identify entities.",
        entity_type_description="site",
        attribute_info_dict={"depth": {"description": "Depth", "units": ["m"]}},
    )
    model_config = ModelConfig(name="fixture-model", model_id="fixture/fixture-model", sampling_params={})
    return dataset_config, model_config, str(ocr_dir)


def test_ablation4_run_writes_step_seconds_matching_base_pipeline_keys(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    output_dir = tmp_path / "results"

    # Stub the inherited base-class steps and ablation4's own overrides so no
    # network call happens; every stub returns an empty result, which the
    # unchanged downstream steps handle fine (mirrors
    # test_pipeline_mode_default_construction_dispatches_original_seven_steps).
    monkeypatch.setattr(MeasurementLM, "_extract_entities", lambda self: [])
    monkeypatch.setattr(MeasurementLM, "_entity_provenance", lambda self, entity_data: {})
    monkeypatch.setattr(MeasurementLM, "_detect_attributes", lambda self: {})
    monkeypatch.setattr(MeasurementLM, "_attribute_provenance", lambda self, doc_attributes: {})
    monkeypatch.setattr(MeasurementLM, "_standardize", lambda self: [])
    monkeypatch.setattr(MeasurementLM, "_parse_quantities", lambda self: [])
    monkeypatch.setattr(MeasurementLM, "_deduplicate", lambda self, data: [])
    monkeypatch.setattr(
        MeasurementLMAblation4, "_resolve_events",
        lambda self, entity_data, doc_attributes, entity_prov, attr_prov: None,
    )
    monkeypatch.setattr(
        MeasurementLMAblation4, "_extract_values_from_text",
        lambda self, *a, **kw: [],
    )
    monkeypatch.setattr(
        MeasurementLMAblation4, "_extract_values_from_tables",
        lambda self, *a, **kw: [],
    )

    run_ablation.run_ablation(dataset_config, model_config, "4", output_dir, ocr_dir=ocr_dir)

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    expected_keys = {
        "entities", "entity_prov", "attributes", "attribute_prov", "events",
        "values_text", "values_tables", "final",
    }
    assert set(metadata["step_seconds"]) == expected_keys
    assert all(v["ran"] is True for v in metadata["step_seconds"].values())
    assert metadata["total_step_seconds"] == sum(v["seconds"] for v in metadata["step_seconds"].values())
