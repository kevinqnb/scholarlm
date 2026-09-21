"""Integration check for run_extraction.py's run_pipeline <-> write_run_metadata
wiring, using a tiny 2-document fixture with every step_* function stubbed
(no LLM calls, no vLLM server).

This is the smoke/tiny-e2e level for the compute-time-recording fix: unit
tests already pin write_run_metadata's merge arithmetic (test_utils_run_metadata.py)
and MeasurementLM._acall's token accounting (test_measurementlm.py) in
isolation. What those don't catch is a plumbing bug in run_pipeline/_run_all_steps
itself -- e.g. step_seconds or token_usage built but never threaded through,
or the wrong dict passed. This exercises the real run_pipeline() end to end
against real files on disk, across two invocations, to catch that class of bug.

Predicted result before running: after invocation 1 (fresh), all 7 steps show
ran=True and token_usage/max_prompt_tokens reflect exactly the 7 stubbed
calls. After deleting attributes.json and re-invoking with resume=True, only
"attributes" re-runs -- the other 6 steps' original timings must survive in
step_seconds (not be reset to ran=False with no data), and token_usage must
be the SUM of both invocations' stubbed calls, not just the second one's.
"""
import json
import sys
import time
from pathlib import Path

import pytest
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

import run_extraction
from scholarlm.config import DatasetConfig, ModelConfig


class _EntitySchema(BaseModel):
    name: str | None


def _make_fixture(tmp_path):
    ocr_dir = tmp_path / "ocr"
    ocr_dir.mkdir()
    (ocr_dir / "doc1.txt").write_text("Site A depth 10m.")
    (ocr_dir / "doc2.txt").write_text("Site B depth 20m.")
    metadata_file = tmp_path / "metadata.json"
    metadata_file.write_text(json.dumps({"doc1": {}, "doc2": {}}))

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


def _stub_steps(monkeypatch, prompt_tokens=100, completion_tokens=20):
    """Replace every module-level step_* function with one that writes a
    minimal valid checkpoint and bumps mlm.token_usage/max_prompt_tokens by a
    fixed amount, simulating exactly one real LLM call per step without any
    network access."""
    calls = {"count": 0}

    def _record_usage(mlm):
        calls["count"] += 1
        mlm.token_usage["prompt_tokens"] += prompt_tokens
        mlm.token_usage["completion_tokens"] += completion_tokens
        mlm.token_usage["successful_calls"] += 1
        mlm.max_prompt_tokens = max(mlm.max_prompt_tokens, prompt_tokens)

    def make_stub(content):
        def stub(mlm, *args, **kwargs):
            _record_usage(mlm)
            time.sleep(0.02)
            outfile = args[-1]  # every step_* function takes outfile as its last positional arg
            outfile.parent.mkdir(parents=True, exist_ok=True)
            with open(outfile, "w") as f:
                json.dump(content, f)
        return stub

    monkeypatch.setattr(run_extraction, "step_extract_entities", make_stub([]))
    monkeypatch.setattr(run_extraction, "step_detect_attributes", make_stub({}))
    monkeypatch.setattr(run_extraction, "step_entity_provenance", make_stub([]))
    monkeypatch.setattr(run_extraction, "step_attribute_provenance", make_stub([]))
    monkeypatch.setattr(run_extraction, "step_resolve_events", make_stub(None))
    monkeypatch.setattr(run_extraction, "step_extract_values", make_stub([]))
    monkeypatch.setattr(run_extraction, "step_standardize_and_deduplicate", make_stub([]))
    return calls


def test_fresh_then_resumed_run_accumulates_step_seconds_and_token_usage(tmp_path, monkeypatch):
    dataset_config, model_config, ocr_dir = _make_fixture(tmp_path)
    output_dir = tmp_path / "results"

    # --- Invocation 1: fresh run, all 7 steps execute ---
    _stub_steps(monkeypatch)
    run_extraction.run_pipeline(dataset_config, model_config, output_dir, ocr_dir=ocr_dir, resume=False)

    metadata = json.loads((output_dir / "run_metadata.json").read_text())
    step_names = {"entities", "attributes", "entity_prov", "attribute_prov", "events", "values", "final"}
    assert set(metadata["step_seconds"]) == step_names
    assert all(v["ran"] is True and v["seconds"] is not None for v in metadata["step_seconds"].values())
    assert metadata["resume"] is False
    assert metadata["token_usage"] == {
        "prompt_tokens": 700, "completion_tokens": 140, "successful_calls": 7, "failed_calls": 0,
    }
    assert metadata["max_prompt_tokens"] == 100
    first_run_step_seconds = metadata["step_seconds"]

    # --- Force exactly one step to re-run: delete its checkpoint ---
    (output_dir / "attributes.json").unlink()

    # --- Invocation 2: resumed run ---
    calls = _stub_steps(monkeypatch)
    run_extraction.run_pipeline(dataset_config, model_config, output_dir, ocr_dir=ocr_dir, resume=True)
    assert calls["count"] == 1, "only the deleted checkpoint's step should have re-run"

    metadata2 = json.loads((output_dir / "run_metadata.json").read_text())
    assert metadata2["resume"] is True
    # The 6 untouched steps keep their original timing -- being "skipped this
    # invocation" must not erase a real timing recorded by an earlier one.
    for name in step_names - {"attributes"}:
        assert metadata2["step_seconds"][name] == first_run_step_seconds[name]
    assert metadata2["step_seconds"]["attributes"]["ran"] is True
    # token_usage sums BOTH invocations' calls (700+100, 140+20, 7+1) -- not
    # just the second invocation's single call, which is the bug this fixes.
    assert metadata2["token_usage"] == {
        "prompt_tokens": 800, "completion_tokens": 160, "successful_calls": 8, "failed_calls": 0,
    }
    assert metadata2["total_step_seconds"] == pytest.approx(
        sum(v["seconds"] for v in metadata2["step_seconds"].values())
    )
