"""Rung-1 (model-free) unit tests for experiments/run_attribution.py.

See ``notes/scholarlm/builds/2026-08-31-attribution-runner-01.md`` staged-gate
ladder. These cover the runner plumbing only — input loading, the npz/sidecar
schema, and the measurement_id <-> interp-judge responses.json join — against a
stub ``AttributionMethod``. The GPU rungs (smoke / tiny-e2e / full) and the
pH=8.2 known-answer fixture run as submitted jobs, not here.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import run_attribution as ra


# ---------------------------------------------------------------------------
# Fixtures / stubs
# ---------------------------------------------------------------------------


class StubMethod:
    """Deterministic fake AttributionMethod: score length keyed on context text."""

    def __init__(self, n_by_context: dict[str, int], scalar_key: str):
        self.n_by_context = n_by_context
        self.scalar_key = scalar_key
        self.calls: list[tuple[str, str, str]] = []

    def attribute(self, instructions: str, context: str, query: str) -> dict:
        self.calls.append((instructions, context, query))
        n = self.n_by_context[context]
        return {
            "scores": (np.arange(n, dtype=np.float64) * 0.1),
            "context_token_indices": list(range(5, 5 + n)),
            self.scalar_key: 0.5,
        }


def _data():
    return [
        {"measurement_id": 0, "document_id": "docA"},
        {"measurement_id": 1, "document_id": "docB"},
    ]


def _chat_entries():
    return [
        {"custom_id": "0", "document_id": "docA", "system": "S", "page_text": "ctxA", "user_query": "Q0"},
        {"custom_id": "1", "document_id": "docB", "system": "S", "page_text": "ctxB", "user_query": "Q1"},
    ]


def _responses_by_mid():
    return {
        "0": {"measurement_id": 0, "document_id": "docA", "judgement": True,
              "judgement_p_true": 0.9, "judgement_p_false": 0.1},
        "1": {"measurement_id": 1, "document_id": "docB", "judgement": False,
              "judgement_p_true": 0.2, "judgement_p_false": 0.8},
    }


# ---------------------------------------------------------------------------
# npz / sidecar schema
# ---------------------------------------------------------------------------


def test_npz_and_sidecar_schema(tmp_path):
    method = StubMethod({"ctxA": 3, "ctxB": 5}, scalar_key="target")
    summary = ra.attribute_dataset(
        method=method,
        method_name="contrastive_gradient",
        data=_data(),
        chat_entries=_chat_entries(),
        responses_by_mid=_responses_by_mid(),
        output_dir=tmp_path,
    )

    npz = np.load(tmp_path / "attribution_scores.npz", allow_pickle=True)
    keys = set(npz.files)
    assert keys == {"measurement_ids", "0", "0__context_token_indices", "1", "1__context_token_indices"}
    assert list(npz["measurement_ids"]) == ["0", "1"]

    # every {mid} has a matching index array, dtypes as documented, lengths agree
    for mid, n in (("0", 3), ("1", 5)):
        assert npz[mid].shape == (n,)
        assert npz[mid].dtype == np.float32
        assert npz[f"{mid}__context_token_indices"].shape == (n,)
        assert npz[f"{mid}__context_token_indices"].dtype == np.int32
        assert list(npz[f"{mid}__context_token_indices"]) == list(range(5, 5 + n))

    sidecar = json.loads((tmp_path / "attribution.json").read_text())
    assert [r["measurement_id"] for r in sidecar] == [0, 1]
    assert sidecar[0] == {
        "measurement_id": 0,
        "document_id": "docA",
        "method": "contrastive_gradient",
        "scalar_name": "target",
        "target": 0.5,
        "judgement": True,
        "judgement_p_true": 0.9,
        "judgement_p_false": 0.1,
        "n_context_tokens": 3,
    }
    assert summary["n_measurements"] == 2
    assert summary["context_token_count_min"] == 3
    assert summary["context_token_count_max"] == 5


def test_probe_scalar_key_handled(tmp_path):
    method = StubMethod({"ctxA": 2, "ctxB": 2}, scalar_key="probe_output")
    ra.attribute_dataset(
        method=method,
        method_name="probe",
        data=_data(),
        chat_entries=_chat_entries(),
        responses_by_mid=_responses_by_mid(),
        output_dir=tmp_path,
    )
    sidecar = json.loads((tmp_path / "attribution.json").read_text())
    assert sidecar[0]["scalar_name"] == "probe_output"
    assert sidecar[0]["probe_output"] == 0.5
    assert "target" not in sidecar[0]


def test_wrong_scalar_key_fails_loud(tmp_path):
    # method_name says probe (expects "probe_output") but stub returns "target"
    method = StubMethod({"ctxA": 2, "ctxB": 2}, scalar_key="target")
    with pytest.raises(AssertionError, match="probe_output"):
        ra.attribute_dataset(
            method=method, method_name="probe", data=_data(),
            chat_entries=_chat_entries(), responses_by_mid=_responses_by_mid(),
            output_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# boundary assert
# ---------------------------------------------------------------------------


def test_scores_index_length_mismatch_fails_loud(tmp_path):
    class BadStub:
        def attribute(self, instructions, context, query):
            return {"scores": np.zeros(3), "context_token_indices": [1, 2], "target": 0.0}

    with pytest.raises(AssertionError, match="len\\(scores\\)"):
        ra.attribute_dataset(
            method=BadStub(), method_name="contrastive_gradient", data=_data(),
            chat_entries=_chat_entries(), responses_by_mid=_responses_by_mid(),
            output_dir=tmp_path,
        )


def test_nonfinite_scores_fail_loud(tmp_path):
    class NanStub:
        def attribute(self, instructions, context, query):
            return {"scores": np.array([1.0, np.nan]), "context_token_indices": [1, 2], "target": 0.0}

    with pytest.raises(AssertionError, match="non-finite"):
        ra.attribute_dataset(
            method=NanStub(), method_name="contrastive_gradient", data=_data(),
            chat_entries=_chat_entries()[:1], responses_by_mid=_responses_by_mid(),
            output_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# measurement_id <-> responses.json join
# ---------------------------------------------------------------------------


def test_join_miss_fails_loud(tmp_path):
    method = StubMethod({"ctxA": 2, "ctxB": 2}, scalar_key="target")
    responses = {"0": _responses_by_mid()["0"]}  # missing "1"
    with pytest.raises(AssertionError, match="no record in the paired interp-judge"):
        ra.attribute_dataset(
            method=method, method_name="contrastive_gradient", data=_data(),
            chat_entries=_chat_entries(), responses_by_mid=responses,
            output_dir=tmp_path,
        )


def test_join_to_skipped_judge_row_fails_loud(tmp_path):
    method = StubMethod({"ctxA": 2, "ctxB": 2}, scalar_key="target")
    responses = _responses_by_mid()
    responses["1"] = {"measurement_id": 1, "judgement": False, "judgement_p_true": None, "judgement_p_false": None}
    with pytest.raises(AssertionError, match="judgement_p_true=None"):
        ra.attribute_dataset(
            method=method, method_name="contrastive_gradient", data=_data(),
            chat_entries=_chat_entries(), responses_by_mid=responses,
            output_dir=tmp_path,
        )


def test_join_document_id_mismatch_fails_loud(tmp_path):
    method = StubMethod({"ctxA": 2, "ctxB": 2}, scalar_key="target")
    responses = _responses_by_mid()
    responses["1"]["document_id"] = "docWRONG"  # not row-aligned with chat entry docB
    with pytest.raises(AssertionError, match="not row-aligned"):
        ra.attribute_dataset(
            method=method, method_name="contrastive_gradient", data=_data(),
            chat_entries=_chat_entries(), responses_by_mid=responses,
            output_dir=tmp_path,
        )


def test_duplicate_measurement_id_in_input_fails_loud(tmp_path):
    method = StubMethod({"ctxA": 2, "ctxB": 2}, scalar_key="target")
    data = [{"measurement_id": 0, "document_id": "docA"}, {"measurement_id": 0, "document_id": "docB"}]
    with pytest.raises(AssertionError, match="appears twice"):
        ra.attribute_dataset(
            method=method, method_name="contrastive_gradient", data=data,
            chat_entries=_chat_entries(), responses_by_mid=_responses_by_mid(),
            output_dir=tmp_path,
        )


def test_load_responses_by_mid_dedup(tmp_path):
    p = tmp_path / "responses.json"
    p.write_text(json.dumps([{"measurement_id": 0}, {"measurement_id": 0}]))
    with pytest.raises(AssertionError, match="duplicate measurement_id"):
        ra._load_responses_by_mid(p)


# ---------------------------------------------------------------------------
# seed: no fallback (CLAUDE.md; run_judge_interp.py:263 is the anti-pattern)
# ---------------------------------------------------------------------------


def test_analysis_loaders_importable_via_runner():
    # `--method probe` does `from analysis.loaders import load_trained_probe` at
    # runtime (run_attribution.run_attribution). `analysis` is a repo-root
    # package; when the script runs as `python experiments/run_attribution.py`,
    # sys.path[0] is experiments/, not the repo root — the runner must add the
    # repo root itself or the probe path ModuleNotFoundErrors deep into a GPU job.
    import importlib

    importlib.import_module("run_attribution")  # triggers its sys.path setup
    importlib.import_module("analysis.loaders")


def test_seed_read_has_no_fallback():
    src = (_REPO_ROOT / "experiments" / "run_attribution.py").read_text()
    assert 'cfg["defaults"]["seed"]' in src
    assert not re.search(r"\.get\(\s*[\"']seed[\"']", src)
    assert "342" not in src  # no hardcoded seed default anywhere


# ---------------------------------------------------------------------------
# input loading — real pond data, model-free (skipped if data absent)
# ---------------------------------------------------------------------------


def test_load_inputs_real_pond_extraction():
    final = _REPO_ROOT / "data/experiments/pond/extraction/gemma-3-27b/2026_05_05/final.json"
    ocr = _REPO_ROOT / "data/pond/ocr_output_raw"
    if not final.exists() or not ocr.exists():
        pytest.skip("pond extraction / OCR data not present")

    from run_extraction import load_dataset_config
    cfg = load_dataset_config("pond")
    data, chat_entries = ra._load_inputs(cfg, final, ocr_dir=str(ocr), limit=3)
    assert len(data) == 3
    # prepare_chat_entries may skip records; it never invents them
    assert 0 < len(chat_entries) <= 3
    for e in chat_entries:
        assert int(e["custom_id"]) < 3
        assert {"system", "page_text", "user_query", "document_id"} <= set(e)
