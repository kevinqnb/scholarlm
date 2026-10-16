"""Rung-1 tests for analysis/runtime.py on hand-built run_metadata dicts and a fake HF cache."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))

from analysis import runtime as rt  # noqa: E402

GPU = [{"index": 0, "name": "NVIDIA L40S", "memory_total_gib": 44.4}]


def _pipeline_meta(**overrides) -> dict:
    steps = {s: {"ran": True, "seconds": 10.0} for s in rt.PIPELINE_STEPS}
    meta = {
        "gpu_info": GPU, "model_id": "org/m", "hf_revision": None, "dataset": "nfix",
        "step_seconds": steps, "total_step_seconds": 70.0,
        "token_usage": {"prompt_tokens": 2_500_000, "completion_tokens": 125_000, "successful_calls": 9, "failed_calls": 0},
    }
    meta.update(overrides)
    return meta


def _baseline_meta(**overrides) -> dict:
    meta = {
        "gpu_info": GPU, "model_id": "org/m", "hf_revision": None, "dataset": "nfix",
        "runtime_seconds": 1234.5, "token_accounting": "n/a: local GLiNER2, no API calls",
    }
    meta.update(overrides)
    return meta


def test_pipeline_stats_sum_steps():
    s = rt.run_stats(_pipeline_meta(), "extraction", "x")
    assert s["seconds"] == 70.0
    assert (s["prompt_tokens"], s["completion_tokens"]) == (2_500_000, 125_000)
    assert s["gpu"] == "NVIDIA L40S"


def test_final_only_run_rejected():
    # The pond gemma case: only `final` recorded, steps 1-6 came from another run's checkpoints.
    meta = _pipeline_meta(step_seconds={"final": {"ran": True, "seconds": 2457.0}}, total_step_seconds=2457.0)
    with pytest.raises(ValueError, match="another run's checkpoints"):
        rt.run_stats(meta, "extraction", "x")


def test_pipeline_step_not_run_rejected():
    meta = _pipeline_meta()
    meta["step_seconds"]["events"] = {"ran": False, "seconds": None}
    with pytest.raises(ValueError, match="events"):
        rt.run_stats(meta, "extraction", "x")


def test_pipeline_total_mismatch_rejected():
    with pytest.raises(ValueError, match="total_step_seconds"):
        rt.run_stats(_pipeline_meta(total_step_seconds=71.0), "extraction", "x")


def test_pipeline_runtime_shorter_than_steps_rejected():
    with pytest.raises(ValueError, match="runtime_seconds"):
        rt.run_stats(_pipeline_meta(runtime_seconds=60.0), "extraction", "x")


def test_baseline_na_tokens_render_dash():
    s = rt.run_stats(_baseline_meta(), "baseline_gliner", "x")
    assert s["seconds"] == 1234.5 and s["prompt_tokens"] is None
    s["weight_bytes"] = 1_945_828_140
    row = rt.format_row("GLiNER", s, {"seconds": 0, "tokens_millions": 1, "size_gb": 1}, "L40S", None)
    assert row == "GLiNER & 1,234 & -- & -- & 1.9 & L40S \\\\"


def test_missing_token_accounting_raises():
    meta = _baseline_meta()
    del meta["token_accounting"]
    with pytest.raises(ValueError, match="neither token_usage nor token_accounting"):
        rt.run_stats(meta, "ablation", "x")


def test_baseline_with_step_seconds_raises():
    with pytest.raises(ValueError, match="has step_seconds"):
        rt.run_stats(_baseline_meta(step_seconds={}), "ablation", "x")


def test_unknown_type_and_multi_gpu_raise():
    with pytest.raises(ValueError, match="unknown experiment type"):
        rt.run_stats(_baseline_meta(), "judge", "x")
    with pytest.raises(ValueError, match="exactly one GPU"):
        rt.run_stats(_baseline_meta(gpu_info=GPU * 2), "ablation", "x")


def test_pipeline_row_with_footnote():
    s = rt.run_stats(_pipeline_meta(), "extraction", "x")
    s["weight_bytes"] = 16_060_522_496
    row = rt.format_row("Pipeline", s, {"seconds": 0, "tokens_millions": 2, "size_gb": 1}, "L40S", "a")
    assert row == "Pipeline & 70 & 2.50$^{a}$ & 0.12$^{a}$ & 16.1 & L40S \\\\"


def test_build_table_footnote_text():
    spec = {
        "decimals": {"seconds": 0, "tokens_millions": 1, "size_gb": 1}, "gpu_labels": {"NVIDIA L40S": "L40S"},
        "footnotes": {"p": "Includes one discarded final step."}, "caption": "cap", "label": "tab:rt",
        "blocks": [{"rows": [{"label": "B", "id": "b"}]}, {"rows": [{"label": "P", "id": "p"}]}],
    }
    b = rt.run_stats(_baseline_meta(), "baseline_gliner", "b") | {"weight_bytes": 2_000_000_000}
    p = rt.run_stats(_pipeline_meta(), "extraction", "p") | {"weight_bytes": 4_000_000_000}
    text = rt.build_table(spec, {"b": b, "p": p})
    assert "B & 1,234 & -- & -- & 2.0 & L40S \\\\" in text
    assert "P & 70 & 2.5$^{a}$ & 0.1$^{a}$ & 4.0 & L40S \\\\" in text
    assert "$^{a}$Includes one discarded final step." in text
    assert text.count("\\midrule") == 2


def test_footnote_on_na_row_raises():
    spec = {
        "decimals": {"seconds": 0, "tokens_millions": 1, "size_gb": 1}, "gpu_labels": {"NVIDIA L40S": "L40S"},
        "footnotes": {"b": "x"}, "caption": "c", "label": "l", "blocks": [{"rows": [{"label": "B", "id": "b"}]}],
    }
    b = rt.run_stats(_baseline_meta(), "baseline_gliner", "b") | {"weight_bytes": 1}
    with pytest.raises(ValueError, match="footnote"):
        rt.build_table(spec, {"b": b})


# --- model size from a fake HF cache -------------------------------------------------


def _snapshot(hub: Path, model_id: str, sha: str, files: dict[str, int], index: bool) -> None:
    snap = hub / f"models--{model_id.replace('/', '--')}" / "snapshots" / sha
    snap.mkdir(parents=True)
    for name, n in files.items():
        (snap / name).write_bytes(b"\0" * n)
    if index:
        (snap / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {f"t{i}": name for i, name in enumerate(files)}}))


def _ref(hub: Path, model_id: str, sha: str) -> None:
    refs = hub / f"models--{model_id.replace('/', '--')}" / "refs"
    refs.mkdir(parents=True, exist_ok=True)
    (refs / "main").write_text(sha)


def test_size_sums_indexed_shards_only(tmp_path):
    _snapshot(tmp_path, "org/m", "aaa", {"model-1.safetensors": 300, "model-2.safetensors": 200}, index=True)
    (tmp_path / "models--org--m/snapshots/aaa/stray.bin").write_bytes(b"\0" * 999)  # not in the index
    _ref(tmp_path, "org/m", "aaa")
    assert rt.model_weight_bytes(tmp_path, "org/m", None) == 500


def test_size_single_file_and_equal_other_snapshot(tmp_path):
    _snapshot(tmp_path, "org/m", "aaa", {"model.safetensors": 42}, index=False)
    _snapshot(tmp_path, "org/m", "bbb", {"model.safetensors": 42}, index=False)
    _ref(tmp_path, "org/m", "bbb")
    assert rt.model_weight_bytes(tmp_path, "org/m", None) == 42


def test_size_differing_snapshots_raise(tmp_path):
    _snapshot(tmp_path, "org/m", "aaa", {"model.safetensors": 42}, index=False)
    _snapshot(tmp_path, "org/m", "bbb", {"model.safetensors": 43}, index=False)
    _ref(tmp_path, "org/m", "bbb")
    with pytest.raises(ValueError, match="cannot tell"):
        rt.model_weight_bytes(tmp_path, "org/m", None)


def test_size_missing_model_or_shard_raise(tmp_path):
    with pytest.raises(FileNotFoundError):
        rt.model_weight_bytes(tmp_path, "org/absent", None)
    _snapshot(tmp_path, "org/m", "aaa", {"model-1.safetensors": 1}, index=True)
    (tmp_path / "models--org--m/snapshots/aaa/model-1.safetensors").unlink()
    _ref(tmp_path, "org/m", "aaa")
    with pytest.raises(FileNotFoundError, match="not downloaded"):
        rt.model_weight_bytes(tmp_path, "org/m", None)


# --- interp-judge table ----------------------------------------------------------------


def _judge_meta(**overrides) -> dict:
    meta = {
        "gpu_info": GPU, "dataset": "supermat", "extraction_id": "ext", "judge_model": "llama-3.1-8b",
        "judge_model_id": "org/j", "max_prompt_tokens": 30, "runtime_seconds": 6393.1,
    }
    meta.update(overrides)
    return meta


def test_judge_stats_and_wrong_extraction():
    s = rt.judge_run_stats(_judge_meta(), "j", "ext")
    assert (s["seconds"], s["gpu"], s["model_id"], s["max_prompt_tokens"]) == (6393.1, "NVIDIA L40S", "org/j", 30)
    with pytest.raises(ValueError, match="judged 'ext'"):
        rt.judge_run_stats(_judge_meta(), "j", "other")
    meta = _judge_meta()
    del meta["runtime_seconds"]
    with pytest.raises(ValueError, match="runtime_seconds"):
        rt.judge_run_stats(meta, "j", "ext")


def test_prompt_length_checks():
    assert rt.check_prompt_lengths([10, 30, 20], 3, 30, "j") == 60
    with pytest.raises(ValueError, match="rebuilt 3 prompts"):
        rt.check_prompt_lengths([10, 30, 20], 4, 30, "j")
    with pytest.raises(ValueError, match="longest rebuilt prompt is 30"):
        rt.check_prompt_lengths([10, 30, 20], 3, 31, "j")


def test_activation_bytes(tmp_path):
    (tmp_path / "attention_outputs.npz").write_bytes(b"\0" * 300)
    with pytest.raises(FileNotFoundError, match="layer_outputs"):
        rt.activation_bytes(tmp_path)
    (tmp_path / "layer_outputs.npz").write_bytes(b"\0" * 200)
    assert rt.activation_bytes(tmp_path) == 500


def test_judge_table_by_inspection():
    spec = {
        "decimals": {"seconds": 0, "tokens_millions": 1, "size_gb": 2}, "gpu_labels": {"NVIDIA L40S": "L40S"},
        "caption": "cap", "label": "tab:j", "rows": [{"label": "Llama", "id": "j"}],
    }
    s = rt.judge_run_stats(_judge_meta(), "j", "ext") | {
        "prompt_tokens": 43_031_662, "weight_bytes": 16_060_522_496, "activation_bytes": 2_033_221_806}
    text = rt.build_judge_table(spec, {"j": s})
    assert "Judge & Time (s) & Input tok.\\ (M) & Weights (GB) & Activations (GB) & GPU \\\\" in text
    assert "Llama & 6,393 & 43.0 & 16.06 & 2.03 & L40S \\\\" in text
