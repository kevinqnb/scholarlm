"""Rung-1 unit tests for experiments/utils.py's new id-based addressing and
model-config loading (the new-framework counterparts to the legacy
(dataset, model, date) helpers covered by tests/test_utils_paths.py).

Hand-built fixtures, no real experiment-configs/model-configs data touched --
result_dir/find_result_dir/experiment_config_dir are tested against a
monkeypatched tmp_path root; load_model_config/classify_gpu_need are tested
against literal dicts plus one real model-config file to catch schema drift.
"""
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import utils


# ---------------------------------------------------------------------------
# result_dir / find_result_dir / experiment_config_dir
# ---------------------------------------------------------------------------


VALID_ID = "2026-05-05-pond-gemma-3-27b-extraction-01"


def test_result_dir_shape():
    p = utils.result_dir("pond", "extraction", VALID_ID)
    assert p.parts[-3:] == ("pond", "extraction", VALID_ID)
    assert p.parent.parent.parent == utils.RESULTS_ROOT


def test_result_dir_rejects_malformed_id():
    for bad in ("not-an-id", "2026-05-05-pond", "05-05-2026-pond-01"):
        with pytest.raises(ValueError):
            utils.result_dir("pond", "extraction", bad)


def test_experiment_config_dir_shape():
    p = utils.experiment_config_dir("pond", "extraction", VALID_ID)
    assert p.parts[-3:] == ("pond", "extraction", VALID_ID)
    assert p.parent.parent.parent == utils.EXPERIMENT_CONFIGS_ROOT


def test_find_result_dir_locates_by_id_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "RESULTS_ROOT", tmp_path)
    target = tmp_path / "pond" / "extraction" / VALID_ID
    target.mkdir(parents=True)
    (target / "final.json").write_text("[]")

    found = utils.find_result_dir(VALID_ID)
    assert found == target


def test_find_result_dir_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "RESULTS_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        utils.find_result_dir(VALID_ID)


def test_find_result_dir_collision_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "RESULTS_ROOT", tmp_path)
    (tmp_path / "pond" / "extraction" / VALID_ID).mkdir(parents=True)
    (tmp_path / "nfix" / "ablation" / VALID_ID).mkdir(parents=True)
    with pytest.raises(ValueError, match="more than one"):
        utils.find_result_dir(VALID_ID)


# ---------------------------------------------------------------------------
# load_model_config
# ---------------------------------------------------------------------------


def test_load_model_config_real_file():
    # Not a fixture -- exercises the real experiments/model-configs/ tree so
    # schema drift between the generator that built it and this reader is
    # actually caught.
    mc = utils.load_model_config("extraction", "gemma-3-27b")
    assert mc["model_id"] == "gaunernst/gemma-3-27b-it-int4-awq"
    assert "serve" in mc and "resources" in mc


def test_load_model_config_missing_raises_with_available_list():
    with pytest.raises(FileNotFoundError, match="Available extraction models"):
        utils.load_model_config("extraction", "not-a-real-model")


# ---------------------------------------------------------------------------
# classify_gpu_need
# ---------------------------------------------------------------------------


def test_classify_frontier_api_model_needs_no_gpu():
    assert utils.classify_gpu_need({"model_id": "gpt-5-mini", "api_base": "https://api.openai.com/v1"}) == "none"


def test_classify_vllm_served_model():
    mc = {"model_id": "x", "serve": {"port": 8081}, "resources": {"gpu_memory": "48G"}}
    assert utils.classify_gpu_need(mc) == "vllm_server"


def test_classify_vllm_served_model_missing_resources_raises():
    mc = {"model_id": "x", "serve": {"port": 8081}}
    with pytest.raises(ValueError, match="resources"):
        utils.classify_gpu_need(mc)


def test_classify_direct_gpu_model():
    mc = {"model_id": "x", "nnsight_kwargs": {"torch_dtype": "bfloat16"}, "resources": {"gpu_memory": "24G"}}
    assert utils.classify_gpu_need(mc) == "direct_gpu"


def test_classify_ambiguous_model_raises():
    # Today's real state for interp_judge/jacobian_lens/representation_lm and
    # the GLiNER baselines -- no api_base, no serve, no resources. Must raise,
    # not silently return "none" (that would qsub with no GPU request at all
    # for a model that needs one).
    mc = {"model_id": "x", "nnsight_kwargs": {"torch_dtype": "bfloat16"}}
    with pytest.raises(ValueError, match="cannot determine"):
        utils.classify_gpu_need(mc)


def test_classify_error_message_includes_source_when_given():
    mc = {"model_id": "x"}
    with pytest.raises(ValueError, match="interp_judge/qwen-2.5-7b.yaml"):
        utils.classify_gpu_need(mc, source="interp_judge/qwen-2.5-7b.yaml")


def test_classify_all_real_model_configs_are_resolvable_or_flagged():
    """Every model-config file in the repo either classifies cleanly, or is
    one of the known-missing-resources files this restructure's own commit
    history documents (interp_judge/*, jacobian_lens/*, representation_lm/*,
    baseline/gliner-*). Catches an accidental new unresolvable model-config
    slipping in silently.
    """
    known_missing = {
        ("baseline", "gliner-large-v1"), ("baseline", "gliner-base-v1"),
        ("interp_judge", "llama-3.1-8b"), ("interp_judge", "mistral-7b"),
        ("interp_judge", "qwen-2.5-7b"), ("interp_judge", "qwen-2.5-7b-base"),
        ("interp_judge", "qwen-2.5-7b-base-cued"), ("interp_judge", "llama-3.1-8b-base-cued"),
        ("jacobian_lens", "llama-3.1-8b-base"),
        ("representation_lm", "llama-3.1-8b-base"),
    }
    for kind_dir in sorted(utils.MODEL_CONFIGS_ROOT.iterdir()):
        if not kind_dir.is_dir():
            continue
        for model_file in sorted(kind_dir.glob("*.yaml")):
            mc = utils.load_model_config(kind_dir.name, model_file.stem)
            key = (kind_dir.name, model_file.stem)
            if key in known_missing:
                with pytest.raises(ValueError):
                    utils.classify_gpu_need(mc, source=f"{kind_dir.name}/{model_file.name}")
            else:
                # Must not raise.
                utils.classify_gpu_need(mc, source=f"{kind_dir.name}/{model_file.name}")
