"""Rung-1 unit tests for experiments/utils.py's experiment-config loading and
job resolution (find_experiment_config, load_experiment_config, resolve_job).

Hand-built fixture trees under tmp_path, monkeypatching
EXPERIMENT_CONFIGS_ROOT and MODEL_CONFIGS_ROOT so no real repo data is
touched. Exercises every EXPERIMENT_TYPES branch: vllm_server, direct_gpu,
none-with-model (frontier), and none-without-model (no model at all).
"""
import sys
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import utils


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _write_experiment_config(root: Path, dataset: str, exp_type: str, exp_id: str, params: dict, **extra) -> Path:
    path = root / dataset / exp_type / exp_id / f"{exp_id}.yaml"
    cfg = {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342, "params": params, **extra}
    _write_yaml(path, cfg)
    return path


@pytest.fixture
def fixture_roots(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment-configs"
    model_root = tmp_path / "model-configs"
    monkeypatch.setattr(utils, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(utils, "MODEL_CONFIGS_ROOT", model_root)
    return exp_root, model_root


# ---------------------------------------------------------------------------
# find_experiment_config
# ---------------------------------------------------------------------------


def test_find_experiment_config_locates_by_id(fixture_roots):
    exp_root, _ = fixture_roots
    exp_id = "2026-09-12-test-extraction-01"
    path = _write_experiment_config(exp_root, "pond", "extraction", exp_id, {"model": "gemma-3-27b"})
    assert utils.find_experiment_config(exp_id) == path


def test_find_experiment_config_missing_raises(fixture_roots):
    with pytest.raises(FileNotFoundError):
        utils.find_experiment_config("2026-09-12-nope-01")


def test_find_experiment_config_rejects_bad_id(fixture_roots):
    with pytest.raises(ValueError):
        utils.find_experiment_config("not-an-id")


# ---------------------------------------------------------------------------
# load_experiment_config
# ---------------------------------------------------------------------------


def test_load_experiment_config_valid(fixture_roots):
    exp_root, _ = fixture_roots
    exp_id = "2026-09-12-test-extraction-01"
    path = _write_experiment_config(exp_root, "pond", "extraction", exp_id, {"model": "gemma-3-27b"})
    cfg = utils.load_experiment_config(path)
    assert cfg["id"] == exp_id
    assert cfg["params"]["model"] == "gemma-3-27b"


def test_load_experiment_config_missing_required_key_raises(tmp_path):
    path = tmp_path / "2026-09-12-bad-01.yaml"
    _write_yaml(path, {"id": "2026-09-12-bad-01", "project": "scholarlm", "params": {}})
    with pytest.raises(ValueError, match="missing required key"):
        utils.load_experiment_config(path)


def test_load_experiment_config_id_mismatch_raises(tmp_path):
    path = tmp_path / "2026-09-12-bad-01.yaml"
    _write_yaml(path, {"id": "2026-09-12-other-01", "project": "scholarlm",
                        "description": "x", "seed": 342, "params": {}})
    with pytest.raises(ValueError, match="does not match filename stem"):
        utils.load_experiment_config(path)


def test_load_experiment_config_params_not_mapping_raises(tmp_path):
    path = tmp_path / "2026-09-12-bad-01.yaml"
    _write_yaml(path, {"id": "2026-09-12-bad-01", "project": "scholarlm",
                        "description": "x", "seed": 342, "params": ["not", "a", "dict"]})
    with pytest.raises(ValueError, match="params must be a mapping"):
        utils.load_experiment_config(path)


# ---------------------------------------------------------------------------
# resolve_job
# ---------------------------------------------------------------------------


def test_resolve_job_vllm_server_type(fixture_roots):
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-12-test-extraction-01"
    _write_experiment_config(exp_root, "pond", "extraction", exp_id, {"model": "gemma-3-27b"})
    _write_yaml(model_root / "extraction" / "gemma-3-27b.yaml", {
        "model_id": "gaunernst/gemma-3-27b-it-int4-awq",
        "serve": {"port": 8081, "max_model_len": 90000, "gpu_memory_utilization": 0.9,
                  "quantization": "awq_marlin", "dtype": "bfloat16", "sif_image": "x.sif"},
        "resources": {"gpu_memory": "48G", "gpu_capability": "8.9", "walltime": "48:00:00", "omp": 8},
    })

    job = utils.resolve_job(exp_id)
    assert job["dataset"] == "pond"
    assert job["experiment_type"] == "extraction"
    assert job["runner"] == "run_extraction.py"
    assert job["gpu_need"] == "vllm_server"
    assert job["model"] == "gemma-3-27b"
    assert job["model_config"]["model_id"] == "gaunernst/gemma-3-27b-it-int4-awq"


def test_resolve_job_direct_gpu_type(fixture_roots):
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-12-test-judgeinterp-01"
    _write_experiment_config(exp_root, "pond", "judge_interp", exp_id, {"model": "qwen-2.5-7b"})
    _write_yaml(model_root / "interp_judge" / "qwen-2.5-7b.yaml", {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "nnsight_kwargs": {"torch_dtype": "bfloat16"},
        "resources": {"gpu_memory": "24G", "gpu_capability": "8.0", "walltime": "12:00:00", "omp": 4},
    })

    job = utils.resolve_job(exp_id)
    assert job["gpu_need"] == "direct_gpu"
    assert job["runner"] == "run_judge_interp.py"


def test_resolve_job_frontier_model_needs_no_gpu(fixture_roots):
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-12-test-gptmini-01"
    _write_experiment_config(exp_root, "pond", "extraction", exp_id,
                              {"model": "gpt-5-mini", "walltime": "6:00:00"})
    _write_yaml(model_root / "extraction" / "gpt-5-mini.yaml", {
        "model_id": "gpt-5-mini", "api_base": "https://api.openai.com/v1",
    })

    job = utils.resolve_job(exp_id)
    assert job["gpu_need"] == "none"
    assert job["model"] == "gpt-5-mini"


def test_resolve_job_no_model_type(fixture_roots):
    exp_root, _ = fixture_roots
    exp_id = "2026-09-12-test-combine-01"
    _write_experiment_config(exp_root, "pond", "judge_combine",
                              exp_id, {"extraction_model": "gemma-3-27b"})

    job = utils.resolve_job(exp_id)
    assert job["gpu_need"] == "none"
    assert job["model_config"] is None
    assert "model" not in job
    assert job["runner"] == "run_judge_combine.py"


def test_resolve_job_unknown_experiment_type_raises(fixture_roots):
    exp_root, _ = fixture_roots
    exp_id = "2026-09-12-test-weird-01"
    _write_experiment_config(exp_root, "pond", "some_composite_type", exp_id, {})
    with pytest.raises(ValueError, match="no run_\\{type\\}.py automation"):
        utils.resolve_job(exp_id)


def test_resolve_job_missing_model_param_raises(fixture_roots):
    exp_root, _ = fixture_roots
    exp_id = "2026-09-12-test-extraction-01"
    _write_experiment_config(exp_root, "pond", "extraction", exp_id, {})  # no model
    with pytest.raises(ValueError, match="params.model is required"):
        utils.resolve_job(exp_id)


def test_resolve_job_missing_resources_propagates_from_classify(fixture_roots):
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-12-test-judgeinterp-01"
    _write_experiment_config(exp_root, "pond", "judge_interp", exp_id, {"model": "qwen-2.5-7b"})
    _write_yaml(model_root / "interp_judge" / "qwen-2.5-7b.yaml", {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "nnsight_kwargs": {"torch_dtype": "bfloat16"},
        # no resources: block -- today's real state for this exact model
    })
    with pytest.raises(ValueError, match="cannot determine"):
        utils.resolve_job(exp_id)


# ---------------------------------------------------------------------------
# require_params -- used by every run_{type}.py runner's main() now that
# individual CLI flags are gone and params come only from a config file.
# ---------------------------------------------------------------------------


def test_require_params_all_present_is_noop():
    utils.require_params({"dataset": "pond", "model": "x"}, "dataset", "model")


def test_require_params_missing_raises_with_names():
    with pytest.raises(ValueError, match=r"\['model', 'date'\]"):
        utils.require_params({"dataset": "pond"}, "dataset", "model", "date")


def test_require_params_includes_config_path_in_message():
    with pytest.raises(ValueError, match="myconfig.yaml"):
        utils.require_params({}, "dataset", config_path="myconfig.yaml")
