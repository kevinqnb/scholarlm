"""Rung-1 unit tests for experiments/_resolve_job.py's CLI emission layer.

utils.resolve_job() itself is covered by tests/test_utils_resolve_job.py --
this file covers the thin shell-eval-line-printing wrapper on top of it,
including gpu_type: emitted only when a model-config's resources: block sets
one (most models don't need to pin a GPU type, only a capability floor).
"""
import importlib.util
import io
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import utils

_spec = importlib.util.spec_from_file_location("_resolve_job", _REPO_ROOT / "experiments" / "_resolve_job.py")
_resolve_job = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_resolve_job)


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def _run_and_capture(argv: list[str]) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = _resolve_job.main(argv)
    return rc, buf.getvalue()


@pytest.fixture
def fixture_roots(tmp_path, monkeypatch):
    exp_root = tmp_path / "experiment-configs"
    model_root = tmp_path / "model-configs"
    monkeypatch.setattr(utils, "EXPERIMENT_CONFIGS_ROOT", exp_root)
    monkeypatch.setattr(utils, "MODEL_CONFIGS_ROOT", model_root)
    return exp_root, model_root


def test_gpu_type_emitted_when_present(fixture_roots):
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-13-test-gputype-01"
    _write_yaml(
        exp_root / "pond" / "judge_interp" / exp_id / f"{exp_id}.yaml",
        {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342,
         "params": {"dataset": "pond", "judge": "qwen-2.5-7b", "extraction_id": "x",
                     "walltime": "24:00:00"}},
    )
    _write_yaml(model_root / "interp_judge" / "qwen-2.5-7b.yaml", {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "nnsight_kwargs": {"torch_dtype": "bfloat16"},
        "resources": {"gpu_memory": "32G", "gpu_capability": 7.0, "gpu_type": "L40S", "omp": 8},
    })
    rc, out = _run_and_capture([exp_id])
    assert rc == 0
    assert "GPU_TYPE=L40S" in out
    assert "GPU_C=7.0" in out
    assert "WALLTIME=24:00:00" in out


def test_gpu_type_omitted_when_absent(fixture_roots):
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-13-test-nogputype-01"
    _write_yaml(
        exp_root / "pond" / "extraction" / exp_id / f"{exp_id}.yaml",
        {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342,
         "params": {"dataset": "pond", "model": "gemma-3-27b", "walltime": "48:00:00"}},
    )
    _write_yaml(model_root / "extraction" / "gemma-3-27b.yaml", {
        "model_id": "gaunernst/gemma-3-27b-it-int4-awq",
        "serve": {"port": 8081, "max_model_len": 90000, "gpu_memory_utilization": 0.9,
                  "quantization": "awq_marlin", "dtype": "bfloat16", "sif_image": "x.sif"},
        "resources": {"gpu_memory": "48G", "gpu_capability": "8.9", "omp": 8},
    })
    rc, out = _run_and_capture([exp_id])
    assert rc == 0
    assert "GPU_TYPE" not in out


def test_walltime_required_for_vllm_server_job(fixture_roots):
    # walltime used to fall back to the model-config's resources.walltime for
    # GPU jobs -- now it's required in params, same as the frontier path,
    # since it's a function of (model x dataset size x experiment type), not
    # just the model.
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-13-test-missing-walltime-01"
    _write_yaml(
        exp_root / "pond" / "extraction" / exp_id / f"{exp_id}.yaml",
        {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342,
         "params": {"dataset": "pond", "model": "gemma-3-27b"}},
    )
    _write_yaml(model_root / "extraction" / "gemma-3-27b.yaml", {
        "model_id": "gaunernst/gemma-3-27b-it-int4-awq",
        "serve": {"port": 8081, "max_model_len": 90000, "gpu_memory_utilization": 0.9,
                  "quantization": "awq_marlin", "dtype": "bfloat16", "sif_image": "x.sif"},
        "resources": {"gpu_memory": "48G", "gpu_capability": "8.9", "omp": 8},
    })
    rc, out = _run_and_capture([exp_id])
    assert rc == 0  # _fail prints an `exit 1` eval line rather than raising, see _resolve_job.py
    assert "params.walltime is required" in out


def test_walltime_required_for_frontier_job(fixture_roots):
    exp_root, model_root = fixture_roots
    exp_id = "2026-09-13-test-missing-walltime-frontier-01"
    _write_yaml(
        exp_root / "pond" / "extraction" / exp_id / f"{exp_id}.yaml",
        {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342,
         "params": {"dataset": "pond", "model": "gpt-5-mini"}},
    )
    _write_yaml(model_root / "extraction" / "gpt-5-mini.yaml", {
        "model_id": "gpt-5-mini", "api_base": "https://api.openai.com/v1",
    })
    rc, out = _run_and_capture([exp_id])
    assert rc == 0
    assert "params.walltime is required" in out


def test_real_qwen_2_5_7b_model_config_emits_l40s():
    # Exercises the real, committed model-config (not a fixture) -- catches
    # drift between this test and the actual file filled in for Phase C.
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        exp_id = "2026-09-13-test-real-qwen-01"
        _write_yaml(
            td / f"{exp_id}.yaml",
            {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342,
             "params": {"dataset": "pond", "judge": "qwen-2.5-7b", "extraction_id": "x"}},
        )
        import utils as real_utils
        cfg_path = td / f"{exp_id}.yaml"

        # Bypass find_experiment_config (real EXPERIMENT_CONFIGS_ROOT wouldn't
        # have this fixture) by resolving the job directly against the real
        # model-configs/ tree.
        cfg = real_utils.load_experiment_config(cfg_path)
        params = cfg["params"]
        model_config = real_utils.load_model_config("interp_judge", params["judge"])
        gpu_need = real_utils.classify_gpu_need(model_config, source="interp_judge/qwen-2.5-7b.yaml")
        assert gpu_need == "direct_gpu"
        assert model_config["resources"]["gpu_type"] == "L40S"
