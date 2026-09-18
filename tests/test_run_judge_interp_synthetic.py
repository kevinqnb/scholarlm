"""Rung-1 unit tests for run_judge_interp.py's synthetic_file mode migration
(2026-09-16): output now goes through the id-addressed
experiments/results/{dataset}/judge_interp/{id}/ tree instead of the old
date-addressed data/experiments/ tree, and params.synthetic_name is retired
(a config that still sets it is a hard error, not a silent no-op).

run_interp_judge itself (which loads a real NNsight model) is monkeypatched
to a stub -- these tests only exercise main()'s param validation and
output_dir resolution, not the judge pass.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "experiments"))

import utils as paths  # noqa: E402
import run_judge_interp  # noqa: E402


def _write_config(tmp_path: Path, exp_id: str, params: dict) -> Path:
    path = tmp_path / f"{exp_id}.yaml"
    cfg = {"id": exp_id, "project": "scholarlm", "description": "test", "seed": 342, "params": params}
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f)
    return path


def test_synthetic_name_is_a_hard_error(tmp_path):
    exp_id = "2026-09-16-test-synthetic-judge-01"
    config_path = _write_config(tmp_path, exp_id, {
        "dataset": "pond", "judge": "qwen-2.5-7b",
        "synthetic_file": "does/not/matter.json",
        "synthetic_name": "some_label",
    })
    with pytest.raises(ValueError, match="synthetic_name is retired"):
        run_judge_interp.main([str(config_path)])


def test_synthetic_file_resolves_id_addressed_output_dir(tmp_path, monkeypatch):
    exp_id = "2026-09-16-test-synthetic-judge-01"
    probe_file = tmp_path / "fixture.json"
    probe_file.write_text("[]")
    config_path = _write_config(tmp_path, exp_id, {
        "dataset": "pond", "judge": "qwen-2.5-7b", "synthetic_file": str(probe_file),
    })

    captured = {}

    def _fake_run_interp_judge(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(run_judge_interp, "run_interp_judge", _fake_run_interp_judge)
    run_judge_interp.main([str(config_path)])

    assert captured["output_dir"] == paths.result_dir("pond", "judge_interp", exp_id)
    assert captured["input_file"] == probe_file
    assert captured.get("extraction_id") is None  # not passed -> run_interp_judge's own default


def test_synthetic_file_missing_input_raises(tmp_path):
    exp_id = "2026-09-16-test-synthetic-judge-01"
    config_path = _write_config(tmp_path, exp_id, {
        "dataset": "pond", "judge": "qwen-2.5-7b", "synthetic_file": "does/not/exist.json",
    })
    with pytest.raises(FileNotFoundError, match="synthetic_file"):
        run_judge_interp.main([str(config_path)])
