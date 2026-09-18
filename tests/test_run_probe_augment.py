"""Rung-1 unit tests for experiments/run_probe_augment.py.

build_augment_argv is checked against gen_augment.sh's own historical
per-mode invocations (pond / nfix / supermat / rung3) verbatim, so this
runner is a faithful replacement, not a reinterpretation. main()'s dispatch
is tested by mocking run_probe_augment() itself (the orchestrator) rather
than subprocess.run directly -- subprocess.check_output (used internally by
utils.write_run_metadata's get_git_info()) is implemented as a wrapper
around subprocess.run in the stdlib, so patching subprocess.run globally
silently corrupts unrelated git-info gathering instead of raising, which
this file's development surfaced the hard way.
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import utils
import run_probe_augment as rpa


def _flagset(argv: list[str]) -> dict[str, list[str]]:
    """argv (script path first) -> {flag: [values]}, order-independent."""
    d: dict[str, list[str]] = {}
    i = 1
    while i < len(argv):
        if argv[i].startswith("--"):
            flag = argv[i]
            vals = []
            j = i + 1
            while j < len(argv) and not argv[j].startswith("--"):
                vals.append(argv[j])
                j += 1
            d[flag] = vals
            i = j
        else:
            i += 1
    return d


# ---------------------------------------------------------------------------
# build_augment_argv vs. gen_augment.sh's real historical invocations
# ---------------------------------------------------------------------------


def test_argv_matches_gen_augment_pond_mode():
    argv = rpa.build_augment_argv("pond", 42, True, 3, "http://localhost:8081/v1", {})
    assert _flagset(argv) == {
        "--augment": [], "--seed": ["42"], "--augment-prompt-budget-multiple": ["3"],
        "--gpt-oss-api-base": ["http://localhost:8081/v1"], "--reviewed": [],
    }


def test_argv_matches_gen_augment_nfix_mode():
    argv = rpa.build_augment_argv(
        "nfix", 42, True, 10, "http://localhost:8081/v1",
        {"valid_floor": 1200, "diag_valid_floor": 650},
    )
    assert _flagset(argv) == {
        "--augment": [], "--seed": ["42"], "--augment-prompt-budget-multiple": ["10"],
        "--gpt-oss-api-base": ["http://localhost:8081/v1"], "--reviewed": [],
        "--augment-valid-floor": ["1200"], "--augment-diag-valid-floor": ["650"],
    }


def test_argv_matches_gen_augment_supermat_mode_no_reviewed():
    # supermat has no ground_truth_review.json -- gen_augment.sh's run_ds
    # deliberately omits --reviewed only for this dataset.
    argv = rpa.build_augment_argv(
        "supermat", 42, False, 12, "http://localhost:8081/v1", {"valid_floor": 1200}
    )
    flags = _flagset(argv)
    assert "--reviewed" not in flags
    assert flags == {
        "--augment": [], "--seed": ["42"], "--augment-prompt-budget-multiple": ["12"],
        "--gpt-oss-api-base": ["http://localhost:8081/v1"], "--augment-valid-floor": ["1200"],
    }


def test_argv_matches_gen_augment_rung3_mode():
    argv = rpa.build_augment_argv(
        "pond", 42, True, 1, "http://localhost:8081/v1",
        {"sample_gt": 80, "valid_floor": 0, "diag_valid_floor": 0, "out_suffix": "_rung3"},
    )
    assert _flagset(argv) == {
        "--augment": [], "--seed": ["42"], "--augment-prompt-budget-multiple": ["1"],
        "--gpt-oss-api-base": ["http://localhost:8081/v1"], "--reviewed": [],
        "--augment-sample-gt": ["80"], "--augment-valid-floor": ["0"],
        "--augment-diag-valid-floor": ["0"], "--augment-out-suffix": ["_rung3"],
    }


def test_argv_omits_optional_flags_when_absent_from_params():
    # No duplicated defaults -- create_probe_dataset.py's own defaults apply
    # for anything not explicitly set in params.
    argv = rpa.build_augment_argv("pond", 42, True, 1, "http://localhost:8081/v1", {})
    flags = _flagset(argv)
    for absent in ("--augment-valid-floor", "--augment-diag-valid-floor",
                   "--augment-sample-gt", "--augment-out-suffix", "--augment-stub"):
        assert absent not in flags


def test_argv_stub_and_pos_axes_flags():
    argv = rpa.build_augment_argv(
        "pond", 42, True, 1, "http://localhost:8081/v1",
        {"stub": True, "pos_axes": ["change_value", "change_entity"]},
    )
    flags = _flagset(argv)
    assert "--augment-stub" in flags
    assert flags["--augment-pos-axes"] == ["change_value", "change_entity"]


# ---------------------------------------------------------------------------
# main() dispatch
# ---------------------------------------------------------------------------


def _write_cfg(path: Path, exp_id: str, seed: int, params: dict) -> Path:
    cfg_path = path / f"{exp_id}.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(
            {"id": exp_id, "project": "scholarlm", "description": "test", "seed": seed, "params": params},
            f,
        )
    return cfg_path


def test_main_dispatches_with_resolved_params_and_output_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(rpa.paths, "RESULTS_ROOT", tmp_path / "results")
    cfg_path = _write_cfg(
        tmp_path, "2026-09-12-test-augment-01", 42,
        {"dataset": "pond", "model": "gpt-oss-120b", "reviewed": True, "prompt_budget_multiple": 3},
    )
    with patch("run_probe_augment.run_probe_augment") as m:
        rpa.main([str(cfg_path)])
        kw = m.call_args.kwargs
        assert kw["dataset"] == "pond"
        assert kw["seed"] == 42
        assert kw["reviewed"] is True
        assert kw["prompt_budget_multiple"] == 3
        assert str(kw["output_dir"]).endswith("results/pond/probe_augment/2026-09-12-test-augment-01")


def test_main_rejects_unsupported_dataset(tmp_path):
    cfg_path = _write_cfg(
        tmp_path, "2026-09-12-test-augment-02", 42,
        {"dataset": "measeval", "model": "gpt-oss-120b", "reviewed": False, "prompt_budget_multiple": 1},
    )
    with pytest.raises(ValueError, match="no create_probe_dataset.py"):
        rpa.main([str(cfg_path)])


def test_main_requires_prompt_budget_multiple(tmp_path):
    cfg_path = _write_cfg(
        tmp_path, "2026-09-12-test-augment-03", 42,
        {"dataset": "pond", "model": "gpt-oss-120b", "reviewed": True},  # missing prompt_budget_multiple
    )
    with pytest.raises(ValueError, match="missing required params key"):
        rpa.main([str(cfg_path)])


def test_main_does_not_enforce_repo_seed_consistency(tmp_path, monkeypatch):
    # Unlike every other Tier-1 runner: probe augmentation legitimately uses
    # its own per-run seed (historically 42, not the repo-wide 342), so a
    # config declaring seed: 42 must NOT be rejected the way run_extraction.py
    # etc. would reject a seed mismatch.
    monkeypatch.setattr(utils, "RESULTS_ROOT", tmp_path / "results")
    monkeypatch.setattr(rpa.paths, "RESULTS_ROOT", tmp_path / "results")
    cfg_path = _write_cfg(
        tmp_path, "2026-09-12-test-augment-04", 42,
        {"dataset": "pond", "model": "gpt-oss-120b", "reviewed": True, "prompt_budget_multiple": 1},
    )
    with patch("run_probe_augment.run_probe_augment") as m:
        rpa.main([str(cfg_path)])  # must not raise
        assert m.call_args.kwargs["seed"] == 42
