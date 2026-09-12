"""Unit tests for experiments/paths.py's ablation-aware judge-output readers.

find_activations / find_layer_outputs / find_judge_responses previously lacked
the `ablation` parameter that judge()/judge_base() (and find_extraction_final/
find_combined) already had, so they couldn't read
ablations/ablation{N}/{model}/{date}/judge/{judge_model}/{judge_date}/ trees.
These tests build both a non-ablation and an ablation-1 judge tree under a
tmp_path EXPERIMENTS_ROOT and check both the resolved path and the
latest-date-wins behavior are unchanged/correct in each tree.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import paths


@pytest.fixture
def fake_root(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "EXPERIMENTS_ROOT", tmp_path)
    return tmp_path


def _write_judge_run(base, judge_model, judge_date, *, with_activations=True):
    d = base / judge_model / judge_date
    d.mkdir(parents=True)
    (d / "responses.json").write_text("[]")
    if with_activations:
        np.savez(d / "attention_outputs.npz", x=np.zeros(1))
        np.savez(d / "layer_outputs.npz", x=np.zeros(1))
    return d


def test_find_activations_no_ablation(fake_root):
    base = fake_root / "pond" / "judge" / "gemma-3-27b" / "2026_05_05"
    _write_judge_run(base, "qwen-2.5-7b", "2026_05_06")
    path = paths.find_activations("pond", "gemma-3-27b", "2026_05_05", "qwen-2.5-7b")
    assert path == base / "qwen-2.5-7b" / "2026_05_06" / "attention_outputs.npz"


def test_find_activations_with_ablation(fake_root):
    base = (
        fake_root / "pond" / "ablations" / "ablation1"
        / "gpt-oss-120b" / "2026_05_03" / "judge"
    )
    _write_judge_run(base, "qwen-2.5-7b", "2026_09_11")
    path = paths.find_activations(
        "pond", "gpt-oss-120b", "2026_05_03", "qwen-2.5-7b", ablation="1"
    )
    assert path == base / "qwen-2.5-7b" / "2026_09_11" / "attention_outputs.npz"


def test_find_activations_ablation_latest_date_wins(fake_root):
    base = (
        fake_root / "pond" / "ablations" / "ablation1"
        / "gpt-oss-120b" / "2026_05_03" / "judge"
    )
    _write_judge_run(base, "qwen-2.5-7b", "2026_05_09")  # stale, pre-full-paper judge
    _write_judge_run(base, "qwen-2.5-7b", "2026_09_11")  # fresh
    path = paths.find_activations(
        "pond", "gpt-oss-120b", "2026_05_03", "qwen-2.5-7b", ablation="1"
    )
    assert path.parent.name == "2026_09_11"


def test_find_activations_ablation_and_no_ablation_dont_collide(fake_root):
    # Same (dataset, model, date, judge) string but one lives under ablations/,
    # the other under the flat judge/ tree -- must not find the wrong one.
    real_base = fake_root / "pond" / "judge" / "gpt-oss-120b" / "2026_05_03"
    abl_base = (
        fake_root / "pond" / "ablations" / "ablation1"
        / "gpt-oss-120b" / "2026_05_03" / "judge"
    )
    _write_judge_run(real_base, "qwen-2.5-7b", "2026_05_04")
    _write_judge_run(abl_base, "qwen-2.5-7b", "2026_09_11")

    real_path = paths.find_activations("pond", "gpt-oss-120b", "2026_05_03", "qwen-2.5-7b")
    abl_path = paths.find_activations(
        "pond", "gpt-oss-120b", "2026_05_03", "qwen-2.5-7b", ablation="1"
    )
    assert real_path.parent.name == "2026_05_04"
    assert abl_path.parent.name == "2026_09_11"
    assert real_path != abl_path


def test_find_layer_outputs_with_ablation(fake_root):
    base = (
        fake_root / "nfix" / "ablations" / "ablation2"
        / "gemma-3-27b" / "2026_04_01" / "judge"
    )
    _write_judge_run(base, "mistral-7b", "2026_09_11")
    path = paths.find_layer_outputs(
        "nfix", "gemma-3-27b", "2026_04_01", "mistral-7b", ablation="2"
    )
    assert path == base / "mistral-7b" / "2026_09_11" / "layer_outputs.npz"


def test_find_judge_responses_with_ablation_pins_date_and_returns_it(fake_root):
    base = (
        fake_root / "pond" / "ablations" / "ablation1"
        / "gpt-oss-120b" / "2026_05_03" / "judge"
    )
    _write_judge_run(base, "gpt-oss-120b", "2026_05_09", with_activations=False)
    _write_judge_run(base, "gpt-oss-120b", "2026_09_11", with_activations=False)

    path, resolved_date = paths.find_judge_responses(
        "pond", "gpt-oss-120b", "2026_05_03", "gpt-oss-120b",
        judge_date="2026_05_09", ablation="1",
    )
    assert resolved_date == "2026_05_09"
    assert path == base / "gpt-oss-120b" / "2026_05_09" / "responses.json"


def test_find_activations_missing_ablation_dir_raises(fake_root):
    with pytest.raises(FileNotFoundError):
        paths.find_activations("pond", "gpt-oss-120b", "2026_05_03", "qwen-2.5-7b", ablation="1")
