"""Rung-1 unit tests for run_judge_combine.py under the 2026-09-27 judge batch:
every production judge_combine config now names 6 judge_ids (3 voting: gpt-oss-120b,
llama-3.3-70b, qwen-2.5-72b + 3 interp: llama-3.1-8b, qwen-2.5-7b, mistral-nemo-12b),
one more non-voting judge than any prior combine run (mistral-7b -> mistral-nemo-12b
swap added a slot, it did not replace one already there). Neither
combine_judge_results nor run_combine hard-codes a judge count anywhere in the
source (checked by inspection before writing this), so this is confirmation, not a
suspected bug: fixtures below hand-compute the expected majority vote for 6 judges
and check run_combine's output matches exactly.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT / "experiments"))

import utils as paths  # noqa: E402
import run_judge_combine as rjc  # noqa: E402


VOTING = ["gpt-oss-120b", "llama-3.3-70b", "qwen-2.5-72b"]
INTERP = ["llama-3.1-8b", "qwen-2.5-7b", "mistral-nemo-12b"]
ALL_SIX = VOTING + INTERP


def _write_responses(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        json.dump(rows, f)


def test_combine_judge_results_six_judges_hand_computed(tmp_path):
    """2 measurements x 6 judges (3 voting, 3 interp). voting_threshold=2, matching
    every production judge_combine config in the 2026-09-27 batch.

    Row 0: 2/3 voting judges say True (gpt-oss-120b, qwen-2.5-72b), llama-3.3-70b
        says False -> affirmative=2 >= threshold=2 -> combined True.
        All 3 interp judges say False -- must NOT count toward the vote.
    Row 1: 1/3 voting judges say True (qwen-2.5-72b only) -> affirmative=1 < 2 ->
        combined False, even though all 3 interp judges say True -- non-voting
        judges must never flip the outcome.
    """
    judge_files: dict[str, Path] = {}
    votes_row0 = {"gpt-oss-120b": True, "llama-3.3-70b": False, "qwen-2.5-72b": True,
                  "llama-3.1-8b": False, "qwen-2.5-7b": False, "mistral-nemo-12b": False}
    votes_row1 = {"gpt-oss-120b": False, "llama-3.3-70b": False, "qwen-2.5-72b": True,
                  "llama-3.1-8b": True, "qwen-2.5-7b": True, "mistral-nemo-12b": True}

    for judge_key in ALL_SIX:
        rows = [
            {"measurement_id": 0, "attribute": "tp", "judgement": votes_row0[judge_key],
             "judgement_model": judge_key, "judgement_prob": 0.9},
            {"measurement_id": 1, "attribute": "tn", "judgement": votes_row1[judge_key],
             "judgement_model": judge_key, "judgement_prob": 0.8},
        ]
        p = tmp_path / f"{judge_key}.json"
        _write_responses(p, rows)
        judge_files[judge_key] = p

    voting_judges = set(VOTING)
    combined = rjc.combine_judge_results(judge_files, voting_judges, voting_threshold=2)
    by_id = {r["measurement_id"]: r for r in combined}

    assert len(combined) == 2
    assert by_id[0]["judgement_combined"] is True
    assert by_id[1]["judgement_combined"] is False

    # Every one of the 6 judges' fields must be present and namespaced correctly,
    # including the new mistral-nemo-12b slot.
    for judge_key in ALL_SIX:
        assert by_id[0][f"judgement_{judge_key}"] == votes_row0[judge_key]
        assert by_id[1][f"judgement_{judge_key}"] == votes_row1[judge_key]

    # Base (non-judge) fields come from whichever judge file is merged first,
    # and are not duplicated/namespaced.
    assert by_id[0]["attribute"] == "tp"
    assert by_id[1]["attribute"] == "tn"


def test_run_combine_end_to_end_six_judges(tmp_path, monkeypatch):
    """Full run_combine() with monkeypatched find_result_dir/load_run_metadata,
    standing in for 6 real judge_local/judge_interp result directories -- confirms
    the id-resolution plumbing (not just combine_judge_results in isolation) has
    no assumption baked in about exactly 2 interp judges.
    """
    fake_dirs: dict[str, Path] = {}
    for i, judge_key in enumerate(ALL_SIX):
        judge_id = f"2026-09-27-fake-judge-{i:02d}-01"
        d = tmp_path / judge_id
        d.mkdir()
        _write_responses(d / "responses.json", [
            {"measurement_id": 0, "attribute": "tp", "judgement": True,
             "judgement_model": judge_key, "judgement_prob": 0.9},
        ])
        with open(d / "run_metadata.json", "w") as f:
            json.dump({"judge_model": judge_key}, f)
        fake_dirs[judge_id] = d

    def _fake_find_result_dir(experiment_id):
        return fake_dirs[experiment_id]

    monkeypatch.setattr(paths, "find_result_dir", _fake_find_result_dir)
    monkeypatch.setattr(rjc.paths, "find_result_dir", _fake_find_result_dir)

    output_dir = tmp_path / "out"
    all_true_votes = list(fake_dirs.keys())
    result_path = rjc.run_combine(all_true_votes, output_dir, voting_threshold=2)

    with open(result_path) as f:
        combined = json.load(f)
    assert len(combined) == 1
    assert combined[0]["judgement_combined"] is True
    for judge_key in ALL_SIX:
        assert combined[0][f"judgement_{judge_key}"] is True


def test_run_combine_raises_with_no_voting_judge(tmp_path, monkeypatch):
    """If judge_ids resolves to interp-only judges (no voting judge present at
    all), run_combine must fail loud rather than silently produce an
    unfounded ground-truth label -- this is the one real hard constraint on
    a judge_ids list, distinct from "how many interp judges" (unconstrained).
    """
    fake_dirs: dict[str, Path] = {}
    for i, judge_key in enumerate(INTERP):
        judge_id = f"2026-09-27-fake-interp-only-{i:02d}-01"
        d = tmp_path / judge_id
        d.mkdir()
        _write_responses(d / "responses.json", [
            {"measurement_id": 0, "judgement": True, "judgement_model": judge_key},
        ])
        with open(d / "run_metadata.json", "w") as f:
            json.dump({"judge_model": judge_key}, f)
        fake_dirs[judge_id] = d

    monkeypatch.setattr(rjc.paths, "find_result_dir", lambda eid: fake_dirs[eid])

    with pytest.raises(ValueError, match="Cannot compute ground truth"):
        rjc.run_combine(list(fake_dirs.keys()), tmp_path / "out")
