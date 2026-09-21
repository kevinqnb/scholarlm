"""Unit tests for experiments/utils.py's write_run_metadata step_seconds handling.

write_run_metadata used to overwrite run_metadata.json wholesale on every
call, so a checkpointed pipeline resumed across separate script invocations
(e.g. after a killed SGE job) would report the last invocation's wall-clock
time as if it were the whole run's -- silently wrong in the direction
CLAUDE.md calls out (runs clean, produces a number that's wrong). These tests
verify the fix: step_seconds is merged by step name across calls instead of
replaced, and total_step_seconds -- the resume-safe total -- sums only the
steps that actually ran.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))

from utils import write_run_metadata


def test_step_seconds_merges_across_calls_instead_of_overwriting(tmp_path):
    write_run_metadata(
        tmp_path,
        step_seconds={
            "entities": {"ran": True, "seconds": 10.0},
            "attributes": {"ran": True, "seconds": 5.0},
        },
    )

    # Simulates a resumed second invocation: entities/attributes were already
    # on disk and skipped this time (ran=False), only entity_prov is new.
    write_run_metadata(
        tmp_path,
        step_seconds={
            "entities": {"ran": False, "seconds": None},
            "attributes": {"ran": False, "seconds": None},
            "entity_prov": {"ran": True, "seconds": 3.0},
        },
    )

    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    # entities/attributes keep their real first-invocation timings: being
    # skipped (ran=False) on the second call must not erase them.
    assert metadata["step_seconds"] == {
        "entities": {"ran": True, "seconds": 10.0},
        "attributes": {"ran": True, "seconds": 5.0},
        "entity_prov": {"ran": True, "seconds": 3.0},
    }
    # total_step_seconds must reflect the *original* 10.0/5.0 timings (from
    # the first invocation, merged in) plus the new 3.0 -- not just the
    # second call's own numbers, and not double-counted.
    assert metadata["total_step_seconds"] == 18.0


def test_step_seconds_first_time_skipped_with_no_prior_record_stays_skipped(tmp_path):
    """A step skipped on the very first recorded invocation (e.g. its
    checkpoint predates this instrumentation) has no real timing to preserve
    -- it should land in step_seconds as ran=False, not be silently dropped."""
    write_run_metadata(
        tmp_path,
        step_seconds={"entities": {"ran": False, "seconds": None}},
    )
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["step_seconds"] == {"entities": {"ran": False, "seconds": None}}
    assert metadata["total_step_seconds"] == 0.0


def test_total_step_seconds_excludes_skipped_steps(tmp_path):
    write_run_metadata(
        tmp_path,
        step_seconds={
            "entities": {"ran": True, "seconds": 10.0},
            "attributes": {"ran": False, "seconds": None},
        },
    )
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["total_step_seconds"] == 10.0


def test_runtime_seconds_and_step_seconds_are_independent(tmp_path):
    write_run_metadata(
        tmp_path,
        start_time=__import__("time").time() - 2.0,
        step_seconds={"final": {"ran": True, "seconds": 1.5}},
    )
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["runtime_seconds"] >= 2.0
    assert metadata["total_step_seconds"] == 1.5


def test_no_step_seconds_argument_leaves_field_absent(tmp_path):
    write_run_metadata(tmp_path, dataset="pond")
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert "step_seconds" not in metadata
    assert "total_step_seconds" not in metadata


def test_token_usage_accumulates_across_invocations(tmp_path):
    """A fresh MeasurementLM per script invocation only knows about calls it
    made itself -- token_usage must sum across invocations on the same
    output_dir, the same resume hazard as step_seconds."""
    write_run_metadata(
        tmp_path,
        token_usage={"prompt_tokens": 100, "completion_tokens": 20, "successful_calls": 2, "failed_calls": 0},
    )
    write_run_metadata(
        tmp_path,
        token_usage={"prompt_tokens": 50, "completion_tokens": 10, "successful_calls": 1, "failed_calls": 1},
    )
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["token_usage"] == {
        "prompt_tokens": 150,
        "completion_tokens": 30,
        "successful_calls": 3,
        "failed_calls": 1,
    }


def test_max_prompt_tokens_takes_max_across_invocations_not_last_write(tmp_path):
    write_run_metadata(tmp_path, max_prompt_tokens=5000)
    write_run_metadata(tmp_path, max_prompt_tokens=1200)  # resumed run, smaller steps only
    metadata = json.loads((tmp_path / "run_metadata.json").read_text())
    assert metadata["max_prompt_tokens"] == 5000
